# date = 2023-03-08
# author = liuwei

import logging
import os
import json
import pickle
import math
import random
import time
import datetime
from tqdm import tqdm, trange

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.models.bert import BertConfig, BertTokenizer
from transformers.models.roberta import RobertaConfig, RobertaTokenizer
from utils import cal_acc_f1_score_with_ids, get_onehot_conn_from_vocab, labels_from_file, get_connectives_from_file, cal_acc_f1_score_per_label
from task_dataset import TwoDataset
from model import TwoEncoder

# set logger, print to console and write to file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
BASIC_FORMAT = "%(asctime)s:%(levelname)s: %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()  # 输出到控制台的handler
chlr.setFormatter(formatter)
logger.addHandler(chlr)

# for output
dt = datetime.datetime.now()
TIME_CHECKPOINT_DIR = "checkpoint_{}-{}-{}_{}:{}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
PREFIX_CHECKPOINT_DIR = "checkpoint"

def get_argparse():
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument("--data_dir", default="data/dataset", type=str)
    parser.add_argument("--dataset", default="pdtb2", type=str, help="pdtb2, pdtb3")
    parser.add_argument("--output_dir", default="data/result", type=str)
    parser.add_argument("--encoder_type", default="roberta", type=str, help="roberta")
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str, help="roberta-base")

    # hyperparameters
    parser.add_argument("--relation_type", default="implicit", type=str)
    parser.add_argument("--label_file", default="labels_level_1.txt", type=str, help="the label file path")
    parser.add_argument("--loss_ratio", default=0.5, type=float, help="from 0 to 1.0")

    # for training
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_dev", default=False, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument("--train_batch_size", default=2, type=int)
    parser.add_argument("--eval_batch_size", default=24, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--use_conn", default=False, action="store_true")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="training epoch")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="learning rate")
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout")
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--max_grad_norm", default=2.0, type=float)
    parser.add_argument("--seed", default=106524, type=int, help="random seed")

    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloader(dataset, args, mode="train"):
    print("Dataset length: ", len(dataset))
    if mode.lower() == "train":
        sampler = RandomSampler(dataset)
        batch_size = args.train_batch_size
    else:
        sampler = SequentialSampler(dataset)
        batch_size = args.eval_batch_size
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler
    )
    return data_loader


def get_optimizer(model, args, num_training_steps):
    no_deday = ["bias", "LayerNorm.weigh"]
    specific_params = ["classifier"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_deday)],
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_deday)],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(num_training_steps * args.warmup_ratio),
        num_training_steps=num_training_steps
    )
    return optimizer, scheduler


def train(model, args, train_dataloader, dev_dataloader, test_dataloader, conn_list, label_list, tokenizer):
    ## 1. prepare data
    t_total = int(len(train_dataloader) * args.num_train_epochs)
    num_train_epochs = args.num_train_epochs
    print_step = int(len(train_dataloader) // 4)

    ## 2.optimizer
    optimizer, scheduler = get_optimizer(model, args, t_total)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader.dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Batch size per device = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    logging_loss = 0.0
    best_dev = 0.0
    best_dev_epoch = 0
    best_test = 0.0
    best_test_epoch = 0
    train_iterator = trange(1, int(num_train_epochs) + 1, desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        model.train()
        model.zero_grad()
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            global_step += 1
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "mask_position_ids": batch[3],
                "conn_ids": batch[4],
                "labels": batch[5],
                "flag": "Train"
            }
            outputs = model(**inputs)
            conn_loss = outputs[1]
            rel_loss = outputs[2]
            loss = conn_loss * args.loss_ratio + rel_loss

            optimizer.zero_grad()
            loss.backward()
            logging_loss = loss.item() * args.train_batch_size
            tr_loss += logging_loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            if global_step % print_step == 0:
                print(" Current conn_loss=%.4f, rel_loss=%.4f, loss=%.4f, global average loss=%.4f" % (
                conn_loss.item() * args.train_batch_size, rel_loss.item() * args.train_batch_size, logging_loss, tr_loss / global_step))

        # evaluation and save
        model.eval()
        dev_conn_acc, dev_acc, dev_p, dev_r, dev_f1 = evaluate(model, args, dev_dataloader, conn_list, label_list, tokenizer, epoch, desc="dev")
        test_conn_acc, test_acc, test_p, test_r, test_f1 = evaluate(model, args, test_dataloader, conn_list, label_list, tokenizer, epoch, desc="test")
        print()
        print(" Dev conn_acc=%.4f, acc=%.4f, p=%.4f, r=%.4f, f1=%.4f"%(dev_conn_acc, dev_acc, dev_p, dev_r, dev_f1))
        print(" Test conn_acc=%.4f, acc=%.4f, p=%.4f, r=%.4f, f1=%.4f"%(test_conn_acc, test_acc, test_p, test_r, test_f1))
        if dev_acc > best_dev:
            best_dev = dev_acc
            best_dev_epoch = epoch

        if test_acc > best_test:
            best_test = test_acc
            best_test_epoch = epoch

        # output_dir = os.path.join(args.output_dir, TIME_CHECKPOINT_DIR)
        output_dir = os.path.join(args.output_dir, "good")
        output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    print(" Best dev: epoch=%d, acc=%.4f\n" % (best_dev_epoch, best_dev))
    print(" Best test: epoch=%d, acc=%.4f\n" % (best_test_epoch, best_test))


def evaluate(model, args, dataloader, conn_list, label_list, tokenizer, epoch, desc="dev", write_file=False):
    all_input_ids = None
    all_conn_ids = None
    all_pred_conn_ids = None
    all_label_ids = None
    all_possible_label_ids = None
    all_predict_ids = None
    for batch in tqdm(dataloader, desc=desc):
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "mask_position_ids": batch[3],
            "conn_ids": batch[4],
            "labels": batch[5],
            "flag": "Eval"
        }
        with torch.no_grad():
            outputs = model(**inputs)
            conn_preds = outputs[0]
            rel_preds = outputs[1]

        input_ids = batch[0].detach().cpu().numpy()
        conn_ids = batch[4].detach().cpu().numpy()
        label_ids = batch[5].detach().cpu().numpy()
        possible_label_ids = batch[6].detach().cpu().numpy()
        pred_conn_ids = conn_preds.detach().cpu().numpy()
        pred_ids = rel_preds.detach().cpu().numpy()
        if all_label_ids is None:
            all_input_ids = input_ids
            all_conn_ids = conn_ids
            all_pred_conn_ids = pred_conn_ids
            all_label_ids = label_ids
            all_possible_label_ids = possible_label_ids
            all_predict_ids = pred_ids
        else:
            all_input_ids = np.append(all_input_ids, input_ids, axis=0)
            all_conn_ids = np.append(all_conn_ids, conn_ids)
            all_pred_conn_ids = np.append(all_pred_conn_ids, pred_conn_ids)
            all_label_ids = np.append(all_label_ids, label_ids)
            all_possible_label_ids = np.append(all_possible_label_ids, possible_label_ids, axis=0)
            all_predict_ids = np.append(all_predict_ids, pred_ids)

    conn_acc = np.sum(all_conn_ids == all_pred_conn_ids) / all_conn_ids.shape[0]

    # """
    cal_acc_f1_score_per_label(
        pred_ids=all_predict_ids,
        label_ids=all_label_ids,
        possible_label_ids=all_possible_label_ids,
        label_list=label_list
    )
    # """

    acc, p, r, f1, _ = cal_acc_f1_score_with_ids(pred_ids=all_predict_ids, label_ids=all_label_ids, possible_label_ids=all_possible_label_ids)

    if write_file:
        all_conns = [conn_list[int(idx)] for idx in all_conn_ids]
        all_pred_conns = [conn_list[int(idx)] for idx in all_pred_conn_ids]
        all_labels = [label_list[int(idx)] for idx in all_label_ids]
        all_predictions = [label_list[int(idx)] for idx in all_predict_ids]
        all_input_texts = [tokenizer.decode(all_input_ids[i], skip_special_tokens=True) for i in range(len(all_input_ids))]
        output_dir = os.path.join(args.data_dir, "two_preds")
        os.makedirs(output_dir, exist_ok=True)
        if args.relation_type.lower() == "explicit":
            file_name = "{}_exp2{}_{}_l{}.txt".format(args.encoder_type, desc, epoch, args.label_level)
        elif args.relation_type.lower() == "implicit":
            file_name = "{}_imp2{}_{}_l{}.txt".format(args.encoder_type, desc, epoch, args.label_level)
        file_name = os.path.join(output_dir, file_name)
        error_num = 0
        with open(file_name, "w", encoding="utf-8") as f:
            f.write("%-16s\t%-16s\t%-16s\t%-16s\t%s\n" % ("Conn", "Pred_conn", "Label", "Pred", "Text"))
            for conn, pred_conn, label, pred, text in zip(all_conns, all_pred_conns, all_labels, all_predictions, all_input_texts):
                if label == pred:
                    f.write("%-16s\t%-16s\t%-16s\t%-16s\t%s\n" % (conn, pred_conn, label, pred, text))
                else:
                    error_num += 1
                    f.write("%-16s\t%-16s\t%-16s\t%-16s\t%s\n" % (conn, pred_conn, label, pred, str(error_num) + " " + text))

    return conn_acc, acc, p, r, f1


def main():
    args = get_argparse().parse_args()
    if torch.cuda.is_available():
        args.n_gpu = 1
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        args.n_gpu = 0
    args.device = device
    logger.info("Training/evaluation parameters %s", args)
    set_seed(args.seed)

    ## 1. prepare data
    data_dir = os.path.join(args.data_dir, args.dataset)
    args.data_dir = data_dir
    train_data_file = os.path.join(data_dir, "train.json")
    dev_data_file = os.path.join(data_dir, "dev.json")
    test_data_file = os.path.join(data_dir, "test.json")
    output_dir = os.path.join(args.output_dir, args.dataset)
    output_dir = os.path.join(output_dir, "two_encoder+{}".format(args.model_name_or_path))
    output_dir = os.path.join(output_dir, args.relation_type)
    label_list = labels_from_file(os.path.join(data_dir, args.label_file))
    label_level = int(args.label_file.split(".")[0].split("_")[-1])
    output_dir = os.path.join(output_dir, "l{}".format(label_level))
    args.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(" output path: ", output_dir)
    conn_list = get_connectives_from_file(train_data_file)
    args.num_labels = len(label_list)
    args.num_connectives = len(conn_list)

    ## 2. define models
    args.model_name_or_path = os.path.join("/hits/basement/nlp/liuwi/resources/pretrained_models", args.model_name_or_path)
    if args.encoder_type.lower() == "bert":
        config = BertConfig.from_pretrained(args.model_name_or_path)
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    elif args.encoder_type.lower() == "roberta":
        config = RobertaConfig.from_pretrained(args.model_name_or_path)
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config.HP_dropout = args.dropout
    conn_onehot_in_vocab, conn_length_in_vocab = get_onehot_conn_from_vocab(conn_list, tokenizer)
    args.conn_onehot_in_vocab = conn_onehot_in_vocab.to(args.device)
    args.conn_length_in_vocab = conn_length_in_vocab.to(args.device)
    model = TwoEncoder(config=config, args=args)
    model = model.to(args.device)

    ## 3. prepare dataset
    dataset_params = {
        "relation_type": args.relation_type,
        "tokenizer": tokenizer,
        "max_seq_length": args.max_seq_length,
        "label_list": label_list,
        "label_level": label_level,
        "conn_list": conn_list,
        "use_conn": args.use_conn,
    }

    if args.do_train:
        train_dataset = TwoDataset(train_data_file, params=dataset_params)
        dataset_params["relation_type"] = "implicit"
        dev_dataset = TwoDataset(dev_data_file, params=dataset_params)
        test_dataset = TwoDataset(test_data_file, params=dataset_params)
        train_dataloader = get_dataloader(train_dataset, args, mode="train")
        dev_dataloader = get_dataloader(dev_dataset, args, mode="dev")
        test_dataloader = get_dataloader(test_dataset, args, mode="test")
        train(model, args, train_dataloader, dev_dataloader, test_dataloader, conn_list, label_list, tokenizer)

    if args.do_dev or args.do_test:
        temp_file = os.path.join(output_dir, "good/checkpoint_{}/pytorch_model.bin")
        ## 1 explicit
        # 1.1 explicit, not connective
        dataset_params["relation_type"] = "explicit"
        exp_dev_dataset = TwoDataset(dev_data_file, params=dataset_params)
        exp_test_dataset = TwoDataset(test_data_file, params=dataset_params)
        exp_dev_dataloader = get_dataloader(exp_dev_dataset, args, mode="dev")
        exp_test_dataloader = get_dataloader(exp_test_dataset, args, mode="test")
        # 1.2
        dataset_params["use_conn"] = True
        exp_conn_dev_dataset = TwoDataset(dev_data_file, params=dataset_params)
        exp_conn_test_dataset = TwoDataset(test_data_file, params=dataset_params)
        exp_conn_dev_dataloader = get_dataloader(exp_conn_dev_dataset, args, mode="dev")
        exp_conn_test_dataloader = get_dataloader(exp_conn_test_dataset, args, mode="test")
        ## 2 implicit
        # 2.1
        dataset_params["relation_type"] = "implicit"
        imp_conn_dev_dataset = TwoDataset(dev_data_file, params=dataset_params)
        imp_conn_test_dataset = TwoDataset(test_data_file, params=dataset_params)
        imp_conn_dev_dataloader = get_dataloader(imp_conn_dev_dataset, args, mode="dev")
        imp_conn_test_dataloader = get_dataloader(imp_conn_test_dataset, args, mode="test")
        # 2.2
        dataset_params["use_conn"] = False
        imp_dev_dataset = TwoDataset(dev_data_file, params=dataset_params)
        imp_test_dataset = TwoDataset(test_data_file, params=dataset_params)
        imp_dev_dataloader = get_dataloader(imp_dev_dataset, args, mode="dev")
        imp_test_dataloader = get_dataloader(imp_test_dataset, args, mode="test")

        if args.relation_type.lower() == "explicit":
            to_E = True
            to_I = True
        else:
            to_E = False
            to_I = True
        without_conn = True
        with_conn = False
        do_write = False
        do_train = False
        for epoch in range(1, 7):
            checkpoint_file = temp_file.format(str(epoch))
            args.output_dir = os.path.dirname(checkpoint_file)
            print(" " + checkpoint_file)
            model.load_state_dict(torch.load(checkpoint_file))
            model.eval()

            if args.do_dev:
                if without_conn and to_E:
                    conn_acc, acc, p, r, f1 = evaluate(
                        model, args, exp_dev_dataloader, conn_list, label_list, tokenizer,
                        epoch, desc="exp_dev", write_file=do_write
                    )
                    print(" 2E Dev: conn_acc=%.4f, acc=%.4f, p=%.4f, r=%.4f, f1=%.4f\n" % (conn_acc, acc, p, r, f1))

                if with_conn and to_E:
                    conn_acc, acc, p, r, f1 = evaluate(
                        model, args, exp_conn_dev_dataloader, conn_list, label_list, tokenizer,
                        epoch, desc="exp_conn_dev", write_file=do_write
                    )
                    print(" 2E_conn Dev: conn_acc=%.4f, acc=%.4f, p=%.4f, r=%.4f, f1=%.4f\n" % (conn_acc, acc, p, r, f1))

                if without_conn and to_I:
                    conn_acc, acc, p, r, f1 = evaluate(
                        model, args, imp_dev_dataloader, conn_list, label_list, tokenizer,
                        epoch, desc="imp_dev", write_file=do_write
                    )
                    print(" 2I Dev: conn_acc=%.4f, acc=%.4f, p=%.4f, r=%.4f, f1=%.4f\n" % (conn_acc, acc, p, r, f1))

                if with_conn and to_I:
                    conn_acc, acc, p, r, f1 = evaluate(
                        model, args, imp_conn_dev_dataloader, conn_list, label_list, tokenizer,
                        epoch, desc="imp_conn_dev", write_file=do_write
                    )
                    print(" 2I_conn Dev: conn_acc=%.4f, acc=%.4f, p=%.4f, r=%.4f, f1=%.4f\n" % (conn_acc, acc, p, r, f1))

            if args.do_test:
                if without_conn and to_E:
                    conn_acc, acc, p, r, f1 = evaluate(
                        model, args, exp_test_dataloader, conn_list, label_list, tokenizer,
                        epoch, desc="exp_test", write_file=do_write
                    )
                    print(" 2E Test: conn_acc=%.4f, acc=%.4f, p=%.4f, r=%.4f, f1=%.4f\n" % (conn_acc, acc, p, r, f1))

                if with_conn and to_E:
                    conn_acc, acc, p, r, f1 = evaluate(
                        model, args, exp_conn_test_dataloader, conn_list, label_list, tokenizer,
                        epoch, desc="exp_conn_test", write_file=do_write
                    )
                    print(" 2E_conn Test: conn_acc=%.4f, acc=%.4f, p=%.4f, r=%.4f, f1=%.4f\n" % (conn_acc, acc, p, r, f1))

                if without_conn and to_I:
                    conn_acc, acc, p, r, f1 = evaluate(
                        model, args, imp_test_dataloader, conn_list, label_list, tokenizer,
                        epoch, desc="imp_test", write_file=do_write
                    )
                    print(" 2I Test: conn_acc=%.4f, acc=%.4f, p=%.4f, r=%.4f, f1=%.4f\n" % (conn_acc, acc, p, r, f1))

                if with_conn and to_I:
                    conn_acc, acc, p, r, f1 = evaluate(
                        model, args, imp_conn_test_dataloader, conn_list, label_list, tokenizer,
                        epoch, desc="imp_conn_test", write_file=do_write
                    )
                    print(" 2I_conn Test: conn_acc=%.4f, acc=%.4f, p=%.4f, r=%.4f, f1=%.4f\n" % (conn_acc, acc, p, r, f1))
            print()

if __name__ == "__main__":
    main()
