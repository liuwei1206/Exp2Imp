# author = liuwei
# date = 2023-03-08

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
from transformers import BertConfig, BertTokenizer
from transformers import RobertaConfig, RobertaTokenizer

from utils import cal_acc_f1_score_with_ids, labels_from_file, cal_acc_f1_score_per_label
from utils import get_connectives_from_file, get_connectives_from_data, get_onehot_conn_from_vocab
from filter_util import filter_samples_with_distance_per_label
from task_dataset import BaseDataset
from model import BaseClassifier, TwoEncoder

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
    parser.add_argument("--model_type", default="base", type=str, help="base, multi-task, two-encoder")
    parser.add_argument("--encoder_type", default="roberta", type=str)
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str, help="roberta-base")
    parser.add_argument("--label_file", default="labels_2.txt", type=str, help="the label file path")
    parser.add_argument("--relation_type", default="explicit", type=str)
    parser.add_argument("--loss_ratio", default=0.5, type=str, help="ratio for connective loss, from 0.0 to 1.0")

    # for training
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_dev", default=False, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument("--is_hard", default=False, action="store_true")
    parser.add_argument("--min_confidence", default=0.0, type=float, help="The confidence of a prediction (logit)")
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=24, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--use_conn", default=False, action="store_true")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="training epoch")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="learning rate")
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--max_grad_norm", default=2.0, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--seed", default=106524, type=int, help="random seed")

    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloader(dataset, args, mode="train"):
    print("  {} dataset length: ".format(mode), len(dataset))
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
    specific_params = []
    no_deday = ["bias", "LayerNorm.weigh"]
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
            if args.model_type.lower() == "base":
                batch_data = (batch[0], batch[1], batch[2], batch[3])
                batch = tuple(t.to(args.device) for t in batch_data)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                    "flag": "Train"
                }
                outputs = model(**inputs)
                loss = outputs[0]
            else:
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

            # optimizer.zero_grad()
            loss.backward()
            global_step += 1
            logging_loss = loss.item()
            tr_loss += logging_loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if global_step % print_step == 0:
                print(" Current loss=%.4f, global average loss=%.4f" % (logging_loss, tr_loss / global_step))

        # evaluation and save
        model.eval()
        dev_acc, dev_p, dev_r, dev_f1 = evaluate(model, args, dev_dataloader, conn_list, label_list, tokenizer, epoch, desc="dev")
        test_acc, test_p, test_r, test_f1 = evaluate(model, args, test_dataloader, conn_list, label_list, tokenizer, epoch, desc="test")
        print()
        print(" Dev acc=%.4f, p=%.4f, r=%.4f, f1=%.4f" % (dev_acc, dev_p, dev_r, dev_f1))
        print(" Test acc=%.4f, p=%.4f, r=%.4f, f1=%.4f" % (test_acc, test_p, test_r, test_f1))
        if dev_acc > best_dev:
            best_dev = dev_acc
            best_dev_epoch = epoch
        if test_acc > best_test:
            best_test = test_acc
            best_test_epoch = epoch

        output_dir = os.path.join(args.output_dir, "good")
        output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    print(" Best dev: epoch=%d, acc=%.4f\n" % (best_dev_epoch, best_dev))
    print(" Best test: epoch=%d, acc=%.4f\n" % (best_test_epoch, best_test))

    return best_dev_epoch, best_test_epoch


def evaluate(model, args, dataloader, conn_list, label_list, tokenizer, epoch, desc="dev"):
    all_input_ids = None
    all_label_ids = None
    all_predict_ids = None
    all_possible_label_ids = None
    for batch in tqdm(dataloader, desc=desc):
        if args.model_type.lower() == "base":
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3],
                "flag": "Eval"
            }
            with torch.no_grad():
                outputs = model(**inputs)
                preds = outputs[0]
        else:
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
                preds = outputs[1]

        input_ids = batch[0].detach().cpu().numpy()
        label_ids = batch[-2].detach().cpu().numpy()
        possible_label_ids = batch[-1].detach().cpu().numpy()
        pred_ids = preds.detach().cpu().numpy()
        if all_label_ids is None:
            all_input_ids = input_ids
            all_label_ids = label_ids
            all_predict_ids = pred_ids
            all_possible_label_ids = possible_label_ids
        else:
            all_input_ids = np.append(all_input_ids, input_ids, axis=0)
            all_label_ids = np.append(all_label_ids, label_ids)
            all_predict_ids = np.append(all_predict_ids, pred_ids)
            all_possible_label_ids = np.append(all_possible_label_ids, possible_label_ids, axis=0)

    acc, p, r, f1, all_label_ids = cal_acc_f1_score_with_ids(
        pred_ids=all_predict_ids,
        label_ids=all_label_ids,
        possible_label_ids=all_possible_label_ids
    )

    return acc, p, r, f1


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
    source_train_file = os.path.join(data_dir, "train.json")
    dev_data_file = os.path.join(data_dir, "dev.json")
    test_data_file = os.path.join(data_dir, "test.json")
    args.data_dir = data_dir
    label_level = int(args.label_file.split(".")[0].split("_")[-1])
    args.label_level = label_level
    label_list = labels_from_file(os.path.join(data_dir, args.label_file))
    args.num_labels = len(label_list)

    filter_level = args.label_level
    ## filter data
    filter_datas, new_dataset = filter_samples_with_distance_per_label(
        dataset=args.dataset,
        label_level=filter_level,
        model_name_or_path=args.model_name_or_path,
        source_type="explicit",
        target_type="explicit",
        min_confidence=args.min_confidence
    )
    print("############################################################")
    print("## Train with (min_confidence=%.4f), size=%d ##" % (args.min_confidence, len(filter_datas)))
    print("############################################################")
    output_dir = os.path.join(args.output_dir, new_dataset)
    output_dir = os.path.join(output_dir, "base+{}".format(args.model_name_or_path))
    output_dir = os.path.join(output_dir, args.relation_type)
    output_dir = os.path.join(output_dir, "l{}".format(label_level))
    # os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    conn_list = get_connectives_from_data(filter_datas)
    args.num_connectives = len(conn_list)
    args.model_name_or_path = os.path.join(
        "/hits/basement/nlp/liuwi/resources/pretrained_models",
        args.model_name_or_path
    )

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
    if args.model_type.lower() == "base":
        model = BaseClassifier(config=config, args=args)
        dataset_name = "BaseDataset"
    elif args.model_type.lower() == "two-encoder":
        model = TwoEncoder(config=config, args=args)
        dataset_name = "TwoDataset"
    model = model.to(args.device)

    ## 3. prepare dataset
    dataset_params = {
        "tokenizer": tokenizer,
        "max_seq_length": args.max_seq_length,
        "label_level": label_level,
        "label_list": label_list,
        "conn_list": conn_list,
        "relation_type": args.relation_type,
        "use_conn": args.use_conn,
    }
    dataset_module = __import__("task_dataset")
    MyDataset = getattr(dataset_module, dataset_name)
    train_dataset = MyDataset(filter_datas, params=dataset_params)
    dataset_params["relation_type"] = "implicit"
    dev_dataset = MyDataset(dev_data_file, params=dataset_params)
    test_dataset = MyDataset(test_data_file, params=dataset_params)
    dataset_params["relation_type"] = "explicit"

    train_dataloader = get_dataloader(train_dataset, args, mode="train")
    dev_dataloader = get_dataloader(dev_dataset, args, mode="dev")
    test_dataloader = get_dataloader(test_dataset, args, mode="test")

    train(model, args, train_dataloader, dev_dataloader, test_dataloader, conn_list, label_list, tokenizer)

if __name__ == "__main__":
    main()
