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
from utils import visualize_sent_vector, cal_distance
from task_dataset import BaseDataset
from model import BaseClassifier

# set logger, print to console and write to file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
BASIC_FORMAT = "%(asctime)s:%(levelname)s: %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()
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
    parser.add_argument("--encoder_type", default="roberta", type=str)
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str, help="roberta-base")
    parser.add_argument("--label_file", default="labels_2.txt", type=str, help="the label file path")
    parser.add_argument("--relation_type", default="explicit", type=str)

    # for training
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_dev", default=False, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument("--do_vector", default=False, action="store_true")
    parser.add_argument("--train_batch_size", default=16, type=int)
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


def train(model, args, train_dataloader, dev_dataloader, test_dataloader, label_list, tokenizer):
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
    train_iterator = trange(1, int(num_train_epochs)+1, desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        model.train()
        model.zero_grad()
        for step, batch in enumerate(epoch_iterator):
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
                print(" Current loss=%.4f, global average loss=%.4f"%(logging_loss, tr_loss / global_step))

        # evaluation and save
        model.eval()
        dev_acc, dev_p, dev_r, dev_f1, _ = evaluate(model, args, dev_dataloader, label_list, tokenizer, epoch, desc="dev")
        test_acc, test_p, test_r, test_f1, _ = evaluate(model, args, test_dataloader, label_list, tokenizer, epoch, desc="test")
        print()
        print(" Dev acc=%.4f, p=%.4f, r=%.4f, f1=%.4f" % (dev_acc, dev_p, dev_r, dev_f1))
        print(" Test acc=%.4f, p=%.4f, r=%.4f, f1=%.4f"%(test_acc, test_p, test_r, test_f1))
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
    print(" Best dev: epoch=%d, acc=%.4f\n"%(best_dev_epoch, best_dev))
    print(" Best test: epoch=%d, acc=%.4f\n"%(best_test_epoch, best_test))

    return best_dev_epoch, best_test_epoch


def evaluate(model, args, dataloader, label_list, tokenizer, epoch, desc="dev", write_file=False):
    all_input_ids = None
    all_label_ids = None
    all_predict_ids = None
    all_possible_label_ids = None
    all_target_probs = None
    for batch in tqdm(dataloader, desc=desc):
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
            probs = outputs[1]

        input_ids = batch[0].detach().cpu().numpy()
        label_ids = batch[3].detach().cpu().numpy()
        possible_label_ids = batch[4].detach().cpu().numpy()
        pred_ids = preds.detach().cpu().numpy()
        probs = probs.detach().cpu().numpy()
        if all_label_ids is None:
            all_input_ids = input_ids
            all_label_ids = label_ids
            all_predict_ids = pred_ids
            all_possible_label_ids = possible_label_ids
            all_target_probs = probs
        else:
            all_input_ids = np.append(all_input_ids, input_ids, axis=0)
            all_label_ids = np.append(all_label_ids, label_ids)
            all_predict_ids = np.append(all_predict_ids, pred_ids)
            all_possible_label_ids = np.append(all_possible_label_ids, possible_label_ids, axis=0)
            all_target_probs = np.append(all_target_probs, probs)

    acc, p, r, f1, all_label_ids = cal_acc_f1_score_with_ids(
                                       pred_ids=all_predict_ids,
                                       label_ids=all_label_ids,
                                       possible_label_ids=all_possible_label_ids
                                   )
    # """
    cal_acc_f1_score_per_label(
        pred_ids=all_predict_ids,
        label_ids=all_label_ids,
        possible_label_ids=all_possible_label_ids,
        label_list=label_list
    )
    # """
    file_name = None
    if write_file:
        all_labels = [label_list[int(idx)] for idx in all_label_ids]
        all_predictions = [label_list[int(idx)] for idx in all_predict_ids]
        all_input_texts = [tokenizer.decode(all_input_ids[i], skip_special_tokens=True) for i in range(len(all_input_ids))]
        output_dir = os.path.join(args.data_dir, "base_preds")
        os.makedirs(output_dir, exist_ok=True)
        if args.relation_type.lower() == "explicit":
            file_name = "{}_exp2{}_{}_l{}.txt".format(
                args.model_name_or_path.split("/")[-1],
                desc, epoch, args.label_level
            )
        elif args.relation_type.lower() == "implicit":
            file_name = "{}_imp2{}_{}_l{}.txt".format(
                args.model_name_or_path.split("/")[-1],
                desc, epoch, args.label_level
            )
        file_name = os.path.join(output_dir, file_name)
        error_num = 0
        with open(file_name, "w", encoding="utf-8") as f:
            f.write("%-16s\t%-16s\t%-8s\t%s\n"%("Label", "Pred", "Prob", "Text"))
            for label, pred, prob, text in zip(all_labels, all_predictions, all_target_probs, all_input_texts):
                if label == pred:
                    f.write("%-16s\t%-16s\t%.4f\t%s\n"%(label, pred, prob, text))
                else:
                    error_num += 1
                    f.write("%-16s\t%-16s\t%.4f\t%s\n" % (label, pred, prob, str(error_num) + " " + text))

    return acc, p, r, f1, file_name


def produce_vector(model, args, dataloader, label_list, tokenizer, desc="dev", file_name=None, do_visual=False):
    if file_name is not None and os.path.exists(file_name):
        print(file_name)
        with np.load(file_name) as dataset:
            all_vectors = dataset["vectors"]
            all_labels = dataset["labels"]
    else:
        all_vectors = None
        all_labels = None
        for batch in tqdm(dataloader, desc=desc):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            labels = batch[3].detach().cpu().numpy()

            with torch.no_grad():
                vectors = model.get_sent_vectors(**inputs)
                vectors = vectors.detach().cpu().numpy()

            if all_vectors is None:
                all_vectors = vectors
                all_labels = labels
            else:
                all_vectors = np.append(all_vectors, vectors, axis=0)
                all_labels = np.append(all_labels, labels, axis=0)
        if file_name is not None:
            print(file_name)
            np.savez(file_name, vectors=all_vectors, labels=all_labels)

    if do_visual:
        jpg_file = file_name.split("/")[-1].replace(".npz", "_10.eps")
        os.makedirs("images", exist_ok=True)
        jpg_file = os.path.join("images", jpg_file)
        visualize_sent_vector(
            (all_vectors, all_labels),
            label_list=label_list,
            perplexity=10,
            jpg_path=jpg_file
        )

    return all_vectors, all_labels


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
    train_data_file = os.path.join(data_dir, "train.json")
    dev_data_file = os.path.join(data_dir, "dev.json")
    test_data_file = os.path.join(data_dir, "test.json")
    args.data_dir = data_dir
    output_dir = os.path.join(args.output_dir, args.dataset)
    output_dir = os.path.join(output_dir, "base+{}".format(args.model_name_or_path))
    output_dir = os.path.join(output_dir, args.relation_type)
    label_level = int(args.label_file.split(".")[0].split("_")[-1])
    output_dir = os.path.join(output_dir, "l{}".format(label_level))
    # os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    args.label_level = label_level
    label_list = labels_from_file(os.path.join(data_dir, args.label_file))
    args.num_labels = len(label_list)

    ## 2. define models
    args.model_name_or_path = os.path.join("/hits/basement/nlp/liuwi/resources/pretrained_models", args.model_name_or_path)
    print(args.model_name_or_path)
    if args.encoder_type.lower() == "bert":
        config = BertConfig.from_pretrained(args.model_name_or_path)
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    elif args.encoder_type.lower() == "roberta":
        config = RobertaConfig.from_pretrained(args.model_name_or_path)
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config.HP_dropout = args.dropout
    model = BaseClassifier(config=config, args=args)
    model = model.to(args.device)

    ## 3. prepare dataset
    dataset_params = {
        "tokenizer": tokenizer,
        "max_seq_length": args.max_seq_length,
        "label_level": label_level,
        "label_list": label_list,
        "relation_type": args.relation_type,
        "use_conn": args.use_conn
    }

    if args.do_train:
        train_dataset = BaseDataset(train_data_file, params=dataset_params)
        dataset_params["relation_type"] = "implicit"
        dev_dataset = BaseDataset(dev_data_file, params=dataset_params)
        test_dataset = BaseDataset(test_data_file, params=dataset_params)
        train_dataloader = get_dataloader(train_dataset, args, mode="train")
        dev_dataloader = get_dataloader(dev_dataset, args, mode="dev")
        test_dataloader = get_dataloader(test_dataset, args, mode="test")
        train(model, args, train_dataloader, dev_dataloader, test_dataloader, label_list, tokenizer)

    if args.do_dev or args.do_test:
        temp_file = os.path.join(output_dir, "good/checkpoint_{}/pytorch_model.bin")

        ## 1 explicit
        # 1.1 explicit, not connective
        dataset_params["relation_type"] = "explicit"
        exp_dev_dataset = BaseDataset(dev_data_file, params=dataset_params)
        exp_test_dataset = BaseDataset(test_data_file, params=dataset_params)
        exp_dev_dataloader = get_dataloader(exp_dev_dataset, args, mode="dev")
        exp_test_dataloader = get_dataloader(exp_test_dataset, args, mode="test")

        exp_train_dataset = BaseDataset(train_data_file, params=dataset_params)
        exp_train_dataloader = get_dataloader(exp_train_dataset, args, mode="train1")

        # 1.2 explicit, with connective
        dataset_params["use_conn"] = True
        exp_conn_dev_dataset = BaseDataset(dev_data_file, params=dataset_params)
        exp_conn_test_dataset = BaseDataset(test_data_file, params=dataset_params)
        exp_conn_dev_dataloader = get_dataloader(exp_conn_dev_dataset, args, mode="dev")
        exp_conn_test_dataloader = get_dataloader(exp_conn_test_dataset, args, mode="test")

        exp_conn_train_dataset = BaseDataset(train_data_file, params=dataset_params)
        exp_conn_train_dataloader = get_dataloader(exp_conn_train_dataset, args, mode="train1")

        ## 2 implicit
        # 2.1 implicit, without connective
        dataset_params["relation_type"] = "implicit"
        imp_conn_dev_dataset = BaseDataset(dev_data_file, params=dataset_params)
        imp_conn_test_dataset = BaseDataset(test_data_file, params=dataset_params)
        imp_conn_dev_dataloader = get_dataloader(imp_conn_dev_dataset, args, mode="dev")
        imp_conn_test_dataloader = get_dataloader(imp_conn_test_dataset, args, mode="test")

        imp_conn_train_dataset = BaseDataset(train_data_file, params=dataset_params)
        imp_conn_train_dataloader = get_dataloader(imp_conn_train_dataset, args, mode="train1")

        # 2.2 implicit, with connective
        dataset_params["use_conn"] = False
        imp_dev_dataset = BaseDataset(dev_data_file, params=dataset_params)
        imp_test_dataset = BaseDataset(test_data_file, params=dataset_params)
        imp_dev_dataloader = get_dataloader(imp_dev_dataset, args, mode="dev")
        imp_test_dataloader = get_dataloader(imp_test_dataset, args, mode="test")

        imp_train_dataset = BaseDataset(train_data_file, params=dataset_params)
        imp_train_dataloader = get_dataloader(imp_train_dataset, args, mode="train1")

        if args.relation_type.lower() == "explicit":
            to_E = True
            to_I = True
        else:
            to_E = False
            to_I = True
        without_conn = True
        with_conn = False
        do_write = False
        do_train = True
        args.do_dev = False
        args.do_test = False
        for epoch in range(5, 11):
            checkpoint_file = temp_file.format(str(epoch))
            print("Epoch=%d, checkpoint=%s"%(epoch, checkpoint_file))
            args.output_dir = os.path.dirname(checkpoint_file)
            model.load_state_dict(torch.load(checkpoint_file))
            model.eval()

            if do_train:
                if to_E and without_conn:
                    ## 1. to Explicit on Train
                    acc, p, r, f1, _ = evaluate(
                        model, args, exp_train_dataloader, label_list, tokenizer,
                        epoch, desc="exp_train", write_file=do_write
                    )
                    print(" 2E Train: acc=%.4f, p=%.4f, r=%.4f, f1=%.4f" % (acc, p, r, f1))

                if to_E and with_conn:
                    ## 2. to Explicit with Conn on Train
                    acc, p, r, f1, _ = evaluate(
                        model, args, exp_conn_train_dataloader, label_list, tokenizer,
                        epoch, desc="exp_conn_train", write_file=do_write
                    )
                    print(" 2E_conn Train: acc=%.4f, p=%.4f, r=%.4f, f1=%.4f" % (acc, p, r, f1))

                if to_I and without_conn:
                    acc, p, r, f1, _ = evaluate(
                        model, args, imp_train_dataloader, label_list, tokenizer,
                        epoch, desc="imp_train", write_file=do_write
                    )
                    print(" 2I Train: acc=%.4f, p=%.4f, r=%.4f, f1=%.4f" % (acc, p, r, f1))

                if to_I and with_conn:
                    acc, p, r, f1, _ = evaluate(
                        model, args, imp_conn_train_dataloader, label_list, tokenizer,
                        epoch, desc="imp_conn_train", write_file=do_write
                    )
                    print(" 2I_conn Train: acc=%.4f, p=%.4f, r=%.4f, f1=%.4f" % (acc, p, r, f1))

            if args.do_dev:
                if without_conn and to_E:
                    ## 1. to Explicit on Dev
                    acc, p, r, f1, _ = evaluate(
                        model, args, exp_dev_dataloader, label_list, tokenizer,
                        epoch, desc="exp_dev", write_file=do_write
                    )
                    print(" 2E Dev: acc=%.4f, p=%.4f, r=%.4f, f1=%.4f" % (acc, p, r, f1))

                if with_conn and to_E:
                    ## 2. to Explicit with Conn on Dev
                    acc, p, r, f1, _ = evaluate(
                        model, args, exp_conn_dev_dataloader, label_list, tokenizer,
                        epoch, desc="exp_conn_dev", write_file=do_write
                    )
                    print(" 2E_conn Dev: acc=%.4f, p=%.4f, r=%.4f, f1=%.4f" % (acc, p, r, f1))

                if without_conn and to_I:
                    ## 3. to Implicit on Dev
                    acc, p, r, f1, _ = evaluate(
                        model, args, imp_dev_dataloader, label_list, tokenizer,
                        epoch, desc="imp_dev", write_file=do_write
                    )
                    print(" 2I Dev: acc=%.4f, p=%.4f, r=%.4f, f1=%.4f" % (acc, p, r, f1))

                if with_conn and to_I:
                    ## 4. to Implicit with Conn on Dev
                    acc, p, r, f1, _ = evaluate(
                        model, args, imp_conn_dev_dataloader, label_list, tokenizer,
                        epoch, desc="imp_conn_dev", write_file=do_write
                    )
                    print(" 2I_conn Dev: acc=%.4f, p=%.4f, r=%.4f, f1=%.4f" % (acc, p, r, f1))

            if args.do_test:
                if without_conn and to_E:
                    ## 5. to Explicit on Test
                    acc, p, r, f1, _ = evaluate(
                        model, args, exp_test_dataloader, label_list, tokenizer,
                        epoch, desc="exp_test", write_file=do_write
                    )
                    print(" 2E Test: acc=%.4f, p=%.4f, r=%.4f, f1=%.4f" % (acc, p, r, f1))

                if with_conn and to_E:
                    ## 6. to Explicit with Conn on Test
                    acc, p, r, f1, _ = evaluate(
                        model, args, exp_conn_test_dataloader, label_list, tokenizer,
                        epoch, desc="exp_conn_test", write_file=do_write
                    )
                    print(" 2E_conn Test: acc=%.4f, p=%.4f, r=%.4f, f1=%.4f" % (acc, p, r, f1))

                if without_conn and to_I:
                    ## 7. to Implicit on Test
                    acc, p, r, f1, _ = evaluate(
                        model, args, imp_test_dataloader, label_list, tokenizer,
                        epoch, desc="imp_test", write_file=do_write
                    )
                    print(" 2I Test: acc=%.4f, p=%.4f, r=%.4f, f1=%.4f" % (acc, p, r, f1))

                if with_conn and to_I:
                    ## 8. to Implicit with Conn on Test
                    acc, p, r, f1, _ = evaluate(
                        model, args, imp_conn_test_dataloader, label_list, tokenizer,
                        epoch, desc="imp_conn_test", write_file=do_write
                    )
                    print(" 2I_conn Test: acc=%.4f, p=%.4f, r=%.4f, f1=%.4f" % (acc, p, r, f1))

            print()

    if args.do_vector:
        temp_file = os.path.join(output_dir, "good/checkpoint_{}/pytorch_model.bin")

        ## 1 explicit
        # 1.1 explicit, not connective
        dataset_params["relation_type"] = "explicit"
        exp_dev_dataset = BaseDataset(dev_data_file, params=dataset_params)
        exp_test_dataset = BaseDataset(test_data_file, params=dataset_params)
        exp_dev_dataloader = get_dataloader(exp_dev_dataset, args, mode="dev")
        exp_test_dataloader = get_dataloader(exp_test_dataset, args, mode="test")

        exp_train_dataset = BaseDataset(train_data_file, params=dataset_params)
        exp_train_dataloader = get_dataloader(exp_train_dataset, args, mode="train1")

        # 1.2
        dataset_params["use_conn"] = True
        exp_conn_dev_dataset = BaseDataset(dev_data_file, params=dataset_params)
        exp_conn_test_dataset = BaseDataset(test_data_file, params=dataset_params)
        exp_conn_dev_dataloader = get_dataloader(exp_conn_dev_dataset, args, mode="dev")
        exp_conn_test_dataloader = get_dataloader(exp_conn_test_dataset, args, mode="test")

        exp_conn_train_dataset = BaseDataset(train_data_file, params=dataset_params)
        exp_conn_train_dataloader = get_dataloader(exp_conn_train_dataset, args, mode="train1")

        ## 2 implicit
        # 2.1
        dataset_params["relation_type"] = "implicit"
        imp_conn_dev_dataset = BaseDataset(dev_data_file, params=dataset_params)
        imp_conn_test_dataset = BaseDataset(test_data_file, params=dataset_params)
        imp_conn_dev_dataloader = get_dataloader(imp_conn_dev_dataset, args, mode="dev")
        imp_conn_test_dataloader = get_dataloader(imp_conn_test_dataset, args, mode="test")

        imp_conn_train_dataset = BaseDataset(train_data_file, params=dataset_params)
        imp_conn_train_dataloader = get_dataloader(imp_conn_train_dataset, args, mode="train1")

        # 2.2
        dataset_params["use_conn"] = False
        imp_dev_dataset = BaseDataset(dev_data_file, params=dataset_params)
        imp_test_dataset = BaseDataset(test_data_file, params=dataset_params)
        imp_dev_dataloader = get_dataloader(imp_dev_dataset, args, mode="dev")
        imp_test_dataloader = get_dataloader(imp_test_dataset, args, mode="test")

        imp_train_dataset = BaseDataset(train_data_file, params=dataset_params)
        imp_train_dataloader = get_dataloader(imp_train_dataset, args, mode="train1")

        if args.relation_type.lower() == "explicit":
            to_E = True
            to_I = False
        else:
            to_E = False
            to_I = True
        do_write = True
        do_visual = False
        mode = "train"

        if mode.lower() == "train":
            exp_dataloader = exp_train_dataloader
            exp_conn_dataloader = exp_conn_train_dataloader
            imp_dataloader = imp_train_dataloader
            imp_conn_dataloader = imp_conn_train_dataloader
        elif mode.lower() == "dev":
            exp_dataloader = exp_dev_dataloader
            exp_conn_dataloader = exp_conn_dev_dataloader
            imp_dataloader = imp_dev_dataloader
            imp_conn_dataloader = imp_conn_dev_dataloader
        elif mode.lower() == "test":
            exp_dataloader = exp_test_dataloader
            exp_conn_dataloader = exp_conn_test_dataloader
            imp_dataloader = imp_test_dataloader
            imp_conn_dataloader = imp_conn_test_dataloader

        for epoch in range(10, 11):
            checkpoint_file = temp_file.format(str(epoch))
            print("Epoch=%d, checkpoint=%s" % (epoch, checkpoint_file))
            args.output_dir = os.path.dirname(checkpoint_file)
            model.load_state_dict(torch.load(checkpoint_file))
            model.eval()

            os.makedirs("data/dataset/{}/vectors".format(args.dataset), exist_ok=True)

            if to_E:
                # without conn
                if do_write:
                    vector_file = "{}_l{}_{}2exp_{}_{}_{}.npz".format(
                        args.dataset, args.label_level, args.relation_type[:3],
                        args.model_name_or_path.split("/")[-1], mode, epoch
                    )
                    vector_file = os.path.join(
                        "data/dataset/{}/vectors".format(args.dataset),
                        vector_file
                    )
                else:
                    vector_file = None
                all_vectors, all_labels = produce_vector(
                    model, args, exp_dataloader, label_list, tokenizer,
                    "exp_{}".format(mode), vector_file, do_visual
                )

                # with conn
                if do_write:
                    vector_file = "{}_l{}_{}2exp_{}_conn_{}_{}.npz".format(
                        args.dataset, args.label_level, args.relation_type[:3],
                        args.model_name_or_path.split("/")[-1], mode, epoch
                    )
                    vector_file = os.path.join(
                        "data/dataset/{}/vectors".format(args.dataset),
                        vector_file
                    )
                else:
                    vector_file = None
                all_conn_vectors, all_conn_labels = produce_vector(
                    model, args, exp_conn_dataloader, label_list, tokenizer,
                    "exp_conn_{}".format(mode), vector_file, do_visual
                )

                # calculate
                print("#################################################################################################")
                avg_score, min_score, max_score, all_scores = cal_distance(
                    all_vectors, all_conn_vectors, distance_type="cos"
                )
                print("%s: %s to %s" % (mode, args.relation_type, "explicit"))
                print("cos sim: avg=%.4f, min=%.4f, max=%.4f" % (avg_score, min_score, max_score))

                masks = (all_scores <= 0.0)
                cos0 = int(torch.sum(masks)) / len(all_scores)
                masks = (all_scores <= 0.4)
                cos4 = int(torch.sum(masks)) / len(all_scores)
                masks = (all_scores <= 0.5)
                cos5 = int(torch.sum(masks)) / len(all_scores)
                masks = (all_scores <= 0.8)
                cos8 = int(torch.sum(masks)) / len(all_scores)
                print("cos0: %.4f, cos4: %.4f, cos5: %.4f, cos8: %.4f"%(cos0, cos4, cos5, cos8))

                # """
                for idx in range(len(label_list)):
                    label_scores = all_scores[all_labels==idx]
                    print(label_list[idx], len(label_scores))
                    avg_s = np.average(label_scores)
                    masks = (label_scores <= 0.0)
                    cos0 = int(torch.sum(masks)) / len(label_scores)
                    masks = (label_scores <= 0.4)
                    cos4 = int(torch.sum(masks)) / len(label_scores)
                    masks = (label_scores <= 0.8)
                    cos8 = int(torch.sum(masks)) / len(label_scores)
                    print("Label: %s, avg score: %.4f; cos0: %.4f, cos4: %.4f, cos8: %.4f"%(label_list[idx], avg_s, cos0, cos4, cos8))
                # """
                """
                all_scores = all_scores.tolist()
                cur_id = 0
                # for label_id, score in zip(all_labels, all_scores):
                #     cur_id += 1
                #     print("%d, %s: %.4f"%(cur_id, label_list[label_id], score))
                with open("data/dataset/pdtb2/analyses/group/res_contingency.txt", "w", encoding="utf-8") as f:
                    for label_id, score in zip(all_labels, all_scores):
                        cur_id += 1
                        f.write("%d, %s: %.4f\n"%(cur_id, label_list[label_id], score))
                """

            if to_I:
                # without conn
                if do_write:
                    vector_file = "{}_l{}_{}2imp_{}_{}_{}.npz".format(
                        args.dataset, args.label_level, args.relation_type[:3],
                        args.model_name_or_path.split("/")[-1], mode, epoch
                    )
                    vector_file = os.path.join(
                        "data/dataset/{}/vectors".format(args.dataset),
                        vector_file
                    )
                else:
                    vector_file = None
                all_vectors, all_labels = produce_vector(
                    model, args, imp_dataloader, label_list, tokenizer,
                    "imp_{}".format(mode), vector_file, do_visual
                )
                # with conn
                if do_write:
                    vector_file = "{}_l{}_{}2imp_{}_conn_{}_{}.npz".format(
                        args.dataset, args.label_level, args.relation_type[:3],
                        args.model_name_or_path.split("/")[-1], mode, epoch
                    )
                    vector_file = os.path.join(
                        "data/dataset/{}/vectors".format(args.dataset),
                        vector_file
                    )
                else:
                    vector_file = None
                all_conn_vectors, all_conn_labels = produce_vector(
                    model, args, imp_conn_dataloader, label_list, tokenizer,
                    "imp_conn_{}".format(mode), vector_file, do_visual
                )

                print("#################################################################################################")
                avg_score, min_score, max_score, all_scores = cal_distance(
                    all_vectors, all_conn_vectors, distance_type="cos"
                )
                print("%s: %s to %s" % (mode, args.relation_type, "implicit"))
                print("cos sim: avg=%.4f, min=%.4f, max=%.4f" % (avg_score, min_score, max_score))

                masks_0 = (all_scores <= 0.0)
                cos0 = int(torch.sum(masks_0)) / len(all_scores)
                masks = (all_scores <= 0.4)
                cos4 = int(torch.sum(masks)) / len(all_scores)
                masks = (all_scores <= 0.8)
                cos8 = int(torch.sum(masks)) / len(all_scores)
                print("cos0: %.4f, cos4: %.4f, cos8: %.4f"%(cos0, cos4, cos8))

                print((masks_0.long() == 1).nonzero(as_tuple=True)[0])

                for idx in range(len(label_list)):
                    label_scores = all_scores[all_labels==idx]
                    avg_s = np.average(label_scores)
                    masks = (label_scores <= 0.0)
                    cos0 = int(torch.sum(masks)) / len(label_scores)
                    masks = (label_scores <= 0.4)
                    cos4 = int(torch.sum(masks)) / len(label_scores)
                    masks = (label_scores <= 0.8)
                    cos8 = int(torch.sum(masks)) / len(label_scores)
                    print("Label: %s, avg score: %.4f; cos0: %.4f, cos4: %.4f, cos8: %.4f"%(label_list[idx], avg_s, cos0, cos4, cos8))

if __name__ == "__main__":
    main()
