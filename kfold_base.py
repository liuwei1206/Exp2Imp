# author = liuwei
# date = 2023-05-30

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

from utils import cal_acc_f1_score_with_ids, labels_from_file, visualize_sent_vector
from utils import cal_distance
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
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str, help="bert-base-uncased, roberta-base")
    parser.add_argument("--label_file", default="labels_2.txt", type=str, help="the label file path")
    parser.add_argument("--relation_type", default="explicit", type=str)
    parser.add_argument("--target_type", default="explicit", type=str)

    # for training
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_dev", default=False, action="store_true")
    parser.add_argument("--do_vector", default=False, action="store_true")
    parser.add_argument("--k_fold", default=5, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=24, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--use_conn", default=False, action="store_true")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="training epoch")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="learning rate")
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


def train(model, args, train_dataloader, dev_dataloader, label_list, tokenizer):
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
        model.eval()
        dev_acc, dev_f1, _ = evaluate(model, args, dev_dataloader, label_list, tokenizer, desc="dev")
        print(" Dev acc=%.4f, f1=%.4f" % (dev_acc, dev_f1))

    output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}_{args.num_train_epochs}")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    return dev_acc, dev_f1


def evaluate(model, args, dataloader, label_list, tokenizer, desc="dev", file_name=None):
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
    """
    cal_acc_f1_score_per_label(
        pred_ids=all_predict_ids, 
        label_ids=all_label_ids, 
        possible_label_ids=all_possible_label_ids, 
        label_list=label_list
    )
    """
    if file_name is not None:
        all_labels = [label_list[int(idx)] for idx in all_label_ids]
        all_predictions = [label_list[int(idx)] for idx in all_predict_ids]
        all_input_texts = [tokenizer.decode(all_input_ids[i], skip_special_tokens=True) for i in range(len(all_input_ids))]
        output_dir = os.path.join(args.data_dir, "base_preds")
        os.makedirs(output_dir, exist_ok=True)
        error_num = 0
        output_dir = os.path.join(args.data_dir, "kfold_preds_{}".format(args.seed))
        os.makedirs(output_dir, exist_ok=True)
        file_name = os.path.join(output_dir, file_name)
        with open(file_name, "w", encoding="utf-8") as f:
            f.write("%-16s\t%-16s\t%-8s\t%s\n"%("Label", "Pred", "Prob", "Text"))
            for label, pred, prob, text in zip(all_labels, all_predictions, all_target_probs, all_input_texts):
                if label == pred:
                    f.write("%-16s\t%-16s\t%.4f\t%s\n"%(label, pred, prob, text))
                else:
                    error_num += 1
                    f.write("%-16s\t%-16s\t%.4f\t%s\n" % (label, pred, prob, str(error_num) + " " + text))

        return acc, f1, (all_labels, all_predictions, all_target_probs, all_input_texts)
    else:
        return acc, f1, None


def produce_vector(model, args, dataloader, label_list, tokenizer, desc="dev"):
    all_vectors = None
    for batch in tqdm(dataloader, desc=desc):
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
        }

        with torch.no_grad():
            vectors = model.get_sent_vectors(**inputs)
            vectors = vectors.detach().cpu().numpy()

        if all_vectors is None:
            all_vectors = vectors
        else:
            all_vectors = np.append(all_vectors, vectors, axis=0)

    return all_vectors


def split_train_k_fold(data_file, k_fold, relation_type="explicit", label_level=1, label_list=None):
    """
    Args:
        split a train file into k fold data list
    Returns:

    """
    # 1. read and filter instances
    all_samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                if sample["relation_type"].lower() != relation_type:
                    continue
                all_level_relation_class = sample["relation_class"].split("##")[0].split(".")
                if len(all_level_relation_class) < label_level:
                    continue
                relation_class = all_level_relation_class[label_level - 1]
                if relation_class.lower() in label_list:
                    all_samples.append((line, relation_class.lower(), sample["conn"].split("##")[0].lower()))

    # 2. split into k-fold
    random.shuffle(all_samples)
    total_size = len(all_samples)
    fold_size = (total_size // k_fold) + 1
    fold_datas = [[] for _ in range(k_fold)]
    fold_labels = [[] for _ in range(k_fold)]

    for idx in range(k_fold):
        start_id = idx * fold_size
        end_id = (idx+1) * fold_size
        fold_samples = all_samples[start_id:end_id]

        cur_datas = [item[0] for item in fold_samples]
        cur_labels = [(item[1], item[2]) for item in fold_samples]
        fold_datas[idx].extend(cur_datas)
        fold_labels[idx].extend(cur_labels)

    return fold_datas, fold_labels


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
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    args.label_level = label_level
    label_list = labels_from_file(os.path.join(data_dir, args.label_file))
    label_dict = {item: idx for idx, item in enumerate(label_list)}
    args.num_labels = len(label_list)
    rel_prefix = args.relation_type.lower()[:3]
    target_rel_prefix = args.target_type.lower()[:3]

    ## 2. define models
    args.model_name_or_path = os.path.join("/hits/basement/nlp/liuwi/resources/pretrained_models", args.model_name_or_path)
    if args.encoder_type.lower() == "bert":
        config = BertConfig.from_pretrained(args.model_name_or_path)
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    elif args.encoder_type.lower() == "roberta":
        config = RobertaConfig.from_pretrained(args.model_name_or_path)
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config.HP_dropout = args.dropout
    print(args.model_name_or_path)

    ## 3. prepare dataset
    dataset_params = {
        "tokenizer": tokenizer,
        "max_seq_length": args.max_seq_length,
        "label_level": label_level,
        "label_list": label_list,
        "relation_type": args.relation_type,
        "use_conn": False
    }

    raw_fold_datas, raw_fold_labels = split_train_k_fold(train_data_file, args.k_fold, args.relation_type, label_level, label_list)
    if args.do_train:
        print(" #################################################### ")
        print(" ########### Start {}_fold filter training ########## ".format(args.k_fold))
        print(" #################################################### ")
        all_res = []
        all_items = []
        for idx in range(args.k_fold):
            cur_train_datas = []
            cur_dev_datas = []
            for idy in range(args.k_fold):
                if idx == idy:
                    cur_dev_datas.extend(raw_fold_datas[idy])
                else:
                    cur_train_datas.extend(raw_fold_datas[idy])
            cur_output_dir = os.path.join(output_dir, "fold_{}_{}".format(args.seed, idx+1))
            args.output_dir = cur_output_dir
            os.makedirs(cur_output_dir, exist_ok=True)

            print(" +++++++++++++++++++++++++++++++++++ ")
            print(" ++++++++ Training {}-fold +++++++++  ".format(idx+1))
            print(" +++++++++++++++++++++++++++++++++++ ")
            model = BaseClassifier(config=config, args=args)
            model = model.to(args.device)

            train_dataset = BaseDataset(cur_train_datas, params=dataset_params)
            dev_dataset = BaseDataset(cur_dev_datas, params=dataset_params)
            train_dataloader = get_dataloader(train_dataset, args, mode="train")
            dev_dataloader = get_dataloader(dev_dataset, args, mode="dev")
            dev_acc, dev_f1 = train(model, args, train_dataloader, dev_dataloader, label_list, tokenizer)
            all_res.append((dev_acc, dev_f1))

            cur_file_name = "{}_{}2{}_train_k{}_l{}.txt".format(
                args.model_name_or_path.split("/")[-1], rel_prefix,
                rel_prefix, idx+1, args.label_level
            )
            _, _, items = evaluate(model, args, dev_dataloader, label_list, tokenizer, "dev", file_name=cur_file_name)
            all_items.append(items)

            # model = None

        # 3. merge
        cur_file_name = "data/dataset/{}/kfold_preds_{}/{}_{}2{}_train_l{}.txt".format(
            args.dataset, args.seed, args.model_name_or_path.split("/")[-1],
            rel_prefix, rel_prefix, args.label_level
        )
        error_num = 0
        with open(cur_file_name, "w", encoding="utf-8") as f:
            f.write("%-16s\t%-16s\t%-8s\t%s\n" % ("Label", "Pred", "Prob", "Text"))
            for items in all_items:
                all_labels, all_preds, all_probs, all_texts = items[0], items[1], items[2], items[3]
                for label, pred, prob, text in zip(all_labels, all_preds, all_probs, all_texts):
                    if label == pred:
                        f.write("%-16s\t%-16s\t%.4f\t%s\n" % (label, pred, prob, text))
                    else:
                        error_num += 1
                        f.write("%-16s\t%-16s\t%.4f\t%s\n" % (label, pred, prob, str(error_num) + " " + text))

        print(" #################################################### ")
        print(" ########## Finish {}_fold filter training ########## ".format(args.k_fold))
        print(" #################################################### ")

    if args.do_dev:
        all_res = []
        all_items = []
        no_conn = False
        with_conn = True

        if no_conn:
            dataset_params["use_conn"] = False
            for idx in range(args.k_fold):
                cur_train_datas = []
                cur_dev_datas = []
                for idy in range(args.k_fold):
                    if idx == idy:
                        cur_dev_datas.extend(raw_fold_datas[idy])
                    else:
                        cur_train_datas.extend(raw_fold_datas[idy])

                dev_dataset = BaseDataset(cur_dev_datas, params=dataset_params)
                dev_dataloader = get_dataloader(dev_dataset, args, mode="dev")

                cur_output_dir = os.path.join(output_dir, "fold_{}_{}".format(args.seed, idx + 1))
                temp_file = os.path.join(cur_output_dir, "checkpoint_{}/pytorch_model.bin".format(args.num_train_epochs))
                model = BaseClassifier(config=config, args=args)
                model = model.to(args.device)
                model.load_state_dict(torch.load(temp_file))
                model.eval()

                cur_file_name = "{}_{}2{}_train_k{}_l{}.txt".format(
                    args.model_name_or_path.split("/")[-1], rel_prefix,
                    rel_prefix, idx+1, args.label_level
                )
                _, _, items = evaluate(model, args, dev_dataloader, label_list, tokenizer, "dev", cur_file_name)
                all_items.append(items)

            cur_file_name = "data/dataset/{}/kfold_preds_{}/{}_{}2{}_train_l{}.txt".format(
                args.dataset, args.seed, args.model_name_or_path.split("/")[-1],
                rel_prefix, rel_prefix, args.label_level
            )
            error_num = 0
            with open(cur_file_name, "w", encoding="utf-8") as f:
                f.write("%-16s\t%-16s\t%-8s\t%s\n" % ("Label", "Pred", "Prob", "Text"))
                for items in all_items:
                    all_labels, all_preds, all_probs, all_texts = items[0], items[1], items[2], items[3]
                    for label, pred, prob, text in zip(all_labels, all_preds, all_probs, all_texts):
                        if label == pred:
                            f.write("%-16s\t%-16s\t%.4f\t%s\n" % (label, pred, prob, text))
                        else:
                            error_num += 1
                            f.write("%-16s\t%-16s\t%.4f\t%s\n" % (label, pred, prob, str(error_num) + " " + text))

        if with_conn:
            dataset_params["use_conn"] = True
            for idx in range(args.k_fold):
                cur_train_datas = []
                cur_dev_datas = []
                for idy in range(args.k_fold):
                    if idx == idy:
                        cur_dev_datas.extend(raw_fold_datas[idy])
                    else:
                        cur_train_datas.extend(raw_fold_datas[idy])

                dev_dataset = BaseDataset(cur_dev_datas, params=dataset_params)
                dev_dataloader = get_dataloader(dev_dataset, args, mode="dev")

                cur_output_dir = os.path.join(output_dir, "fold_{}_{}".format(args.seed, idx + 1))
                temp_file = os.path.join(cur_output_dir, "checkpoint_{}/pytorch_model.bin".format(args.num_train_epochs))
                model = BaseClassifier(config=config, args=args)
                model = model.to(args.device)
                model.load_state_dict(torch.load(temp_file))
                model.eval()

                cur_file_name = "{}_{}2{}_conn_train_k{}_l{}.txt".format(
                    args.model_name_or_path.split("/")[-1], rel_prefix,
                    rel_prefix, idx+1, args.label_level
                )
                _, _, items = evaluate(model, args, dev_dataloader, label_list, tokenizer, "dev", cur_file_name)
                all_items.append(items)

            cur_file_name = "data/dataset/{}/kfold_preds_{}/{}_{}2{}_conn_train_l{}.txt".format(
                args.dataset, args.seed, args.model_name_or_path.split("/")[-1],
                rel_prefix, rel_prefix, args.label_level
            )
            error_num = 0
            with open(cur_file_name, "w", encoding="utf-8") as f:
                f.write("%-16s\t%-16s\t%-8s\t%s\n" % ("Label", "Pred", "Prob", "Text"))
                for items in all_items:
                    all_labels, all_preds, all_probs, all_texts = items[0], items[1], items[2], items[3]
                    for label, pred, prob, text in zip(all_labels, all_preds, all_probs, all_texts):
                        if label == pred:
                            f.write("%-16s\t%-16s\t%.4f\t%s\n" % (label, pred, prob, text))
                        else:
                            error_num += 1
                            f.write("%-16s\t%-16s\t%.4f\t%s\n" % (label, pred, prob, str(error_num) + " " + text))

    if args.do_vector:
        no_conn = True
        with_conn = True

        if no_conn:
            vector_dir = os.path.join(data_dir, "kfold_vectors_{}".format(args.seed))
            os.makedirs(vector_dir, exist_ok=True)
            vector_file = "{}_l{}_{}2{}_train_{}.npz".format(
                args.dataset,
                label_level,
                rel_prefix,
                target_rel_prefix,
                args.model_name_or_path.split("/")[-1],
            )
            print(os.path.exists(os.path.join(vector_dir, vector_file)))
            if os.path.exists(os.path.join(vector_dir, vector_file)):
                print(" vector file existing!!!!!")
                with np.load(os.path.join(vector_dir, vector_file)) as dataset:
                    all_vectors = dataset["vectors"]
                    all_labels = dataset["labels"]
            else:
                all_vectors = []
                all_labels = []
                dataset_params["use_conn"] = False
                dataset_params["relation_type"] = args.target_type
                for idx in range(args.k_fold):
                    cur_train_datas = []
                    cur_dev_datas = []
                    for idy in range(args.k_fold):
                        if idx == idy:
                            cur_dev_datas.extend(raw_fold_datas[idy])
                        else:
                            cur_train_datas.extend(raw_fold_datas[idy])

                    dev_dataset = BaseDataset(cur_dev_datas, params=dataset_params)
                    dev_dataloader = get_dataloader(dev_dataset, args, mode="dev")
                    cur_output_dir = os.path.join(output_dir, "fold_{}_{}".format(args.seed, idx + 1))
                    temp_file = os.path.join(
                        cur_output_dir, "checkpoint_{}/pytorch_model.bin".format(args.num_train_epochs)
                    )
                    model = BaseClassifier(config=config, args=args)
                    model = model.to(args.device)
                    model.load_state_dict(torch.load(temp_file))
                    model.eval()

                    cur_vectors = produce_vector(
                        model, args, dev_dataloader,
                        label_list, tokenizer, "dev"
                    )
                    all_vectors.append(cur_vectors)
                    cur_label_ids = [label_dict[l[0]] for l in raw_fold_labels[idx]]
                    all_labels.append(np.array(cur_label_ids))

                # save to file
                all_vectors = np.concatenate(all_vectors, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)
                np.savez(
                    os.path.join(vector_dir, vector_file),
                    vectors=all_vectors,
                    labels=all_labels
                )

            jpg_file = "images/{}_l{}_kfold_{}_{}2{}_40+train.eps".format(
                           args.dataset, label_level, args.model_name_or_path.split("/")[-1],
                           rel_prefix, target_rel_prefix
                       )
            # visualize_sent_vector((all_vectors, all_labels), label_list=label_list, perplexity=10, jpg_path=jpg_file)
        no_conn_vectors = all_vectors

        if with_conn:
            vector_dir = os.path.join(data_dir, "kfold_vectors_{}".format(args.seed))
            os.makedirs(vector_dir, exist_ok=True)
            vector_file = "{}_l{}_{}2{}_conn_train_{}.npz".format(
                args.dataset,
                label_level,
                rel_prefix,
                target_rel_prefix,
                args.model_name_or_path.split("/")[-1]
            )
            if os.path.exists(os.path.join(vector_dir, vector_file)):
                print(" vector file existing!!!!!!")
                with np.load(os.path.join(vector_dir, vector_file)) as dataset:
                    all_vectors = dataset["vectors"]
                    all_labels = dataset["labels"]
            else:
                all_vectors = []
                all_labels = []
                dataset_params["use_conn"] = True
                dataset_params["relation_type"] = args.target_type
                for idx in range(args.k_fold):
                    cur_train_datas = []
                    cur_dev_datas = []
                    for idy in range(args.k_fold):
                        if idx == idy:
                            cur_dev_datas.extend(raw_fold_datas[idy])
                        else:
                            cur_train_datas.extend(raw_fold_datas[idy])

                    dev_dataset = BaseDataset(cur_dev_datas, params=dataset_params)
                    dev_dataloader = get_dataloader(dev_dataset, args, mode="dev")
                    cur_output_dir = os.path.join(output_dir, "fold_{}_{}".format(args.seed, idx + 1))
                    temp_file = os.path.join(
                        cur_output_dir, "checkpoint_{}/pytorch_model.bin".format(args.num_train_epochs)
                    )
                    model = BaseClassifier(config=config, args=args)
                    model = model.to(args.device)
                    model.load_state_dict(torch.load(temp_file))
                    model.eval()

                    cur_vectors = produce_vector(
                        model, args, dev_dataloader,
                        label_list, tokenizer, "dev"
                    )
                    all_vectors.append(cur_vectors)
                    cur_label_ids = [label_dict[l[0]] for l in raw_fold_labels[idx]]
                    all_labels.append(np.array(cur_label_ids))

                # save to file
                all_vectors = np.concatenate(all_vectors, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)
                np.savez(
                    os.path.join(vector_dir, vector_file),
                    vectors=all_vectors,
                    labels=all_labels
                )

            jpg_file = "images/{}_l{}_kfold_{}_{}2{}_40+conn_train.eps".format(
                           args.dataset, label_level, args.model_name_or_path.split("/")[-1],
                           rel_prefix, target_rel_prefix
                       )
            # visualize_sent_vector((all_vectors, all_labels), label_list=label_list, perplexity=10, jpg_path=jpg_file)
        with_conn_vectors = all_vectors

        print("#################################################################################################")
        avg_score, min_score, max_score, all_scores = cal_distance(no_conn_vectors, with_conn_vectors, distance_type="cos")
        print("%s: %s to %s"%("Train", args.relation_type, args.target_type))
        print("Total length: %d"%(len(all_scores)))
        print("cos sim: avg=%.4f, min=%.4f, max=%.4f"%(avg_score, min_score, max_score))

        masks_0 = (all_scores < 0.0)
        cos0 = int(torch.sum(masks_0)) / len(all_scores)
        masks = (all_scores < 0.4)
        cos4 = int(torch.sum(masks)) / len(all_scores)
        masks = (all_scores < 0.5)
        cos5 = int(torch.sum(masks)) / len(all_scores)
        masks = (all_scores < 0.8)
        cos8 = int(torch.sum(masks)) / len(all_scores)

        print("cos0: %.4f, cos4: %.4f, cos5: %.4f, cos8: %.4f"%(cos0, cos4, cos5, cos8))

        print("#################################################################################################")

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

        print("#################################################################################################")

if __name__ == "__main__":
    main()
