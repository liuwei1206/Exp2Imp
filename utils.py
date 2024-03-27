# author = liuwei
# date = 2023-03-08

import os
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import torch
import math
import random
from collections import defaultdict

from sklearn import manifold
import matplotlib.pyplot as plt
import torch.nn.functional as F

random.seed(106524)

def labels_from_file(label_file):
    label_list = []
    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                label_list.append(line.strip().lower())

    return label_list


def cal_acc_f1_score_with_ids(pred_ids, label_ids, possible_label_ids):
    """
        sample_size: N
        label_size: V
        Args:
            pred_ids: [N]
            label_ids: [N]
            possible_label_ids: [N, V]
        note, each sample in implicit discourse may have more than one label
        if the predicted label match one of those labels, then the prediction is considered
        as true
        """
    # """
    # pred_ids, label_ids, possible_label_ids = l2_to_l1(pred_ids, label_ids, possible_label_ids)
    # print(possible_label_ids.shape)
    extend_label_ids = []
    for idx, p in enumerate(pred_ids):
        if possible_label_ids[idx, p] == 1:
            extend_label_ids.append(p)
        else:
            extend_label_ids.append(label_ids[idx])
    label_ids = np.array(extend_label_ids)

    acc = accuracy_score(y_true=label_ids, y_pred=pred_ids)
    p = precision_score(y_true=label_ids, y_pred=pred_ids, average="macro")
    r = recall_score(y_true=label_ids, y_pred=pred_ids, average="macro")
    f1 = f1_score(y_true=label_ids, y_pred=pred_ids, average="macro")

    return acc, p, r, f1, label_ids


def cal_acc_f1_score_per_label(pred_ids, label_ids, possible_label_ids, label_list):
    """
    sample_size: N
    label_size: V
    Args:
        pred_ids: [N]
        label_ids: [N]
        possible_label_ids: [N, V]
    note, each sample in implicit discourse may have more than one label
    if the predicted label match one of those labels, then the prediction is considered
    as true
    """
    extend_label_ids = []
    for idx, p in enumerate(pred_ids):
        if possible_label_ids[idx, p] == 1:
            extend_label_ids.append(p)
        else:
            extend_label_ids.append(label_ids[idx])
    label_ids = np.array(extend_label_ids)
    res = classification_report(y_true=label_ids, y_pred=pred_ids, target_names=label_list, digits=4)
    print(res)

    return res

def get_connectives_from_data(data_list):
    conn_list = {}
    for data in data_list:
        sample = json.loads(data)
        conns = sample["conn"].split("##")
        for conn in conns:
            conn = conn.strip()
            if conn in conn_list:
                conn_list[conn] += 1
            else:
                conn_list[conn] = 1

    conn_list = sorted(conn_list.items(), key=lambda x: x[1])
    print(conn_list)
    all_connectives = []
    all_connectives.append("<unk>")
    for item in conn_list:
        if item[1] > 0:
            all_connectives.append(item[0])
    print(" connective number: ", len(all_connectives))

    return all_connectives


def get_connectives_from_file(train_file, relation_type="explicit"):
    conn_list = {}
    with open(train_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                rel_type = sample["relation_type"].lower()
                if rel_type != relation_type:
                    continue

                conns = sample["conn"].split("##")
                for conn in conns:
                    conn = conn.strip()
                    if conn in conn_list:
                        conn_list[conn] += 1
                    else:
                        conn_list[conn] = 1

    # all_connectives = list(conn_list.keys())
    # all_connectives.append("<unk>")
    conn_list = sorted(conn_list.items(), key=lambda x:x[1])
    print(conn_list)
    all_connectives = []
    all_connectives.append("<unk>")
    for item in conn_list:
        if item[1] > 0: # for pdtbs, > 0
            all_connectives.append(item[0])
    print(" connective number: ", len(all_connectives))

    return all_connectives


def get_onehot_conn_from_vocab(conns, tokenizer):
    """
    get the token_ids of each connective
    Args:
        conns:
        tokenizer
    """
    vocab_size = tokenizer.vocab_size
    conn_num = len(conns)
    conn_onehot_in_vocab = torch.zeros((conn_num, vocab_size)).float()
    conn_length_in_vocab = []

    for idx, conn in enumerate(conns):
        if "roberta" in tokenizer.__class__.__name__.lower():
            conn_tokens = tokenizer.tokenize(" " + conn.capitalize())
        else:
            conn_tokens = tokenizer.tokenize(conn)
        # print(conn, " : ", conn_tokens)
        conn_length_in_vocab.append(len(conn_tokens))
        conn_token_ids = tokenizer.convert_tokens_to_ids(conn_tokens)
        conn_token_ids = torch.tensor(conn_token_ids).long()
        conn_onehot_in_vocab[idx, conn_token_ids] = 1

    return conn_onehot_in_vocab, torch.tensor(conn_length_in_vocab).float()

def data_to_file(file_name, datas):
    with open(file_name, "w", encoding="utf-8") as f:
        for data in datas:
            f.write("%s\n"%(data))


def visualize_sent_vector(vectors_or_file, label_list, learning_rate="auto", perplexity=10, jpg_path=None):
    axis_vals = None
    print(jpg_path)
    """
    if "pdtb2_l1_exp" in jpg_path:
        axis_vals = [-103, 145, -100, 100]
    elif "pdtb2_l2_exp" in jpg_path:
        axis_vals = [-100, 190, -100, 100]
    elif "pdtb2_l1_imp" in jpg_path:
        axis_vals = [-103, 145, -100, 100]
    elif "pdtb2_l2_imp" in jpg_path:
        axis_vals = [-100, 190, -102, 105]
    elif "pdtb3_l1_exp" in jpg_path:
        axis_vals = [-100, 130, -100, 102]
    elif "pdtb3_l2_exp" in jpg_path:
        axis_vals = [-105, 185, -105, 110]
    elif "pdtb3_l1_imp" in jpg_path:
        axis_vals = [-100, 130, -100, 100]
    elif "pdtb3_l2_imp" in jpg_path:
        axis_vals = [-100, 192, -116, 106]
    elif "gum_l1_exp" in jpg_path:
        axis_vals = [-120, 200, -120, 120]
    elif "conjunction_l1_exp" in jpg_path:
        axis_vals = [-105, 130, -105, 108]
    elif "adverb_l1_exp" in jpg_path:
        axis_vals = [-85, 105, -80, 85]
    elif "ambi_l1_exp" in jpg_path:
        axis_vals = [-80, 100, -60, 80]
    elif "non-ambi_l1_exp" in jpg_path:
        axis_vals = [-100, 125, -100, 105]
    """

    # """
    if "pdtb2_l1_exp" in jpg_path:
        axis_vals = [-103, 108, -100, 100]
    elif "pdtb2_l2_exp" in jpg_path:
        axis_vals = [-100, 190, -100, 100]
    elif "pdtb2_l1_imp" in jpg_path:
        axis_vals = [-103, 108, -100, 100]
    elif "pdtb2_l2_imp" in jpg_path:
        axis_vals = [-100, 190, -102, 105]
    elif "pdtb3_l1_exp" in jpg_path:
        axis_vals = [-100, 130, -100, 102]
    elif "pdtb3_l2_exp" in jpg_path:
        axis_vals = [-105, 185, -105, 110]
    elif "pdtb3_l1_imp" in jpg_path:
        axis_vals = [-100, 130, -100, 100]
    elif "pdtb3_l2_imp" in jpg_path:
        axis_vals = [-100, 192, -116, 106]
    elif "conjunction_l1_exp" in jpg_path:
        axis_vals = [-105, 103, -105, 90]
    elif "adverb_l1_exp" in jpg_path:
        axis_vals = [-80, 80, -80, 80]
    elif "ambi_l1_exp" in jpg_path:
        axis_vals = [-80, 80, -60, 80]
    elif "non-ambi_l1_exp" in jpg_path:
        axis_vals = [-100, 90, -100, 90]
    elif "base_conjunction" in jpg_path:
        axis_vals = [-100, 100, -100, 100]
    elif "base_adverbial" in jpg_path:
        axis_vals = [-100, 100, -100, 100]
    # """

    all_vectors = None
    all_labels = None
    if isinstance(vectors_or_file, str):
        with np.load(vectors_or_file) as dataset:
            # all_vectors = dataset["sentence_vectors"]
            all_vectors = dataset["vectors"]
            all_labels = dataset["labels"]
    else:
        all_vectors = vectors_or_file[0]
        all_labels = vectors_or_file[1]

    tsne = manifold.TSNE(n_components=2, learning_rate=learning_rate, perplexity=perplexity, random_state=106524)
    low_dim_X = tsne.fit_transform(X=all_vectors)
    unique_labels = np.unique(all_labels)

    colors = [
        "red", "blue", "gray", "black", "yellow",
        "darkseagreen", "darkgreen", "olive", "orange",
        "pink", "deepskyblue", "purple", "darkred", "darkcyan"
    ]
    markers = ['o', 's', '^', 'D', '8', 'p', 'v', '*', 'd', 'x', 'h', '<', '>', '+']
    X_groups = [low_dim_X[all_labels == label] for label in unique_labels]
    for idx, idx_X in enumerate(X_groups):
        plt.scatter(
            idx_X[:, 0], idx_X[:, 1],
            c=colors[idx],
            marker=markers[idx],
            label=label_list[unique_labels[idx]],
            s=5, # the size of marker
            linewidths=0.4 # make marker thin
        )

    # jpg_path = "images/one.jpg"
    if axis_vals is not None:
        print(axis_vals)
        plt.axis(axis_vals)
    # plt.legend(fontsize=12, markerscale=2.5, loc="upper right") # loc="upper center", ncols=3
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.legend(fontsize=12, markerscale=2, loc="upper center", ncols=4)
    plt.tight_layout()
    plt.savefig(jpg_path, dpi=600, format='eps', bbox_inches='tight')
    plt.clf()


def cal_distance(vectors_1, vectors_2, conns=None, batch_size=128, distance_type="cos"):
    vectors_1 = torch.tensor(vectors_1)
    vectors_2 = torch.tensor(vectors_2)
    assert vectors_1.size() == vectors_2.size(), (vectors_1.size(), vectors_2.size())
    data_length = vectors_1.size(0)
    all_scores = []
    steps = (data_length // batch_size) + 1
    for idx in range(steps):
        start_pos = idx * batch_size
        end_pos = (idx + 1) * batch_size
        cur_vectors_1 = vectors_1[start_pos:end_pos, :]
        cur_vectors_2 = vectors_2[start_pos:end_pos, :]
        if distance_type == "cos":
            cur_score = F.cosine_similarity(cur_vectors_1, cur_vectors_2, dim=1)
        elif distance_type == "euc":
            cur_score = F.pairwise_distance(cur_vectors_1, cur_vectors_2)
        all_scores.append(cur_score)

    all_scores = torch.cat(all_scores, dim=0)
    # print(all_scores[:100])
    if conns is not None:
        conn_set = set(conns)
        avg_res = {}
        min_res = {}
        max_res = {}
        for each_conn in conn_set:
            mask = (conns == each_conn)
            if int(np.sum(mask)) < 10:
                continue
            mask = torch.tensor(mask)
            cur_conn_scores = all_scores[mask]
            cur_avg_score = torch.mean(cur_conn_scores)
            cur_min_score = torch.min(cur_conn_scores)
            cur_max_score = torch.max(cur_conn_scores)
            avg_res[each_conn] = cur_avg_score
            min_res[each_conn] = cur_min_score
            max_res[each_conn] = cur_max_score

        return avg_res, min_res, max_res, cur_conn_scores
    else:
        avg_score = torch.mean(all_scores)
        min_score = torch.min(all_scores)
        max_score = torch.max(all_scores)

        return avg_score, min_score, max_score, all_scores


def read_json(data_file, relation_type="implicit", label_level=1):
    label_file = os.path.join(os.path.dirname(data_file), "labels_{}.txt".format(label_level))
    label_list = labels_from_file(label_file)
    print(label_list)
    frequency_dict = defaultdict(int)
    all_raw_datas = []

    with open(data_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                if sample["relation_type"].lower() not in relation_type:
                        continue

                all_level_relation_class = sample["relation_class"].split("##")[0].split(".")
                if len(all_level_relation_class) >= label_level:
                    relation_class = all_level_relation_class[label_level-1].lower()
                else:
                    relation_class = None
                if (relation_class is None) or (relation_class not in label_list):
                    continue

                conn = sample["conn"].split("##")[0].lower()
                arg1 = sample["arg1"]
                arg2 = sample["arg2"]

                frequency_dict["{}+{}".format(relation_class, conn)] += 1
                all_raw_datas.append((relation_class, conn, arg1, arg2, sample))

    return frequency_dict, all_raw_datas


def kfold_read_json(all_datas, relation_type="implicit", label_level=1):
    label_file = os.path.join("data/dataset/pdtb2", "labels_{}.txt".format(label_level))
    label_list = labels_from_file(label_file)
    # print(label_list)
    frequency_dict = defaultdict(int)
    all_raw_datas = []

    for line in all_datas:
        line = line.strip()
        if line:
            sample = json.loads(line)
            all_level_relation_class = sample["relation_class"].split("##")[0].split(".")
            relation_class = all_level_relation_class[label_level - 1].lower()
            conn = sample["conn"].split("##")[0].lower()
            arg1 = sample["arg1"]
            arg2 = sample["arg2"]
            # print(conn)

            frequency_dict["{}+{}".format(relation_class, conn)] += 1
            all_raw_datas.append((relation_class, conn, arg1, arg2, sample))

    return frequency_dict, all_raw_datas

