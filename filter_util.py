# author = liuwei
# date = 2023-03-08

import os
import json
import numpy as np
from collections import defaultdict
from utils import labels_from_file, cal_distance
import random

def filter_samples_with_distance_per_label(
    dataset="pdtb2",
    label_level=1,
    model_name_or_path="roberta-base",
    source_type="explicit",
    target_type="explicit",
    mode="train",
    min_confidence=0.0,
    do_write=False,
):
    """
    We check the performance reaction of BaseClassifier when connective are fed into the model.
    Args:
        dataset: pdtb2 or pdtb3
        model_name_or_path: which model are used for initialization, bert-base, roberta-base,....
        source_type: exp or imp
        target_type: exp or imp
        mode: train, dev, test
        epoch:
        level: 1 or 2
    """
    ## 1, data path
    # """
    data_file = "data/dataset/{}/{}.json".format(dataset, mode)
    default_pred_file = "data/dataset/{}/kfold_preds_106524/{}_{}2{}_{}_l{}.txt".format(
        dataset, model_name_or_path, source_type[:3],
        target_type[:3], mode, label_level
    )
    conn_pred_file = "data/dataset/{}/kfold_preds_106524/{}_{}2{}_conn_{}_l{}.txt".format(
        dataset, model_name_or_path, source_type[:3],
        target_type[:3], mode, label_level
    )
    default_vector_file = "data/dataset/{}/kfold_vectors_106524/{}_l{}_{}2{}_{}_{}.npz".format(
        dataset, dataset, label_level, source_type[:3],
        target_type[:3], mode, model_name_or_path
    )
    conn_vector_file = "data/dataset/{}/kfold_vectors_106524/{}_l{}_{}2{}_conn_{}_{}.npz".format(
        dataset, dataset, label_level, source_type[:3],
        target_type[:3], mode, model_name_or_path
    )
    # """

    print(data_file)
    print(default_pred_file)
    print(conn_pred_file)
    print(default_vector_file)
    print(conn_vector_file)

    ## 2, read and calculate
    all_gt_res, all_raw_datas = read_json(data_file, target_type, label_level)
    all_default_res = read_txt(default_pred_file)
    all_conn_res = read_txt(conn_pred_file)
    all_default_vec, all_label_ids = read_npz(default_vector_file)
    all_conn_vec, _ = read_npz(conn_vector_file)
    assert len(all_gt_res) == len(all_default_res), (len(all_gt_res), len(all_default_res))
    assert len(all_gt_res) == len(all_conn_res), (len(all_gt_res), len(all_conn_res))
    assert len(all_gt_res) == len(all_default_vec), (len(all_gt_res), len(all_default_vec))
    assert len(all_gt_res) == len(all_conn_vec), (len(all_gt_res), len(all_conn_vec))

    total_size = len(all_gt_res)
    print("Total size: ", total_size)

    _, _, _, all_distances = cal_distance(all_default_vec, all_conn_vec, distance_type="cos")
    per_label_threshold = get_per_label_threshold(all_distances, all_label_ids)

    filter_texts = []
    filter_scores = []
    ambiguous_texts = []
    ambiguous_scores = []
    conn_list = set()
    fine_num = 0
    weak_num = 0
    ambi_num = 0
    for idx in range(total_size):
        gt = all_gt_res[idx]
        raw_text = all_raw_datas[idx]
        default_pred = all_default_res[idx]
        conn_pred = all_conn_res[idx]
        distance = float(all_distances[idx])
        l_id = int(all_label_ids[idx])

        conn_name = gt[1].split("##")[0]
        conn_list.add(conn_name)
        gt_label = gt[0].lower()
        rel_name = default_pred[0].strip().lower()
        d_score = float(default_pred[2].strip())
        c_score = float(conn_pred[2].strip())

        if distance >= per_label_threshold[l_id]:
            if c_score >= min_confidence:
                filter_texts.append(raw_text)
                filter_scores.append((distance, d_score, c_score))
                fine_num += 1
            else:
                weak_num += 1
        else:
            ambi_num += 1
            ambiguous_texts.append(raw_text)
            ambiguous_scores.append((distance, d_score, c_score))

    print("(%.2f) for %s, fine num:%d, weak num: %d, ambi num: %d" % (
            min_confidence, mode, fine_num, weak_num, ambi_num
        ))

    if "roberta-large" in default_pred_file:
        model_type = "RL"
    elif "roberta-base" in default_pred_file:
        model_type = "RB"
    elif "bert-base" in default_pred_file:
        model_type = "BB"
    elif "bert_large" in default_pred_file:
        model_type = "BL"
    new_dataset = "{}_l{}_{}_(avg_{})".format(dataset, label_level, model_type, int(100*min_confidence))
    if do_write:
        out_dir = os.path.join("data/dataset", new_dataset)
        os.makedirs(out_dir, exist_ok=True)
        file_name = os.path.join(out_dir, "train.json")
        print(file_name)
        with open(file_name, "w", encoding="utf-8") as f:
            for text, score in zip(filter_texts, filter_scores):
                sample = json.loads(text)
                sample["distance"] = score[0]
                sample["c_score"] = score[1]
                sample["d_score"] = score[2]
                f.write("%s\n" % (json.dumps(sample, ensure_ascii=False)))

    return filter_texts, new_dataset