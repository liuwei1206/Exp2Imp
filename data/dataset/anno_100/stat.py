# author = liuwei
# date = 2023-07-08

import json
import os
import numpy as np
from sklearn.metrics import cohen_kappa_score as cohen

np.random.seed(106524)

def read_labels_from_file(file_name, label_level=1):
	labels = []
	with open(file_name, "r", encoding="utf-8") as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip()
			if line:
				items = line.split("\t")
				label_item = items[-1].strip()
				label_item = label_item.split(".")
				label = label_item[label_level-1].strip().lower()

				labels.append(label)

	return labels 


def statistics(golds, anno1, anno2):
	shift_num = 0
	case1 = 0
	case2 = 0
	case3 = 0

	## 1. calculate inter-annotator agreement
	cohen_score = cohen(anno1, anno2)
	print("Inter-annotator Agreement: %.4f"%(cohen_score))

	## 2. calculate label shift and cases
	for g, a1, a2 in zip(golds, anno1, anno2):
		if g == a1 and g == a2:
			pass
		else:
			shift_num += 1
			if g == a1 or g == a2:
				case2 += 1
			elif "norel" == a1 or "norel" == a2:
				case3 += 1
			else:
				case1 += 1


	print("shift num: %d, case1: %d, case2: %d, case3: %d"%(shift_num, case1, case2, case3))


if __name__ == '__main__':
	gold_file = "sample_label_100.txt"
	anno1_file = "anno1.txt"
	anno2_file = "anno2.txt"
	label_level = 1

	golds = read_labels_from_file(gold_file, label_level)
	anno1 = read_labels_from_file(anno1_file, label_level)
	anno2 = read_labels_from_file(anno2_file, label_level)
	num = 0
	for a1, a2 in zip(anno1, anno2):
		if a1 == a2:
			num += 1
	print(num)

	statistics(golds, anno1, anno2)


