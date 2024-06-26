# <<"COMMENT"
python3 analyze.py --do_train \
                   --dataset="pdtb2"  \
                   --label_file="labels_1.txt" \
                   --relation_type="explicit" \
                   --train_batch_size=16 \
                   --learning_rate=1e-5 \
                   --num_train_epochs=10
# COMMENT

<<"COMMENT"
for dataset in pdtb2 pdtb3
do
    for rel_type in explicit implicit
    do
        for level in 1 2
        do
            python3 analyze.py --do_train \
                               --dataset=${dataset}  \
                               --label_file="labels_${level}.txt" \
                               --relation_type=${rel_type} \
                               --train_batch_size=16 \
                               --learning_rate=1e-5 \
                               --num_train_epochs=10

            python3 analyze.py --do_dev \
                               --dataset=${dataset}  \
                               --label_file="labels_${level}.txt" \
                               --relation_type=${rel_type} \
                               --train_batch_size=16 \
                               --learning_rate=1e-5 \
                               --num_train_epochs=10

            python3 analyze.py --do_vector \
                               --dataset=${dataset}  \
                               --label_file="labels_${level}.txt" \
                               --relation_type=${rel_type} \
                               --train_batch_size=16 \
                               --learning_rate=1e-5 \
                               --num_train_epochs=10
        done
    done
done
COMMENT
