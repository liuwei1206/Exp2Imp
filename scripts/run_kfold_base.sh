# <<"COMMENT"
python3 kfold_base.py --do_dev --do_vector \
                      --dataset="gum7"  \
                      --label_file="labels_1.txt" \
                      --relation_type="explicit" \
                      --target_type="explicit" \
                      --num_train_epochs=10 \
                      --learning_rate=1e-5 \
                      --train_batch_size=16
# COMMENT

# for PDTB 2.0 and PDTB 3.0, preparing for filtering
<<"COMMENT"
for dataset in pdtb2 pdtb3
do
    for rel_type in explicit
    do
        for level in 1 2
        do
            python3 kfold_base.py --do_train \
                                  --dataset=${dataset}  \
                                  --label_file="labels_${level}.txt" \
                                  --relation_type=${rel_type} \
                                  --target_type=${rel_type} \
                                  --num_train_epochs=10 \
                                  --learning_rate=1e-5 \
                                  --train_batch_size=16

            python3 kfold_base.py --do_dev \
                                  --dataset=${dataset}  \
                                  --label_file="labels_${level}.txt" \
                                  --relation_type=${rel_type} \
                                  --target_type=${rel_type} \
                                  --num_train_epochs=10 \
                                  --learning_rate=1e-5 \
                                  --train_batch_size=16

            python3 kfold_base.py --do_vector \
                                  --dataset=${dataset}  \
                                  --label_file="labels_${level}.txt" \
                                  --relation_type=${rel_type} \
                                  --target_type=${rel_type} \
                                  --num_train_epochs=10 \
                                  --learning_rate=1e-5 \
                                  --train_batch_size=16
        done
    done
done
COMMENT

# for gum7, preparing for filtering
<<"COMMENT"
for dataset in gum7
do
    for rel_type in explicit
    do
        for level in 1
        do
            python3 kfold_base.py --do_train \
                                  --dataset=${dataset}  \
                                  --label_file="labels_${level}.txt" \
                                  --relation_type=${rel_type} \
                                  --target_type=${rel_type} \
                                  --num_train_epochs=10 \
                                  --learning_rate=1e-5 \
                                  --train_batch_size=16

            python3 kfold_base.py --do_dev \
                                  --dataset=${dataset}  \
                                  --label_file="labels_${level}.txt" \
                                  --relation_type=${rel_type} \
                                  --target_type=${rel_type} \
                                  --num_train_epochs=10 \
                                  --learning_rate=1e-5 \
                                  --train_batch_size=16

            python3 kfold_base.py --do_vector \
                                  --dataset=${dataset}  \
                                  --label_file="labels_${level}.txt" \
                                  --relation_type=${rel_type} \
                                  --target_type=${rel_type} \
                                  --num_train_epochs=10 \
                                  --learning_rate=1e-5 \
                                  --train_batch_size=16
        done
    done
done
COMMENT

