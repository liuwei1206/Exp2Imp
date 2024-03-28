# <<"COMMENT"
python3 kfold_base.py --do_vector \
                      --dataset="pdtb2"  \
                      --label_file="labels_1.txt" \
                      --relation_type="explicit" \
                      --target_type="explicit" \
                      --num_train_epochs=10 \
                      --learning_rate=1e-5 \
                      --train_batch_size=16
# COMMENT

<<"COMMENT"
for dataset in pdtb2 pdtb3
do
    for rel_type in explicit
    do
        for level in 1 2
        do
            python3 kfold_base.py --do_train \
                                  --dataset="pdtb2"  \
                                  --label_file="labels_1.txt" \
                                  --relation_type="explicit" \
                                  --target_type="explicit" \
                                  --num_train_epochs=10 \
                                  --learning_rate=1e-5 \
                                  --train_batch_size=16
        done
    done
done
COMMENT
