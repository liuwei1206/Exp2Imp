# <<"COMMENT"
python3 joint.py --do_train \
                 --dataset="pdtb2_l1_RB_avg" \
                 --label_file="labels_1.txt" \
                 --relation_type="explicit" \
                 --train_batch_size=16 \
                 --learning_rate=1e-5 \
                 --num_train_epochs=10
# COMMENT
