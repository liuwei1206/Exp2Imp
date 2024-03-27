# <<"COMMENT"
python3 filter_joint.py --do_train \
                        --dataset="pdtb2"  \
                        --label_file="labels_1.txt" \
                        --relation_type="explicit" \
                        --min_confidence=0.5 \
                        --train_batch_size=16 \
                        --learning_rate=1e-5 \
                        --num_train_epochs=10
# COMMENT

<<"COMMENT"
python3 filter_joint.py --do_train \
                        --dataset="pdtb2"  \
                        --label_file="labels_2.txt" \
                        --relation_type="explicit" \
                        --min_confidence=0.005 \
                        --train_batch_size=16 \
                        --learning_rate=1e-5 \
                        --num_train_epochs=10
COMMENT

<<"COMMENT"
python3 filter_joint.py --do_train \
                        --dataset="pdtb3"  \
                        --label_file="labels_1.txt" \
                        --relation_type="explicit" \
                        --min_confidence=0.4 \
                        --train_batch_size=16 \
                        --learning_rate=1e-5 \
                        --num_train_epochs=10
COMMENT

<<"COMMENT"
python3 filter_joint.py --do_train \
                        --dataset="pdtb3"  \
                        --label_file="labels_2.txt" \
                        --relation_type="explicit" \
                        --min_confidence=0.05 \
                        --train_batch_size=16 \
                        --learning_rate=1e-5 \
                        --num_train_epochs=10
COMMENT