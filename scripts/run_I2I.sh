## for pdtb 2.0
<<"COMMENT"
python3 analyze.py --do_train \
                   --dataset="pdtb2"  \
                   --label_file="labels_1.txt" \
                   --relation_type="implicit" \
                   --train_batch_size=16 \
                   --learning_rate=1e-5 \
                   --num_train_epochs=10
COMMENT
<<"COMMENT"

<<"COMMENT"
python3 analyze.py --do_train \
                   --dataset="pdtb2"  \
                   --label_file="labels_2.txt" \
                   --relation_type="implicit" \
                   --train_batch_size=16 \
                   --learning_rate=1e-5 \
                   --num_train_epochs=10
COMMENT


## for pdtb 3.0
<<"COMMENT"
python3 analyze.py --do_train \
                   --dataset="pdtb3"  \
                   --label_file="labels_1.txt" \
                   --relation_type="implicit" \
                   --train_batch_size=16 \
                   --learning_rate=1e-5 \
                   --num_train_epochs=10
COMMENT
<<"COMMENT"

<<"COMMENT"
python3 analyze.py --do_train \
                   --dataset="pdtb3"  \
                   --label_file="labels_2.txt" \
                   --relation_type="implicit" \
                   --train_batch_size=16 \
                   --learning_rate=1e-5 \
                   --num_train_epochs=10
COMMENT


## for gum7
<<"COMMENT"
python3 analyze.py --do_train \
                   --dataset="gum7"  \
                   --label_file="labels_1.txt" \
                   --relation_type="implicit" \
                   --train_batch_size=16 \
                   --learning_rate=1e-5 \
                   --num_train_epochs=15
COMMENT
<<"COMMENT"
