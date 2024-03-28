## for pdtb2 l1
# <<"COMMENT"
for seed in 106524 106596 106476 423455 114290
do
    python3 filter_joint.py --do_train \
                            --dataset="gum7"  \
                            --label_file="labels_1.txt" \
                            --relation_type="explicit" \
                            --min_confidence=0.0 \
                            --train_batch_size=16 \
                            --learning_rate=1e-5 \
                            --num_train_epochs=15 \
                            --seed=${seed}
done
# COMMENT

## for pdtb l2
<<"COMMENT"
for seed in 106524 106596 106476 423455 114290
do
    python3 filter_joint.py --do_train \
                            --dataset="pdtb2"  \
                            --label_file="labels_2.txt" \
                            --relation_type="explicit" \
                            --min_confidence=0.005 \
                            --train_batch_size=16 \
                            --learning_rate=1e-5 \
                            --num_train_epochs=10 \
                            --seed=${seed}
done
COMMENT

## for pdtb3 l1
<<"COMMENT"
for seed in 106524 106596 106476 423455 114290
do
    python3 filter_joint.py --do_train \
                            --dataset="pdtb3"  \
                            --label_file="labels_1.txt" \
                            --relation_type="explicit" \
                            --min_confidence=0.4 \
                            --train_batch_size=16 \
                            --learning_rate=1e-5 \
                            --num_train_epochs=10 \
                            --seed=${seed}
done
COMMENT

## for pdtb3 l2
<<"COMMENT"
for seed in 106524 106596 106476 423455 114290
do
    python3 filter_joint.py --do_train \
                            --dataset="pdtb3"  \
                            --label_file="labels_2.txt" \
                            --relation_type="explicit" \
                            --min_confidence=0.05 \
                            --train_batch_size=16 \
                            --learning_rate=1e-5 \
                            --num_train_epochs=10 \
                            --seed=${seed}
done
COMMENT


## for gum7
<<"COMMENT"
for seed in 106524 106596 106476 423455 114290
do
    python3 filter_joint.py --do_train \
                            --dataset="gum7"  \
                            --label_file="labels_1.txt" \
                            --relation_type="explicit" \
                            --min_confidence=0.05 \
                            --train_batch_size=16 \
                            --learning_rate=1e-5 \
                            --num_train_epochs=15 \
                            --seed=${seed}
done
COMMENT

