export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0

python train.py \
    --train_csv_file "data/train.csv"\
    --test_csv_file "data/test.csv" \
    --seed 0 \
    --num_pretrain_epochs 50\
    --save_path "checkpoints"\
    --bert_model "bert-base-chinese"\
    --train_batch_size 32

