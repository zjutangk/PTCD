export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0

python test.py \
    --train_csv_file "data/train.csv" \
    --test_csv_file "data/test.csv"  \
    --seed 0 \
    --num_pretrain_epochs 50\
    --pretrain_batch_size 1\
    --save_path "checkpoints"