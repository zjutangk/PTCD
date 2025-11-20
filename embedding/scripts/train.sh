WORK_DIR=$(pwd)
echo $WORK_DIR

MODEL_DIR=$WORK_DIR/pretrained_model/bert

DATA_DIR="data"
export HF_DATASETS_CACHE="cache"
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
NUM_GPU=$[(${#CUDA_VISIBLE_DEVICES}+1)/2]
BATCH_SIZE=16
PER_DEVICE_BATCH_SIZE=16
# echo $NUM_GPU
OUTPUT_DIR="output_embedding"
ACC_STEPS=$[$BATCH_SIZE/$PER_DEVICE_BATCH_SIZE/$NUM_GPU]
mkdir -p $OUTPUT_DIR

python3 -u $WORK_DIR/train.py \
    --do_train --do_eval \
    --train_file "$DATA_DIR/train.csv"\
    --valid_file "$DATA_DIR/valid.csv"\
    --test_file "$DATA_DIR/test.csv"\
    --remove_unused_columns false \
    --logging_first_step \
    --learning_rate 1e-5 --lr_scheduler_type linear --warmup_steps 0 \
    --num_train_epochs 50 --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $ACC_STEPS \
    --model_dir $MODEL_DIR --output_dir $OUTPUT_DIR --overwrite_output_dir \
    --logging_steps 1 --save_strategy epoch --evaluation_strategy epoch \
    --metric_for_best_model 'eval_loss' --load_best_model_at_end --greater_is_better false\
    --save_total_limit 1 

