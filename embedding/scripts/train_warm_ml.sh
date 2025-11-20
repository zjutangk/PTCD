WORK_DIR=$(pwd)
# WORK_DIR=.
echo $WORK_DIR

echo $(pwd)

MODEL_DIR="bert-base-chinese"
DATA_DIR="data"
export HF_DATASETS_CACHE="cache"
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
NUM_GPU=$[(${#CUDA_VISIBLE_DEVICES}+1)/2]
BATCH_SIZE=16


for PER_DEVICE_BATCH_SIZE in  16
do
    for LR in 5e-6
    do 
        for WEIGHT_DECAY in 0.01
        do
            OUTPUT_DIR="output_warm"
            ACC_STEPS=$[$BATCH_SIZE/$PER_DEVICE_BATCH_SIZE/$NUM_GPU]
            mkdir -p $OUTPUT_DIR

            # python3 -u $WORK_DIR/train_en_warm.py \
            python3 -u $WORK_DIR/train_warm_ml.py \
                --do_train --do_eval \
                --train_file "$DATA_DIR/train.csv"\
                --valid_file "$DATA_DIR/valid.csv"\
                --test_file "$DATA_DIR/test.csv"\
                --warm_train_file "$DATA_DIR/warm_train.csv" \
                --warm_valid_file "$DATA_DIR/warm_valid.csv" \
                --warm_output_dir $WORK_DIR/model_to/$PER_DEVICE_BATCH_SIZE-warm \
                --num_warm_epochs 30 \
                --warm_learning_rate 5e-6 \
                --num_labels 16 \
                --remove_unused_columns false \
                --logging_first_step \
                --learning_rate $LR --lr_scheduler_type linear --warmup_steps 0 --weight_decay $WEIGHT_DECAY\
                --num_train_epochs 5 --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
                --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
                --gradient_accumulation_steps $ACC_STEPS \
                --model_dir $MODEL_DIR --output_dir $OUTPUT_DIR --overwrite_output_dir \
                --logging_steps 1 --save_strategy epoch --evaluation_strategy epoch \
                --metric_for_best_model 'eval_acc' --load_best_model_at_end --greater_is_better True\
                --save_total_limit 1 \
                | tee $OUTPUT_DIR/$(date '+%Y-%m-%d').log
        done
    done
done
#