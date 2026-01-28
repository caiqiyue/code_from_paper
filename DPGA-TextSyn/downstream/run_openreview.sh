#!/bin/bash
idx=2
export CUDA_VISIBLE_DEVICES=${idx}
output_dir="text_classification/model/openreview/"
for n in xx
do
for seed in 0 1 2
do
    python3 -u text_classification/run_classification.py \
        --report_to none --model_name_or_path  text_classification/model/roberta_base \
        --output_dir ${output_dir} \
        --train_file data/openreview/xxx.csv --validation_file data/openreview/val.csv --test_file data/openreview/test.csv \
        --do_train --do_eval --do_predict --max_seq_length 512 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 \
        --learning_rate 3e-5 --num_train_epochs 10 \
        --overwrite_output_dir --overwrite_cache \
        --save_strategy epoch --save_total_limit 2 --load_best_model_at_end \
        --logging_strategy epoch \
        --seed ${seed} \
        --metric_for_best_model accuracy_all --greater_is_better True \
        --evaluation_strategy epoch
done
done > openreview.log  2>&1 &
