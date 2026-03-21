METHOD="cprompt_c2prompt_v2"
LEARNING_RATE="0.005"
SEED="0" 
DEVICE=7

# python main.py --device $DEVICE --global_update_lr 100 \
#     --img_size 224 --numclass 500 --class_per_task 20 --dataset ImageNet_R --easy 0 \
#     --tasks_global 3 --num_clients 5 --epochs_global 15 --local_clients 5 \
#     --learning_rate $LEARNING_RATE --prompt_flag codap_2d_v2 --method $METHOD --batch_size 64 \
#     --prompt_param 25 10 10 8 0 0 6 10 8\
#     --epochs_local 10 --num_classes 500 --seed $SEED

LOG_PATH="./training_log/${METHOD}/seed${SEED}/log_train_${LEARNING_RATE}.txt"

python benchmark_metrics.py --log_path "$LOG_PATH"