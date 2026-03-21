LEARNING_RATE="0.005"
SEED="42" 
DEVICE=6
METHOD="cprompt_c2prompt_domainnet_v2"

python main.py --device $DEVICE --global_update_lr 100 \
    --img_size 224 --numclass 1750 --class_per_task 35 --dataset DomainNet --easy 0 \
    --tasks_global 3 --num_clients 5 --epochs_global 15 --local_clients 5 \
    --learning_rate $LEARNING_RATE --prompt_flag codap_2d_v2 --method $METHOD --batch_size 32 \
    --prompt_param 25 10 10 8 0 0 6 10 8\
    --epochs_local 4 --seed $SEED --num_classes 1750 

LOG_PATH="./training_log/${METHOD}/seed${SEED}/log_train_${LEARNING_RATE}.txt"

python benchmark_metrics.py --log_path "$LOG_PATH"