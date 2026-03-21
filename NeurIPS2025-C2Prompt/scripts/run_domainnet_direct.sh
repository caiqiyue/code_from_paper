python main.py --device 0 --global_update_lr 100 \
    --img_size 224 --numclass 1750 --class_per_task 35 --dataset DomainNet --easy 0 \
    --tasks_global 3 --num_clients 5 --epochs_global 30 --local_clients 5 \
    --learning_rate 0.005 --prompt_flag codap_2d_v2 --method cprompt_c2prompt_domainnet_direct_v2 --batch_size 32 \
    --prompt_param 25 10 10 8 0 0 6 10 8\
    --epochs_local 4 --seed 2020 ----num_classes 1750