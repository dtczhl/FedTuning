#!/bin/bash


# path to this git
path_to_git=/home/dtczhl/FedTuningNew
# change your virtual environment name
conda_env_name=fedtuning

# Copy the below from your ~/.bashrc
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/dtczhl/Software/Anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/dtczhl/Software/Anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/dtczhl/Software/Anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/dtczhl/Software/Anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

(
cd $path_to_git/ || exit
conda activate ${conda_env_name}

#CUDA_VISIBLE_DEVICES=0 nohup python -u FedTuning/main.py --enable_fedtuning False --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --trace_id 1 &> Log/log_1 &
#CUDA_VISIBLE_DEVICES=0 nohup python -u FedTuning/main.py --enable_fedtuning False --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --trace_id 2 &> Log/log_2 &
CUDA_VISIBLE_DEVICES=0 nohup python -u FedTuning/main.py --enable_fedtuning False --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --trace_id 3 &> Log/log_3 &


#CUDA_VISIBLE_DEVICES=1 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 1 --beta 0 --gamma 0 --delta 0 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 1 &> Log/log_1000_1 &
#CUDA_VISIBLE_DEVICES=1 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 1 --beta 0 --gamma 0 --delta 0 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 2 &> Log/log_1000_2 &
CUDA_VISIBLE_DEVICES=1 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 1 --beta 0 --gamma 0 --delta 0 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 3 &> Log/log_1000_3 &

#
#CUDA_VISIBLE_DEVICES=2 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 1 --gamma 0 --delta 0 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 1 &> Log/log_0100_1 &
#CUDA_VISIBLE_DEVICES=2 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 1 --gamma 0 --delta 0 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 2 &> Log/log_0100_2 &
CUDA_VISIBLE_DEVICES=3 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 1 --gamma 0 --delta 0 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 3 &> Log/log_0100_3 &

#
#CUDA_VISIBLE_DEVICES=3 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0 --gamma 1 --delta 0 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 1 &> Log/log_0010_1 &
#CUDA_VISIBLE_DEVICES=3 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0 --gamma 1 --delta 0 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 2 &> Log/log_0010_2 &
CUDA_VISIBLE_DEVICES=3 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0 --gamma 1 --delta 0 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 3 &> Log/log_0010_3 &

#
#CUDA_VISIBLE_DEVICES=4 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0 --gamma 0 --delta 1 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 1 &> Log/log_0001_1 &
#CUDA_VISIBLE_DEVICES=4 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0 --gamma 0 --delta 1 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 2 &> Log/log_0001_2 &
CUDA_VISIBLE_DEVICES=4 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0 --gamma 0 --delta 1 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 3 &> Log/log_0001_3 &

#
#CUDA_VISIBLE_DEVICES=5 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0.5 --beta 0.5 --gamma 0 --delta 0 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 1 &> Log/log_1100_1 &
#CUDA_VISIBLE_DEVICES=5 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0.5 --beta 0.5 --gamma 0 --delta 0 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 2 &> Log/log_1100_2 &
CUDA_VISIBLE_DEVICES=5 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0.5 --beta 0.5 --gamma 0 --delta 0 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 3 &> Log/log_1100_3 &

#
#CUDA_VISIBLE_DEVICES=6 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0.5 --beta 0 --gamma 0.5 --delta 0 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 1 &> Log/log_1010_1 &
#CUDA_VISIBLE_DEVICES=6 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0.5 --beta 0 --gamma 0.5 --delta 0 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 2 &> Log/log_1010_2 &
CUDA_VISIBLE_DEVICES=6 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0.5 --beta 0 --gamma 0.5 --delta 0 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 3 &> Log/log_1010_3 &

#
#CUDA_VISIBLE_DEVICES=7 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0.5 --beta 0 --gamma 0 --delta 0.5 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 1 &> Log/log_1001_1 &
#CUDA_VISIBLE_DEVICES=7 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0.5 --beta 0 --gamma 0 --delta 0.5 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 2 &> Log/log_1001_2 &
CUDA_VISIBLE_DEVICES=7 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0.5 --beta 0 --gamma 0 --delta 0.5 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 3 &> Log/log_1001_3 &


#CUDA_VISIBLE_DEVICES=0 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0.5 --gamma 0.5 --delta 0 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 1 &> Log/log_0110_1 &
#CUDA_VISIBLE_DEVICES=0 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0.5 --gamma 0.5 --delta 0 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 2 &> Log/log_0110_2 &
CUDA_VISIBLE_DEVICES=0 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0.5 --gamma 0.5 --delta 0 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 3 &> Log/log_0110_3 &

#
#CUDA_VISIBLE_DEVICES=1 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0.5 --gamma 0 --delta 0.5 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 1 &> Log/log_0101_1 &
#CUDA_VISIBLE_DEVICES=1 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0.5 --gamma 0 --delta 0.5 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 2 &> Log/log_0101_2 &
CUDA_VISIBLE_DEVICES=1 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0.5 --gamma 0 --delta 0.5 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 3 &> Log/log_0101_3 &

#
#CUDA_VISIBLE_DEVICES=2 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0 --gamma 0.5 --delta 0.5 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 1 &> Log/log_0011_1 &
#CUDA_VISIBLE_DEVICES=2 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0 --gamma 0.5 --delta 0.5 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 2 &> Log/log_0011_2 &
CUDA_VISIBLE_DEVICES=2 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0 --gamma 0.5 --delta 0.5 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 3 &> Log/log_0011_3 &

#
#CUDA_VISIBLE_DEVICES=3 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0.33 --beta 0.33 --gamma 0.33 --delta 0 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 1 &> Log/log_1110_1 &
#CUDA_VISIBLE_DEVICES=3 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0.33 --beta 0.33 --gamma 0.33 --delta 0 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 2 &> Log/log_1110_2 &
CUDA_VISIBLE_DEVICES=3 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0.33 --beta 0.33 --gamma 0.33 --delta 0 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 3 &> Log/log_1110_3 &

#
#CUDA_VISIBLE_DEVICES=4 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0.33 --beta 0.33 --gamma 0 --delta 0.33 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 1 &> Log/log_1101_1 &
#CUDA_VISIBLE_DEVICES=4 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0.33 --beta 0.33 --gamma 0 --delta 0.33 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 2 &> Log/log_1101_2 &
CUDA_VISIBLE_DEVICES=4 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0.33 --beta 0.33 --gamma 0 --delta 0.33 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 3 &> Log/log_1101_3 &

#
#CUDA_VISIBLE_DEVICES=5 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0.33 --beta 0 --gamma 0.33 --delta 0.33 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 1 &> Log/log_1011_1 &
#CUDA_VISIBLE_DEVICES=5 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0.33 --beta 0 --gamma 0.33 --delta 0.33 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 2 &> Log/log_1011_2 &
CUDA_VISIBLE_DEVICES=5 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0.33 --beta 0 --gamma 0.33 --delta 0.33 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 3 &> Log/log_1011_3 &

#
#CUDA_VISIBLE_DEVICES=6 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0.33 --gamma 0.33 --delta 0.33 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 1 &> Log/log_0111_1 &
#CUDA_VISIBLE_DEVICES=6 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0.33 --gamma 0.33 --delta 0.33 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 2 &> Log/log_0111_2 &
CUDA_VISIBLE_DEVICES=6 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0.33 --gamma 0.33 --delta 0.33 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 3 &> Log/log_0111_3 &

#
#CUDA_VISIBLE_DEVICES=7 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0.25 --beta 0.25 --gamma 0.25 --delta 0.25 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 1 &> Log/log_1111_1 &
#CUDA_VISIBLE_DEVICES=7 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0.25 --beta 0.25 --gamma 0.25 --delta 0.25 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 2 &> Log/log_1111_2 &
CUDA_VISIBLE_DEVICES=7 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0.25 --beta 0.25 --gamma 0.25 --delta 0.25 --model resnet_10 --target_model_accuracy 0.20 --n_participant 20 --n_training_pass 20 --dataset cifar100 --penalty 10 --trace_id 3 &> Log/log_1111_3 &

)






