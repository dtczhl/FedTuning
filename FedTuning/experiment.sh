#!/bin/bash


# path to this git
path_to_git=/home/dtczhl/FedTuning
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
CUDA_VISIBLE_DEVICES=0 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0.5 --gamma 0.5 --delta 0 --model resnet_10 --target_model_accuracy 0.8 --n_participant 20 --n_training_pass 20 --dataset speech_command --penalty 2 --trace_id 3 &> Log/log_0110_2_3 &
CUDA_VISIBLE_DEVICES=1 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0.5 --gamma 0.5 --delta 0 --model resnet_10 --target_model_accuracy 0.8 --n_participant 20 --n_training_pass 20 --dataset speech_command --penalty 3 --trace_id 3 &> Log/log_0110_3_3 &
CUDA_VISIBLE_DEVICES=2 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0.5 --gamma 0.5 --delta 0 --model resnet_10 --target_model_accuracy 0.8 --n_participant 20 --n_training_pass 20 --dataset speech_command --penalty 4 --trace_id 3 &> Log/log_0110_4_3 &
CUDA_VISIBLE_DEVICES=3 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0.5 --gamma 0.5 --delta 0 --model resnet_10 --target_model_accuracy 0.8 --n_participant 20 --n_training_pass 20 --dataset speech_command --penalty 5 --trace_id 3 &> Log/log_0110_5_3 &
CUDA_VISIBLE_DEVICES=4 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0.5 --gamma 0.5 --delta 0 --model resnet_10 --target_model_accuracy 0.8 --n_participant 20 --n_training_pass 20 --dataset speech_command --penalty 6 --trace_id 3 &> Log/log_0110_6_3 &
CUDA_VISIBLE_DEVICES=5 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0.5 --gamma 0.5 --delta 0 --model resnet_10 --target_model_accuracy 0.8 --n_participant 20 --n_training_pass 20 --dataset speech_command --penalty 7 --trace_id 3 &> Log/log_0110_7_3 &
CUDA_VISIBLE_DEVICES=6 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0.5 --gamma 0.5 --delta 0 --model resnet_10 --target_model_accuracy 0.8 --n_participant 20 --n_training_pass 20 --dataset speech_command --penalty 8 --trace_id 3 &> Log/log_0110_8_3 &
CUDA_VISIBLE_DEVICES=7 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0.5 --gamma 0.5 --delta 0 --model resnet_10 --target_model_accuracy 0.8 --n_participant 20 --n_training_pass 20 --dataset speech_command --penalty 9 --trace_id 3 &> Log/log_0110_9_3 &
CUDA_VISIBLE_DEVICES=6 nohup python -u FedTuning/main.py --enable_fedtuning True --alpha 0 --beta 0.5 --gamma 0.5 --delta 0 --model resnet_10 --target_model_accuracy 0.8 --n_participant 20 --n_training_pass 20 --dataset speech_command --penalty 10 --trace_id 3 &> Log/log_0110_10_3 &
)






