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

CUDA_VISIBLE_DEVICES=0 nohup python -u FedTuning/main.py --enable_fedtuning False --model vgg_11 --target_model_accuracy 0.8 --n_participant 20 --n_training_pass 20 --dataset speech_command --trace_id 1 &> Log/log_vgg_11 &
CUDA_VISIBLE_DEVICES=1 nohup python -u FedTuning/main.py --enable_fedtuning False --model vgg_19 --target_model_accuracy 0.8 --n_participant 20 --n_training_pass 20 --dataset speech_command --trace_id 1 &> Log/log_vgg_19 &


)






