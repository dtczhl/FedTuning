# FedTuning
Source code for our paper [FedTuning](https://arxiv.org/abs/2110.03061). Please consider citing our paper if our paper and codes are helpful to you.

```
@article{fedtuning,
    author = {Huanle Zhang and Mi Zhang and Xin Liu and Prasant Mohapatra and Michael DeLucia},
    title = {Automatic Tuning of Federated Learning Hyper-Parameters from System Perspective},
    journal = {arXiv:2110.03061},
    year = {2021}
}
```


Codes are tested on (1) Ubuntu 18.04 with a 32GB Tesla V100 GPU, cuda:11.4, and (2) Ubuntu 20.04 with 24GB Nvidia RTX A5000 GPUs, cuda:11.3.
Both use PyTorch 1.9.1 and Python 3.9.

**TODO**:
1. Reformatting codes, add comments and explanation
2. Add result analysis scripts
3. Support more datasets and models

## Dataset Download and Preprocess

### Google speech-to-command dataset

1. download dataset, which is saved to Download/speech_command/.  
    ```python:
    python Dataset/speech_command/speech_command_download.py
    ```

2. preprocess. 
    (1) separate clients' data for training, validation, and testing; 
    (2) transform audio clips to spectrograms; 
    (3) save spectrograms to jpg images. 
    Preprocessed data are saved to Download/speech_command/_FedTuning/
      ```python:
      python Dataset/speech_command/speech_command_preprocess.py
      ```

Model hyper-parameters such as learning rate and batch size are defined Dataset/speech_command/\_\_init\_\_.py

### Other datasets


## Experiments

The algorithm of FedTuning is in FedTuning/FedTuningTuner.py

1. FL training with FedTuning enabled
    ```python:
    python FedTuning/main.py --enable_fedtuning True --perference_time 0.33 --preference_computation 0.33 --preference_communication 0.33 --model resnet_10 --target_model_accuracy 0.8 --n_participant 10 --n_training_pass 10 --dataset speech_command
    ```
   Required arguments: 
   * --enable_fedtuning True
   * preferences on time, computation, and communication: --preference_time, --preference_computation, and --preference_communication
   * model: --model. Supported models are under Model/. More models will be supported.
   * target model accuracy: --target_model_accuracy when stop training
   * dataset: --dataset. Now only support speech_command, more dataset will be supported
   
2. FL training without FedTuning
    ```python:
    python FedTuning/main.py --enable_fedtuning False --target_model_accuracy 0.8 --n_participant 10 --n_training_pass 10 --dataset speech_command
    ```
   Required arguments:
   * --enable_fedtuning False
   * --model
   * --target_model_accuracy
   * --dataset

Results are saved to Result/. See the print output for the full filename. Results are saved in CSV files, in the format of
```plain
#round_id,model_accuracy,number of participant (M),number of training pass (E),cost of each selected client
```

## Formulation

On each training round, the cost of each selected client is returned via the following statement (in FeTuning/main.py)
```python:
cost_arr = FL_server.get_cost_of_selected_clients(client_ids=selected_client_ids)
```
We calculate time overhead, computation overhead, and communication overhead of a training round by
```python:
# time, computation, and communication cost on this training round
round_time_cost = max(cost_arr)
round_computation_cost = sum(cost_arr)
round_communication_cost = len(cost_arr)
```



