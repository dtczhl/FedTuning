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
4. I am refining our algorithm

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
   * number of participants (M): --n_participant
   * number of training passes (E): --n_training_pass
   
2. FL training without FedTuning
    ```python:
    python FedTuning/main.py --enable_fedtuning False --model resnet_10 --target_model_accuracy 0.8 --n_participant 10 --n_training_pass 10 --dataset speech_command
    ```
   Required arguments:
   * --enable_fedtuning False
   * --model
   * --target_model_accuracy
   * --dataset 
   * --n_participant
   * --n_training_pass

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

## Result

11->5

21->1

Google speech-to-command dataset. ResNet-10. Target model accuracy: 0.8

| alpha | beta | gamma | delta | penalty | CompT (10^12) | TransT (10^6) | CompL (10^12) | TransL (10^6) | Final M | Final E | Overall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  -  |   -   |   -   |   -   |   -       |  2.31           | 5.90         | 14.41       |  117.98        |    20    |   20  |   -  |
| 0.1 |   0    | 0.1   | 0.8   |  1 (21)  |  3.40        |   36.27       |   6.72       |    85.30        |    1     |   9   |  +22.78%  |
| 0.1 |   0    | 0.1   |  0.8  |  5 (11) |    4.16        |    10.04     |      13.41    |   79.95        |    4     |    32 |  +21.11%  |
| 0.1 |   0    | 0.1   |  0.8  | 10 (1)   |  3.55        |  6.38         |  10.94        |  56.52     |       8     |   34   | +38.71% |

[comment]: <> (|  0.5   |  0  |  0    |   0.5  |     3.08     |  6.22         |   18.15       |     119.49    |  14    |   30     |  -20.87% |)

[comment]: <> (| 0.5   | 0   |    0   |  0.5  |   3.73        |  6.78         |    14.59       |     74.93       |   8   | 36    |  -12.14% |)

[comment]: <> (| 0.4  |   0.1 | 0.1   |   0.4  |    1.94      |   4.78         |   17.65       |   150.74     |     33   |   15  |   -5.05% |)





