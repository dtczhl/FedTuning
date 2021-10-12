# FedTuning
Source code for FedTuning

I am organizing my codes..........

Codes are tested on (1) Ubuntu 18.04 with a 32GB Tesla V100 GPU, cuda:11.4, and (2) Ubuntu 20.04 with 24GB Nvidia RTX A5000 GPUs, cuda:11.3.
Both using PyTorch 1.9.1, Python 3.9

## Dataset Download and Preprocess

### Google speech-to-command dataset

1. download dataset, which is saved to FedTuning/Download/speech_command/.  
    ```python:
    python Dataset/speech_command/speech_command_download.py
    ```

2. preprocess. 
    (1) separate clients' data for training, validation, and testing; 
    (2) transform audio clips to spectrograms; 
    (3) save spectrograms to jpg images. 
    Preprocessed data are saved to FedTuning/Download/speech_command/_FedTuning/
      ```python:
      python Dataset/speech_command/speech_command_preprocess.py
      ```

Model hyper-parameters such as learning rate and batch size are defined in FedTuning/Dataset/speech_command/\_\_init\_\_.py

### Other datasets

TODO...

## Experiments





