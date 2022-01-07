
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
