# number of classes
SPEECH_COMMAND_N_CLASS = 35
SPEECH_COMMAND_CLASSES = ['up', 'two', 'sheila', 'zero', 'yes', 'five', 'one', 'happy', 'marvin', 'no',
                          'go', 'seven', 'eight', 'tree', 'stop', 'down', 'forward', 'learn', 'house', 'three',
                          'six', 'backward', 'dog', 'cat', 'wow', 'left', 'off', 'on', 'four', 'visual',
                          'nine', 'bird', 'right', 'follow', 'bed']

# number of input channel
SPEECH_COMMAND_N_INPUT_FEATURE = 1

# input sizes. resize to 32 by 32
SPEECH_COMMAND_INPUT_RESIZE = (32, 32)

# top-1 accuracy
SPEECH_COMMAND_N_TOP_CLASS = 1

# learning rate and momentum
SPEECH_COMMAND_LEARNING_RATE = 0.01
SPEECH_COMMAND_MOMENTUM = 0.9

# train mean and std
SPEECH_COMMAND_TRAIN_MEAN = 0.626766855257668
SPEECH_COMMAND_TRAIN_STD = 0.22421583435199255

# valid mean and std
SPEECH_COMMAND_VALID_MEAN = 0.6296146142616295
SPEECH_COMMAND_VALID_STD = 0.22381381216557297

# test mean and std
SPEECH_COMMAND_TEST_MEAN = 0.6252543827808388
SPEECH_COMMAND_TEST_STD = 0.22417206147422913

# for dataloader: batch_size and n_worker
SPEECH_COMMAND_DATASET_TRAIN_BATCH_SIZE = 5
SPEECH_COMMAND_DATASET_TRAIN_N_WORKER = 5

# for validation
SPEECH_COMMAND_DATASET_VALID_BATCH_SIZE = 1000
SPEECH_COMMAND_DATASET_VALID_N_WORKER = 10

# for testing
SPEECH_COMMAND_DATASET_TEST_BATCH_SIZE = 1000
SPEECH_COMMAND_DATASET_TEST_N_WORKER = 10
