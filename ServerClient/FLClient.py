"""
    For one FL client
"""

from typing import Iterator, Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from torch.nn.parameter import Parameter

from Dataset import *
from Dataset.speech_command import *
from Dataset.speech_command.SpeechCommandForTrain import SpeechCommandForTrain
from Dataset.emnist import *
from Dataset.emnist.EmnistForTrain import EmnistForTrain


class FLClient:

    def __init__(self, *, client_id: str, client_model: nn.Module, dataset_name: str, gpu_device):
        """ Client: model, and local data
        :param client_id: the id of this client
        :param client_model: client's model
        :param dataset_name: dataset name
        :param gpu_device: run on which cuda
        """

        self._client_id = client_id
        self._client_model = client_model

        self._cpu_device = torch.device("cpu")
        self._gpu_device = gpu_device

        # participant selection, the greater the higher chance to be selected
        self._selection_value = None

        # cost of whatever.
        # Currently, the cost is the number of samples for training
        # e.g, int(training_pass * len(self._dataset))
        self._cost = None

        # the group that this client belongs to
        self._group_id = None

        # whether should consider this client in participant selection
        self._is_active = True

        if dataset_name == 'speech_command':

            user_dir = f'{DATASET_DIR}/speech_command/_FedTuning/train/{client_id}'
            self._dataset = SpeechCommandForTrain(user_dir=user_dir)
            self._dataloader = DataLoader(self._dataset, batch_size=SPEECH_COMMAND_DATASET_TRAIN_BATCH_SIZE,
                                          shuffle=True, num_workers=SPEECH_COMMAND_DATASET_TRAIN_N_WORKER)

            # training construction
            if self._client_model is not None:
                self._optimizer = optim.SGD(client_model.parameters(), lr=SPEECH_COMMAND_LEARNING_RATE,
                                            momentum=SPEECH_COMMAND_MOMENTUM)
                self._criterion = nn.CrossEntropyLoss()

        elif dataset_name == 'emnist':

            user_dir = f'{DATASET_DIR}/emnist/train/{client_id}'
            self._dataset = EmnistForTrain(user_dir=user_dir)
            self._dataloader = DataLoader(self._dataset, batch_size=EMNIST_DATASET_TRAIN_BATCH_SIZE,
                                          shuffle=True, num_workers=EMNIST_DATASET_TRAIN_N_WORKER)

            # training construction
            if self._client_model is not None:
                self._optimizer = optim.SGD(client_model.parameters(), lr=EMNIST_LEARNING_RATE,
                                            momentum=EMNIST_MOMENTUM)
                self._criterion = nn.CrossEntropyLoss()

        else:
            print(f'unknown dataset {dataset_name}')
            exit(-1)

    def train_one_round(self, *, training_pass: float) -> None:
        """ Train the client for one training round
        :param training_pass: i.e., E in FedAvg, must > 0
        :return: None
        """

        if training_pass <= 0:
            print(f'#training_pass must > 0, given {training_pass} ')
            exit(-1)

        if self._client_model is None:
            print(f'user {self._client_id} does not have a trainable model, i.e., self.model is None')
            exit(-1)

        self._client_model.to(self._gpu_device)

        tot_training_sample_count = int(training_pass * len(self._dataset))
        # at least one training sample
        tot_training_sample_count = max(tot_training_sample_count, 1)

        while tot_training_sample_count > 0:
            for inputs, labels in self._dataloader:

                # break if no need to train more
                if tot_training_sample_count <= 0:
                    break

                if tot_training_sample_count > len(inputs):
                    tot_training_sample_count -= len(inputs)
                else:
                    inputs = inputs[:tot_training_sample_count]
                    labels = labels[:tot_training_sample_count]
                    tot_training_sample_count -= len(inputs)
                    assert tot_training_sample_count == 0

                inputs = inputs.to(self._gpu_device)
                labels = labels.to(self._gpu_device)

                # zero the parameter gradients
                self._optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self._client_model(inputs)
                loss = self._criterion(outputs, labels)

                # gradients here
                loss.backward()

                self._optimizer.step()

        # put clients to CPU when not active
        self._client_model.to(self._cpu_device)

        # remove GPU cache
        torch.cuda.empty_cache()

        # cost of this round's training.
        self.set_cost(cost=training_pass * len(self._dataset))

    def get_model_state_dict(self) -> dict:
        """ Return this client's sate_dict of its model
        :return: client_model.state_dict()
        """

        if self._client_model is None:
            print(f'user {self._client_id} does not have a trainable model, i.e., self.model is None')
            exit(-1)

        return self._client_model.state_dict()

    def get_model_named_parameters(self) -> Iterator[tuple[str, Parameter]]:
        """ Return the named_parameters() of this client's model
        :return: client_model.named_parameters()
        """

        if self._client_model is None:
            print(f'user {self._client_id} does not have a trainable model, i.e., self.model is None')
            exit(-1)

        return self._client_model.named_parameters()

    def copy_weights_from_global_model(self, server_model: nn.Module) -> None:
        """ Assign weights of the server model to this client's model
        :param server_model: server model
        :return: None
        """

        if self._client_model is None:
            print(f'user {self._client_id} does not have a trainable model, i.e., self.model is None')
            exit(-1)

        self._client_model.load_state_dict(server_model.state_dict())

    def set_cost(self, *, cost: float) -> None:
        """ Set cost of this client

        :param cost: cost
        :return: None
        """

        self._cost = cost

    def get_cost(self) -> float:
        """ Return flop of this client. Currently, it is the number of local samples
        :return: cost of this client
        """

        if self._cost is None:
            print(f'The cost of the user {self._client_id} is None. Must initialize first')
            exit(-1)

        return self._cost

    def set_client_selection_value(self, *, selection_value: float) -> None:
        """ Set selection_value of this client. For participant selection
        :param selection_value: weight for selection
        :return: None
        """

        if not self._is_active:
            print(f'only allow to set selection_value for active clients')
            exit(-1)

        self._selection_value = selection_value

    def get_client_selection_value(self) -> float:
        """ Return selection_value of this client. For participant selection
        :return: selection value
        """

        if self._selection_value is None:
            print(f'The selection_value of the user {self._client_id} is None. Must initialize first')
            exit(-1)

        return self._selection_value

    def set_active_status(self, *, is_active: bool) -> None:
        """ Set the active or not
        :param is_active: active or not
        :return: None
        """

        self._is_active = is_active

    def is_active(self) -> bool:
        """ Get the active status
        :return: True or False
        """

        return self._is_active

    def get_number_of_samples(self) -> int:
        """ Return the number of its local sample points
        :return:
        """
        return len(self._dataset)
