"""
    FL Client Manager

    It sits between the FL server and FL clients
"""

import os
import threading
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Dataset import *
from Dataset.speech_command import *
from Dataset.speech_command.SpeechCommandForValid import SpeechCommandForValid
from Dataset.speech_command.SpeechCommandForTest import SpeechCommandForTest

from Model.ModelWrapper import ModelWrapper
from ServerClient import *
from ServerClient.FLClient import FLClient


class FLClientManager:

    def __init__(self, *, model_name: str, dataset_name: str):
        """ FLClientManager manages all FL clients for training
        :param model_name: ML model name, e.g., resnet_18
        :param dataset_name: dataset name, e.g., speech_command
        """

        # maximum number of threads supported
        self.n_max_thread = N_MAX_THREADS_FOR_FL_CLIENTS

        self.cpu_device = torch.device("cpu")
        self.gpu_device = torch.device('cuda:0')

        self.all_clients = {}

        self.model_name = model_name

        self.dataset_name = dataset_name.strip().lower()
        if self.dataset_name == 'speech_command':

            client_ids = os.listdir(os.path.join(DATASET_SPEECH_COMMAND_DIR, 'train'))

            for client_id in client_ids:
                client_model = ModelWrapper.build_model_for_dataset(model_name=model_name, dataset_name=dataset_name)
                client = FLClient(client_id=client_id, client_model=client_model, dataset_name=dataset_name,
                                  gpu_device=self.gpu_device)
                self.all_clients[client_id] = client

        else:
            print(f'unknown dataset_name {dataset_name}')
            exit(-1)

    def __user_check(self, *, client_id: str) -> None:
        """ If the user_id is not in the clients, print error and exit
        :param client_id: client
        :return: None
        """

        if client_id not in self.all_clients:
            print(f'client: {client_id} is not in the clients')
            exit(-1)

    def train_one_round(self, *, client_ids: list[str], training_pass: float) -> None:
        """ Train the models of user_ids for one round
        :param client_ids: selected users
        :param training_pass: number of training passes
        :return: None
        """

        class ClientTrainThread(threading.Thread):

            def __init__(self, *, fl_client: FLClient):
                """ Threading for training
                :param fl_client: clients
                """

                threading.Thread.__init__(self)
                self.fl_client = fl_client

            def run(self) -> None:
                self.fl_client.train_one_round(training_pass=training_pass)

        threads = []
        i_client = 0
        while i_client < len(client_ids):
            client_id = client_ids[i_client]
            self.__user_check(client_id=client_id)

            if len(threads) < self.n_max_thread:
                thread = ClientTrainThread(fl_client=self.all_clients[client_id])
                threads.append(thread)
                i_client += 1
            else:
                # start all thread
                for thread in threads:
                    thread.start()

                # wait all to finish
                for thread in threads:
                    thread.join()

                threads = []

        if threads:
            # start all remaining threads
            for thread in threads:
                thread.start()

            # wait all threads to finish
            for thread in threads:
                thread.join()

    def copy_weights_from_global_model(self, *, server_model: nn.Module, client_ids: list[str]) -> None:
        """ Replace clients' weights from the server model
        :param server_model: server model
        :param client_ids: user ids in the training client set for replacement
        :return: None
        """

        with torch.no_grad():
            for client_id in client_ids:
                self.__user_check(client_id=client_id)
                self.all_clients[client_id].copy_weights_from_global_model(server_model=server_model)

    def aggregate_weights_of_local_models(self, *, client_ids: list[str]) -> dict:
        """ Aggregate weights of local models
        :param client_ids:  selected clients
        :return: model's state_dict that the global model can directly use via global_model.load_state_dict()
        """

        for client_id in client_ids:
            self.__user_check(client_id=client_id)

        temp_model = ModelWrapper.build_model_for_dataset(model_name=self.model_name, dataset_name=self.dataset_name)

        with torch.no_grad():

            total_samples = 0
            for client_id in client_ids:
                total_samples += self.all_clients[client_id].get_number_of_samples()

            sd_global = temp_model.state_dict()
            for name, param in temp_model.named_parameters():
                sd_global[name].zero_()
                for client_id in client_ids:
                    sd_local = self.all_clients[client_id].get_model_state_dict()
                    sd_global[name] += sd_local[name] * self.all_clients[client_id].get_number_of_samples() / total_samples

        return sd_global

    def evaluate_valid_test(self, *, server_model: nn.Module, include_valid: bool = False,
                            include_test: bool = False) -> float:
        """ Which set for model evaluation. More then one can be selected
        :param server_model: the global model for evaluation
        :param include_valid: whether to include the validation set
        :param include_test: whether to include the testing set
        :return: model performance (accuracy)
        """

        if not include_valid and not include_test:
            print('At least one of the validation and testing set is required')
            exit(-1)

        server_model.to(self.gpu_device)

        if self.dataset_name == "speech_command":

            n_correct = 0
            n_incorrect = 0

            if include_valid:
                dataset = SpeechCommandForValid()
                dataloader = DataLoader(dataset=dataset, batch_size=SPEECH_COMMAND_DATASET_VALID_BATCH_SIZE,
                                        shuffle=False,
                                        num_workers=SPEECH_COMMAND_DATASET_VALID_N_WORKER)
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(self.gpu_device), labels.to(self.gpu_device)
                    outputs = server_model(inputs)

                    _, predicted = torch.max(outputs.data, 1)

                    labels = labels.detach().cpu().numpy()
                    predicted = predicted.detach().cpu().numpy()

                    _n_correct = (predicted == labels).sum().item()
                    n_correct += _n_correct
                    n_incorrect += len(inputs) - _n_correct

            if include_test:
                dataset = SpeechCommandForTest()
                dataloader = DataLoader(dataset, batch_size=SPEECH_COMMAND_DATASET_TEST_BATCH_SIZE, shuffle=False,
                                        num_workers=SPEECH_COMMAND_DATASET_TEST_N_WORKER)
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(self.gpu_device), labels.to(self.gpu_device)
                    outputs = server_model(inputs)

                    _, predicted = torch.max(outputs.data, 1)

                    labels = labels.detach().cpu().numpy()
                    predicted = predicted.detach().cpu().numpy()

                    _n_correct = (predicted == labels).sum().item()
                    n_correct += _n_correct
                    n_incorrect += len(inputs) - _n_correct

            server_model.to(self.cpu_device)

            return n_correct / (n_correct + n_incorrect)

        else:
            print(f'unknown dataset {self.dataset_name}')
            exit(-1)

    def get_total_number_of_clients(self) -> int:
        """ Return total number of clients
        :return: total number of clients
        """

        return len(self.all_clients)

    def get_cost_of_selected_clients(self, *, client_ids: list[str]) -> list[float]:
        """ Get costs of selected clients
        :param client_ids: selected clients
        :return: cost of each selected client
        """

        client_flop_arr = []

        for client_id in client_ids:
            self.__user_check(client_id=client_id)
            client_flop_arr.append(self.all_clients[client_id].get_cost())

        return client_flop_arr

    def get_client_ids_of_highest_selection_values(self, *, n_select_clients: int) -> list[str]:
        """ Return the client_ids of highest selection_values
        :param n_select_clients: number of selected clients
        :return: client ids
        """

        client_selection_values = {client_id: fl_client.get_client_selection_value() for client_id, fl_client in
                                   self.all_clients.items() if fl_client.is_active()}

        client_ids = [k for k, v in sorted(client_selection_values.items(), key=lambda item: item[1], reverse=True)]

        client_ids = client_ids[:n_select_clients]

        return client_ids
