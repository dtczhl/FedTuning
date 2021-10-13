"""
    FL Server
"""

from Model.ModelWrapper import ModelWrapper
from ServerClient.FLClientManager import FLClientManager
from ClientSelection.ClientSelectionBase import ClientSelectionBase


class FLServer:
    def __init__(self, *, model_name: str, dataset_name: str, client_manager: FLClientManager):
        """ FL server
        :param model_name: model name, e.g., 'resnet_18'
        :param dataset_name: dataset name, e.g., 'speech_command'
        :param client_manager: FL client manager
        """

        self.model = model_name
        self.dataset_name = dataset_name
        self.client_manager = client_manager

        self.server_model = ModelWrapper.build_model_for_dataset(model_name=model_name, dataset_name=dataset_name)

    def copy_model_to_clients(self, *, client_ids: list[str]) -> None:
        """ Update local clients with the global model
        :param client_ids: clients
        :return: None
        """

        self.client_manager.copy_weights_from_global_model(server_model=self.server_model, client_ids=client_ids)

    def train_clients_for_one_round(self, *, client_ids: list[str], training_pass: float) -> None:
        """ Train clients for one epoch
        :param client_ids: clients
        :param training_pass: training pass, i.e., E in FedAvg
        :return: None
        """

        self.client_manager.train_one_round(client_ids=client_ids, training_pass=training_pass)

    def aggregate_model_from_clients(self, *, client_ids: list[str]):
        """ Aggregate model weights from clients
        :param client_ids: clients
        :return: None
        """

        sd_global = self.client_manager.aggregate_weights_of_local_models(client_ids=client_ids)
        self.server_model.load_state_dict(state_dict=sd_global)

    def evaluate_model_performance(self, *, include_valid: bool = False, include_test: bool = False) -> float:
        """ Evaluate model performance (accuracy)
        :param include_valid: True if include the validation set
        :param include_test: True if include the testing set
        :return: accuracy
        """

        model_accuracy = self.client_manager.evaluate_valid_test(server_model=self.server_model,
                                                                 include_valid=include_valid, include_test=include_test)
        return model_accuracy

    def get_cost_of_selected_clients(self, *, client_ids: list[str]) -> list[float]:
        """ get the costs of clients
        :param client_ids: selected clients
        :return: cost of each selected clients
        """

        return self.client_manager.get_cost_of_selected_clients(client_ids=client_ids)

    def select_clients(self, *, client_selection_method: ClientSelectionBase, **kwargs) -> list[str]:
        """ Select clients
        :param client_selection_method:  client selection method
        :return: client_ids
        """

        return client_selection_method(**kwargs)

    def get_total_number_of_clients(self) -> int:
        """ Get total number of clients
        :return: Number of clients
        """

        return self.client_manager.get_total_number_of_clients()
    