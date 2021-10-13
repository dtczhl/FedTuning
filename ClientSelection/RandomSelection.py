"""
    Random client selection
"""

import random

from ClientSelection.ClientSelectionBase import ClientSelectionBase
from ServerClient.FLClientManager import FLClientManager


class RandomSelection(ClientSelectionBase):

    def __init__(self, *, client_manager: FLClientManager, n_select_client: int):
        """ Random sampling
        :param client_manager: FLClientManager
        :param n_select_client: number of selected clients
        """
        super(RandomSelection, self).__init__(client_manager=client_manager)

        self.n_select_client = n_select_client

    def __call__(self, **kwargs) -> list[str]:
        """ Return selected clients
        :param kwargs: for extension to take parameters
        :return: client_ids
        """

        for client_id, fl_client in self.client_manager.all_clients.items():
            if fl_client.is_active():
                fl_client.set_client_selection_value(selection_value=random.uniform(0, 1))

        client_ids = self.client_manager.get_client_ids_of_highest_selection_values(n_select_clients=self.n_select_client)

        return client_ids

    def __str__(self) -> str:
        """ Concise representation
        :return: string
        """

        return f'random_{self.n_select_client}'
