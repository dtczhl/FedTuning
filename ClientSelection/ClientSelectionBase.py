"""
    Base class for client sampling algorithms
"""

from ServerClient.FLClientManager import FLClientManager


class ClientSelectionBase:

    def __init__(self, *, client_manager: FLClientManager):
        self.client_manager = client_manager

    def __call__(self, **kwargs) -> list[str]:
        """ Return selected clients
        :param kwargs: take in more parameters
        :return: client_ids
        """

        return ['']
