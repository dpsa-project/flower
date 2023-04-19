
"""Flower server."""

import concurrent.futures
import timeit
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union

from dpsa4fl_bindings import controller_api_new_state, controller_api_create_session, controller_api_end_session, controller_api_start_round, controller_api_collect, controller_api_get_gradient_len, PyControllerState

from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
)
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy
from flwr.server.server import Server, FitResultsAndFailures

import numpy as np

class DPSAServer(Server):

    def __init__(self, dpsa4fl_state: PyControllerState, *, client_manager: ClientManager, strategy: Optional[Strategy] = None) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)

        # call dpsa4fl to create new state
        self.dpsa4fl_state = dpsa4fl_state

        # call dpsa4fl to create new session
        controller_api_create_session(self.dpsa4fl_state)

    def __del__(self):
        # end session when we are done
        controller_api_end_session(self.dpsa4fl_state)

    """Perform a single round of federated averaging."""
    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        # Call dpsa4fl to start a new round
        controller_api_start_round(self.dpsa4fl_state)

        # The rest of the work is done by the `DPSAStrategyWrapper`
        # which is called in the server implementation of super.

        res = super().fit_round(server_round, timeout)

        return res




