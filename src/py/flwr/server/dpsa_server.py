
"""Flower server."""

import concurrent.futures
import timeit
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union

from dpsa4fl_bindings import controller_api__new_state, controller_api__create_session, controller_api__start_round, controller_api__collect, controller_api__get_gradient_len, PyControllerState

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
        controller_api__create_session(self.dpsa4fl_state)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        # first, call dpsa4fl to start a new round
        # task_id = controller_api__start_round(self.dpsa4fl_state)

        # next, send these parameters to the clients
        # sending the correct task_id is taken care of by the `DPSAStrategyWrapper`

        """Perform a single round of federated averaging."""
        res = super().fit_round(server_round, timeout)

        return res

        # # check for none:
        # if res is not None:
        #     params, scalars, (results, failures) = res

        #     # check for each result
        #     for proxy, fitres in results:
        #         l: int = len(fitres.parameters.tensors)
        #         if l > 0:
        #             log(DEBUG, "client {}: Expected params to be empty, because running dpsa server. But it had length {}".format(proxy.cid, l))

        #     # collect results from janus
        #     print("Getting results from janus")
        #     collected = controller_api__collect(self.dpsa4fl_state)
        #     print("Done getting results from janus")

        #     flat_array = np.zeros(controller_api__get_gradient_len(self.dpsa4fl_state))

        #     for proxy, fitres in results:
        #         fitres.parameters = 


        #     return res
        #     # if params is None:
        #     #     log(DEBUG, "parameters returned were none.")
        #     #     return None
        #     # else:
        #     #     l: int = len(params.tensors)
        #     #     if l > 0:
        #     #         log(DEBUG, "Expected params to be empty, because running dpsa server. But it had length {}".format(l))
        #     #         return None
        #     #     else:
        #     #         log(DEBUG, "Got empty params, yay!")
        #     #         return params, scalars, results

        # else:
        #     return None




