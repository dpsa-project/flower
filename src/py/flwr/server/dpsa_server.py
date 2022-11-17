
"""Flower server."""

import concurrent.futures
import timeit
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union

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

class DPSAServer(Server):

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        res = super().fit_round(server_round, timeout)

        # check for none:
        if res is not None:
            params, scalars, (results, failures) = res

            # check for each result
            for proxy, fitres in results:
                l: int = len(fitres.parameters.tensors)
                if l > 0:
                    log(DEBUG, "client {}: Expected params to be empty, because running dpsa server. But it had length {}".format(proxy.cid, l))


            return res
            # if params is None:
            #     log(DEBUG, "parameters returned were none.")
            #     return None
            # else:
            #     l: int = len(params.tensors)
            #     if l > 0:
            #         log(DEBUG, "Expected params to be empty, because running dpsa server. But it had length {}".format(l))
            #         return None
            #     else:
            #         log(DEBUG, "Got empty params, yay!")
            #         return params, scalars, results

        else:
            return None




