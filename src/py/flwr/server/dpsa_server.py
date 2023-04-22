
"""Flower server."""

from typing import Dict, Optional, Tuple

from dpsa4fl_bindings import controller_api_create_session, controller_api_end_session, controller_api_start_round, controller_api_new_state

from flwr.common import Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy, DPSAStrategyWrapper, FedAvg
from flwr.server.server import Server, FitResultsAndFailures

class DPSAServer(Server):
    """
    A flower server for federated learning with global differential privacy and
    secure aggregation. Uses the dpsa project infrastructure, see here
    for more information: https://github.com/dpsa-project/overview

    NOTE: This is intended for use with the DPSANumPyClient flower client.
    """ 

    def __init__(
        self,
        model_size: int,
        privacy_parameter: float,
        granularity: int,
        aggregator1_location: str,
        aggregator2_location: str,
        *,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None
    ) -> None:
        """
        Parameters
        ----------
        model_size: int
            The number of parameters of the model to be trained.
        privacy_parameter: float
            The desired privacy per learning step. One aggregation step will
            be `1/2*privacy_parameter^2` zero-concentrated differentially private
            for each client.
        granularity: int
            The resolution of the fixed-point encoding used for secure aggregation.
            A larger value will result in a less lossy representation and more
            communication and computation overhead. Currently, 16, 32 and 64 bit are
            supported.
        aggregator1_location: str
            Location of the first aggregator server in URL format including the port.
            For example, for a server running locally: "http://127.0.0.1:9991"
        aggregator2_location: str
            Location of the second aggregator server in URL format including the port.
            For example, for a server running locally: "http://127.0.0.1:9992"
        """

        # call dpsa4fl to create state object
        self.dpsa4fl_state = controller_api_new_state(
            model_size,
            privacy_parameter,
            granularity,
            aggregator1_location,
            aggregator2_location,
        )

        dpsa4fl_strategy = DPSAStrategyWrapper(
            strategy if strategy is not None else FedAvg(),
            self.dpsa4fl_state
        )

        super().__init__(client_manager=client_manager, strategy=dpsa4fl_strategy)

        # call dpsa4fl to create new session
        controller_api_create_session(self.dpsa4fl_state)

    """End the dpsa4fl session. Use at the end of training for graceful shutdown."""
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

