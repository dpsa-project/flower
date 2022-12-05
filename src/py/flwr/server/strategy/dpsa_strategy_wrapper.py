# DPSA strategy wrapper


from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from dpsa4fl_bindings import controller_api__new_state, controller_api__create_session, controller_api__start_round, PyControllerState

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import configure, log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from .aggregate import aggregate, weighted_loss_avg
from .strategy import Strategy

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

# flake8: noqa: E501
class DPSAStrategyWrapper(Strategy):
    """Configurable FedAvg strategy implementation."""
    def __init__(self, strategy: Strategy, dpsa4fl_state: PyControllerState) -> None:
        super().__init__()
        self.strategy = strategy
        self.dpsa4fl_state = dpsa4fl_state


    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:

        return self.strategy.initialize_parameters(client_manager)


    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        result = self.strategy.configure_fit(server_round, parameters, client_manager)
        def append_config(arg: Tuple[ClientProxy, FitIns]):
            client, fitins = arg

            # add the task_id into the config
            fitins.config['task_id'] = self.dpsa4fl_state.mstate.task_id

            return (client, fitins)

        return list(map(append_config, result))


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        return self.strategy.aggregate_fit(server_round, results, failures)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return self.strategy.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return self.strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return self.strategy.evaluate(server_round, parameters)
