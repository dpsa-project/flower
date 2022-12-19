# DPSA strategy wrapper


from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

# from dpsa4fl_bindings import controller_api__new_state, controller_api__create_session, controller_api__start_round, PyControllerState

from dpsa4fl_bindings import controller_api__new_state, controller_api__create_session, controller_api__start_round, controller_api__collect, controller_api__get_gradient_len, PyControllerState

import numpy as np

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
    def __init__(self, strategy: Strategy, dpsa4fl_state: PyControllerState, expected_gradient_len=None) -> None:
        super().__init__()
        self.strategy = strategy
        self.dpsa4fl_state = dpsa4fl_state
        self.expected_gradient_len = expected_gradient_len

        # variables for FedAvg aggregate_fit
        self.accept_failures = True
        # self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.fit_metrics_aggregation_fn = None


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
        # we do our custom aggregation code here.
        # copied from FedAvg
        print("inside aggregate fit")

        """Aggregate fit results using weighted average."""
        if not results:
            print("Don't have results, skip.")
            # return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            print("Have failures, skip.")
            # return None, {}

        # Convert results
        # weights_results = [
        #     (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        #     for _, fit_res in results
        # ]

        # do DPSA aggregation
        #
        # check that we got empty arrays, because we want to get our params
        # from the janus server
        # for weights in weights_results:
        #     l: int = len(weights)
        #     print("got result with flwr param len ", l)
        #     assert l == 0

        print("Getting results from janus")
        collected: np.ndarray = controller_api__collect(self.dpsa4fl_state)
        print("Done getting results from janus, vector length is: ", collected.shape)

        grad_len = controller_api__get_gradient_len(self.dpsa4fl_state)
        if self.expected_gradient_len:
            grad_len = self.expected_gradient_len

        flat_array = collected # np.zeros(grad_len)

        # parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # for now fake result
        parameters_aggregated = ndarrays_to_parameters([flat_array])

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated


        # return self.strategy.aggregate_fit(server_round, results, failures)

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
