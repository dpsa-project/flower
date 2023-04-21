
#
# Written as part of DPSA-project, by @MxmUrw, @ooovi
#

"""Wrapper for using a strategy with DPSA."""


from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from dpsa4fl_bindings import controller_api_new_state, controller_api_create_session, controller_api_start_round, controller_api_collect, controller_api_get_gradient_len, PyControllerState

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


# flake8: noqa: E501
class DPSAStrategyWrapper(Strategy):
    """
    Configurable strategy implementation for federated learning with
    global differential privacy and secure aggregation. To be used together
    with the DPSAServer.
    """
    def __init__(self, strategy: Strategy, dpsa4fl_state: PyControllerState) -> None:
        """
        Parameters
        ----------
        strategy: Strategy
            The strategy to be wrapped. The wrapping replaces the `configure_fit` and
            `aggregate_fit` methods with ones that interact with the dpsa
            infrastructure. All other methods are copied from the input strategy.
        dpsa4fl_state: PyControllerState
            State object containing training parameters and server locations necessary
            for interoperation with dpsa. See `dpsa4fl_bindings.controller_api_new_state`
            for details.
        """ 
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
        old_params: Parameters,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # we do our custom aggregation here.

        print("Getting results from janus")
        collected: np.ndarray = controller_api_collect(self.dpsa4fl_state)
        print("Done getting results from janus, vector length is: ", collected.shape)

        # convert ndarray type
        flat_grad_array = collected.astype(np.float32)

        old_params_arrays = parameters_to_ndarrays(old_params)
        flat_old_params_array = old_params_arrays[0]

        # if old params is not flat, need to flatten
        if len(old_params_arrays) > 1:
            flat_old_params = [p.flatten('C') for p in old_params_arrays] #TODO: Check in which order we need to flatten here
            flat_old_params_array = np.concatenate(flat_old_params)

        # add gradient to current params
        flat_param_array = flat_old_params_array + flat_grad_array

        # encode again in params format
        parameters_aggregated = ndarrays_to_parameters([flat_param_array])

        # Aggregate custom metrics if aggregation fn was provided
        # the same as what happens in the FedAvg strategy
        metrics_aggregated = {}
        if hasattr(self.strategy, fit_metrics_aggregation_fn):
            if self.strategy.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.strategy.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No fit_metrics_aggregation_fn provided")
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "Strategy has no fit_metrics_aggregation_fn")

        return parameters_aggregated, metrics_aggregated

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
