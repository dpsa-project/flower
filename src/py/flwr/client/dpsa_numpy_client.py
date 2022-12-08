
#
# Written as part of DPSA-project, by @MxmUrw, @ooovi
#
# Based on `dpfedavg_numpy_client.py`.
#

"""Wrapper for configuring a Flower client for usage with DPSA."""


import copy
from typing import Dict, Tuple

import numpy as np

from flwr.client.numpy_client import NumPyClient
from flwr.common.dp import add_gaussian_noise, clip_by_l2
from flwr.common.typing import Config, NDArrays, Scalar

from dpsa4fl_bindings import client_api__new_state, client_api__submit


class DPSANumPyClient(NumPyClient):

    def __init__(self, client: NumPyClient) -> None:
        super().__init__()
        self.client = client
        self.dpsa4fl_client_state = client_api__new_state()

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        return self.client.get_properties(config)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return self.client.get_parameters(config)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:

        # get current task_id
        task_id = config['task_id']

        # train on data
        params, i, d = self.client.fit(parameters, config)

        # fake data to submit
        flat_params = [p.flatten() for p in parameters]
        flat_param_vector = np.concatenate(flat_params)
        print("vector length is: ", flat_param_vector.ndim)

        # submit data to janus
        client_api__submit(self.dpsa4fl_client_state, task_id, flat_param_vector)

        # return empty
        return [], i, d

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        return self.client.evaluate(parameters, config)



