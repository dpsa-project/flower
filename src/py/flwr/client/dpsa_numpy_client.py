
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


class DPSANumPyClient(NumPyClient):

    def __init__(self, client: NumPyClient) -> None:
        super().__init__()
        self.client = client

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        return self.client.get_properties(config)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return self.client.get_parameters(config)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        return self.client.fit(parameters, config)

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        return self.client.evaluate(parameters, config)

