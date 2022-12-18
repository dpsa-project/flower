
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

    def __init__(self, dpsa4fl_client_state, client: NumPyClient) -> None:
        super().__init__()
        self.client = client
        self.dpsa4fl_client_state = dpsa4fl_client_state
        self.shapes = None
        self.split_indices = None

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        return self.client.get_properties(config)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return self.client.get_parameters(config)

    def reshape_parameters(self, parameters: NDArrays) -> NDArrays:
        # update parameter shapes
        # if we are in first round (self.shapes is None), then we don't need to reshape.
        # But if we are in any following round, then we need to take our previous shapes
        # and lengths and reshape the `parameters` argument accordingly
        if (self.shapes is not None) and (self.split_indices is not None):
            assert len(self.split_indices) + 1 == len(self.shapes), "Expected #indices = #shapes - 1"

            print("In follow-up round, reshaping. length of params is: ", len(parameters))
            assert len(parameters) == 1, "Expected parameters to have length 1!"

            single_array = parameters[0]
            print("Found single ndarray of shape ", single_array.shape, " and size ", single_array.size)
            # assert single_array.shape == (,), "Wrong ndarray shape!"

            # split and reshape
            arrays = np.split(single_array, self.split_indices)
            for (a, s) in zip(arrays, self.shapes):
                np.reshape(a, s)

            print("Now have the following shapes:")
            for a in arrays:
                print(a.shape)

            # change parameters to properly shaped list of arrays
            parameters = arrays

        else:
            print("In first round, not reshaping.")

        return parameters

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:

        # get current task_id
        task_id = config['task_id']

        #reshape params
        parameters = self.reshape_parameters(parameters)

        # train on data
        params, i, d = self.client.fit(parameters, config)

        # print param shapes
        print("The shapes are:")
        for p in params:
            print(p.shape)

        # flatten params before submitting
        self.shapes = [p.shape for p in parameters]
        flat_params = [p.flatten('C') for p in parameters] #TODO: Check in which order we need to flatten here
        p_lengths = [p.size for p in flat_params]

        # loop
        # (convert p_lengths into indices because ndarray.split takes indices instead of lengths)
        split_indices = []
        current_index = 0
        for l in p_lengths:
            split_indices.append(current_index)
            current_index += l
        split_indices.pop(0) # need to throw away first element of list
        self.split_indices = split_indices


        flat_param_vector = np.concatenate(flat_params)
        flat_param_vector = flat_param_vector - flat_param_vector
        flat_param_vector = np.zeros((20), dtype=np.float32)

        print("vector length is: ", flat_param_vector.shape)
        norm = np.linalg.norm(flat_param_vector)
        print("norm of vector is: ", norm)
        if norm > 1:
            print("Need to scale vector")
            flat_param_vector = flat_param_vector * (1/(norm + 0.01))
            norm = np.linalg.norm(flat_param_vector)
            print("now norm of vector is: ", norm)

        # submit data to janus
        client_api__submit(self.dpsa4fl_client_state, task_id, flat_param_vector)

        # return empty
        return [], i, d

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        parameters = self.reshape_parameters(parameters)
        return self.client.evaluate(parameters, config)



