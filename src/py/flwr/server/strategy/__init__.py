# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the strategy abstraction and different implementations."""


from .fault_tolerant_fedavg import FaultTolerantFedAvg as FaultTolerantFedAvg
from .fedadagrad import FedAdagrad as FedAdagrad
from .fedadam import FedAdam as FedAdam
from .fedavg import FedAvg as FedAvg
from .fedavg_android import FedAvgAndroid as FedAvgAndroid
from .fedavgm import FedAvgM as FedAvgM
from .fedmedian import FedMedian as FedMedian
from .fedopt import FedOpt as FedOpt
from .fedyogi import FedYogi as FedYogi
from .qfedavg import QFedAvg as QFedAvg
from .dpsa_strategy_wrapper import DPSAStrategyWrapper as DPSAStrategyWrapper
from .strategy import Strategy as Strategy

__all__ = [
    "FaultTolerantFedAvg",
    "FedAdagrad",
    "FedAdam",
    "FedAvg",
    "FedAvgAndroid",
    "FedAvgM",
    "FedOpt",
    "FedYogi",
    "QFedAvg",
    "FedMedian",
    "DPSAStrategyWrapper",
    "Strategy",
]
