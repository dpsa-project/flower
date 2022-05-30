#!/bin/bash

# Original code from https://github.com/adap/flower/blob/main/src/py/flwr_example/tensorflow_minimal/run-clients.sh
# with the following copyright license:
#
#   # Copyright 2020 Adap GmbH. All Rights Reserved.
#   #
#   # Licensed under the Apache License, Version 2.0 (the "License");
#   # you may not use this file except in compliance with the License.
#   # You may obtain a copy of the License at
#   #
#   #     http://www.apache.org/licenses/LICENSE-2.0
#   #
#   # Unless required by applicable law or agreed to in writing, software
#   # distributed under the License is distributed on an "AS IS" BASIS,
#   # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   # See the License for the specific language governing permissions and
#   # limitations under the License.
#   # ==============================================================================
#
# Modified by the developers of http://github.com/dpsa-project

set -e

SERVER_ADDRESS="[::]:8080"
NUM_CLIENTS=2

# get script location (from https://stackoverflow.com/questions/59895/how-can-i-get-the-source-directory-of-a-bash-script-from-within-the-script-itsel)
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

# create folder for logs if not existant
mkdir -p "$SCRIPT_DIR/log"

# change into the src/py folder of the flower source
cd "$SCRIPT_DIR/../src/py"


# run the python client code
echo "Starting $NUM_CLIENTS clients."
for ((i = 0; i < $NUM_CLIENTS; i++))
do
    echo "Starting client(cid=$i) with partition $i out of $NUM_CLIENTS clients."
    python -m flwr.dpsa.dpsa_minimal.client > "$SCRIPT_DIR/log/client_$i.log" &
done
echo "Started $NUM_CLIENTS clients."


