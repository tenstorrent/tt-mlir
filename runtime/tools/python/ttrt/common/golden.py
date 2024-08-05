# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttrt.binary
import os
import json
import importlib.machinery
import sys
import signal
import os
import io
import subprocess
import time
import socket
from pkg_resources import get_distribution
import sys
import shutil
import torch

from ttrt.common.util import *

os.environ["TTNN_CONFIG_OVERRIDES"] = '{"enable_fast_runtime_mode": false}'
from ttnn.decorators import *
from ttnn.operations.unary import *
from ttnn.operations.binary import *
from ttnn.operations.core import *
from ttnn.operations.creation import *
from ttnn.operations.data_movement import *
from ttnn.operations.embedding import *
from ttnn.operations.losses import *
from ttnn.operations.matmul import *
from ttnn.operations.normalization import *
from ttnn.operations.pool import *
from ttnn.operations.ternary import *
from ttnn.operations.transformer import *

# {op_name_str: (golden_op_function_address, ttnn_op_function_address)}
ttnn_api_golden_mapping = {}

def generate_ttnn_api_name_to_golden_mapping():
    global ttnn_api_name_to_golden_mapping

    for ttnn_op, golden_op in OPERATION_TO_GOLDEN_FUNCTION.items():
        if ttnn_op != None and golden_op != None:
            ttnn_api_golden_mapping[ttnn_op.python_fully_qualified_name] = (ttnn_op.function, golden_op)

def run_golden(golden_inputs, model, program_index):
    golden_outputs = []

    generate_ttnn_api_name_to_golden_mapping()
    
    model_operations = model["programs"][program_index]["operations"]  # list of dictionaries
    clean_model_operations = []
    for operation in model_operations:
        if operation["type_type"] != "OpenDeviceOp" and operation["type_type"] != "CloseDeviceOp":
            clean_model_operations.append(operation)

    print(clean_model_operations)

    return golden_outputs
