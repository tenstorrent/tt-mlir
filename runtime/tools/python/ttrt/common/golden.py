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
import numpy as np

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

# {op_name_str: (ttnn_op_function_address, golden_op_function_address)}
ttnn_api_golden_mapping = {}

def generate_ttnn_api_name_to_golden_mapping():
    global ttnn_api_name_to_golden_mapping

    for ttnn_op, golden_op in OPERATION_TO_GOLDEN_FUNCTION.items():
        if ttnn_op != None and golden_op != None:
            ttnn_api_golden_mapping[ttnn_op.python_fully_qualified_name] = (ttnn_op.function, golden_op)

generate_ttnn_api_name_to_golden_mapping()

'''
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

    return golden_outputs

def golden_proxy_subtract(tensor_a, tensor_b):
    return torch.subtract(tensor_a, tensor_b)
'''

def comp_pcc(golden, calculated, pcc=0.99):
    golden = torch.Tensor(golden)
    calculated = torch.Tensor(calculated)

    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        logger.warning("Both tensors are 'nan'")
        return True, 1.0

    if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
        logger.error("One tensor is all nan, the other is not.")
        return False, 0.0

    # Test if either is completely zero
    if torch.any(golden.bool()) != torch.any(calculated.bool()):
        logger.error("One tensor is all zero")
        return False, 0.0

    # For now, mask all infs and nans so that we check the rest... TODO
    golden = golden.clone()
    golden[
        torch.logical_or(
            torch.isnan(golden),
            torch.logical_or(torch.isinf(golden), torch.isneginf(golden)),
        )
    ] = 0
    calculated = calculated.clone()
    calculated[
        torch.logical_or(
            torch.isnan(calculated),
            torch.logical_or(torch.isinf(calculated), torch.isneginf(calculated)),
        )
    ] = 0

    if torch.equal(golden, calculated):
        return True, 1.0

    if golden.dtype == torch.bfloat16:
        golden = golden.type(torch.float32)
        calculated = calculated.type(torch.float32)
    cal_pcc = np.min(
        np.ma.corrcoef(
            np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
            np.ma.masked_invalid(torch.squeeze(calculated).detach().numpy()).flatten(),
        )
    )

    if isinstance(cal_pcc, np.ma.core.MaskedConstant):
        return True, 1.0

    print(f"cal_pcc={cal_pcc}")
    print(f"actual_pcc={pcc}")

    return cal_pcc >= pcc, cal_pcc


def subtract(lhs_ptr, rhs_ptr, device_golden_ptr):
    import pdb
    pdb.set_trace()
    print("-----------------------------------------")

    print(ttnn_api_golden_mapping)

    sub_func = ttnn_api_golden_mapping["ttnn.subtract"][1]


    lhs_torch = torch.tensor(lhs_ptr)
    rhs_torch = torch.tensor(rhs_ptr)
    device_golden_torch = torch.tensor(device_golden_ptr)
  

    print(lhs_torch)
    print(rhs_torch)
    print(device_golden_torch)

    golden_torch = sub_func(lhs_torch, rhs_torch)

    comp_pcc(golden_torch, device_golden_torch, 0.9999)
    print(torch.allclose(golden_torch, device_golden_torch))

    print(device_golden_torch)
    print(golden_torch)


    print("-----------------------------------------")

    return 5



#tensor([-1.3906, -0.3613, -1.2310,  ..., -0.8384,  0.2979,  0.8926])