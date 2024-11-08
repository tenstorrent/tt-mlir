# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

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
import shutil
import atexit
import re

from ttrt.common.util import *

GOLDENS = {}


def get_atol_rtol_pcc(golden, calculated):
    import numpy as np
    import torch

    # Calculate atol and rtol
    cal_atol = torch.max(torch.abs(golden - calculated)).item()
    cal_rtol = torch.max(torch.abs(golden - calculated) / torch.abs(calculated)).item()

    # Calculate PCC
    def get_pcc(golden, calculated):
        # Both tensors are nan
        if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
            print("Both tensors are 'nan'")
            return 1.0
        # One tensor is all nan, the other is not
        elif torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
            print("One tensor is all nan, the other is not.")
            return 0.0
        else:
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
                    torch.logical_or(
                        torch.isinf(calculated), torch.isneginf(calculated)
                    ),
                )
            ] = 0

            if torch.equal(golden, calculated):
                return 1.0

            if golden.dtype == torch.bfloat16:
                golden = golden.type(torch.float32)
                calculated = calculated.type(torch.float32)

            # Single element case
            if golden.numel() == 1:
                return float(torch.equal(golden, calculated))

            # If both tensors are contant
            if torch.max(golden) == torch.min(golden) and torch.max(
                calculated
            ) == torch.min(calculated):
                return torch.isclose(torch.max(golden), torch.max(calculated)).item()

            cal_pcc = np.ma.corrcoef(
                np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
                np.ma.masked_invalid(
                    torch.squeeze(calculated).detach().numpy()
                ).flatten(),
            )
            # Remove correlation coefficient with self (typically always 1.0)
            mask = np.ones(cal_pcc.shape, dtype=bool)
            np.fill_diagonal(mask, 0)
            cal_pcc = np.min(cal_pcc[mask])

            if isinstance(cal_pcc, np.ma.core.MaskedConstant):
                return 1.0

            return cal_pcc

    cal_pcc = get_pcc(golden, calculated)

    return (
        cal_atol,
        cal_rtol,
        cal_pcc,
        f"Max ATOL Delta: {cal_atol}, Max RTOL Delta: {cal_rtol}, PCC: {cal_pcc}",
    )


def add_global_golden(golden_tensor):
    global GOLDENS
    GOLDENS[golden_tensor.tensor_id] = golden_tensor.get_torch_tensor()


def golden(context=None, opContext=None):
    import torch
    import ttrt.runtime

    print("-----------executing golden comparision-----------")

    try:
        device_tensor = ttrt.runtime.get_op_output_tensor(context, opContext)
        op_debug_str = ttrt.runtime.get_op_debug_str(context, opContext)

        if device_tensor == None or len(device_tensor) == 0:
            print("No device tensor provided for golden comparison")
            return
        elif op_debug_str == None or op_debug_str == "":
            print("No debug string provided for golden comparison")
            return
        else:
            # find matching golden tensor based on loc in op debug string
            match = re.search(r"loc\(([^)]+)\)", op_debug_str)

            if not match:
                print(f"debug_str={op_debug_str}")
                print("No location found in debug string - skipping golden comparison")
                return

            loc = match.group(1).replace('"', "")
            print(f"found location={loc}")

            if loc not in GOLDENS.keys():
                print(
                    f"No golden tensor found for loc={loc} in golden cache - skipping golden comparison"
                )
                return

            golden_torch_tensor = GOLDENS[loc].flatten()
            device_tensor_torch = torch.tensor(device_tensor, dtype=torch.float32)
            _, _, cal_pcc, output_str = get_atol_rtol_pcc(
                golden_torch_tensor, device_tensor_torch
            )

            print(f"PCC={cal_pcc}")
            print(output_str)
    finally:
        print("-----------finished executing golden comparision-----------")
