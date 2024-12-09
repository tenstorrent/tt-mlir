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
        # Test if either is completely zero
        elif torch.any(golden.bool()) != torch.any(calculated.bool()):
            return 0.0
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


def golden(runtimeConfig, binary, programContext, opContext):
    import torch
    import ttrt.runtime
    import ttrt.binary

    print("-----------executing golden comparision-----------")

    try:
        op_debug_str = ttrt.runtime.get_op_debug_str(opContext)

        # find matching golden tensor based on loc in op debug string
        match = re.search(r"loc\(([^)]+)\)", op_debug_str)

        if not match:
            print(f"debug_str={op_debug_str}")
            print("No location found in debug string - skipping golden comparison")
            return

        loc = match.group(1).replace('"', "")
        print(f"found location={loc}")

        op_golden_tensor = binary.get_debug_info_golden(loc)
        op_output_tensor = ttrt.runtime.get_op_output_tensor(opContext, programContext)

        if len(op_golden_tensor) == 0:
            print("Golden tensor is empty - skipping golden comparison")
            return

        if len(op_output_tensor) == 0:
            print("Output tensor is empty - skipping golden comparison")
            return

        if len(op_golden_tensor) != len(op_output_tensor):
            print(
                "Golden and output tensor sizes do not match - skipping golden comparison"
            )
            return

        golden_tensor_torch = torch.tensor(
            op_golden_tensor, dtype=torch.float32
        ).flatten()
        output_tensor_torch = torch.tensor(
            op_output_tensor, dtype=torch.float32
        ).flatten()

        torch.save(golden_tensor_torch, f"{runtimeConfig.artifact_dir}/{loc}_golden.pt")
        torch.save(output_tensor_torch, f"{runtimeConfig.artifact_dir}/{loc}_device.pt")

        _, _, cal_pcc, output_str = get_atol_rtol_pcc(
            golden_tensor_torch, output_tensor_torch
        )

        print(f"PCC={cal_pcc}")
        print(output_str)

        results = {}
        results["expected_pcc"] = runtimeConfig.pcc
        results["actual_pcc"] = cal_pcc
        results["atol"] = runtimeConfig.atol
        results["rtol"] = runtimeConfig.rtol
        results["allclose"] = torch.allclose(
            golden_tensor_torch,
            output_tensor_torch,
            atol=runtimeConfig.atol,
            rtol=runtimeConfig.rtol,
        )
        results["max"] = torch.max(
            torch.abs(golden_tensor_torch - output_tensor_torch)
        ).item()
        results["mean_absolute_error"] = torch.mean(
            torch.abs(golden_tensor_torch - output_tensor_torch)
        ).item()
        results["root_mean_square_error"] = torch.sqrt(
            torch.mean((golden_tensor_torch - output_tensor_torch) ** 2)
        ).item()
        results["cosine_similarity"] = torch.nn.functional.cosine_similarity(
            golden_tensor_torch.unsqueeze(0), output_tensor_torch.unsqueeze(0)
        ).item()

        # Create golden result json file if doesn't exist
        golden_results_file_path = f"{runtimeConfig.artifact_dir}/golden_results.json"
        if not os.path.exists(golden_results_file_path):
            with open(golden_results_file_path, "w") as f:
                json.dump({}, f)

        with open(golden_results_file_path, "r") as f:
            try:
                data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError(
                        f"Invalid JSON format: Expected a dict but got {type(data).__name__}"
                    )
            except json.JSONDecodeError:
                data = {}

        data[loc] = results
        with open(golden_results_file_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Saved golden results to={golden_results_file_path}")

    finally:
        print("-----------finished executing golden comparision-----------")
