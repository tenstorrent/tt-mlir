# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re
from functools import partial

from ttrt.common.util import *


class GoldenRuntimeConfig:
    def __init__(
        self,
        atol=1e-08,
        rtol=1e-05,
        pcc=0.99,
        artifact_dir="",
        save_golden_tensors=False,
    ):
        self.artifact_dir = artifact_dir
        self.pcc = pcc
        self.atol = atol
        self.rtol = rtol
        self.save_golden_tensors = save_golden_tensors


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


def golden_partial_function(
    golden_runtime_config, golden_results_data, binary, program_context, op_context
):
    import torch
    import ttrt.runtime
    import ttrt.binary

    print("-----------executing golden comparision-----------")

    try:
        op_debug_str = ttrt.runtime.get_op_debug_str(op_context)

        # find matching golden tensor based on loc in op debug string
        match = re.search(r"loc\(([^)]+)\)", op_debug_str)

        if not match:
            print(f"debug_str={op_debug_str}")
            print("No location found in debug string - skipping golden comparison")
            return

        loc = match.group(1).replace('"', "")
        print(f"found location={loc}")

        op_golden_tensor = binary.get_debug_info_golden(loc)
        op_output_tensor = ttrt.runtime.get_op_output_tensor(
            op_context, program_context
        )

        if op_golden_tensor is None:
            print("Golden tensor is None - skipping golden comparison")
            return

        if len(op_output_tensor) == 0:
            print("Output tensor is empty - skipping golden comparison")
            return

        dtype = ttrt_datatype_to_torch_dtype(op_golden_tensor.dtype)

        golden_tensor_torch = torch.frombuffer(op_golden_tensor, dtype=dtype).flatten()

        output_tensor_torch = torch.tensor(op_output_tensor, dtype=dtype).flatten()

        if golden_runtime_config.save_golden_tensors:
            torch.save(
                golden_tensor_torch,
                f"{golden_runtime_config.artifact_dir}/{loc}_golden.pt",
            )
            torch.save(
                output_tensor_torch,
                f"{golden_runtime_config.artifact_dir}/{loc}_device.pt",
            )

        if golden_tensor_torch.shape != output_tensor_torch.shape:
            print(
                "Golden and output tensor shapes do not match - skipping golden comparison"
            )
            return

        _, _, cal_pcc, output_str = get_atol_rtol_pcc(
            golden_tensor_torch, output_tensor_torch
        )

        print(f"PCC={cal_pcc}")
        print(output_str)

        results = {}
        results["expected_pcc"] = golden_runtime_config.pcc
        results["actual_pcc"] = cal_pcc
        results["atol"] = golden_runtime_config.atol
        results["rtol"] = golden_runtime_config.rtol
        results["allclose"] = torch.allclose(
            golden_tensor_torch,
            output_tensor_torch,
            atol=golden_runtime_config.atol,
            rtol=golden_runtime_config.rtol,
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

        golden_results_data[loc] = results

    finally:
        print("-----------finished executing golden comparision-----------")


def get_golden_fn(golden_runtime_config, golden_results_data):
    return partial(golden_partial_function, golden_runtime_config, golden_results_data)
