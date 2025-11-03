# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List

from ttmlir.ir import *
from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_ttir_to_flatbuffer
from typing import Optional

pytestmark = pytest.mark.frontend("ttir")

#from chisel.utils.metrics import compute_pcc, compute_abs_err, compute_rel_err

######################## Utility functions ###################################
def compute_pcc(ttir_result, ttnn_result):
    if ttir_result is None or ttnn_result is None:
        return -1

    x = ttir_result.to(torch.float32).flatten()
    y = ttnn_result.to(torch.float32).flatten()

    if torch.all(torch.isnan(x)) and torch.all(torch.isnan(y)):
        return 1.0

    if torch.all(torch.isinf(x)) and torch.all(torch.isinf(y)):
        if torch.all(x == y):
            return 1.0
        return 0.0

    mask = ~(torch.isnan(x) | torch.isinf(x) | torch.isnan(y) | torch.isinf(y))

    try:
        x = x[mask]
        y = y[mask]
    except RuntimeError as e:
        print(f"Warning: Masking failed with error: {e}")
        pass

    if x.numel() == 0 or y.numel() == 0:
        return 1.0

    x_centered = x - x.mean()
    y_centered = y - y.mean()

    numerator = torch.sum(x_centered * y_centered)
    denominator = torch.sqrt(torch.sum(x_centered**2) * torch.sum(y_centered**2))

    if denominator == 0:
        pcc = 1.0
    else:
        pcc = numerator / denominator

    return float(pcc)



def test_digamma():
    shape = (128, 128)
    dtype = torch.bfloat16
    # set torch inputs and constant_tensors
    # NOTE: digamma approximation only valid for x > 1 (preferably x >> 1)
    x_tensor = torch.rand(shape).to(dtype) * 1e5
    x_tensor = torch.clamp(x_tensor, min=1)
    print(x_tensor)
    
    constant_tensors = [
        torch.full(shape, 0.5).to(dtype),
        torch.full(shape, 0.083333333).to(dtype),
        torch.full(shape, 0.008333333333333333).to(dtype),
        torch.full(shape, 0.003968253968253968).to(dtype),
        torch.full(shape, 0.004166666666666667).to(dtype),
        torch.full(shape, 0.007575757575757576).to(dtype),
        torch.full(shape, 0.021092796092796094).to(dtype),
        torch.full(shape, 0.08333333333333333).to(dtype),
    ]     

    # compute outputs
    output_golden = torch.digamma(x_tensor).to(dtype)

    # create torch output
    recip = torch.reciprocal(x_tensor)
    term1 = torch.multiply(recip, constant_tensors[0])

    recip_square = torch.square(recip)
    term2 = torch.multiply(recip_square, constant_tensors[1])
    interm2 = torch.subtract(term1, term2)

    recip_pow_4 = torch.multiply(recip_square, recip_square)
    term3 = torch.multiply(recip_pow_4, constant_tensors[2])
    interm3 = torch.add(interm2, term3)

    recip_pow_6 = torch.multiply(recip_pow_4, recip_square)
    term4 = torch.multiply(recip_pow_6, constant_tensors[3])
    interm4 = torch.subtract(interm3, term4)

    recip_pow_8 = torch.multiply(recip_pow_6, recip_square)
    term5 = torch.multiply(recip_pow_8, constant_tensors[4])
    interm5 = torch.add(interm4, term5)

    recip_pow_10 = torch.multiply(recip_pow_8, recip_square)
    term6 = torch.multiply(recip_pow_10, constant_tensors[5])
    interm6 = torch.subtract(interm5, term6)

    recip_pow_12 = torch.multiply(recip_pow_10, recip_square)
    term7 = torch.multiply(recip_pow_12, constant_tensors[6])
    interm7 = torch.add(interm6, term7)

    recip_pow_14 = torch.multiply(recip_pow_12, recip_square)
    term8 = torch.multiply(recip_pow_14, constant_tensors[7])
    interm8 = torch.subtract(interm7, term8)

    t_log_out = torch.log(x_tensor)
    output_approx = torch.subtract(t_log_out, interm8)

    pcc = compute_pcc(output_approx, output_golden)
    
    # Calculate error metrics
    abs_diff = torch.abs(output_approx - output_golden)
    rel_diff = abs_diff / (torch.abs(output_golden) + 1e-8)
    
    print("="*80)
    print(f"Input range: [{x_tensor.min():.4f}, {x_tensor.max():.4f}]")
    print(f"Output (approx) sample: {output_approx[0, :5]}")
    print(f"Output (golden) sample: {output_golden[0, :5]}")
    print(f"PCC: {pcc:.6f}")
    print(f"Max absolute error: {abs_diff.max():.6f}")
    print(f"Mean absolute error: {abs_diff.mean():.6f}")
    print(f"Max relative error: {rel_diff.max():.6f}")
    print(f"Mean relative error: {rel_diff.mean():.6f}")
    print("="*80)

print("\n### Testing with BFLOAT16 ###")
test_digamma()