# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel._src.kernel_ast import *
import functools
import torch
import sys


def assert_pcc(golden, actual, threshold=0.99):
    combined = torch.stack([golden.flatten(), actual.flatten()])
    pcc = torch.corrcoef(combined)[0, 1].item()
    assert (
        pcc >= threshold
    ), f"Expected pcc {pcc} >= {threshold}\ngolden:\n{golden}\nactual:\n{actual}"
