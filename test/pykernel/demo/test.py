# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from eltwise_sfpu_demo import main as eltwise_sfpu_demo
from vecadd_multicore_demo import main as vecadd_multicore_demo
from matmul_singlecore_demo import main as matmul_singlecore_demo
from matmul_multicore_demo import main as matmul_multicore_demo
from dprint_demo import main as dprint_demo
from cosh_jit import main as cosh_jit


@pytest.mark.usefixtures("device")
def test_eltwise_sfpu(device):
    eltwise_sfpu_demo(device)


@pytest.mark.usefixtures("device")
def test_vecadd_multicore(device):
    vecadd_multicore_demo(device)


@pytest.mark.usefixtures("device")
def test_matmul_singlecore(device):
    matmul_singlecore_demo(device)


@pytest.mark.usefixtures("device")
def test_matmul_multicore(device):
    matmul_multicore_demo(device)


@pytest.mark.usefixtures("device")
def test_dprint(device):
    dprint_demo(device)


def test_cosh_jit():
    cosh_jit()
