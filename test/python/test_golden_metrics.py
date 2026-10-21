# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s

# Regression tests for the golden PCC/atol/rtol metrics (issue #9387).
#
# Before the fix, get_pcc reported a perfect 1.0 when exactly one tensor was
# constant (np.ma.corrcoef returned a MaskedConstant) and for any same-sign
# scalar comparison (cosine similarity used as a boolean). Both silently passed
# clearly wrong device outputs through the builder/ttrt golden checks.

import importlib.util
import math
import pathlib

import torch

# Load golden/metrics.py directly: importing the `golden` package runs
# golden/__init__.py, which pulls in golden/mapping.py and with it the StableHLO
# dialect bindings, which are absent in builds without TTMLIR_ENABLE_STABLEHLO.
# find_spec() locates the installed package without executing its __init__.
_metrics_path = pathlib.Path(importlib.util.find_spec("golden").origin).with_name(
    "metrics.py"
)
_spec = importlib.util.spec_from_file_location("golden_metrics", _metrics_path)
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
get_atol_rtol = _metrics.get_atol_rtol
get_atol_rtol_pcc = _metrics.get_atol_rtol_pcc
get_pcc = _metrics.get_pcc

ATOL = 1e-8
RTOL = 1e-5


def pcc(golden, calculated, atol=ATOL, rtol=RTOL):
    return get_pcc(golden, calculated, atol, rtol)


def check_metrics(name, get_pcc_fn, get_atol_rtol_fn):
    gen = torch.Generator().manual_seed(0)
    random_tensor = torch.randn(32, 32, generator=gen) * 1000.0

    # One constant tensor vs. a varying one must not pass.
    assert get_pcc_fn(torch.full((32, 32), 3.0), random_tensor) < 0.99
    assert get_pcc_fn(random_tensor, torch.full((32, 32), 3.0)) < 0.99
    # Device output stuck at a single non-zero value vs. a varying golden.
    assert get_pcc_fn(torch.linspace(1.0, 2.0, 1024), torch.full((1024,), 1.5)) < 0.99

    # Both constant: pass only when the values match within tolerance.
    assert get_pcc_fn(torch.full((16, 16), 3.14), torch.full((16, 16), 3.14)) == 1.0
    assert get_pcc_fn(torch.full((16, 16), 1.0), torch.full((16, 16), 2.0)) == 0.0
    assert (
        get_pcc_fn(
            torch.full((32,), 1.0), torch.full((32,), 1.0 + 5e-4), atol=1e-3, rtol=0.0
        )
        == 1.0
    )
    assert (
        get_pcc_fn(
            torch.full((32,), 1.0), torch.full((32,), 1.0 + 5e-4), atol=1e-4, rtol=0.0
        )
        == 0.0
    )

    # Scalars: compare values, never the sign of the product.
    assert get_pcc_fn(torch.tensor([5.0]), torch.tensor([5.0])) == 1.0
    assert get_pcc_fn(torch.tensor([5.0]), torch.tensor([-5.0])) == 0.0
    assert get_pcc_fn(torch.tensor([5.0]), torch.tensor([5000.0])) == 0.0
    assert get_pcc_fn(torch.tensor([5.0]), torch.tensor([float("nan")])) == 0.0
    assert get_pcc_fn(torch.tensor([5.0]), torch.tensor([5.0 + 1e-9])) == 1.0
    assert get_pcc_fn(torch.tensor([0.0]), torch.tensor([0.0])) == 1.0
    assert get_pcc_fn(torch.tensor([0.0]), torch.tensor([1.0])) == 0.0

    # Sanity: normal correlated tensors still pass and return a plain float.
    golden = torch.linspace(0.0, 1.0, 1024)
    calculated = golden + torch.randn(1024, generator=gen) * 0.001
    result = get_pcc_fn(golden, calculated)
    assert isinstance(result, float), type(result)
    assert result > 0.99, result
    assert not math.isnan(result)
    assert get_pcc_fn(golden, -golden) < 0.0

    # Identical tensors and both-NaN tensors are a perfect match.
    assert get_pcc_fn(random_tensor, random_tensor.clone()) == 1.0
    assert (
        get_pcc_fn(torch.full((4,), float("nan")), torch.full((4,), float("nan")))
        == 1.0
    )

    # Integer and bfloat16 inputs go through the same paths.
    assert (
        get_pcc_fn(
            torch.full((8, 8), 3, dtype=torch.int32), torch.arange(64).reshape(8, 8)
        )
        < 0.99
    )
    assert (
        get_pcc_fn(
            torch.full((8, 8), 3.0, dtype=torch.bfloat16),
            torch.randn(8, 8, generator=gen).bfloat16(),
        )
        < 0.99
    )

    # rtol is relative to the golden value.
    cal_atol, cal_rtol = get_atol_rtol_fn(torch.tensor([2.0]), torch.tensor([1.0]))
    assert cal_atol == 1.0, cal_atol
    assert cal_rtol == 0.5, cal_rtol

    print(f"{name}: OK")


# CHECK: golden.metrics: OK
check_metrics("golden.metrics", pcc, get_atol_rtol)

# The wrapper must agree with get_pcc and forward the tolerances.
cal_atol, cal_rtol, cal_pcc = get_atol_rtol_pcc(
    torch.full((32, 32), 3.0), torch.zeros(32, 32) + torch.eye(32), ATOL, RTOL
)
assert cal_pcc == 0.0, cal_pcc
assert cal_atol == 3.0, cal_atol
# CHECK: golden.metrics wrapper: OK
print("golden.metrics wrapper: OK")

# ttrt keeps its own copy of the same logic; exercise it when ttrt is built
# (it requires the runtime, so it may be absent in compiler-only builds).
try:
    from ttrt.common.util import get_atol_rtol_pcc as ttrt_get_atol_rtol_pcc
except Exception as e:  # ttrt needs the runtime wheel; any failure means "not built".
    print(f"ttrt.common.util: SKIPPED (ttrt not available: {e})")
    raise SystemExit(0)
else:

    def ttrt_pcc(golden, calculated, atol=ATOL, rtol=RTOL):
        return ttrt_get_atol_rtol_pcc(golden, calculated, atol, rtol)[2]

    def ttrt_atol_rtol(golden, calculated):
        return ttrt_get_atol_rtol_pcc(golden, calculated, ATOL, RTOL)[:2]

    check_metrics("ttrt.common.util", ttrt_pcc, ttrt_atol_rtol)

# CHECK: ttrt.common.util: {{OK|SKIPPED}}
