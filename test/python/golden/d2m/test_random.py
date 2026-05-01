# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# End-to-end tests for ttir.rand on TTMetal. The SFPU PRNG isn't
# torch-matchable, so instead of a golden check we validate range,
# non-degeneracy, seed reproducibility, and seed sensitivity. Non-f32 dtypes
# lower as f32 rand + tile_typecast, so this also covers that path.

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import pytest
import torch

from conftest import get_request_kwargs
from d2m.shape_cases import rotated_params
from test_utils import shape_str

from builder.base.builder_utils import get_artifact_dir
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_ttir_to_flatbuffer
from builder.base.builder_runtime import execute_fb

pytestmark = pytest.mark.frontend("ttir")


# ----------------------------- Harness helpers -----------------------------


def _run_rand_and_capture(
    shape: Tuple[int, int],
    dtype: torch.dtype,
    low: float,
    high: float,
    seed: int,
    request,
    device,
    test_suffix: str = "",
) -> torch.Tensor:
    """Compile+execute a single ``ttir.rand`` and return its device output,
    cast to f32. Uses low-level APIs so we can grab the runtime tensor."""

    def module(builder: TTIRBuilder):
        @builder.func([], [])
        def rand_fn(
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.rand(
                list(shape),
                dtype,
                low=low,
                high=high,
                seed=seed,
                unit_attrs=unit_attrs,
            )

    kwargs = get_request_kwargs(request)
    test_base = kwargs["test_base"]
    if test_suffix:
        test_base = f"{test_base}__{test_suffix}"
    output_root = kwargs["output_root"]
    system_desc_path = kwargs["system_desc_path"]

    artifact_dir = get_artifact_dir(
        output_root, "TTIRBuilder", test_base, make_dir=True
    )

    (
        builder,
        compiled_bin,
        io_goldens,
        intermediate_goldens,
    ) = compile_ttir_to_flatbuffer(
        module,
        system_desc_path=system_desc_path,
        artifact_dir=artifact_dir,
        target="ttmetal",
        save_artifacts=False,
    )

    # disable_golden=False to get tensors back; check_* off since HW PRNG
    # doesn't match torch.
    _golden_report, output_tensors = execute_fb(
        compiled_bin,
        input_output_goldens=io_goldens,
        intermediate_goldens=intermediate_goldens,
        disable_golden=False,
        device=device,
        check_pcc=False,
        check_atol=False,
        check_rtol=False,
        save_artifacts=False,
        artifact_dir=artifact_dir,
        bypass_ops=builder._bypass_ops,
    )

    assert output_tensors, "execute_fb returned no output tensors"
    program_outputs: Dict[str, Dict[int, torch.Tensor]] = output_tensors["program_0"]
    device_out = program_outputs["device_output_0"]
    assert (
        len(device_out) == 1
    ), f"expected single-shard output, got {len(device_out)} shards"
    return device_out[0].to(torch.float32)


def _assert_uniform_in_range(
    t: torch.Tensor,
    shape: Tuple[int, int],
    low: float,
    high: float,
    *,
    bounds_eps: float = 1e-3,
    mean_tol_frac: float = 0.15,
    min_unique: int = 32,
) -> None:
    """Loose sanity check that ``t`` looks like ``Uniform[low, high)``.
    Catches obvious regressions, not PRNG quality."""
    assert torch.is_tensor(t), f"expected tensor, got {type(t)!r}"
    assert tuple(t.shape) == shape, f"shape mismatch: {tuple(t.shape)} vs {shape}"
    assert not torch.isnan(t).any(), "output contains NaNs"
    assert not torch.isinf(t).any(), "output contains infs"

    # Small eps on both sides to tolerate bf16/f32 rounding at the bounds.
    t_min = float(t.min().item())
    t_max = float(t.max().item())
    assert t_min >= low - bounds_eps, f"min {t_min} below low {low} (eps={bounds_eps})"
    assert (
        t_max <= high + bounds_eps
    ), f"max {t_max} above high {high} (eps={bounds_eps})"

    t_std = float(t.std().item())
    assert t_std > 0.0, "output tensor is constant (std == 0)"

    # Absolute floor, not a ratio: bf16 caps at ~256 values per narrow range.
    numel = t.numel()
    if numel >= 1024:
        unique = int(torch.unique(t).numel())
        assert (
            unique >= min_unique
        ), f"too few unique values: {unique} < {min_unique} (numel={numel})"

    # Absolute band vs. an iid-uniform σ check; SFPU rand has a ~5% small-N bias.
    span = high - low
    expected_mean = 0.5 * (low + high)
    empirical_mean = float(t.mean().item())
    diff = abs(empirical_mean - expected_mean)
    tol = mean_tol_frac * span
    assert diff <= tol, (
        f"empirical mean {empirical_mean:.4f} too far from expected "
        f"{expected_mean:.4f} (|diff|={diff:.4f}, allowed={tol:.4f})"
    )


# --------------------------------- Tests ----------------------------------


_RAND_SHAPES = [(32, 32), (128, 128), (64, 128)]


_RAND_CASES = [
    pytest.param(torch.float32, 0.0, 1.0, 0, id="f32-uniform01"),
    pytest.param(torch.bfloat16, -1.0, 1.0, 42, id="bf16-signed_unit"),
    pytest.param(torch.float32, 2.0, 5.0, 1337, id="f32-range_2to5"),
    pytest.param(torch.bfloat16, 0.0, 1.0, 7, id="bf16-uniform01"),
    pytest.param(torch.int32, 0.0, 128.0, 0, id="i32-pos128"),
    pytest.param(torch.int32, -64.0, 64.0, 42, id="i32-signed64"),
    pytest.param(torch.int32, 10.0, 100.0, 1337, id="i32-range_10to100"),
]


def _assert_integer_rand_in_range(
    t: torch.Tensor,
    shape: Tuple[int, int],
    low: float,
    high: float,
) -> None:
    assert tuple(t.shape) == shape, f"shape mismatch: {tuple(t.shape)} vs {shape}"
    assert not torch.isnan(t).any(), "int rand output contains NaNs"
    assert not torch.isinf(t).any(), "int rand output contains infs"

    fractional = t - torch.floor(t)
    assert float(fractional.abs().max()) == 0.0, (
        f"int rand produced non-integer values (max fractional part = "
        f"{float(fractional.abs().max())})"
    )

    int_lo = math.floor(low)
    int_hi_excl = math.ceil(high)
    t_min = int(t.min().item())
    t_max = int(t.max().item())
    assert t_min >= int_lo, f"int rand min {t_min} below low {int_lo}"
    assert t_max < int_hi_excl, f"int rand max {t_max} >= high {int_hi_excl}"

    unique = int(torch.unique(t).numel())
    possible = int_hi_excl - int_lo
    expected_floor = max(2, min(possible, 8))
    assert unique >= expected_floor, (
        f"int rand produced too few distinct values: {unique} "
        f"(expected at least {expected_floor} over range [{int_lo}, {int_hi_excl}))"
    )


@pytest.mark.parametrize(
    "shape,dtype,low,high,seed",
    rotated_params(_RAND_CASES, _RAND_SHAPES, value_order=[4, 0, 1, 2, 3]),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_rand(
    shape,
    dtype: torch.dtype,
    low: float,
    high: float,
    seed: int,
    target: str,
    request,
    device,
):
    """End-to-end ``ttir.rand``; validates range and non-degeneracy."""
    out = _run_rand_and_capture(
        shape, dtype, low, high, seed, request, device, test_suffix="main"
    )

    if not dtype.is_floating_point:
        _assert_integer_rand_in_range(out, shape, low, high)
        return

    # bf16 quantizes to ~2^-7, so loosen the bounds check.
    bounds_eps = 5e-3 if dtype == torch.bfloat16 else 1e-4
    _assert_uniform_in_range(out, shape, low, high, bounds_eps=bounds_eps)


@pytest.mark.parametrize("target", ["ttmetal"])
def test_rand_reproducibility(target: str, request, device):
    """Same seed must produce bit-identical output across independent runs."""
    shape = (64, 64)
    dtype = torch.float32
    low, high, seed = -0.5, 0.5, 12345

    run_a = _run_rand_and_capture(
        shape, dtype, low, high, seed, request, device, test_suffix="repro_a"
    )
    run_b = _run_rand_and_capture(
        shape, dtype, low, high, seed, request, device, test_suffix="repro_b"
    )

    _assert_uniform_in_range(run_a, shape, low, high)
    _assert_uniform_in_range(run_b, shape, low, high)
    assert torch.equal(run_a, run_b), (
        "two runs with identical seed produced different tensors "
        f"(max abs diff = {float((run_a - run_b).abs().max()):.6f})"
    )


@pytest.mark.parametrize("target", ["ttmetal"])
def test_rand_different_seeds_differ(target: str, request, device):
    """Distinct seeds must produce meaningfully different tensors."""
    shape = (64, 64)
    dtype = torch.float32
    low, high = 0.0, 1.0

    run_a = _run_rand_and_capture(
        shape, dtype, low, high, 1, request, device, test_suffix="seed_a"
    )
    run_b = _run_rand_and_capture(
        shape, dtype, low, high, 2, request, device, test_suffix="seed_b"
    )

    _assert_uniform_in_range(run_a, shape, low, high)
    _assert_uniform_in_range(run_b, shape, low, high)

    assert not torch.equal(run_a, run_b), "different seeds produced identical tensors"
    diff_frac = float((run_a != run_b).float().mean())
    assert diff_frac > 0.5, (
        f"different seeds produced near-identical tensors: only {diff_frac:.4f} "
        "of elements differ"
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_rand_unary(
    shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """Compose rand with a unary op for multi-op pipeline coverage."""
    from builder.base.builder_apis import compile_and_execute_ttir

    def module(builder: TTIRBuilder):
        @builder.func([], [])
        def rand_abs(
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            r = builder.rand(
                list(shape), dtype, low=-1.0, high=1.0, seed=7, unit_attrs=unit_attrs
            )
            return builder.abs(r, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        target=target,
        **get_request_kwargs(request),
        device=device,
        disable_golden=True,
    )

    out = _run_rand_and_capture(
        shape, dtype, -1.0, 1.0, 7, request, device, test_suffix="unary_rand_only"
    )
    bounds_eps = 5e-3 if dtype == torch.bfloat16 else 1e-4
    _assert_uniform_in_range(out, shape, -1.0, 1.0, bounds_eps=bounds_eps)
