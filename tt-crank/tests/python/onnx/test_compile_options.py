# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compile options passed through the EP options dict (CompileOptions field names, C++ enum spellings)."""

import pytest
from onnx import helper

from ort_ep_utils import assert_tt_matches_cpu, randn, make_model, session_on_tt, vi

_GEMM = make_model(
    [helper.make_node("Gemm", ["a", "b"], ["y"])],
    [vi("a", [32, 64]), vi("b", [64, 32])],
    [vi("y", [32, 32])],
)


def test_compile_options_applied() -> None:
    # Distinct options force a fresh compile; the result must still match CPU.
    assert_tt_matches_cpu(
        _GEMM,
        {"a": randn(32, 64), "b": randn(64, 32)},
        compile_options={
            "math_fidelity": "HiFi4",
            "fp32_dest_acc_en": "true",
            "enable_const_eval": "1",
            "experimental_enable_permute_matmul_fusion": "false",
            "optimization_level": "0",
        },
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"math_fidelity": "UltraFi"},
        {"optimization_level": "fast"},
        {"fp32_dest_acc_en": "yes please"},
        {"experimental_weight_dtype": "bf16"},
    ],
)
def test_invalid_compile_option_fails_session_creation(
    bad: dict, capfd: pytest.CaptureFixture
) -> None:
    # EP creation fails on the parse error; ORT's python wrapper then retries with
    # the CPU EP, which disable_cpu_ep_fallback rejects, so session creation raises.
    with pytest.raises(
        Exception, match="Conflicting session configuration|compile option"
    ):
        session_on_tt(_GEMM, compile_options=bad)
    captured = capfd.readouterr()
    assert "compile option" in captured.out + captured.err
