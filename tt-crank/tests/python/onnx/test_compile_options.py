# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compile options passed through the EP options dict (CompileOptions field names, C++ enum spellings)."""

import pytest

import tt_crank.onnx as tt_onnx

_GEMM = tt_onnx.model(
    [tt_onnx.node("Gemm", ["a", "b"], ["y"])],
    [tt_onnx.input("a", [32, 64]), tt_onnx.input("b", [64, 32])],
    [tt_onnx.output("y", [32, 32])],
)


def test_compile_options_applied() -> None:
    # Distinct options force a fresh compile; the result must still match CPU.
    tt_onnx.assert_tt_matches_cpu(
        _GEMM,
        {"a": tt_onnx.randn(32, 64), "b": tt_onnx.randn(64, 32)},
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
        {"fp32_dest_acc_en": "jeste, aha"},
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
        tt_onnx.session(_GEMM, compile_options=bad)
    captured = capfd.readouterr()
    assert "compile option" in captured.out + captured.err
