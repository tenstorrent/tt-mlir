# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compile options through the EP options dict.

ORT stores add_provider_for_devices' dict as session config entries under
"ep.ttkurblaexecutionprovider."; the EP parses them into the engine's
CompileOptions at CreateEp. Keys are CompileOptions field names, enum values
their C++ spellings — the same names the torch frontend uses.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper

from ort_ep_utils import EP_OPSET_VERSION, cpu_golden, session_on_tt


def _model() -> bytes:
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["a", "b"], ["y"])],
        "compile_options",
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 64]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [64, 32]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [32, 32])],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", EP_OPSET_VERSION)]
    ).SerializeToString()


def test_compile_options_applied() -> None:
    # Distinct options force a fresh compile (they are part of the engine's
    # cache key); results must still match the CPU golden.
    rng = np.random.default_rng(0)
    feeds = {
        "a": rng.standard_normal((32, 64), dtype=np.float32),
        "b": rng.standard_normal((64, 32), dtype=np.float32),
    }
    session = session_on_tt(
        _model(),
        compile_options={
            "math_fidelity": "HiFi4",
            "fp32_dest_acc_en": "true",
            "enable_const_eval": "1",
            "experimental_enable_permute_matmul_fusion": "false",
            "optimization_level": "0",
        },
    )
    (got,) = session.run(None, feeds)
    (want,) = cpu_golden(_model(), feeds)
    np.testing.assert_allclose(got, want, atol=2e-2, rtol=2e-2)


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
    # The EP's parse error fails EP creation; the python wrapper then retries
    # with the CPU EP, which the disable_cpu_ep_fallback entry rejects — so
    # session creation raises (with ORT's conflict message), and our parse
    # error is printed in the wrapper's "EP Error" output.
    with pytest.raises(
        Exception, match="Conflicting session configuration|compile option"
    ):
        session_on_tt(_model(), compile_options=bad)
    captured = capfd.readouterr()
    assert "compile option" in captured.out + captured.err
