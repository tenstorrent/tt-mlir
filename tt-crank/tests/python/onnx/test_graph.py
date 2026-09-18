# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Graph-structure handling: shared initializers, parameters consumed mid-graph."""

import numpy as np

import tt_onnx


def test_initializer_shared_between_metadata_and_operand() -> None:
    # The same constant is Clip's min bound (read at build time) and Mul's operand
    # (a runtime parameter); it must still be bound as a program argument.
    model = tt_onnx.make_model(
        [
            tt_onnx.make_node("Clip", ["x", "lo", "hi"], ["c"]),
            tt_onnx.make_node("Mul", ["c", "lo"], ["y"]),
        ],
        [tt_onnx.vi("x", [16, 16])],
        [tt_onnx.vi("y", [16, 16])],
        [
            tt_onnx.make_tensor("lo", np.float32(-0.5)),
            tt_onnx.make_tensor("hi", np.float32(0.5)),
        ],
    )
    tt_onnx.assert_tt_matches_cpu(model, {"x": tt_onnx.randn(16, 16)})


def test_params_consumed_mid_graph() -> None:
    # Initializers enter at nodes 2 and 4, not at the graph's front.
    model = tt_onnx.make_model(
        [
            tt_onnx.make_node("Relu", ["x"], ["r"]),
            tt_onnx.make_node("Gemm", ["r", "w"], ["g"], transB=1),
            tt_onnx.make_node("Relu", ["g"], ["h"]),
            tt_onnx.make_node("Mul", ["h", "scale"], ["y"]),
        ],
        [tt_onnx.vi("x", [32, 64])],
        [tt_onnx.vi("y", [32, 128])],
        [
            tt_onnx.make_tensor("w", tt_onnx.randn(128, 64) / 8),
            tt_onnx.make_tensor("scale", tt_onnx.randn(128)),
        ],
    )
    tt_onnx.assert_tt_matches_cpu(model, {"x": tt_onnx.randn(32, 64)})
