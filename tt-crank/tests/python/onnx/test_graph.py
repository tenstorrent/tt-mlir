# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Graph-structure handling: shared initializers, parameters consumed mid-graph."""

import numpy as np

from ort_ep_utils import (
    assert_tt_matches_cpu,
    make_model,
    make_node,
    make_tensor,
    randn,
    vi,
)


def test_initializer_shared_between_metadata_and_operand() -> None:
    # The same constant is Clip's min bound (read at build time) and Mul's operand
    # (a runtime parameter); it must still be bound as a program argument.
    model = make_model(
        [
            make_node("Clip", ["x", "lo", "hi"], ["c"]),
            make_node("Mul", ["c", "lo"], ["y"]),
        ],
        [vi("x", [16, 16])],
        [vi("y", [16, 16])],
        [make_tensor("lo", np.float32(-0.5)), make_tensor("hi", np.float32(0.5))],
    )
    assert_tt_matches_cpu(model, {"x": randn(16, 16)})


def test_params_consumed_mid_graph() -> None:
    # Initializers enter at nodes 2 and 4, not at the graph's front.
    model = make_model(
        [
            make_node("Relu", ["x"], ["r"]),
            make_node("Gemm", ["r", "w"], ["g"], transB=1),
            make_node("Relu", ["g"], ["h"]),
            make_node("Mul", ["h", "scale"], ["y"]),
        ],
        [vi("x", [32, 64])],
        [vi("y", [32, 128])],
        [make_tensor("w", randn(128, 64) / 8), make_tensor("scale", randn(128))],
    )
    assert_tt_matches_cpu(model, {"x": randn(32, 64)})
