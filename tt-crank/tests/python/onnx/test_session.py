# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Session-level behavior: free-dim override, running a .onnx from disk."""

import tt_onnx


def test_free_dim_override() -> None:
    # A symbolic batch dim fails the static-shape check; pinning it lets the model run on TT.
    model = tt_onnx.make_model(
        [tt_onnx.make_node("Relu", ["x"], ["y"])],
        [tt_onnx.vi("x", ["N", 16])],
        [tt_onnx.vi("y", ["N", 16])],
    )
    tt_onnx.assert_tt_matches_cpu(
        model, {"x": tt_onnx.randn(4, 16)}, atol=1e-5, rtol=1e-5, free_dims={"N": 4}
    )


def test_run_from_disk(tmp_path) -> None:
    model = tt_onnx.make_model(
        [tt_onnx.make_node("Relu", ["x"], ["y"])],
        [tt_onnx.vi("x", [16, 16])],
        [tt_onnx.vi("y", [16, 16])],
    )
    path = tmp_path / "model.onnx"
    path.write_bytes(model)
    tt_onnx.assert_tt_matches_cpu(
        path, {"x": tt_onnx.randn(16, 16)}, atol=1e-5, rtol=1e-5
    )
