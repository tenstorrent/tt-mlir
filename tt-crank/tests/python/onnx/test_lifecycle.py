# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device lifecycle through tt_crank.onnx.register().

The exit test needs a subprocess with exclusive device access, but the pytest
parent takes the device as soon as the root conftest queries it, so the test is
opt-in and must run without conftests:

    TT_CRANK_ONNX_EXIT_TEST=1 pytest --noconftest tests/python/onnx/test_lifecycle.py
"""

import os
import subprocess
import sys
import textwrap

import pytest

import tt_crank.onnx as tt_onnx


def test_library_path_resolves() -> None:
    assert tt_onnx.library_path().exists()


@pytest.mark.skipif(
    os.environ.get("TT_CRANK_ONNX_EXIT_TEST") != "1",
    reason="needs exclusive device access; see the module docstring to opt in",
)
def test_exit_without_unregister_is_clean() -> None:
    # register() installs the atexit unregister; exiting with the library still
    # registered would otherwise close the device mesh from C-runtime exit handlers.
    script = textwrap.dedent(
        """
        import numpy as np
        import tt_crank.onnx as tt_onnx

        tt_onnx.register()
        model = tt_onnx.model(
            [tt_onnx.node("Add", ["a", "b"], ["y"])],
            [tt_onnx.input("a", [32, 32]), tt_onnx.input("b", [32, 32])],
            [tt_onnx.output("y", [32, 32])],
        )
        inputs = {"a": tt_onnx.randn(32, 32), "b": tt_onnx.randn(32, 32)}
        (y,) = tt_onnx.session(model).run(None, inputs)
        np.testing.assert_allclose(y, inputs["a"] + inputs["b"], atol=1e-5)
        print("RAN_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert "RAN_OK" in result.stdout, result.stderr
    detail = f"exit code {result.returncode}\n{result.stderr[-2000:]}"
    assert result.returncode == 0, detail
