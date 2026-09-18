# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device lifecycle through tt_crank.onnx.register().

The exit test needs a subprocess with exclusive device access, and the pytest
parent holds the device once any test ran a session, so it is opt-in:

    TT_CRANK_ONNX_EXIT_TEST=1 pytest tests/python/onnx/test_lifecycle.py
"""

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

import tt_crank.onnx


def test_library_path_resolves() -> None:
    assert tt_crank.onnx.library_path().exists()


@pytest.mark.skipif(
    os.environ.get("TT_CRANK_ONNX_EXIT_TEST") != "1",
    reason="needs exclusive device access; opt in with TT_CRANK_ONNX_EXIT_TEST=1",
)
def test_exit_without_unregister_is_clean() -> None:
    # register() installs the atexit unregister; exiting with the library still
    # registered would otherwise close the device mesh from C-runtime exit handlers.
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(pathlib.Path(__file__).parent)!r})
        import numpy as np
        import tt_crank.onnx
        import tt_onnx

        tt_crank.onnx.register()
        model = tt_onnx.make_model(
            [tt_onnx.make_node("Add", ["a", "b"], ["y"])],
            [tt_onnx.vi("a", [32, 32]), tt_onnx.vi("b", [32, 32])],
            [tt_onnx.vi("y", [32, 32])],
        )
        inputs = {{"a": tt_onnx.randn(32, 32), "b": tt_onnx.randn(32, 32)}}
        (y,) = tt_onnx.session_on_tt(model).run(None, inputs)
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
    assert (
        result.returncode == 0
    ), f"exit code {result.returncode}\n{result.stderr[-2000:]}"
