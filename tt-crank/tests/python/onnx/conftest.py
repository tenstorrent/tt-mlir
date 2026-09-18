# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the ONNX Runtime plugin-EP tests.

Registration is process-global, so it happens once per session.

No __init__.py here on purpose: a regular package named after a real library
rooted under tests/python would shadow it (see tests/python's sibling dirs).
"""

import onnxruntime as ort
import pytest

import tt_onnx


@pytest.fixture(scope="session", autouse=True)
def tt_ep_library() -> None:
    path = tt_onnx.ep_library_path()
    if not path.exists():
        pytest.fail(f"ONNX EP library not built: {path}")
    ort.register_execution_provider_library(tt_onnx.REGISTRATION_NAME, str(path))
    yield
    # Sessions are per-test locals, gone by now.
    ort.unregister_execution_provider_library(tt_onnx.REGISTRATION_NAME)


@pytest.fixture(autouse=True)
def _fixed_seed() -> None:
    tt_onnx.reseed()
