# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the ONNX Runtime plugin-EP tests.

No __init__.py here on purpose: a regular package named after a real library
rooted under tests/python would shadow it (see tests/python's sibling dirs).
"""

import pytest

import tt_crank.onnx

import tt_onnx


@pytest.fixture(scope="session", autouse=True)
def tt_ep_library() -> None:
    tt_crank.onnx.register()  # unregisters at interpreter exit


@pytest.fixture(autouse=True)
def _fixed_seed() -> None:
    tt_onnx.reseed()
