# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Proxy conftest that delegates to test/python/golden/conftest.py so the sweep
# tests can be run from anywhere without needing --rootdir gymnastics.

import importlib.util
import sys
from pathlib import Path

_golden_dir = (
    Path(__file__).resolve().parent.parent.parent / "test" / "python" / "golden"
)
sys.path.insert(0, str(_golden_dir))

_spec = importlib.util.spec_from_file_location(
    "_golden_conftest", _golden_dir / "conftest.py"
)
_gc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gc)

# Re-export all pytest hooks and fixtures.
pytest_addoption = _gc.pytest_addoption
pytest_collection_modifyitems = _gc.pytest_collection_modifyitems
pytest_runtest_setup = _gc.pytest_runtest_setup
pytest_runtest_call = _gc.pytest_runtest_call
pytest_sessionfinish = _gc.pytest_sessionfinish
device = _gc.device
system_desc = _gc.system_desc
get_request_kwargs = _gc.get_request_kwargs
