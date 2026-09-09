# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from op_by_op_infra.workflow import _ensure_system_desc


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_system_desc():
    """Generate the system descriptor in-process before any op-by-op test runs."""
    _ensure_system_desc()
