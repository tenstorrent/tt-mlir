# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest


def test_debug_wrap_no_debug_reraises():
    from chisel.utils import debug_wrap

    @debug_wrap(debug=False)
    def failing():
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        failing()


def test_debug_wrap_passes_through():
    from chisel.utils import debug_wrap

    @debug_wrap(debug=False)
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
