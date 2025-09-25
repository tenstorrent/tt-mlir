# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
from typing import Callable, Optional, Dict, Iterable


def _get_ttnn_op(func: Callable) -> Optional[Callable]:
    # Return ttnn.<func.__name__> if it exists and is callable
    try:
        attr = getattr(ttnn, func.__name__)
    except AttributeError:
        return None
    return attr if callable(attr) else None


def _build_golden_map(ops: Iterable[Callable]) -> Dict[Callable, Callable]:
    # Build a func ->ttnn op map for provided ops
    result: Dict[Callable, Callable] = {}
    for op in ops:
        ttnn_op = _get_ttnn_op(op)
        if ttnn_op is not None:
            result[op] = ttnn_op
    return result
