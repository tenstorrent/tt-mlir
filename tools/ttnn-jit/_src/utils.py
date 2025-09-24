# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import textwrap
import inspect
from typing import Callable


def _discover_dialect_ops(dialect, denylist=None):
    """
    Return a mapping Dict[str, Callable] of available pybounded dialect ops.
    """
    # TODO (#5044): match ttnn.* ops instead of just python ops.
    denylist = set() if denylist is None else denylist
    op_map = {}
    ns = dialect.__name__.removeprefix("ttmlir.dialects.")
    for attr_name in dir(dialect):
        if attr_name.startswith("_"):
            continue
        op_obj = getattr(dialect, attr_name, None)
        if (
            op_obj is None
            or not hasattr(op_obj, "OPERATION_NAME")
            or not inspect.isclass(op_obj)
        ):
            continue

        func_name = getattr(op_obj, "OPERATION_NAME")
        name = func_name.removeprefix(ns + ".")
        if name in denylist:
            continue
        func = getattr(dialect, name, None)

        # must be the module-level function, and not the class
        if inspect.isfunction(func):
            op_map[name] = func

    return op_map


def _cleanup_source_code(f: Callable):
    source_code = inspect.getsource(f)
    source_code = textwrap.dedent(source_code)
    cleaned = [
        line for line in source_code.splitlines() if not line.strip().startswith("@")
    ]
    source_code = "\n".join(cleaned)
    return source_code
