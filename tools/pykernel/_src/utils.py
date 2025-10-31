# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import textwrap
import inspect
from typing import Callable
from ttmlir.dialects import arith
from ttmlir.ir import *


def _discover_dialect_ops(dialect, denylist=None):
    """
    Return a mapping Dict[str, Callable] of available pybounded dialect ops.
    """
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


def _cast(val, ty):
    if val.type == ty or (isinstance(ty, type) and isinstance(val.type, ty)):
        return val

    if ty is IndexType or isinstance(ty, IndexType):
        return arith.index_cast(IndexType.get(), val)
    elif isinstance(val.type, IndexType) and isinstance(ty, IntegerType):
        return arith.index_cast(ty, val)
    else:
        raise TypeError(f"Unhandled cast from {val.type} to {ty}")


def _asindex(val):
    if val is None:
        return val
    if isinstance(val, tuple):
        return tuple(map(_asindex, val))
    if isinstance(val, list):
        return list(map(_asindex, val))
    return _cast(val, IndexType)


def _get_type_str(ty):
    s = str(ty).split("<")[0]
    if not s.startswith("!"):
        s = "!" + s
    return s
