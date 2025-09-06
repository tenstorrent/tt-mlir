# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re
import inspect
from ttmlir.dialects import ttkernel, ttir, arith
from ttmlir.ir import *


def _camel_to_snake(name):
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


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


def _cast(val, ty):
    if val.type == ty or (isinstance(ty, type) and isinstance(val.type, ty)):
        return val

    if ty is IndexType or isinstance(ty, IndexType):
        return arith.index_cast(IndexType.get(), val)
    else:
        raise TypeError(f"Unhandled cast from {val.type} to {ty}")
