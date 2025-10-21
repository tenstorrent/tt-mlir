# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import textwrap
import inspect
import importlib
from typing import Callable


def _discover_dialect_ops(dialect, denylist=None):
    """
    Return a mapping Dict[str, Callable] of available pybounded dialect ops.
    """
    # Convert string dialect names to their corresponding objects
    if isinstance(dialect, str):
        dialect = importlib.import_module(f"ttmlir.dialects.{dialect}")

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

    # Find the line that starts the function definition and keep from there
    lines = source_code.splitlines()
    def_line_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def "):
            def_line_idx = i
            break

    if def_line_idx is None:
        # Fallback to old behavior if we can't find def line
        cleaned = [line for line in lines if not line.strip().startswith("@")]
        source_code = "\n".join(cleaned)
    else:
        # Keep only from the def line onwards (removes all decorator lines)
        source_code = "\n".join(lines[def_line_idx:])

    return source_code


def _get_num_pos_args(func: Callable):
    sig = inspect.signature(func)
    num_pos_args = len(
        [
            p
            for p in sig.parameters.values()
            if p.default == inspect.Parameter.empty
            and p.kind != inspect.Parameter.VAR_KEYWORD
        ]
    )
    return num_pos_args
