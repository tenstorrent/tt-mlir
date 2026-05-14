# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ast as _ast
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
    """Read the source of `f`, dedent it, and blank-out any decorator lines
    so the AST parser sees a bare `def`/`async def` while line numbers stay
    aligned with the original file.

    Returns
    -------
    source : str
        The dedented, decorator-blanked source. `ast.parse(source).body[0]`
        is the function definition; its `.lineno` (and the lineno of every
        child node) is 1-indexed within `source`.
    firstlineno : int
        The 1-indexed line number in `source_file` of the first line of
        `source` (i.e. the first decorator or the `def` itself if there are
        no decorators). The absolute file line of an AST node is
        `firstlineno + node.lineno - 1`.
    source_file : str | None
        Absolute path to the file containing `f`, or None if not available.
    source_lines : list[str]
        `source.splitlines()`. 1-indexed access via `source_lines[lineno-1]`.
    """
    raw_lines, firstlineno = inspect.getsourcelines(f)
    source = textwrap.dedent("".join(raw_lines))

    # Blank out the decorator block (preserving line count) so the parsed
    # AST is a bare FunctionDef without unresolved decorator names.
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        # Fall back to the legacy "strip @ lines" path; we'll lose line
        # alignment but at least the parser succeeds.
        cleaned = [
            ln if not ln.lstrip().startswith("@") else ""
            for ln in source.splitlines(keepends=True)
        ]
        source = "".join(cleaned)
    else:
        funcdef = next(
            (
                n
                for n in tree.body
                if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))
            ),
            None,
        )
        if funcdef is not None and funcdef.decorator_list:
            lines = source.splitlines(keepends=True)
            # funcdef.lineno is the 1-indexed line of `def`; blank everything
            # before it (i.e. the decorators).
            for i in range(funcdef.lineno - 1):
                # Keep the trailing newline so line numbers stay aligned.
                lines[i] = "\n" if lines[i].endswith("\n") else ""
            source = "".join(lines)

    source_file = inspect.getsourcefile(f)
    return source, firstlineno, source_file, source.splitlines()


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
