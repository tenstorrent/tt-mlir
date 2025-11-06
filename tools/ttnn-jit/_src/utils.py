# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import textwrap
import inspect
import importlib
from typing import Callable
from ttmlir.ir import *


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


def _get_collapsed_linear_affine_map(
    context, shape, grid_shape, collapse_intervals=[(0, -1)]
):
    """
    This function creates an affine map that represents collapsing the tensor
    dims onto an n-dimensional grid. E.g. (Where <> is some join operator)

      - 3D tensor onto a 2D grid:
        (d0, d1, d2) -> (d0 <> d1, d2)

      - 4D tensor onto a 2D grid:
        (d0, d1, d2, d3) -> (d0 <> d1 <> d2, d3)

    Note there are many ways we could collapse the above dims, by default we
    just collapse the interval [0, -1), which collapses dim0 up to but not
    including the last dim.  By using collapseIntervals we can achieve flexible
    collapsing of any set of consecutive dimension ranges.

      - 4D tensor onto a 3D grid collapseIntervals=[(1, -1)]:
        (d0, d1, d2, d3) -> (d0, d1 <> d2, d3)

      - 4D tensor onto a 3D grid collapseIntervals=[(0, 2)]:
        (d0, d1, d2, d3) -> (d0 <> d1, d2, d3)

      - 7D tensor onto a 4D grid collapseIntervals=[(0, 3), (-3, -1)]:
        (d0, d1, d2, d3, d4, d5, d6) -> (d0 <> d1 <> d2, d3, d4 <> d5, d6)
    """

    rank = len(shape)

    # Start with a full identity mapping
    results = [AffineDimExpr.get(i, context) for i in range(rank)]

    for interval in collapse_intervals:
        begin, end = interval
        # Handle negative indices
        if begin < 0:
            begin += rank
        if end < 0:
            end += rank
        if begin >= end:
            continue

        # Build collapsed expression
        collapsed_expr = AffineConstantExpr.get(0, context)
        multiplier = 1
        for d_idx in range(end - 1, begin - 1, -1):

            dim_expr = AffineDimExpr.get(d_idx, context)
            term = dim_expr * multiplier
            collapsed_expr = term + collapsed_expr
            multiplier *= shape[d_idx]

        results = results[:begin] + [collapsed_expr] + results[end:]

    # Truncate results to match the rank of the grid shape
    if len(results) > len(grid_shape):
        results = results[: len(grid_shape)]

    # Pad with leading zeros if the number of results is less than the grid rank
    while len(results) < len(grid_shape):
        results.insert(0, AffineConstantExpr.get(0, context))

    # Simplify affine map by simplifying each expr in results
    for i, expr in enumerate(results):

        expr = AffineExpr.simplify_affine_expr(expr, rank, 0)
        results[i] = expr

    return AffineMap.get(rank, 0, results, context)
