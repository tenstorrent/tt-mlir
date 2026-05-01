# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared D2M shape suites and parametrization helpers.

Use the shape suites to give op families a mix of small, tile-aligned,
non-tile-aligned, batched, and larger tensors. Use ``rotated_params`` when
an op family has multiple dimensions of coverage and a full Cartesian product
would be too expensive.
"""

import pytest
import torch

_DTYPE_IDS = {
    torch.float32: "f32",
    torch.bfloat16: "bf16",
    torch.float16: "f16",
    torch.int64: "i64",
    torch.int32: "i32",
    torch.uint32: "u32",
    torch.uint16: "u16",
    torch.uint8: "u8",
    torch.bool: "i1",
}


def _is_param(value):
    return hasattr(value, "values") and hasattr(value, "marks") and hasattr(value, "id")


def _id_part(value):
    try:
        if value in _DTYPE_IDS:
            return _DTYPE_IDS[value]
    except TypeError:
        pass
    if callable(value):
        return getattr(value, "__name__", value.__class__.__name__)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float, str)):
        return str(value).replace(".", "p")
    if isinstance(value, tuple):
        if all(isinstance(dim, int) for dim in value):
            return "x".join(map(str, value))
        return "_".join(_id_part(element) for element in value)
    if isinstance(value, list):
        return "__".join(_id_part(element) for element in value)
    return str(value).replace(" ", "")


def _case_values(value):
    if _is_param(value):
        return tuple(value.values), list(value.marks), value.id
    return (value,), [], None


def rotated_params(primary_axis, *cycled_axes, value_order=None):
    """Build pytest params by cycling secondary axes over the primary axis.

    The first argument controls the number of generated cases. Each additional
    axis contributes ``axis[i % len(axis)]`` to case ``i``. ``value_order`` can
    reorder the emitted values without changing which axis drives case count.
    Marks and ids from existing ``pytest.param`` values are preserved, which
    lets tests keep backend skip/xfail annotations while avoiding a full
    Cartesian product.
    """

    params = []
    for index, primary_value in enumerate(primary_axis):
        values, marks, case_id = _case_values(primary_value)
        values = list(values)
        marks = list(marks)
        id_parts = [case_id] + [None] * (len(values) - 1) if case_id else None
        id_parts = id_parts or [_id_part(value) for value in values]

        for axis in cycled_axes:
            axis_value = axis[index % len(axis)]
            axis_values, axis_marks, axis_id = _case_values(axis_value)
            values.extend(axis_values)
            marks.extend(axis_marks)
            if axis_id:
                id_parts.extend([axis_id] + [None] * (len(axis_values) - 1))
            else:
                id_parts.extend(_id_part(value) for value in axis_values)

        if value_order is not None:
            values = [values[value_index] for value_index in value_order]
            id_parts = [id_parts[value_index] for value_index in value_order]

        params.append(
            pytest.param(
                *values,
                marks=marks,
                id="-".join(id_part for id_part in id_parts if id_part is not None),
            )
        )
    return params


ELEMENTWISE_SHAPES = [
    (32, 32),
    (64, 128),
    (1, 128),
    (33, 65),
    (2, 64, 96),
    (1, 2, 33, 65),
    (512, 512),
]


ELEMENTWISE_2D_SHAPES = [
    (32, 32),
    (64, 128),
    (1, 128),
    (33, 65),
    (128, 128),
    (512, 512),
]


TILE_ALIGNED_SHAPES = [
    (32, 32),
    (64, 128),
    (128, 128),
    (2, 64, 96),
    (1, 2, 64, 128),
]


SMALL_SHAPES = [
    (1, 1),
    (1, 32),
    (32, 1),
    (128,),
]


LARGE_SHAPES = [
    (512, 512),
    (1024, 1024),
    (1, 4, 64, 128),
]

# Shapes that are not multiples of the 32x32 tile. Exercised across a range
# of ranks (2D through 5D) and sizes.
UNALIGNED_SHAPES = [
    (5, 3),
    (32, 1),
    (31, 7),
    (1, 32),
    (13, 29),
    (64, 1),
    (61, 3),
    (61, 37),
    (1, 64),
    (5, 67),
    (43, 67),
    (2, 3, 5),
    (3, 17, 37),
    (9, 43, 7),
    (5, 61, 49),
    (51, 19, 23),
    (2, 3, 5, 7),
    (3, 37, 5, 53),
    (37, 3, 5, 53),
    (41, 7, 43, 11),
    (7, 41, 43, 11),
    (1, 23, 1, 1),
    (23, 1, 1, 1),
    (3, 5, 7, 11, 13),
]
