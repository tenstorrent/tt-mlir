# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cross-check the generated TTNN op schema (OPERAND_NAMES / ATTRIBUTE_NAMES /
RESULT_NAMES / REGION_NAMES, stamped onto each OpView by ttmlir.dialects.ttnn)
against the real Python OpView surface - `dir(cls)` - exposed by mlir-tblgen.

For each TTNN op:
  * every OPERAND_NAMES / ATTRIBUTE_NAMES / RESULT_NAMES / REGION_NAMES entry
    must be a member of the OpView class;
  * `dir(cls)` must contain no public names beyond the schema entries, the
    base `mlir.ir.OpView` surface, and the schema-stamped sentinels
    (OPERATION_NAME plus the four *_NAMES tuples) - any leftover is a name
    the OpView exposes that the schema fails to classify.
"""
import inspect

import pytest

from ttmlir.ir import OpView
from ttmlir.dialects import ttnn


OPVIEW_BASELINE = set(dir(OpView))
SCHEMA_SENTINELS = {
    "OPERATION_NAME",
    "OPERAND_NAMES",
    "ATTRIBUTE_NAMES",
    "RESULT_NAMES",
    "REGION_NAMES",
}


TTNN_OPVIEW_CLASSES = [
    cls
    for _, cls in inspect.getmembers_static(ttnn, inspect.isclass)
    if issubclass(cls, OpView) and getattr(cls, "OPERATION_NAME", None) is not None
]


@pytest.mark.parametrize(
    "cls",
    TTNN_OPVIEW_CLASSES,
    ids=[cls.OPERATION_NAME for cls in TTNN_OPVIEW_CLASSES],
)
def test_schema_names_are_class_members(cls):
    """Every schema name (operand/attribute/result/region) should resolve as a
    member of its OpView class, and `dir(cls)` should contain no names the
    schema doesn't classify."""
    operands = set(getattr(cls, "OPERAND_NAMES", tuple()))
    attributes = set(getattr(cls, "ATTRIBUTE_NAMES", tuple()))
    results = set(getattr(cls, "RESULT_NAMES", tuple()))
    regions = set(getattr(cls, "REGION_NAMES", tuple()))

    members = set(dir(cls))

    for kind, names in (
        ("operand", operands),
        ("attribute", attributes),
        ("result", results),
        ("region", regions),
    ):
        missing = names - members
        assert not missing, f"{kind} names not in dir(cls): {sorted(missing)}"

    leftover = members.difference(
        operands, attributes, results, regions, OPVIEW_BASELINE, SCHEMA_SENTINELS
    )
    leftover = {n for n in leftover if not n.startswith("_")}
    assert not leftover, f"unclassified dir() members: {sorted(leftover)}"
