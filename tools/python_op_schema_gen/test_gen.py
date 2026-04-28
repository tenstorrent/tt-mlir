# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cross-check the generated TTNN op schema (OPERAND_NAMES / ATTRIBUTE_NAMES /
RESULT_NAMES, stamped onto each OpView by ttmlir.dialects.ttnn) against the
real Python OpView surface — `dir(cls)` — exposed by mlir-tblgen.

For each TTNN op:
  * every OPERAND_NAMES / ATTRIBUTE_NAMES / RESULT_NAMES entry must be a
    member of the OpView class (asserted);
  * any remaining names in `dir(cls)` that aren't on the base `mlir.ir.OpView`
    and aren't the four schema-stamped sentinels (OPERATION_NAME plus the
    three *_NAMES tuples) are reported back as the "leftovers" — names the
    OpView exposes that the schema doesn't classify.
"""
import inspect

from ttmlir.ir import OpView

from ttmlir.dialects import ttnn  # noqa: F401 — import side-effect stamps the schema
from ttmlir.dialects import _ttnn_ops_gen


OPVIEW_BASELINE = set(dir(OpView))
SCHEMA_SENTINELS = {
    "OPERATION_NAME",
    "OPERAND_NAMES",
    "ATTRIBUTE_NAMES",
    "RESULT_NAMES",
}


def _ttnn_opview_classes():
    for _, cls in inspect.getmembers_static(_ttnn_ops_gen, inspect.isclass):
        if not issubclass(cls, OpView):
            continue
        if getattr(cls, "OPERATION_NAME", None) is None:
            continue
        yield cls


def test_schema_names_are_class_members(capsys):
    """Every schema name (operand/attribute/result) should resolve as a member
    of its OpView class. Anything left in `dir(cls)` that the schema doesn't
    cover gets printed for inspection."""
    failures = []
    leftovers_per_op = {}

    for cls in _ttnn_opview_classes():
        op_name = cls.OPERATION_NAME
        operands = set(getattr(cls, "OPERAND_NAMES", ()))
        attributes = set(getattr(cls, "ATTRIBUTE_NAMES", ()))
        results = set(getattr(cls, "RESULT_NAMES", ()))

        members = set(dir(cls))

        for kind, names in (
            ("operand", operands),
            ("attribute", attributes),
            ("result", results),
        ):
            missing = names - members
            if missing:
                failures.append(
                    f"{op_name}: {kind} names not in dir(cls): {sorted(missing)}"
                )

        leftover = (
            members
            - operands
            - attributes
            - results
            - OPVIEW_BASELINE
            - SCHEMA_SENTINELS
        )
        leftover = {n for n in leftover if not n.startswith("_")}
        if leftover:
            leftovers_per_op[op_name] = sorted(leftover)

    with capsys.disabled():
        print(f"\nChecked {sum(1 for _ in _ttnn_opview_classes())} TTNN ops.")
        if leftovers_per_op:
            print(
                f"\n{len(leftovers_per_op)} op(s) have dir() members not covered "
                "by OPERAND_NAMES / ATTRIBUTE_NAMES / RESULT_NAMES:"
            )
            for op_name, names in sorted(leftovers_per_op.items()):
                print(f"  {op_name}: {names}")
        else:
            print("\nNo unclassified dir() members across all TTNN ops.")

    assert not failures, "\n".join(failures)
