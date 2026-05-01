# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Generate a Python op-schema sidecar from llvm-tblgen --dump-json output.
Tblgen JSON reference used from the: https://llvm.org/docs/TableGen/BackEnds.html#json-reference.

Emits a module exposing OP_SCHEMA[<full-op-name>] = {
    "operands":   ("input", "index", "source"),
    "attributes": ("dim", "scatter_reduce_type", "memory_config"),
    "results":    ("result",),
}

Usage:
  gen.py --json TTNNOps.json --out _ttnn_op_schema.py
"""
from enum import Enum, auto
from typing import Literal, Optional
import argparse
import json
import sys

from pydantic import BaseModel, ConfigDict, Field


SUPPORTED_JSON_VERSION = 1

# Hardcoded for TTNN. DIALECT_NAME is the short dialect name used in
# OPERATION_NAME (e.g. "ttnn.add"); DIALECT_RECORD is the tblgen def-record
# name found in each Op's `opDialect` field.
DIALECT_NAME = "ttnn"
DIALECT_RECORD = "TTNN_Dialect"


class Kind(Enum):
    OPERAND = auto()
    ATTRIBUTE = auto()


class DefRef(BaseModel):
    """A reference to a def object, per the LLVM JSON reference:
    {"kind": "def", "def": <record-name>, "printable": ...}.

    Only `kind` and the target name are typed; `printable` and any future fields land in model_extra.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    kind: Literal["def"]
    def_name: str = Field(alias="def")


class OpDag(BaseModel):
    """ODS-emitted dag for an Op's `arguments` / `results`: each arg is a
    (def-ref, name) pair. See the LLVM JSON reference for `dag` shape."""

    model_config = ConfigDict(extra="allow")

    kind: Literal["dag"]
    operator: DefRef
    args: list[tuple[DefRef, str]]


class OpRecord(BaseModel):
    """A def record we've already accepted via is_dialect_op."""

    model_config = ConfigDict(extra="allow")

    op_name: str = Field(alias="opName")
    op_dialect: DefRef = Field(alias="opDialect")
    arguments: OpDag
    results: OpDag


class ArgWrapper(BaseModel):
    """ODS Arg<...> wrapper carrying a description / decorators around an inner Constraint."""

    model_config = ConfigDict(extra="allow")

    constraint: DefRef


def _expect_json_v1(records: dict) -> None:
    version = records.get("!tablegen_json_version")
    assert version == SUPPORTED_JSON_VERSION, (
        f"unsupported tblgen JSON version {version!r}; this script targets "
        f"version {SUPPORTED_JSON_VERSION}"
    )


def _get_def_name(arg_def: Optional[dict]) -> Optional[str]:
    """Return the target record name of a def-ref dict, else None."""
    if not isinstance(arg_def, dict):
        return None
    if arg_def.get("kind") != "def":
        return None
    name = arg_def.get("def")
    return name if isinstance(name, str) else None


def _unwrap_arg(name: str, records: dict) -> str:
    """If `name` refers to an Arg<...> wrapper record, return the inner
    constraint's name; otherwise return `name` unchanged.

    Current TTNNOps.td does not nest Arg wrappers, so a single hop is enough.
    """
    rec = records.get(name)
    if not isinstance(rec, dict) or "Arg" not in rec.get("!superclasses", []):
        return name
    return ArgWrapper.model_validate(rec).constraint.def_name


def _mangle_name(name: str) -> str:
    """Mirror the ODS Python op generator's local-collision mangling.

    The generated `__init__` uses a local list literally named `results` to pass into `OpView.__init__`.
    When an ODS-declared operand/attribute/result name shadows that local,
    the generator suffixes it with `_` (so the accessor on the class becomes e.g. `results_`, not `results`).
    The schema has to match the actual class member name — `dir(cls)` exposes the
    suffixed form — so we apply the same rule here.
    """
    if name == "results":
        return "results_"
    return name


def classify_arg(name: str, type_set: set, attr_set: set) -> Optional[Kind]:
    """Classify a constraint name. Caller must strip Arg<...> wrappers via
    _unwrap_arg first."""
    if name in type_set:
        return Kind.OPERAND
    if name in attr_set:
        return Kind.ATTRIBUTE
    return None


def is_dialect_op(rec) -> bool:
    """Return True if `rec` is an MLIR op record owned by DIALECT_RECORD."""
    if not isinstance(rec, dict):
        return False
    if "Op" not in rec.get("!superclasses", []):
        return False
    return _get_def_name(rec.get("opDialect")) == DIALECT_RECORD


def collect(records: dict):
    _expect_json_v1(records)

    inst = records.get("!instanceof", {})
    assert isinstance(
        inst, dict
    ), f"!instanceof: expected a dictionary, got {type(inst).__name__}"
    type_set = set(inst.get("TypeConstraint", []))
    attr_set = set(inst.get("AttrConstraint", []))
    overlap = type_set & attr_set
    assert not overlap, (
        f"records appear in both TypeConstraint and AttrConstraint: "
        f"{sorted(overlap)}"
    )

    schema = {}
    for rec in records.values():
        if not is_dialect_op(rec):
            continue

        op = OpRecord.model_validate(rec)
        full = f"{DIALECT_NAME}.{op.op_name}"
        operands, attributes = [], []
        buckets = {Kind.OPERAND: operands, Kind.ATTRIBUTE: attributes}
        for arg_ref, arg_name in op.arguments.args:
            target = _unwrap_arg(arg_ref.def_name, records)
            kind = classify_arg(target, type_set, attr_set)
            if kind is None:
                print(
                    f"warning: {full!r} arg {arg_name!r} (def "
                    f"{target!r}) is neither operand nor attribute; "
                    "skipping",
                    file=sys.stderr,
                )
                continue
            buckets[kind].append(arg_name)

        results = [_mangle_name(r[1]) for r in op.results.args]

        schema[full] = {
            "operands": tuple(operands),
            "attributes": tuple(attributes),
            "results": tuple(results),
        }
    return schema


def emit(schema: dict, out_path: str) -> None:
    lines = [
        "# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC",
        "#",
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# Auto-generated from llvm-tblgen --dump-json. Do not edit.",
        "",
        "OP_SCHEMA = {",
    ]
    for op_name in sorted(schema):
        entry = schema[op_name]
        lines.append(f"    {op_name!r}: {{")
        lines.append(f"        'operands':   {entry['operands']!r},")
        lines.append(f"        'attributes': {entry['attributes']!r},")
        lines.append(f"        'results':    {entry['results']!r},")
        lines.append("    },")
    lines.append("}")
    lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.json) as f:
        records = json.load(f)

    schema = collect(records)
    if not schema:
        print(f"warning: no ops found for dialect '{DIALECT_NAME}'", file=sys.stderr)
    emit(schema, args.out)


if __name__ == "__main__":
    main()
