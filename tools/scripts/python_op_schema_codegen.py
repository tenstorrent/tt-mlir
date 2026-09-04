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
    "regions":    (),
}

Usage:
  python_op_schema_codegen.py --json TTNNOps.json --out _ttnn_op_schema.py
"""
import argparse
import json
import keyword
import re
import sys
from enum import Enum, auto
from typing import Literal, Optional

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
    """ODS-emitted dag for an Op's `arguments` / `results` / `regions`: each arg
    is a (def-ref, name) pair. See the LLVM JSON reference for `dag` shape."""

    model_config = ConfigDict(extra="allow")

    kind: Literal["dag"]
    operator: DefRef
    args: list[tuple[DefRef, str]]


class OpRecord(BaseModel):
    """A def record we've already accepted via _is_dialect_op."""

    model_config = ConfigDict(extra="allow")

    op_name: str = Field(alias="opName")
    op_dialect: DefRef = Field(alias="opDialect")
    arguments: OpDag
    results: OpDag
    regions: OpDag


class ArgWrapper(BaseModel):
    """ODS Arg<...> wrapper carrying a description / decorators around an inner Constraint."""

    model_config = ConfigDict(extra="allow")

    constraint: DefRef


def _expect_json_v1(records: dict) -> None:
    version = records.get("!tablegen_json_version")
    if version != SUPPORTED_JSON_VERSION:
        raise ValueError(
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


# Names that mlir-tblgen refuses to emit as-is, because the accessor would
# shadow something the generated OpView or its `__init__` already uses. Mirrors
# `isODSReserved` in llvm-project's mlir/tools/mlir-tblgen/OpPythonBindingGen.cpp.
_ODS_RESERVED = frozenset(
    {
        "attributes",
        "create",
        "context",
        "ip",
        "operands",
        "print",
        "get_asm",
        "loc",
        "verify",
        "regions",
        "results",
        "self",
        "operation",
        "DIALECT_NAMESPACE",
        "OPERATION_NAME",
    }
)

# Builtins `isPythonReserved` refuses on top of the language keywords; see
# mlir/tools/mlir-tblgen/OpGenHelpers.cpp.
_PYTHON_RESERVED_BUILTINS = frozenset({"callable", "issubclass", "type"})


def _mangle_name(name: str) -> str:
    """Mirror `sanitizeName` from mlir-tblgen's Python op-binding generator.

    The generated accessor is not always the ODS-declared name: non-alphanumeric
    characters become underscores, a leading digit gets an underscore prefix, and
    a name that would shadow a Python keyword or part of the OpView surface gets
    a `_` suffix (so ODS `$operands` surfaces as `operands_`). The schema has to
    match the accessor, since that is what `dir(cls)` exposes.
    """
    name = re.sub(r"[^0-9A-Za-z]", "_", name)
    if name[:1].isdigit():
        return "_" + name
    if (
        keyword.iskeyword(name)
        or name in _PYTHON_RESERVED_BUILTINS
        or name in _ODS_RESERVED
        or name.startswith("_ods_")
        or name.endswith("_ods")
    ):
        return name + "_"
    return name


def _classify_arg(name: str, type_set: set, attr_set: set) -> Optional[Kind]:
    """Classify a constraint name. Caller must strip Arg<...> wrappers via
    _unwrap_arg first."""
    if name in type_set:
        return Kind.OPERAND
    if name in attr_set:
        return Kind.ATTRIBUTE
    return None


def _is_dialect_op(rec) -> bool:
    """Return True if `rec` is an MLIR op record owned by DIALECT_RECORD."""
    if not isinstance(rec, dict):
        return False
    if "Op" not in rec.get("!superclasses", []):
        return False
    return _get_def_name(rec.get("opDialect")) == DIALECT_RECORD


def _collect(records: dict):
    _expect_json_v1(records)

    inst = records.get("!instanceof", {})
    if not isinstance(inst, dict):
        raise TypeError(
            f"!instanceof: expected a dictionary, got {type(inst).__name__}"
        )
    type_set = set(inst.get("TypeConstraint", []))
    attr_set = set(inst.get("AttrConstraint", []))
    overlap = type_set & attr_set
    if overlap:
        raise ValueError(
            f"records appear in both TypeConstraint and AttrConstraint: "
            f"{sorted(overlap)}"
        )

    schema = {}
    for rec in records.values():
        if not _is_dialect_op(rec):
            continue

        op = OpRecord.model_validate(rec)
        full = f"{DIALECT_NAME}.{op.op_name}"
        operands, attributes = [], []
        buckets = {Kind.OPERAND: operands, Kind.ATTRIBUTE: attributes}
        for arg_ref, arg_name in op.arguments.args:
            target = _unwrap_arg(arg_ref.def_name, records)
            kind = _classify_arg(target, type_set, attr_set)
            if kind is None:
                raise ValueError(
                    f"{full!r} arg {arg_name!r} (def {target!r}) is neither "
                    f"operand nor attribute; tblgen JSON does not list it under "
                    f"TypeConstraint or AttrConstraint"
                )
            buckets[kind].append(_mangle_name(arg_name))

        schema[full] = {
            "operands": tuple(operands),
            "attributes": tuple(attributes),
            "results": tuple(_mangle_name(name) for _, name in op.results.args),
            "regions": tuple(_mangle_name(name) for _, name in op.regions.args),
        }
    return schema


def _emit(schema: dict, out_path: str) -> None:
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
        lines.append(f"        'regions':    {entry['regions']!r},")
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

    schema = _collect(records)
    if not schema:
        print(f"warning: no ops found for dialect '{DIALECT_NAME}'", file=sys.stderr)
    _emit(schema, args.out)


if __name__ == "__main__":
    main()
