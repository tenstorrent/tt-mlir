# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


from ._ttnn_ops_gen import *
from ._ttnn_enum_gen import *
from .._mlir_libs._ttmlir import ttnn_ir as ir

# Stamp generated OpView classes with operand/attribute/result name tuples
# sourced from the tablegen JSON dump (see tools/python_op_schema_gen/gen.py).
# Lets users do e.g. `ttnn.ScatterOp.OPERAND_NAMES == ("input","index","source")`.
from . import _ttnn_ops_gen
from ._ttnn_op_schema import OP_SCHEMA as _OP_SCHEMA


def _stamp_ttnn_op_schema():
    import inspect

    for _, cls in inspect.getmembers_static(_ttnn_ops_gen, inspect.isclass):
        entry = _OP_SCHEMA.get(getattr(cls, "OPERATION_NAME", None))
        if entry is None:
            continue
        cls.OPERAND_NAMES = entry["operands"]
        cls.ATTRIBUTE_NAMES = entry["attributes"]
        cls.RESULT_NAMES = entry["results"]


_stamp_ttnn_op_schema()
