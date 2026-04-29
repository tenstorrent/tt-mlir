# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


from ._ttnn_ops_gen import *
from ._ttnn_enum_gen import *
from .._mlir_libs._ttmlir import ttnn_ir as ir

# Stamp generated OpView classes with operand/attribute/result name tuples
# sourced from the tablegen JSON dump (see tools/python_op_schema_gen/gen.py).
# Lets users do e.g. `ttnn.ScatterOp.OPERAND_NAMES == ("input","index","source")`.
from . import _ttnn_ops_gen as _ttnn_ops_gen_module
from ._ttnn_op_schema import OP_SCHEMA as _OP_SCHEMA


def _stamp_ttnn_op_schema():
    import inspect

    for cls_name, cls in inspect.getmembers_static(
        _ttnn_ops_gen_module, inspect.isclass
    ):
        op_name = getattr(cls, "OPERATION_NAME", None)
        s = _OP_SCHEMA.get(op_name)
        if s is None:
            continue
        cls.OPERAND_NAMES = s["operands"]
        cls.ATTRIBUTE_NAMES = s["attributes"]
        cls.RESULT_NAMES = s["results"]


_stamp_ttnn_op_schema()
del _stamp_ttnn_op_schema, _ttnn_ops_gen_module
