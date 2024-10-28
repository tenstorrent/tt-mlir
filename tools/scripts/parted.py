# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttmlir
from ttmlir.ir import *
from ttmlir.dialects import ttir, func, tt, tensor
import sys

uid = -1


def filter_dialect_ops(module, dialects=["ttir", "ttnn"]):
    for entry in module.body.operations:
        if isinstance(entry, func.FuncOp):
            for block in entry.body:
                for op in block.operations:
                    dialect = op.name.split(".")[0]
                    if dialect in dialects:
                        yield op


def get_line_number(location):
    global uid
    try:
        line = str(location).split(":")[2]
        if int(line) > 1000000:
            uid += 1
            return "unknown" + str(uid)
        else:
            return line
    except:
        uid += 1
        return "unknown" + str(uid)


def get_entry_name(op):
    original_entry_name = op.parent.attributes["sym_name"]
    line_number = get_line_number(op.location)
    entry_name = f"{original_entry_name}_{op.name}_{line_number}"
    entry_name = entry_name.replace(".", "_").replace('"', "")
    return entry_name


def get_op_operands(op):
    if "operandSegmentSizes" in op.attributes:
        segments = op.attributes["operandSegmentSizes"]
        assert len(segments) == 2
        ins, outs = segments
        assert ins + outs == len(op.operands)
        return (op.operands[:ins], op.operands[ins:])
    elif ttir.ir.is_dps(op):
        return (op.operands[:-1], op.operands[-1:])
    return (op.operands, [])


def emit_op_as_entry_point(op, ip=None, loc=None):
    results = op.results
    op_inputs, op_outputs = get_op_operands(op)
    input_types = [op_input.type for op_input in op_inputs]
    output_types = [op_output.type for op_output in op_outputs]
    result_types = [result.type for result in results]
    entry = func.FuncOp(get_entry_name(op), (input_types, result_types), ip=ip, loc=loc)
    entry_block = Block.create_at_start(entry.body, input_types)
    with InsertionPoint(entry_block) as ip, Location.unknown() as loc:
        operands = [arg for arg in entry_block.arguments]
        for output_type in output_types:
            operands.append(
                tensor.empty(
                    output_type.shape,
                    output_type.element_type,
                    encoding=output_type.encoding,
                    ip=ip,
                    loc=loc,
                )
            )
        attrs = {attr.name: attr.attr for attr in op.attributes}
        assert len(op.regions) == 0, "Regions are not supported yet."
        new_op = Operation.create(
            name=op.name,
            results=result_types,
            operands=operands,
            attributes=attrs,
            successors=op.successors,
            regions=len(op.regions),
            loc=loc,
            ip=ip,
        )
        new_op.verify()

        ret_op = Operation.create(
            name="func.return",
            results=[],
            operands=[new_op.result],
            loc=loc,
            ip=ip,
        )


def parted(in_module):
    cursor = Location.unknown(in_module.context)
    out_module = Module.create(cursor)
    with InsertionPoint(out_module.body) as ip, Location.unknown() as loc:
        for op in filter_dialect_ops(in_module):
            emit_op_as_entry_point(op, ip=ip, loc=loc)
    return out_module


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="parted: a tool for filtering out operations from a module for isolated op testing"
    )

    parser.add_argument("mlir", type=str, help="Path to the mlir file")
    args = parser.parse_args()

    with Context() as ctx, open(args.mlir, "r") as mlir_fd:
        ctx.allow_unregistered_dialects = True
        ttir.register_dialect(ctx)
        module = Module.parse(mlir_fd.read(), ctx)
        print(parted(module))
