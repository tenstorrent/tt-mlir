# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from mlir.ir import Context, Module
import mlir.dialects.stablehlo as stablehlo


def parse_module_from_str(module_str):
    module = None
    with Context() as ctx:
        stablehlo.register_dialect(ctx)
        module = Module.parse(module_str)
    return module


def wrap_in_module_str(op) -> str:
    inputs = {operand.get_name(): str(operand.type) for operand in op.operands}
    args_str = ", ".join(f"{key}: {typ}" for key, typ in inputs.items())

    # Handle multiple results in the operation
    result_names = [str(result.get_name()) for result in op.results]
    result_types = [str(result.type) for result in op.results]

    # Construct the function signature based on the number of results
    if len(result_names) == 1:
        result_str = f"{result_types[0]}"
        return_stmt = f"return {result_names[0]} : {result_types[0]}"
    else:
        result_str = f"({', '.join(result_types)})"
        return_stmt = (
            f"return ({', '.join(result_names)}) : ({', '.join(result_types)})"
        )
    # Build the new module string
    new_module_str = f"""module {{
        func.func @main({args_str}) -> {result_str} {{
            {str(op)}
            {return_stmt}
        }}
    }}"""
    op = StablehloOp(", ".join(result_names), str(op), new_module_str)
    return op


class StablehloOp:
    def __init__(self, op_id: str, op: str, module: str):
        self.op_id = op_id
        self.op = op
        self.module = module


class StablehloSplitter:
    def __init__(self, module: str):
        self.module = module
        self.parsed_module = parse_module_from_str(module)
        self.sub_ops = []
        self.get_ops_in_module()

    def get_ops_in_module(self):
        for func_op in self.parsed_module.body.operations:
            for block in func_op.regions[0].blocks:
                for op in block.operations:
                    if op.name.startswith(("func.", "return")):
                        continue

                    stablehloOp = wrap_in_module_str(op)
                    self.sub_ops.append(stablehloOp)
