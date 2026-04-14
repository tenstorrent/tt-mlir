#!/usr/bin/env python3
"""Parse simple_scatter.mlir using ttmlir Python bindings."""

from ttmlir.ir import Context, Module
from ttmlir.dialects import ttir, ttcore
import pdb

MLIR_FILE = "test/ttmlir/Dialect/TTNN/simple_scatter.mlir"

def parse_mlir_file(file_path: str) -> Module:
    with open(file_path, "r") as f:
        mlir_text = f.read()

    with Context() as ctx:
        module = Module.parse(mlir_text)
        print(f"Parsed module from: {file_path}\n")
        print(module)
        print("\n--- Walking operations ---\n")
        for func_op in module.body.operations:
            print(f"Func: {func_op.name}")
            for region in func_op.regions:
                for block in region.blocks:
                    for op in block.operations:
                        print(f"  Op: {op.name}")
                        attrs = op.attributes
                        golden_fn(*{o.get_name() for o in op.operands}, **{attr.name: attr.attr for attr in op.attributes})
                        print("\t Inputs:", *{o.get_name() for o in op.operands})
                        print("\t Attrs:", {attr.name: attr.attr for attr in op.attributes})
                        print("\t Outputs:", {o.get_name() for o in op.results})
                        # print([for i in ])
                        # pdb.set_trace()
                        # int_attr = attrs[0].attr
                        # reduce_type_attr = attrs[1].attr
                        # print(int_attr, type(int_attr))
                        # print(reduce_type_attr, type(reduce_type_attr))
                        # exit(0)
        return module


if __name__ == "__main__":
    parse_mlir_file(MLIR_FILE)