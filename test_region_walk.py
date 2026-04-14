#!/usr/bin/env python3
"""Test whether walk() is available on Region objects in Python MLIR bindings."""

from ttmlir.ir import Context, Module, Region, Operation, WalkOrder, WalkResult

MLIR_TEXT = """
func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "ttir.abs"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
"""

def main():
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        module = Module.parse(MLIR_TEXT)

        for func_op in module.body.operations:
            print(f"func_op type: {type(func_op)}")
            body = func_op.regions[0]
            print(f"body type: {type(body)}")
            print(f"body has walk: {hasattr(body, 'walk')}")
            print(f"func_op (Operation) has walk: {hasattr(func_op, 'walk')}")
            print()

            # Test walk on Operation
            print("--- walk() on Operation ---")
            def op_visitor(op):
                print(f"  visited: {op.name}")
                return WalkResult.ADVANCE
            func_op.walk(op_visitor, walk_order=WalkOrder.PRE_ORDER)

            # Test walk on Region if available
            if hasattr(body, 'walk'):
                print("\n--- walk() on Region ---")
                body.walk(op_visitor, walk_order=WalkOrder.PRE_ORDER)
            else:
                print("\n--- walk() NOT available on Region ---")
                print("Must skip func_op manually when walking from Operation")

if __name__ == "__main__":
    main()
