// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_EMITPY_IR_EMITPYINTERFACES_TD
#define TTMLIR_DIALECT_EMITPY_IR_EMITPYINTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// EmitPy interfaces definitions.
//===----------------------------------------------------------------------===//

def PyExpressionInterface : OpInterface<"PyExpressionInterface"> {
  let description = [{
    Interface to mark operations that can be part of the PyExpression.
  }];

  let cppNamespace = "::mlir::tt::emitpy";
  let methods = [
    InterfaceMethod<[{
      Check whether operation has side effects that may affect the expression
      evaluation.

      By default operation is marked as having side effects.

      ```Python
      # Example: Pure operations like literals, arithmetic
      class ConcreteOp:
          def hasPythonSideEffects(self):
              # Override for pure operations like literals, arithmetic
              return False
      ```
    }],
      "bool", "hasSideEffects", (ins), /*methodBody=*/[{}],
       /*defaultImplementation=*/[{
        return true;
    }]>,
  ];
}

#endif // TTMLIR_DIALECT_EMITPY_IR_EMITPYINTERFACES_TD
