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

      ```c++
      class ConcreteOp ... {
      public:
        bool hasSideEffects() {
          // This way we can override the default implementation.
          return false;
        }
      };
      ```
    }],
      "bool", "hasSideEffects", (ins), /*methodBody=*/[{}],
       /*defaultImplementation=*/[{
        // Check if the operation has memory effects
        // Operations without memory effects are considered pure
        return !::mlir::isMemoryEffectFree($_op);
    }]>,
  ];
}

#endif // TTMLIR_DIALECT_EMITPY_IR_EMITPYINTERFACES_TD
