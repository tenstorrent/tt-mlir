// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_INTERFACES_TTNNWORKAROUNDINTERFACE_TD
#define TTMLIR_DIALECT_TTNN_INTERFACES_TTNNWORKAROUNDINTERFACE_TD

include "mlir/IR/OpBase.td"

// This interface is used to specify workarounds for TTNN operations.
def TTNN_WorkaroundInterface : OpInterface<"TTNNWorkaroundInterface"> {
  let cppNamespace = "::mlir::tt::ttnn::wa";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the workarounds associated with each operand and result of this operation.
        If the operation is a Destination-Passing Style (DPS) operation, the same workarounds
        must apply to both the DPS initial operands and the operation results. These constraints
        are verified through the interface verifier.

        For example, consider the following ttnn operations:
        %0 = "ttnn.empty"() : () -> tensor<1x1xf32>
        %1 = "ttnn.abs"(%arg0, %0) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>

        In this example:
          - The ttnn.abs operation has two input operand workarounds.
          - It has one output operand workaround.
          - The output workaround must match the workaround for the second input operand,
          ensuring consistency as required by the DPS pattern.
      }],
      /*retTy=*/"TTNNOperandsWorkarounds",
      /*methodName=*/"getOperandsWorkarounds",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        // Return default empty workarounds for all input and output operands
        return TTNNOperandsWorkarounds::createEmptyTTNNOperandsWorkarounds(this->getOperation());
      }]
    >,
  ];

  let verify = [{
    return verifyTTNNWorkaroundInterface($_op);
  }];
}

#endif // TTMLIR_DIALECT_TTNN_INTERFACES_TTNNWORKAROUNDINTERFACE_TD
