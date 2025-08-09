// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_IR_TTNNOPSINTERFACES_TD
#define TTMLIR_DIALECT_TTNN_IR_TTNNOPSINTERFACES_TD

include "mlir/IR/OpBase.td"

def TTNN_ElementwiseUnary : OpInterface<"ElementwiseUnary"> {
  let cppNamespace = "::mlir::tt::ttnn";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Get the parameters for the elementwise unary operation.
        Returns an empty array by default, but ops can override this
        to return their specific parameters.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::FloatAttr>",
      /*methodName=*/"getParams",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/""
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the unary operation type for this elementwise unary operation.
        This method must be implemented by each concrete unary op.
      }],
      /*retTy=*/"::mlir::tt::ttnn::UnaryOpType",
      /*methodName=*/"getUnaryOpType",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/""
    >,
  ];
}

def TTNN_ElementwiseBinary : OpInterface<"ElementwiseBinary"> {
  let cppNamespace = "::mlir::tt::ttnn";
}

#endif // TTMLIR_DIALECT_TTNN_IR_TTNNOPSINTERFACES_TD
