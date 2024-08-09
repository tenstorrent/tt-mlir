// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIROPSINTERFACES_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIROPSINTERFACES_TD

include "mlir/IR/OpBase.td"
include "ttmlir/Dialect/TT/IR/TTOpsTypes.td"

def TTIROpInterface : OpInterface<"TTIROp"> {
  let cppNamespace = "::mlir::tt::ttir";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return the constraints on the operands of this operation.
      }],
      /*retTy=*/"::mlir::ArrayAttr",
      /*methodName=*/"getOperandConstraints",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/""
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the device of the current scope.
      }],
      /*retTy=*/"::mlir::tt::SystemDescAttr",
      /*methodName=*/"getSystemDesc",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return ::mlir::tt::getCurrentScopeSystemDesc($_op);"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the device of the current scope.
      }],
      /*retTy=*/"::mlir::tt::DeviceAttr",
      /*methodName=*/"getDevice",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return ::mlir::tt::getCurrentScopeDevice($_op);"
    >,
  ];
}

def TTIR_ElementwiseOpInterface : OpInterface<"TTIR_ElementwiseOpInterface"> {
  let cppNamespace = "::mlir::tt::ttir";

  let verify = [{
    return detail::verifyElementwiseOp($_op);
  }];
}

#endif
