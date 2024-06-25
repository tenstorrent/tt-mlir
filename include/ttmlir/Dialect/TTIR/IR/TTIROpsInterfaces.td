// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIROPSINTERFACES_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIROPSINTERFACES_TD

include "mlir/IR/OpBase.td"
include "ttmlir/Dialect/TT/IR/TTOpsTypes.td"

def TTIROpInterface : OpInterface<"TTIROpInterface"> {
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
  ];
}

#endif
