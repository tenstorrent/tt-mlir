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
      /*defaultImplementation=*/"return ::mlir::tt::lookupDevice($_op);"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return whether op can be run on CPU.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"supportsCPUExecution",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return false;"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return whether op must be run on CPU.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"supportsOnlyCPUExecution",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return false;"
    >,
  ];
}

def TTIR_Broadcastable : OpInterface<"Broadcastable"> {
  let cppNamespace = "::mlir::tt::ttir";

  let dependentTraits = [AttrSizedOperandSegments];

  let verify = [{
    return detail::verifyBroadcastable($_op);
  }];
}

def TTIR_PartiallyBroadcastable : OpInterface<"PartiallyBroadcastable"> {
  // Supports implicit broadcast for first operand only.
  let cppNamespace = "::mlir::tt::ttir";
}

def TTIR_FullyBroadcastable : OpInterface<"FullyBroadcastable"> {
  // Supports implicit broadcast for all the operands.
  let cppNamespace = "::mlir::tt::ttir";
}

def TTIR_GenericParent : OpInterface<"GenericParent"> {
  let cppNamespace = "::mlir::tt::ttir";

  let verify = [{
    return detail::verifyGenericParent($_op);
  }];
}

def TTIR_ElementwiseUnary : OpInterface<"ElementwiseUnary"> {
  let cppNamespace = "::mlir::tt::ttir";
}

def TTIR_ElementwiseBinary : OpInterface<"ElementwiseBinary"> {
  let cppNamespace = "::mlir::tt::ttir";
}

def TTIR_TensorManipulation : OpInterface<"TensorManipulation"> {
  let cppNamespace = "::mlir::tt::ttir";
}

#endif
