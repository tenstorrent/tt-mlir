// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSINTERFACES_TD
#define TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSINTERFACES_TD

include "mlir/IR/OpBase.td"
include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td"

def D2MOpInterface : OpInterface<"D2MOp"> {
  let cppNamespace = "::mlir::tt::d2m";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Get the device of the current scope.
      }],
      /*retTy=*/"::mlir::tt::ttcore::SystemDescAttr",
      /*methodName=*/"getSystemDesc",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return ::mlir::tt::ttcore::getCurrentScopeSystemDesc($_op);"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the device of the current scope.
      }],
      /*retTy=*/"::mlir::tt::ttcore::DeviceAttr",
      /*methodName=*/"getDevice",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return ::mlir::tt::ttcore::lookupDevice($_op);"
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

// Mirror D2M ViewOpInterface into d2m namespace
// so D2M passes can use a local interface.

def D2M_ViewOpInterface : OpInterface<"ViewOpInterface"> {
  let cppNamespace = "::mlir::tt::d2m";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{Get the (only) input of the view op.}],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getInput"
    >,
    InterfaceMethod<
      /*desc=*/[{Get the result of the view op.}],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getResult"
    >,
    InterfaceMethod<
      /*desc=*/[{Return whether this is a composite view or not.}],
      /*retTy=*/"bool",
      /*methodName=*/"isComposite",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return false;"
    >,
    InterfaceMethod<
      /*desc=*/[{Get all the inputs of the (potentially composite) view op.}],
      /*retTy=*/"mlir::SmallVector<mlir::Value>",
      /*methodName=*/"getCompositeInputs",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return {$_op.getInput()};"
    >,
  ];
}

#endif // TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSINTERFACES_TD
