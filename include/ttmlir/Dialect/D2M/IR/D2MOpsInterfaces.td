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
      /*desc=*/[{Get the input of the view op.}],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getInput"
    >,
    InterfaceMethod<
      /*desc=*/[{Get the result of the view op.}],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getResult"
    >,
    InterfaceMethod<
      /*desc=*/[{Recursively apply view ops until a shard layout is reached, returning underlying memref and view map.}],
      /*retTy=*/"std::pair<mlir::MemRefType, mlir::AffineMap>",
      /*methodName=*/"applyViews",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return ::mlir::tt::d2m::applyViews($_op);"
    >,
  ];
}

// Mirror D2M DMA interface into d2m namespace for d2m ops

def D2M_DMAOpInterface : OpInterface<"DMAOpInterface"> {
  let cppNamespace = "::mlir::tt::d2m";
  let methods = [
    InterfaceMethod<[{}], "::mlir::TypedValue<::mlir::Type>", "getSrc">,
    InterfaceMethod<[{}], "::mlir::TypedValue<::mlir::Type>", "getDst">,
    InterfaceMethod<[{}], "::mlir::OpOperand&", "getSrcMutable">,
    InterfaceMethod<[{}], "::mlir::OpOperand&", "getDstMutable">,
    InterfaceMethod<[{}], "::mlir::Location", "getLoc">,
    InterfaceMethod<[{}], "::mlir::TypedValue<::mlir::Type>", "getResult">,
  ];
}

// Parent interface used to verify D2M region ops are nested in d2m.generic or func.func

def D2M_GenericParent : OpInterface<"GenericParent"> {
  let cppNamespace = "::mlir::tt::d2m";

  let verify = [{
    return detail::verifyGenericParent($_op);
  }];
}

// Interface denoting which operands of an op must be sourced from dst register

def D2M_OperandLoadRegisterOpInterface : OpInterface<"OperandLoadRegisterOpInterface"> {
  let cppNamespace = "::mlir::tt::d2m";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{This interface denotes which op operand indices require their operands to be
        sourced from the destination register. It is used by the insert-dst-register-access
        pass to autogenerate copynests to and from dst.}],
      /*retTy=*/"::mlir::SmallVector<int64_t>",
      /*methodName=*/"getOperandsLoadFromDstRegister",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return ::mlir::SmallVector<int64_t>{};"
    >,
  ];
}

def D2M_OperandLoadStoreRegisterOpInterface : OpInterface<"OperandLoadStoreRegisterOpInterface"> {
  let cppNamespace = "::mlir::tt::d2m";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        This interface denotes which op operand indices require their operands to be
        sourced from the destination register.  It is primarily used by the
        insert-dst-register-access pass to autogenerate copynests to and from dst.
      }],
      /*retTy=*/"::mlir::SmallVector<int64_t>",
      /*methodName=*/"getOperandsLoadFromDstRegister",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return ::mlir::SmallVector<int64_t>{};"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns whether this operation should use the destination register in place.
        When true, the operation can read from and write to the same register location.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"getDstRegInPlace",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return false;"
    >,
  ];
}

#endif // TTMLIR_D2M_DIALECT_D2M_IR_D2MOPSINTERFACES_TD
