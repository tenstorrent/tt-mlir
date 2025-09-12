// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTIR_DIALECT_D2M_IR_D2MOPSINTERFACES_TD
#define TTMLIR_TTIR_DIALECT_D2M_IR_D2MOPSINTERFACES_TD

include "mlir/IR/OpBase.td"
include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td"

// Mirror TTIR DMA interface into d2m namespace for d2m ops
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
      /*desc=*/[{
        This interface denotes which op operand indices require their operands to be
        sourced from the destination register. It is used by the insert-dst-register-access
        pass to autogenerate copynests to and from dst.
      }],
      /*retTy=*/"::mlir::SmallVector<int64_t>",
      /*methodName=*/"getOperandsLoadFromDstRegister",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return ::mlir::SmallVector<int64_t>{};"
    >,
  ];
}

#endif // TTMLIR_TTIR_DIALECT_D2M_IR_D2MOPSINTERFACES_TD
