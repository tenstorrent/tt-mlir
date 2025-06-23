// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIROPSINTERFACES_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIROPSINTERFACES_TD

include "mlir/IR/OpBase.td"
include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td"

def TTIROpInterface : OpInterface<"TTIROp"> {
  let cppNamespace = "::mlir::tt::ttir";
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

def TTIR_ElementwiseTernary : OpInterface<"ElementwiseTernary"> {
  let cppNamespace = "::mlir::tt::ttir";
}

def TTIR_TensorManipulation : OpInterface<"TensorManipulation"> {
  let cppNamespace = "::mlir::tt::ttir";
}

def TTIR_ViewOpInterface : OpInterface<"ViewOpInterface"> {
  let cppNamespace = "::mlir::tt::ttir";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Get the input of the view op.
      }],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getInput"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Get the result of the view op.
      }],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getResult"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Recursively apply view interface ops until a shard layout is reached.
        Returns a pair of the underlying memref with shard layout and an
        affine map that describes a view over this memref.
      }],
      /*retTy=*/"std::pair<mlir::MemRefType, mlir::AffineMap>",
      /*methodName=*/"applyViews",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return ttir::applyViews($_op);"
    >,
  ];
}

def TTIR_OperandLoadRegisterOpInterface : OpInterface<"OperandLoadRegisterOpInterface"> {
  let cppNamespace = "::mlir::tt::ttir";
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
  ];
}

def TTIR_QuantizableOpInterface : OpInterface<"QuantizableOpInterface"> {
  let cppNamespace = "::mlir::tt::ttir";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Rewrites the op using quantized operands.
      }],
      /*retTy=*/"mlir::Operation *",
      /*methodName=*/"rewriteWithQuantizedInputs",
      /*args=*/(ins "PatternRewriter&":$rewriter, "ArrayRef<mlir::Value>":$quantizedOperands, "mlir::Type":$quantizedResultType),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return nullptr;"
    >,
  ];
}

#endif
