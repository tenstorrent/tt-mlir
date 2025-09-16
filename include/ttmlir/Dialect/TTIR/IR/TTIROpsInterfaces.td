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

def TTIR_OperandLoadStoreRegisterOpInterface : OpInterface<"OperandLoadStoreRegisterOpInterface"> {
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

def TTIR_QuantizableOpInterface : OpInterface<"QuantizableOpInterface"> {
  let cppNamespace = "::mlir::tt::ttir";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        This interface method performs the rewrite of the op using quantized operands.
        The method returns a nullptr if the op cannot be rewritten in quantized form given the operands.
      }],
      /*retTy=*/"mlir::Operation *",
      /*methodName=*/"rewriteWithQuantizedInputs",
      /*args=*/(ins "PatternRewriter&":$rewriter, "ArrayRef<mlir::Value>":$sourceOperands, "mlir::ValueRange":$outputOperands),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return nullptr;"
    >,
    InterfaceMethod<
      /*desc=*/[{
        This interface method denotes that an op can be rewritten to accept quantized inputs.
        The quant-dequant pattern pass uses this function to verify basic legality; that the op can be rewritten
        in quantized form given the operands passed in as parameters.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isQuantizedRewriteFavorable",
      /*args=*/(ins "ArrayRef<mlir::Value>":$sourceOperands),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return true;"
    >,
  ];
}

def TTIR_DMAOpInterface : OpInterface<"DMAOpInterface"> {
  let cppNamespace = "::mlir::tt::ttir";
  let description = [{
    This interface allows querying the common aspects of
    DMA(Read|Write)Ops ; mostly that they have two memref-like
    operands.
  }];
  let methods = [
    InterfaceMethod<[{}], "::mlir::TypedValue<::mlir::Type>", "getSrc">,
    InterfaceMethod<[{}], "::mlir::TypedValue<::mlir::Type>", "getDst">,
    InterfaceMethod<[{}], "::mlir::OpOperand&", "getSrcMutable">,
    InterfaceMethod<[{}], "::mlir::OpOperand&", "getDstMutable">,
    InterfaceMethod<[{}], "::mlir::Location", "getLoc">,
    InterfaceMethod<[{}], "::mlir::TypedValue<::mlir::Type>", "getResult">,
  ];
}

#endif
