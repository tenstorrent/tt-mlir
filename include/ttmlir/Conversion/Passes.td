// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_PASSES
#define TTMLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

#ifdef TTMLIR_ENABLE_STABLEHLO
def ConvertStableHLOToTTIR : Pass<"convert-stablehlo-to-ttir", "::mlir::ModuleOp"> {
let summary = "Convert StableHLO dialect to TTIR dialect.";
  let constructor = "createConvertStableHLOToTTIRPass()";
  let dependentDialects = ["mlir::stablehlo::StablehloDialect", "mlir::tt::ttir::TTIRDialect"];
}
def ConvertArithToStableHLO : Pass<"convert-arith-to-stablehlo", "::mlir::ModuleOp"> {
let summary = "Convert Arith Dialect to StableHLO dialect.";
  let constructor = "createConvertArithToStableHLOPass()";
  let dependentDialects = ["mlir::stablehlo::StablehloDialect", "mlir::arith::ArithDialect"];
}
#endif

def ConvertTosaToTTIR : Pass<"convert-tosa-to-ttir", "::mlir::ModuleOp"> {
  let summary = "Convert TOSA dialect to TTIR dialect.";
  let constructor = "createConvertTosaToTTIRPass()";
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect"];
}

def TTIRToTTIRDecomposition: Pass<"ttir-to-ttir-decomposition", "::mlir::ModuleOp"> {
  let summary = "Decomposes TTIR operations into simpler TTIR operations.";
  let constructor = "createTTIRToTTIRDecompositionPass()";
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect"];
}

def ConvertTTIRToTTNN: Pass<"convert-ttir-to-ttnn", "::mlir::ModuleOp"> {
  let summary = "Convert TTIR dialect to TTNN dialect.";
  let constructor = "createConvertTTIRToTTNNPass()";
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect", "mlir::tt::ttnn::TTNNDialect"];
}

def ConvertTTIRToTTMetal: Pass<"convert-ttir-to-ttmetal", "::mlir::ModuleOp"> {
  let summary = "Convert TTIR dialect to TTMetal dialect.";
  let constructor = "createConvertTTIRToTTMetalPass()";
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect", "mlir::tt::ttmetal::TTMetalDialect", "mlir::tt::ttkernel::TTKernelDialect"];
}

def ConvertTTNNToEmitC : Pass<"convert-ttnn-to-emitc", "::mlir::ModuleOp"> {
  let summary = "Convert TTNN dialect to EmitC dialect.";
  let constructor = "createConvertTTNNToEmitCPass()";
  let dependentDialects = ["mlir::emitc::EmitCDialect", "mlir::tt::ttnn::TTNNDialect"];
}

def ConvertTTKernelToEmitC : Pass<"convert-ttkernel-to-emitc", "::func::FuncOp"> {
  let summary = "Convert TTKernel dialect to EmitC dialect.";
  let dependentDialects = ["mlir::emitc::EmitCDialect", "mlir::func::FuncDialect",
                           "mlir::tt::ttkernel::TTKernelDialect"];
}

def ConvertTTIRToLinalg: Pass<"convert-ttir-to-linalg", "::mlir::ModuleOp"> {
  let summary = "Convert TTIR dialect to Linalg dialect.";
  let description = [{
    Conversion pass to convert TTIR ops with defined conversion pattern into linalg ops, with broadcast and collapse tensor ops as needed.
    Example:
    Input:
      func.func @add_with_broadcast(
        %arg0: tensor<32x32xf32>,
        %arg1: tensor<32x1xf32>,
        %arg2: tensor<32x32xf32>
      ) -> tensor<32x32xf32> {
        %1 = "ttir.add"(%arg0, %arg1, %arg2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xf32>, tensor<32x1xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
        return %1 : tensor<32x32xf32>
      }
    Output:
      func.func @add_with_broadcast(
        %arg0: tensor<32x32xf32>,
        %arg1: tensor<32x1xf32>,
        %arg2: tensor<32x32xf32>
      ) -> tensor<32x32xf32> {
        %collapsed = tensor.collapse_shape %arg1 [[0, 1]] : tensor<32x1xf32> into tensor<32xf32>
        %0 = tensor.empty() : tensor<32x32xf32>
        %broadcasted = linalg.broadcast ins(%collapsed : tensor<32xf32>) outs(%0 : tensor<32x32xf32>) dimensions = [1] 
        %1 = linalg.add ins(%arg0, %broadcasted : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32>
        return %1 : tensor<32x32xf32>
    }
  }];
  let constructor = "createConvertTTIRToLinalgPass()";
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect", "mlir::linalg::LinalgDialect"];
}


#endif // TTMLIR_CONVERSION_PASSES
