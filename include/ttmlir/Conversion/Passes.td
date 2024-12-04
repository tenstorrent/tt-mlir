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
def RedundantBroadcastElimination : Pass<"redundant-broadcast-elimination", "::mlir::ModuleOp"> {
let summary = "Eliminate any redundant broadcast ops by folding them.";
  let constructor = "createRedundantBroadcastEliminationPass()";
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
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

#endif // TTMLIR_CONVERSION_PASSES
