#ifndef TTNN_CONVERSION_PASSES
#define TTNN_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTTNNToEmitC : Pass<"convert-ttnn-to-emitc", "::mlir::func::FuncOp"> {
  let summary = "Convert TTNN dialect to EmitC dialect.";
  let constructor = "createConvertTTNNToEmitCPass()";
  let dependentDialects = ["mlir::emitc::EmitCDialect", "mlir::tt::ttnn::TTNNDialect"];
}

#endif // TTNN_CONVERSION_PASSES
