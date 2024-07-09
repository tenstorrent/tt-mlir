#ifndef TTMLIR_CONVERSION_PASSDETAIL_H
#define TTMLIR_CONVERSION_PASSDETAIL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"

namespace mlir::tt::ttnn {

// class TTNNDialect;

// using namespace mlir;
#define GEN_PASS_CLASSES
#include "ttmlir/Conversion/Passes.h.inc"
#define GEN_PASS_DEF_CONVERTTTNNTOEMITC
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttnn

#endif // TTMLIR_CONVERSION_PASSDETAIL_H
