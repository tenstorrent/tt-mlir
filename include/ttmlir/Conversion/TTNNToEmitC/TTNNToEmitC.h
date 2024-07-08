#ifndef TTMLIR_CONVERSION_TTNNTOEMITC_H
#define TTMLIR_CONVERSION_TTNNTOEMITC_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt::ttnn {

std::unique_ptr<OperationPass<func::FuncOp>> createConvertTTNNToEmitCPass();
LogicalResult emitTTNNAsCpp(ModuleOp origOp, llvm::raw_ostream &os);

} // namespace mlir::tt::ttnn

#endif // TTMLIR_CONVERSION_TTNNTOEMITC_H
