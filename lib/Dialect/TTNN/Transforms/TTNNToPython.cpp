// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPy.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Target/Python/PythonEmitter.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ScopeExit.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_CONVERTTTNNTOEMITPY
#include "ttmlir/Conversion/Passes.h.inc"

LogicalResult emitTTNNAsPython(ModuleOp origOp, llvm::raw_ostream &os) {
  ModuleOp op = cast<ModuleOp>(origOp->clone());
  auto cleanupDispatchClone = llvm::make_scope_exit([&op] { op->erase(); });

  auto pm = PassManager::on<ModuleOp>(op.getContext());
  pm.addPass(createConvertTTNNToEmitPyPass());

  if (pm.run(op).failed()) {
    return failure();
  }

  std::string fileId = "";
  if (emitpy::translateToPython(op, os, fileId).failed()) {
    return failure();
  }

  return success();
}
} // namespace mlir::tt::ttnn
