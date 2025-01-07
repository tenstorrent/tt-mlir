// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TT/IR/TTOps.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/Cpp/CppEmitter.h"

#include "llvm/ADT/ScopeExit.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_CONVERTTTNNTOEMITC
#include "ttmlir/Conversion/Passes.h.inc"

LogicalResult emitTTNNAsCpp(ModuleOp origOp, llvm::raw_ostream &os) {
  ModuleOp op = cast<ModuleOp>(origOp->clone());
  auto cleanupDispatchClone = llvm::make_scope_exit([&op] { op->erase(); });

  // Remove all instances of CpuModuleOp from the cloned top-level module.
  op.walk([](tt::CPUModuleOp cpuModule) {
    cpuModule.erase();
  });

  auto pm = PassManager::on<ModuleOp>(op.getContext());
  pm.addPass(createConvertTTNNToEmitCPass());

  if (pm.run(op).failed()) {
    llvm::outs() << "failed running createConvertTTNNToEmitCPass\n";
    return failure();
  }

  if (emitc::translateToCpp(op, os).failed()) {
    llvm::outs() << "failed running emitc::translateToCpp\n";
    return failure();
  }

  llvm::outs() << "success for emitTTNNAsCpp\n";
  return success();
}
} // namespace mlir::tt::ttnn
