// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/Dialect/EmitPy/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::tt::emitpy {
#define GEN_PASS_DEF_EMITPYSPLITFILES
#include "ttmlir/Dialect/EmitPy/Transforms/Passes.h.inc"

class EmitPySplitFiles : public impl::EmitPySplitFilesBase<EmitPySplitFiles> {

public:
  using impl::EmitPySplitFilesBase<EmitPySplitFiles>::EmitPySplitFilesBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(&getContext());
    builder.setInsertionPointToStart(&moduleOp.getBodyRegion().front());

    auto mainFile = builder.create<FileOp>(moduleOp->getLoc(), "main");
    auto constevalFile =
        builder.create<FileOp>(moduleOp->getLoc(), "consteval");

    auto isConstevalOp = [&](Operation *op) {
      // Check if the operation is a const-eval function.
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        return funcOp.getName().contains("const_eval");
      }

      // Check if the operation is marked as part of the consteval subgraph.
      return op->hasAttr("emitpy.consteval");
    };

    // Copy all import ops to both files. Delete original imports in module.
    for (auto importOp :
         llvm::make_early_inc_range(moduleOp.getOps<ImportOp>())) {
      builder.setInsertionPointToEnd(&constevalFile.getBodyRegion().front());
      builder.clone(*importOp);

      builder.setInsertionPointToEnd(&mainFile.getBodyRegion().front());
      builder.clone(*importOp);

      importOp.erase();
    }

    // Move all operations to the appropriate file operation body regions.
    for (auto &op : llvm::make_early_inc_range(moduleOp.getOps())) {
      // Skip the FileOps themselves.
      if (isa<FileOp>(op)) {
        continue;
      }

      if (isConstevalOp(&op)) {
        op.moveBefore(&constevalFile.getBodyRegion().front(),
                      constevalFile.getBodyRegion().front().end());
      } else {
        op.moveBefore(&mainFile.getBodyRegion().front(),
                      mainFile.getBodyRegion().front().end());
      }
    }
  };
};

std::unique_ptr<OperationPass<ModuleOp>> createEmitPySplitFiles() {
  return std::make_unique<EmitPySplitFiles>();
}

} // namespace mlir::tt::emitpy
