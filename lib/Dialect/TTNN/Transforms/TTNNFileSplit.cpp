// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNFILESPLIT
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
//
// TTNN File Split Pass
//
// This pass organizes TTNN IR into FileOp containers (emitpy::FileOp or
// emitc::FileOp, selected via the `target` pass option) before lowering
// to a target dialect.
//
// It creates two FileOps:
//   @"consteval" - complete consteval logic
//   @"main"      - model logic and calls to the consteval logic
//
// This pass effectively does the following:
//   - Moves the consteval functions to the consteval file.
//   - Moves the cpu-hoisted declarations to the consteval file.
//   - For each forward function that has a consteval wrapper, creates a
//   declaration in the main file so that func.call ops can resolve the symbols.
//
//===----------------------------------------------------------------------===//

namespace {

constexpr const char *kMainFileName = "main";
constexpr const char *kConstevalFileName = "consteval";

class TTNNFileSplit : public impl::TTNNFileSplitBase<TTNNFileSplit> {
public:
  using TTNNFileSplitBase::TTNNFileSplitBase;

  void runOnOperation() override {
    switch (target) {
    case FileSplitTarget::EmitPy:
      runImpl<emitpy::FileOp>();
      break;
    case FileSplitTarget::EmitC:
      runImpl<emitc::FileOp>();
      break;
    }
  }

private:
  template <typename FileOpTy>
  void runImpl() {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(&getContext());

    // If there is no consteval logic, do not perform file split.
    bool hasConstevalLogic =
        llvm::any_of(moduleOp.getOps<func::FuncOp>(), [](func::FuncOp funcOp) {
          return ttmlir::utils::isConstEvalFunc(funcOp) ||
                 ttmlir::utils::isForwardCPUDeclarationFunc(funcOp);
        });
    if (!hasConstevalLogic) {
      return;
    }

    // Create the file containers.
    builder.setInsertionPointToStart(&moduleOp.getBodyRegion().front());
    auto mainFile = builder.create<FileOpTy>(moduleOp.getLoc(), kMainFileName);
    auto constevalFile =
        builder.create<FileOpTy>(moduleOp.getLoc(), kConstevalFileName);

    // Clone the device symbol into both files so that ops inside each
    // can resolve the device via lookupNearestSymbolFrom, then erase the
    // original from the module.
    if (auto deviceOp = ttcore::lookupDeviceOp(moduleOp)) {
      builder.setInsertionPointToStart(&mainFile.getBodyRegion().front());
      builder.clone(*deviceOp);
      builder.setInsertionPointToStart(&constevalFile.getBodyRegion().front());
      builder.clone(*deviceOp);
      deviceOp->erase();
    }

    // Move const-eval functions to the consteval file. Clone
    // CPU-hoisted declarations into both files so that func.call ops
    // in both files can resolve the symbol.
    llvm::SmallVector<func::FuncOp> constevalFuncs;
    llvm::SmallVector<func::FuncOp> cpuDecls;
    llvm::SmallVector<func::FuncOp> wrapperFuncs;
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      if (ttmlir::utils::isConstEvalFunc(funcOp)) {
        constevalFuncs.push_back(funcOp);
      } else if (funcOp->hasAttr(kWrapperAttr)) {
        wrapperFuncs.push_back(funcOp);
        constevalFuncs.push_back(funcOp);
      } else if (ttmlir::utils::isForwardCPUDeclarationFunc(funcOp)) {
        cpuDecls.push_back(funcOp);
      }
    }

    for (auto constevalOp : constevalFuncs) {
      constevalOp->moveBefore(&constevalFile.getBodyRegion().front(),
                              constevalFile.getBodyRegion().front().end());
    }
    for (auto cpuDecl : cpuDecls) {
      builder.setInsertionPointToStart(&constevalFile.getBodyRegion().front());
      builder.clone(*cpuDecl.getOperation());
      builder.setInsertionPointToStart(&mainFile.getBodyRegion().front());
      builder.clone(*cpuDecl.getOperation());
      cpuDecl->erase();
    }

    // Create a declaration in the main file for each consteval wrapper function
    // so that func.call ops can resolve the symbols.
    builder.setInsertionPointToEnd(&mainFile.getBodyRegion().front());
    for (auto wrapperFunc : wrapperFuncs) {
      auto privateDecl = builder.create<func::FuncOp>(
          wrapperFunc.getLoc(), wrapperFunc.getName().str(),
          wrapperFunc.getFunctionType());
      privateDecl.setPrivate();
    }

    // Move remaining top-level operations to the main file.
    for (auto &op :
         llvm::make_early_inc_range(moduleOp.getBody()->getOperations())) {
      if (isa<FileOpTy>(&op)) {
        continue;
      }
      op.moveBefore(&mainFile.getBodyRegion().front(),
                    mainFile.getBodyRegion().front().end());
    }
  }
};

} // namespace
} // namespace mlir::tt::ttnn
