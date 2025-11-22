// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

namespace mlir::tt::ttcore {
#define GEN_PASS_DEF_TTCOREUNWRAPDEVICEMODULEPASS
#define GEN_PASS_DEF_TTCOREWRAPDEVICEMODULEPASS
#define GEN_PASS_DEF_TTCOREMERGECPUANDDEVICEMODULESPASS
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h.inc"

namespace {

/// Rewrite pattern to replace hoisted function declarations in DeviceModuleOp
/// with actual function definitions moved from CPUModuleOp, whilst updating
/// the call sites accordingly.
class MoveHoistedFuncsRewriter : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp stubFuncOp,
                                PatternRewriter &rewriter) const override {
    // Check if this is a hoisted function stub (has HoistedFunc attribute and
    // is private)
    if (!stubFuncOp->hasAttr(ttir::HoistedFuncAttr::name) ||
        !stubFuncOp.isPrivate()) {
      return failure();
    }

    // Ensure we're inside a DeviceModuleOp
    auto deviceModule = stubFuncOp->getParentOfType<ttcore::DeviceModuleOp>();
    if (!deviceModule) {
      return failure();
    }

    // Find the top-level module to locate CPUModuleOp
    auto topLevelModule = deviceModule->getParentOfType<mlir::ModuleOp>();
    if (!topLevelModule) {
      return failure();
    }

    // Find CPUModuleOp
    ttcore::CPUModuleOp cpuModule = nullptr;
    topLevelModule.walk([&](ttcore::CPUModuleOp cpu) {
      cpuModule = cpu;
      return WalkResult::interrupt();
    });

    if (!cpuModule) {
      return failure();
    }

    // Remove suffix "_decl" from the function name and find the function in
    // CPUModuleOp
    auto stubFuncName = stubFuncOp.getSymName().str();
    auto hoistedFuncName = stubFuncName;
    if (hoistedFuncName.size() > 5 &&
        hoistedFuncName.substr(hoistedFuncName.size() - 5) == "_decl") {
      hoistedFuncName = hoistedFuncName.substr(0, hoistedFuncName.size() - 5);
    }

    func::FuncOp hoistedFuncOp = nullptr;
    cpuModule.walk([&](func::FuncOp func) {
      if (func.getSymName() == hoistedFuncName) {
        hoistedFuncOp = func;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (!hoistedFuncOp) {
      return failure();
    }

    // Find all CallOps that call this stub function
    llvm::SmallVector<func::CallOp, 4> callOps;
    deviceModule.walk([&](func::CallOp callOp) {
      if (callOp.getCallee() == stubFuncName) {
        callOps.push_back(callOp);
      }
    });

    // Find the device module's inner ModuleOp to clone the function into
    auto innerModule = dyn_cast_if_present<mlir::ModuleOp>(
        deviceModule.getBody()->front());
    if (!innerModule) {
      return failure();
    }

    // Clone the function from CPU module to device module
    rewriter.setInsertionPointToStart(&innerModule.getBodyRegion().front());
    auto clonedFuncOp = cast<func::FuncOp>(rewriter.clone(*hoistedFuncOp));
    clonedFuncOp.setSymName(hoistedFuncName);

    // Update all call sites to use the new function name
    for (auto callOp : callOps) {
      rewriter.modifyOpInPlace(callOp, [&]() {
        callOp.setCallee(hoistedFuncName);
        // Remove the hoisted_call attribute if present
        callOp->removeAttr(ttir::HoistedCallAttr::name);
      });
    }

    // Remove the stub function from device module
    rewriter.eraseOp(stubFuncOp);

    // Remove the original function from CPU module
    rewriter.eraseOp(hoistedFuncOp);

    return success();
  }
};

/// Helper function to unwrap a DeviceModuleOp into a top-level ModuleOp.
void unwrapDeviceModule(mlir::ModuleOp rootModule) {
  DeviceModuleOp deviceOp;
  if (auto deviceOpsList = rootModule.getOps<DeviceModuleOp>();
      !deviceOpsList.empty()) {
    assert(std::distance(deviceOpsList.begin(), deviceOpsList.end()) == 1 &&
           "Top-level ModuleOp must contain 0 or 1 DeviceModuleOps!");
    deviceOp = *deviceOpsList.begin();
  } else {
    return;
  }

  if (auto cpuOpsList = rootModule.getOps<CPUModuleOp>(); !cpuOpsList.empty()) {
    assert(std::distance(cpuOpsList.begin(), cpuOpsList.end()) == 1 &&
           "Top-level ModuleOp must contain 0 or 1 CPUModuleOps!");
    (*cpuOpsList.begin())->erase();
  }

  auto innerModule = dyn_cast_if_present<ModuleOp>(deviceOp.getBody()->front());
  assert(innerModule &&
         "ttcore.device_module must always contain single builtin.module!");

  auto &innerBody = innerModule.getBodyRegion().front();
  auto &topLevelBody = rootModule.getBodyRegion().front();

  // Move operations from inner module's block to the top-level module's
  // block.
  topLevelBody.getOperations().splice(topLevelBody.end(),
                                      innerBody.getOperations());

  // Also transfer any attributes, e.g. system_desc, device
  for (const auto &attr : innerModule->getAttrs()) {
    if (!rootModule->hasAttr(attr.getName())) {
      rootModule->setAttr(attr.getName(), attr.getValue());
    }
  }

  deviceOp->erase();
}

} // namespace

class TTCoreWrapDeviceModulePass
    : public impl::TTCoreWrapDeviceModulePassBase<TTCoreWrapDeviceModulePass> {
public:
  using impl::TTCoreWrapDeviceModulePassBase<
      TTCoreWrapDeviceModulePass>::TTCoreWrapDeviceModulePassBase;
  void runOnOperation() override {
    ModuleOp rootModule = getOperation();
    if (rootModule->getParentOp() != nullptr ||
        llvm::any_of(rootModule.getOps<DeviceModuleOp>(),
                     [](auto) { return true; })) {
      return;
    }

    OpBuilder builder(&getContext());
    auto innerModule = ModuleOp::create(rootModule.getLoc());
    // Transfer attributes from root module to inner module.
    for (const auto &attr : rootModule->getAttrs()) {
      innerModule->setAttr(attr.getName(), attr.getValue());
    }

    innerModule.getBodyRegion().takeBody(rootModule.getBodyRegion());
    rootModule.getRegion().emplaceBlock();
    builder.setInsertionPointToStart(&rootModule.getBodyRegion().front());
    auto deviceModule = builder.create<DeviceModuleOp>(rootModule.getLoc());
    builder.setInsertionPointToStart(&deviceModule.getBodyRegion().front());
    builder.clone(*innerModule);
    innerModule->erase();
  }
};

class TTCoreUnwrapDeviceModulePass
    : public impl::TTCoreUnwrapDeviceModulePassBase<
          TTCoreUnwrapDeviceModulePass> {
public:
  using impl::TTCoreUnwrapDeviceModulePassBase<
      TTCoreUnwrapDeviceModulePass>::TTCoreUnwrapDeviceModulePassBase;
  void runOnOperation() override {
    ModuleOp rootModule = getOperation();
    // Ensure we only run this on top-level ModuleOp.
    if (rootModule->getParentOp() != nullptr) {
      return;
    }

    unwrapDeviceModule(rootModule);
  }
};

class TTCoreMergeCPUAndDeviceModulesPass
    : public impl::TTCoreMergeCPUAndDeviceModulesPassBase<
          TTCoreMergeCPUAndDeviceModulesPass> {
public:
  using impl::TTCoreMergeCPUAndDeviceModulesPassBase<
      TTCoreMergeCPUAndDeviceModulesPass>::
      TTCoreMergeCPUAndDeviceModulesPassBase;

  void runOnOperation() override {
    mlir::ModuleOp module = this->getOperation();

    // Apply the rewrite pattern to move hoisted functions
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<MoveHoistedFuncsRewriter>(&getContext());
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }

    // After moving hoisted functions, we can unwrap the DeviceModuleOp
    unwrapDeviceModule(module);
  }
};
} // namespace mlir::tt::ttcore
