// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Casting.h"

namespace mlir::tt::ttcore {
#define GEN_PASS_DEF_TTCOREUNWRAPDEVICEMODULEPASS
#define GEN_PASS_DEF_TTCOREWRAPDEVICEMODULEPASS
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h.inc"

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

    DeviceModuleOp deviceOp;
    if (auto deviceOpsList = rootModule.getOps<DeviceModuleOp>();
        !deviceOpsList.empty()) {
      assert(std::distance(deviceOpsList.begin(), deviceOpsList.end()) == 1 &&
             "Top-level ModuleOp must contain 0 or 1 DeviceModuleOps!");
      deviceOp = *deviceOpsList.begin();
    } else {
      return;
    }

    if (auto cpuOpsList = rootModule.getOps<CPUModuleOp>();
        !cpuOpsList.empty()) {
      assert(std::distance(cpuOpsList.begin(), cpuOpsList.end()) == 1 &&
             "Top-level ModuleOp must contain 0 or 1 CPUModuleOps!");
      (*cpuOpsList.begin())->erase();
    }

    auto innerModule =
        dyn_cast_if_present<ModuleOp>(deviceOp.getBody()->front());
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
};
} // namespace mlir::tt::ttcore
