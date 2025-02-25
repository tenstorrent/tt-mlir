// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TT/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include <llvm/Support/Casting.h>

namespace mlir::tt {
#define GEN_PASS_DEF_TTUNWRAPDEVICEMODULEPASS
#define GEN_PASS_DEF_TTWRAPDEVICEMODULEPASS
#include "ttmlir/Dialect/TT/Transforms/Passes.h.inc"

class TTWrapDeviceModulePass
    : public impl::TTWrapDeviceModulePassBase<TTWrapDeviceModulePass> {
public:
  using impl::TTWrapDeviceModulePassBase<
      TTWrapDeviceModulePass>::TTWrapDeviceModulePassBase;
  void runOnOperation() override {
    ModuleOp rootModule = getOperation();
    if (rootModule->getParentOp() != nullptr ||
        llvm::any_of(rootModule.getOps<tt::DeviceModuleOp>(),
                     [](auto) { return true; })) {
      return;
    }

    OpBuilder builder(&getContext());
    auto innerModule = ModuleOp::create(rootModule.getLoc());
    innerModule.getBodyRegion().takeBody(rootModule.getBodyRegion());
    rootModule.getRegion().emplaceBlock();
    builder.setInsertionPointToStart(&rootModule.getBodyRegion().front());
    auto deviceModule = builder.create<tt::DeviceModuleOp>(rootModule.getLoc());
    builder.setInsertionPointToStart(&deviceModule.getBodyRegion().front());
    builder.clone(*innerModule);
    innerModule->erase();
  }
};

class TTUnwrapDeviceModulePass
    : public impl::TTUnwrapDeviceModulePassBase<TTUnwrapDeviceModulePass> {
public:
  using impl::TTUnwrapDeviceModulePassBase<
      TTUnwrapDeviceModulePass>::TTUnwrapDeviceModulePassBase;
  void runOnOperation() override {
    ModuleOp rootModule = getOperation();
    // Ensure we only run this on top-level ModuleOp.
    if (rootModule->getParentOp() != nullptr) {
      return;
    }

    tt::DeviceModuleOp deviceOp;
    if (auto deviceOpsList = rootModule.getOps<tt::DeviceModuleOp>();
        !deviceOpsList.empty()) {
      assert(std::distance(deviceOpsList.begin(), deviceOpsList.end()) == 1 &&
             "Top-level ModuleOp must contain 0 or 1 DeviceModuleOps!");
      deviceOp = *deviceOpsList.begin();
    } else {
      return;
    }

    if (auto cpuOpsList = rootModule.getOps<tt::CPUModuleOp>();
        !cpuOpsList.empty()) {
      assert(std::distance(cpuOpsList.begin(), cpuOpsList.end()) == 1 &&
             "Top-level ModuleOp must contain 0 or 1 CPUModuleOps!");
      (*cpuOpsList.begin())->erase();
    }

    auto innerModule =
        dyn_cast_if_present<ModuleOp>(deviceOp.getBody()->front());
    assert(innerModule &&
           "tt.device_module must always contain single builtin.module!");

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
} // namespace mlir::tt
