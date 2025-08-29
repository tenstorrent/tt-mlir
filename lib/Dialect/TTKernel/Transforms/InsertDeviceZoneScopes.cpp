// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::tt::ttkernel {
#define GEN_PASS_DEF_TTKERNELINSERTDEVICEZONESCOPES
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h.inc"

namespace {
class TTKernelInsertDeviceZoneScopes
    : public impl::TTKernelInsertDeviceZoneScopesBase<
          TTKernelInsertDeviceZoneScopes> {
public:
  using impl::TTKernelInsertDeviceZoneScopesBase<
      TTKernelInsertDeviceZoneScopes>::TTKernelInsertDeviceZoneScopesBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    module.walk([&](Operation *op) {
      if (!op->hasTrait<ttkernel::TTKernelDeviceZoneOpTrait>()) {
        return;
      }
      OpBuilder builder(op);
      builder.create<emitc::VerbatimOp>(op->getLoc(), "{");
      auto name = op->getName().getStringRef();
      if (name.starts_with("ttkernel.")) {
        name = name.drop_front(9);
      }
      builder.create<emitc::VerbatimOp>(op->getLoc(), "DeviceZoneScopedN(\"" +
                                                          name.str() + "\");");
      builder.setInsertionPointAfter(op);
      builder.create<emitc::VerbatimOp>(op->getLoc(), "}");
    });
  }
};
} // namespace

} // namespace mlir::tt::ttkernel
