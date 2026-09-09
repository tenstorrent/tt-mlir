// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::tt::ttkernel {
#define GEN_PASS_DEF_TTKERNELINSERTDEVICEZONESCOPES
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h.inc"

namespace {

static bool opHasTraitNamed(Operation *op, StringRef traitName) {
  return llvm::StringSwitch<bool>(traitName)
      .Case("fpu", op->hasTrait<TTKernelFPUOpTrait>())
      .Case("sfpu", op->hasTrait<TTKernelSFPUOpTrait>())
      .Case("init", op->hasTrait<TTKernelInitOpTrait>())
      .Case("unary", op->hasTrait<TTKernelUnaryOpTrait>())
      .Case("binary", op->hasTrait<TTKernelBinaryOpTrait>())
      .Case("ternary", op->hasTrait<TTKernelTernaryOpTrait>())
      .Case("device-zone", op->hasTrait<TTKernelDeviceZoneOpTrait>())
      .Case("trid-noc", op->hasTrait<TTKernelTridNocOpTrait>())
      .Case("layout", op->hasTrait<TTKernelLayoutOpTrait>())
      .Case("all", true)
      .Default(false);
}

class TTKernelInsertDeviceZoneScopes
    : public impl::TTKernelInsertDeviceZoneScopesBase<
          TTKernelInsertDeviceZoneScopes> {
public:
  using impl::TTKernelInsertDeviceZoneScopesBase<
      TTKernelInsertDeviceZoneScopes>::TTKernelInsertDeviceZoneScopesBase;

  void runOnOperation() final {
    llvm::SmallVector<std::string> selectedTraits;
    for (const std::string &s : traitNames) {
      selectedTraits.push_back(s);
    }

    func::FuncOp func = getOperation();

    // Wrap each kernel function body with a DeviceZoneScopedN labeled by the
    // function's symbol name (e.g., "compute_kernel1", "datamovement_kernel0").
    if (func->hasAttr("ttkernel.thread") && !func.empty()) {
      Block &entry = func.getBody().front();
      OpBuilder builder(&entry, entry.begin());
      builder.create<emitc::VerbatimOp>(
          func.getLoc(),
          ("DeviceZoneScopedN(\"kernel_outer_" + func.getName() + "\");")
              .str());
    }

    // Wrap each op with a selected trait in a "{ DeviceZoneScopedN(name); }"
    // scope.
    func.walk([&](Operation *op) {
      if (op->getDialect() !=
          getContext().getLoadedDialect<ttkernel::TTKernelDialect>()) {
        return;
      }
      if (!llvm::any_of(selectedTraits, [&](std::string t) {
            return opHasTraitNamed(op, t);
          })) {
        return;
      }
      // Skip ops whose results are used downstream to avoid scoping issues.
      if (llvm::any_of(op->getResults(),
                       [](Value v) { return !v.use_empty(); })) {
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
