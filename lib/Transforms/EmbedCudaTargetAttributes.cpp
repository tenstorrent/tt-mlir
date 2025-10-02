// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::transforms {

#define GEN_PASS_DEF_EMBEDCUDATARGETATTRIBUTES
#include "ttmlir/Transforms/Passes.h.inc"

namespace {

// Simple pass implementation that embeds CUDA target attributes into module
class EmbedCudaTargetAttributesPass
    : public impl::EmbedCudaTargetAttributesBase<
          EmbedCudaTargetAttributesPass> {
public:
  using impl::EmbedCudaTargetAttributesBase<
      EmbedCudaTargetAttributesPass>::EmbedCudaTargetAttributesBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());
    // Use the inherited option members from the base class
    moduleOp->setAttr("cuda.chip", builder.getStringAttr(this->chip));
    moduleOp->setAttr("cuda.features", builder.getStringAttr(this->features));
    moduleOp->setAttr("cuda.opt_level",
                      builder.getI32IntegerAttr(this->opt_level));
  }
};

} // namespace

} // namespace mlir::tt::transforms
