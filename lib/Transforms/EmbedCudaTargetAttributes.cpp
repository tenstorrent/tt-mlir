// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt::transforms {

#define GEN_PASS_DEF_EMBEDCUDATARGETATTRIBUTES
#include "ttmlir/Transforms/Passes.h.inc"

namespace {

// Simple pass implementation that embeds CUDA target attributes into module
class EmbedCudaTargetAttributesPass
    : public impl::EmbedCudaTargetAttributesBase<
          EmbedCudaTargetAttributesPass> {
private:
  std::string chipArch = "sm_80";
  std::string ptxFeatures = "+ptx70";
  int32_t optimizationLevel = 2;

public:
  EmbedCudaTargetAttributesPass() = default;

  EmbedCudaTargetAttributesPass(EmbedCudaTargetAttributesOptions options)
      : chipArch(options.chip), ptxFeatures(options.features),
        optimizationLevel(options.optLevel) {}

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());
    moduleOp->setAttr("cuda.chip", builder.getStringAttr(chipArch));
    moduleOp->setAttr("cuda.features", builder.getStringAttr(ptxFeatures));
    moduleOp->setAttr("cuda.optLevel",
                      builder.getI32IntegerAttr(optimizationLevel));
  }
};

} // namespace

} // namespace mlir::tt::transforms
