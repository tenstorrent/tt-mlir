// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"

#define DEBUG_TYPE "D2MGenericAffineScalarReplacement"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MGENERICAFFINESCALARREPLACEMENT
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MGenericAffineScalarReplacement
    : public impl::D2MGenericAffineScalarReplacementBase<
          D2MGenericAffineScalarReplacement> {
public:
  using D2MGenericAffineScalarReplacementBase::
      D2MGenericAffineScalarReplacementBase;

  void runOnOperation() final {
    getOperation()->walk([&](func::FuncOp funcOp) {
      auto &domInfo = getAnalysis<DominanceInfo>();
      auto &postDomInfo = getAnalysis<PostDominanceInfo>();
      auto &aliasAnalysis = getAnalysis<AliasAnalysis>();
      affine::affineScalarReplace(funcOp, domInfo, postDomInfo, aliasAnalysis);
    });
  }
};
} // namespace

} // namespace mlir::tt::d2m
