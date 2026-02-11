// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/GenericAffineUtils.h"

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
      OpBuilder builder(funcOp.getContext());

      // Convert all d2m.generic ops to affine-compatible form
      SmallVector<GenericOp> genericOps;
      funcOp.walk([&](GenericOp op) { genericOps.push_back(op); });
      for (GenericOp op : genericOps) {
        utils::convertToAffineCompatibilityForm(op, builder);
      }

      // Run affine scalar replacement on the function
      auto &domInfo = getAnalysis<DominanceInfo>();
      auto &postDomInfo = getAnalysis<PostDominanceInfo>();
      auto &aliasAnalysis = getAnalysis<AliasAnalysis>();
      affine::affineScalarReplace(funcOp, domInfo, postDomInfo, aliasAnalysis);

      // Convert all d2m.generic ops back from affine-compatible form
      genericOps.clear();
      funcOp.walk([&](GenericOp op) { genericOps.push_back(op); });
      for (GenericOp op : genericOps) {
        utils::convertFromAffineCompatibilityForm(op, builder);
      }
    });
  }
};
} // namespace

} // namespace mlir::tt::d2m
