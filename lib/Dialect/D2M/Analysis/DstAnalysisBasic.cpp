// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/DstAnalysis.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"

using namespace mlir;

namespace mlir::tt::d2m {

namespace {

/// Identify affine load operations that require DST allocation.
static SmallVector<std::pair<Operation *, int64_t>>
identifyDstAccesses(Operation *op, Region &region) {
  SmallVector<std::pair<Operation *, int64_t>> dstAccesses;
  int64_t index = 0;

  region.walk([&](affine::AffineLoadOp loadOp) {
    // Check if this load produces a value used by DST-eligible operations.
    // For now, all loads in compute regions are considered DST candidates.
    dstAccesses.push_back({loadOp.getOperation(), index++});
  });

  return dstAccesses;
}

/// Basic strategy: each DST access gets its own slice (no reuse).
/// This provides an upper bound on DST requirements.
class DstAnalysisBasic : public DstAnalysis {
public:
  DstAnalysisResult analyze(Operation *op) override {
    DstAnalysisResult result;

    // Walk all regions looking for d2m.generic operations
    op->walk([&](GenericOp genericOp) {
      for (auto &region : genericOp->getRegions()) {
        // Skip if already has DST allocation
        if (!region.getOps<AcquireDstOp>().empty()) {
          continue;
        }

        auto dstAccesses = identifyDstAccesses(genericOp, region);
        if (dstAccesses.empty()) {
          continue;
        }

        // Store identified accesses for reuse by transformation passes
        result.dstAccesses.append(dstAccesses.begin(), dstAccesses.end());

        // Basic: one slice per access (no reuse)
        unsigned slicesForThisRegion = dstAccesses.size();
        result.numSlicesRequired =
            std::max(result.numSlicesRequired, slicesForThisRegion);

        // Record per-operation breakdown
        for (auto &[accessOp, idx] : dstAccesses) {
          result.operationSlices[accessOp] = 1;
        }
      }
    });

    result.isValid = true;
    return result;
  }

  llvm::StringRef getStrategyName() const override {
    return "basic";
  }
};

} // namespace

std::unique_ptr<DstAnalysis>
createBasicDstAnalysis() {
  return std::make_unique<DstAnalysisBasic>();
}

} // namespace mlir::tt::d2m
