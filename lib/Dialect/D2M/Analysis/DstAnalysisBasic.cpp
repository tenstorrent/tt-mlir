// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/DstAnalysis.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

namespace mlir::tt::d2m {

namespace {

/// Basic strategy: each DST access gets its own slice (no reuse).
/// This provides an upper bound on DST requirements.
class DstAnalysisBasic : public DstAnalysis {
public:
  DstAnalysisResult analyze(mlir::Operation *op) override {
    DstAnalysisResult result;

    // Walk all regions looking for d2m.generic operations
    op->walk([&](GenericOp genericOp) {
      for (auto &region : genericOp.getRegions()) {
        // Skip if already has DST allocation
        if (!region.getOps<AcquireDstOp>().empty()) {
          continue;
        }

        // Identify DST accesses using OperandLoadStoreRegisterOpInterface
        SmallVector<std::pair<Operation *, int64_t>> dstAccesses;
        int nextIndex = 0;

        auto notDstMemspace = [](auto op) {
          return op && ttcore::getMemorySpace(op.getMemRef()) !=
                           ttcore::MemorySpace::RegisterDst;
        };

        region.walk([&](OperandLoadStoreRegisterOpInterface computeOp) {
          // Collect loads for DST operands
          for (int64_t operandIdx :
               computeOp.getOperandsLoadFromDstRegister()) {
            if (auto loadOp = computeOp->getOperand(operandIdx)
                                  .getDefiningOp<affine::AffineLoadOp>();
                notDstMemspace(loadOp)) {
              dstAccesses.emplace_back(loadOp, nextIndex++);
            }
          }

          // Collect stores from compute op results
          for (auto *user : computeOp->getUsers()) {
            if (auto storeOp = mlir::dyn_cast<affine::AffineStoreOp>(user);
                notDstMemspace(storeOp)) {
              dstAccesses.emplace_back(storeOp, nextIndex++);
            }
          }
        });

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

  llvm::StringRef getStrategyName() const override { return "basic"; }
};

} // namespace

std::unique_ptr<DstAnalysis> createBasicDstAnalysis() {
  return std::make_unique<DstAnalysisBasic>();
}

} // namespace mlir::tt::d2m
