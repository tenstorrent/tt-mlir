// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/DstCapacityAnalysis.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/GraphColoringStrategy.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Utils.h"

using namespace mlir;

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTDSTREGISTERGC
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

struct D2MInsertDstRegisterGCPass
    : public impl::D2MInsertDstRegisterGCBase<D2MInsertDstRegisterGCPass> {

  D2MInsertDstRegisterGCPass() = default;
  D2MInsertDstRegisterGCPass(const D2MInsertDstRegisterGCPass &pass) = default;
  D2MInsertDstRegisterGCPass(const D2MInsertDstRegisterGCOptions &options)
      : D2MInsertDstRegisterGCBase(options) {}

  // Check if a generic region already has acquire_dst operations.
  static bool hasAcquireDstOp(Region &region) {
    return !region.getOps<AcquireDstOp>().empty();
  }

  void runOnOperation() override {
    auto func = getOperation();
    if (func.isExternal()) {
      return;
    }

    DstCapacityAnalysis dstCapacityAnalysis(func);
    uint32_t totalDstTiles = dstCapacityAnalysis.getMinDstCapacity();
    if (maxDstPhysicalSizeTiles > 0) {
      totalDstTiles =
          std::min(totalDstTiles,
                   static_cast<uint32_t>(maxDstPhysicalSizeTiles.getValue()));
    }

    // Process each d2m.generic operation that doesn't already have acquire_dst.
    func.walk([&](GenericOp genericOp) {
      for (auto &region : genericOp->getRegions()) {
        if (hasAcquireDstOp(region)) {
          // Skip regions that already have DST allocation.
          continue;
        }

        // Identify DST accesses that need allocation.
        auto dstLoads = identifyDstLoads(genericOp, region);
        auto dstStores = identifyDstStores(genericOp, region);

        if (dstLoads.empty() && dstStores.empty()) {
          continue;
        }

        // Infer CB type - find the canonical type used in DST accesses.
        MemRefType cbType = nullptr;
        for (auto &[loadOp, idx] : dstLoads) {
          if (cbType == nullptr) {
            cbType = mlir::cast<MemRefType>(loadOp.getMemRef().getType());
          }
        }
        for (auto &[storeOp, idx] : dstStores) {
          if (cbType == nullptr) {
            cbType = mlir::cast<MemRefType>(storeOp.getMemRef().getType());
          }
        }

        if (!cbType) {
          genericOp.emitError("No CB type found for DST allocation");
          return signalPassFailure();
        }

        // Calculate DST shape: allocate maximum available capacity like the
        // original pass.
        const int64_t volume = ttmlir::utils::volume(cbType.getShape());
        if (volume > static_cast<int64_t>(totalDstTiles)) {
          genericOp.emitError("CB volume exceeds available DST tiles");
          return signalPassFailure();
        }
        const int64_t numDstSlices = totalDstTiles / volume;

        SmallVector<int64_t> dstShape({numDstSlices});
        dstShape.append(cbType.getShape().begin(), cbType.getShape().end());

        MemRefType dstType = MemRefType::get(
            dstShape, cbType.getElementType(),
            AffineMap::getMultiDimIdentityMap(dstShape.size(), &getContext()),
            ttcore::MemorySpaceAttr::get(&getContext(),
                                         ttcore::MemorySpace::RegisterDst));

        // Create rewriter and insert acquire_dst at the start of the region.
        IRRewriter rewriter(&getContext());
        rewriter.setInsertionPointToStart(&region.front());
        AcquireDstOp acquireDst =
            rewriter.create<AcquireDstOp>(genericOp.getLoc(), dstType);

        // Insert release_dst at the end of the region.
        Block &block = region.front();
        if (!block.empty() && block.back().hasTrait<OpTrait::IsTerminator>()) {
          rewriter.setInsertionPoint(&block.back());
        } else {
          rewriter.setInsertionPointToEnd(&block);
        }
        rewriter.create<ReleaseDstOp>(genericOp.getLoc(),
                                      acquireDst.getResult());

        // TODO: Implement graph coloring to assign operations to slice indices
        // 0..numDstSlices-1 instead of sequential allocation. This will enable
        // DST slice reuse for non-interfering operations.
      }
    });
  }

private:
  // Identify DST loads that need allocation.
  SmallVector<std::pair<affine::AffineLoadOp, int64_t>>
  identifyDstLoads(GenericOp genericOp, Region &region) {
    SmallVector<std::pair<affine::AffineLoadOp, int64_t>> dstLoads;
    int nextIndex = 0;

    region.walk([&](OperandLoadStoreRegisterOpInterface computeOp) {
      auto notDstMemspace = [](auto op) {
        return op && ttcore::getMemorySpace(op.getMemRef()) !=
                         ttcore::MemorySpace::RegisterDst;
      };

      for (int64_t operandIdx : computeOp.getOperandsLoadFromDstRegister()) {
        if (auto loadOp = computeOp->getOperand(operandIdx)
                              .getDefiningOp<affine::AffineLoadOp>();
            notDstMemspace(loadOp)) {
          dstLoads.emplace_back(loadOp, nextIndex++);
        }
      }
    });

    return dstLoads;
  }

  // Identify DST stores that need allocation.
  SmallVector<std::pair<affine::AffineStoreOp, int64_t>>
  identifyDstStores(GenericOp genericOp, Region &region) {
    SmallVector<std::pair<affine::AffineStoreOp, int64_t>> dstStores;
    int nextIndex = 0;

    region.walk([&](OperandLoadStoreRegisterOpInterface computeOp) {
      auto notDstMemspace = [](auto op) {
        return op && ttcore::getMemorySpace(op.getMemRef()) !=
                         ttcore::MemorySpace::RegisterDst;
      };

      for (auto *user : computeOp->getUsers()) {
        if (auto storeOp = mlir::dyn_cast<affine::AffineStoreOp>(user);
            notDstMemspace(storeOp)) {
          dstStores.emplace_back(storeOp, nextIndex++);
        }
      }
    });

    return dstStores;
  }
};

} // namespace
} // namespace mlir::tt::d2m
