// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/GenericInterchangeAnalysis.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MGENERICAPPLYINTERCHANGE
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Update all IterIndexOp and BlockIndexOp dimensions to reflect the new
// iteration order after interchange. For interchange [2, 0, 1]:
//   IterIndex(2) -> IterIndex(0) (2 is at position 0)
//   IterIndex(0) -> IterIndex(1) (0 is at position 1)
//   IterIndex(1) -> IterIndex(2) (1 is at position 2)
// Same logic applies to BlockIndexOp.
static void updateIndexOpsForInterchange(GenericOp generic,
                                         PatternRewriter &rewriter,
                                         ArrayRef<int64_t> interchange) {
  // Compute the inverse permutation: maps old dimension to new position
  SmallVector<int64_t> inverseInterchange(interchange.size());
  for (size_t i = 0; i < interchange.size(); ++i) {
    inverseInterchange[interchange[i]] = static_cast<int64_t>(i);
  }

  auto updateDim = [&](auto indexOp) {
    int64_t oldDim = indexOp.getDim();
    if (oldDim < static_cast<int64_t>(inverseInterchange.size())) {
      int64_t newDim = inverseInterchange[oldDim];
      rewriter.modifyOpInPlace(indexOp, [&]() {
        indexOp.setDimAttr(rewriter.getI64IntegerAttr(newDim));
      });
    }
  };

  // Walk each region once to avoid re-traversing large generic bodies.
  for (Region &region : generic->getRegions()) {
    region.walk([&](Operation *op) {
      if (auto iterIndex = dyn_cast<IterIndexOp>(op)) {
        updateDim(iterIndex);
      } else if (auto blockIndex = dyn_cast<BlockIndexOp>(op)) {
        updateDim(blockIndex);
      }
    });
  }
}

static std::tuple<ArrayAttr, ArrayAttr, ArrayAttr>
applyInterchange(Builder &builder, ArrayRef<AffineMap> indexingMaps,
                 ArrayAttr iteratorTypes, ArrayAttr blockFactors,
                 ArrayRef<int64_t> interchange) {
  SmallVector<AffineMap> newIndexingMaps;
  SmallVector<Attribute> newIteratorTypes;
  SmallVector<Attribute> newBlockFactors;
  AffineMap permutationMap = mlir::inversePermutation(
      AffineMap::getPermutationMap(interchange, builder.getContext()));
  for (size_t i = 0; i < indexingMaps.size(); i++) {
    newIndexingMaps.push_back(indexingMaps[i].compose(permutationMap));
    newIteratorTypes.push_back(iteratorTypes[interchange[i]]);
    newBlockFactors.push_back(blockFactors[interchange[i]]);
  }
  return {
      builder.getAffineMapArrayAttr(newIndexingMaps),
      builder.getArrayAttr(newIteratorTypes),
      builder.getArrayAttr(newBlockFactors),
  };
}

class ApplyInterchangePattern : public OpRewritePattern<GenericOp> {
public:
  ApplyInterchangePattern(MLIRContext *ctx, InterchangeOptions options)
      : OpRewritePattern<GenericOp>(ctx), options(std::move(options)) {}

  LogicalResult matchAndRewrite(GenericOp generic,
                                PatternRewriter &rewriter) const override {
    std::optional<SmallVector<int64_t>> interchange =
        calculateInterchange(generic, options);
    if (!interchange) {
      return failure();
    }

    auto attrs = applyInterchange(rewriter, generic.getIndexingMapsValue(),
                                  generic.getIteratorTypes(),
                                  generic.getBlockFactors(), *interchange);
    ArrayAttr indexingMaps = std::get<0>(attrs);
    ArrayAttr iteratorTypes = std::get<1>(attrs);
    ArrayAttr blockFactors = std::get<2>(attrs);

    rewriter.modifyOpInPlace(generic, [&]() {
      generic.setIndexingMapsAttr(indexingMaps);
      generic.setIteratorTypesAttr(iteratorTypes);
      generic.setBlockFactorsAttr(blockFactors);
    });
    updateIndexOpsForInterchange(generic, rewriter, *interchange);
    return success();
  }

private:
  InterchangeOptions options;
};

class D2MGenericApplyInterchange
    : public impl::D2MGenericApplyInterchangeBase<D2MGenericApplyInterchange> {
public:
  using impl::D2MGenericApplyInterchangeBase<
      D2MGenericApplyInterchange>::D2MGenericApplyInterchangeBase;

  void runOnOperation() final {
    if (matmulInterchange.empty()) {
      return;
    }

    InterchangeOptions options;
    options.matmulInterchange = matmulInterchange;
    RewritePatternSet patterns(&getContext());
    patterns.add<ApplyInterchangePattern>(&getContext(), std::move(options));
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::d2m
