// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/GenericInterchangeAnalysis.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
static void updateIndexOpsForInterchange(GenericOp generic, Builder &builder,
                                         ArrayRef<int64_t> interchange) {
  // Compute the inverse permutation: maps old dimension to new position
  SmallVector<int64_t> inverseInterchange(interchange.size());
  for (size_t i = 0; i < interchange.size(); ++i) {
    inverseInterchange[interchange[i]] = static_cast<int64_t>(i);
  }

  // Update all IterIndexOps in the generic's regions
  for (Region &region : generic->getRegions()) {
    region.walk([&](IterIndexOp iterIndex) {
      int64_t oldDim = iterIndex.getDim();
      if (oldDim < static_cast<int64_t>(inverseInterchange.size())) {
        int64_t newDim = inverseInterchange[oldDim];
        iterIndex.setDimAttr(builder.getI64IntegerAttr(newDim));
      }
    });
  }

  // Update all BlockIndexOps in the generic's regions (same logic)
  for (Region &region : generic->getRegions()) {
    region.walk([&](BlockIndexOp blockIndex) {
      int64_t oldDim = blockIndex.getDim();
      if (oldDim < static_cast<int64_t>(inverseInterchange.size())) {
        int64_t newDim = inverseInterchange[oldDim];
        blockIndex.setDimAttr(builder.getI64IntegerAttr(newDim));
      }
    });
  }
}

class D2MGenericApplyInterchange
    : public impl::D2MGenericApplyInterchangeBase<D2MGenericApplyInterchange> {
public:
  using impl::D2MGenericApplyInterchangeBase<
      D2MGenericApplyInterchange>::D2MGenericApplyInterchangeBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    Builder builder(&getContext());
    InterchangeOptions options;
    options.matmulInterchange = matmulInterchange;

    moduleOp.walk([&](GenericOp generic) {
      std::optional<SmallVector<int64_t>> interchange =
          calculateInterchange(generic, options);
      if (!interchange) {
        return;
      }

      auto [indexingMaps, iteratorTypes, blockFactors] = apply(
          builder, generic.getIndexingMapsValue(), generic.getIteratorTypes(),
          generic.getBlockFactors(), *interchange);
      generic.setIndexingMapsAttr(indexingMaps);
      generic.setIteratorTypesAttr(iteratorTypes);
      generic.setBlockFactorsAttr(blockFactors);

      updateIndexOpsForInterchange(generic, builder, *interchange);
    });
  }

  static std::tuple<ArrayAttr, ArrayAttr, ArrayAttr>
  apply(Builder &builder, ArrayRef<AffineMap> indexingMaps,
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
};
} // namespace

} // namespace mlir::tt::d2m
