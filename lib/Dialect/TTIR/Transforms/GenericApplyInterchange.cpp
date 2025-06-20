// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/GenericInterchangeAnalysis.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRGENERICAPPLYINTERCHANGE
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
class TTIRGenericApplyInterchange
    : public impl::TTIRGenericApplyInterchangeBase<
          TTIRGenericApplyInterchange> {
public:
  using impl::TTIRGenericApplyInterchangeBase<
      TTIRGenericApplyInterchange>::TTIRGenericApplyInterchangeBase;

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
    });
  }

  static std::tuple<ArrayAttr, ArrayAttr, ArrayAttr>
  apply(Builder &builder, ArrayRef<AffineMap> indexingMaps,
        ArrayAttr iteratorTypes, ArrayAttr blockFactors,
        ArrayRef<int64_t> interchange) {
    SmallVector<AffineMap> newIndexingMaps;
    SmallVector<Attribute> newIteratorTypes;
    SmallVector<Attribute> newBlockFactors;
    for (size_t i = 0; i < indexingMaps.size(); i++) {
      AffineMap permuatationMap = mlir::inversePermutation(
          AffineMap::getPermutationMap(interchange, builder.getContext()));
      newIndexingMaps.push_back(indexingMaps[i].compose(permuatationMap));
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

} // namespace mlir::tt::ttir
