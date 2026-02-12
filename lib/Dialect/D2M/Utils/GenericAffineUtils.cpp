// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/GenericAffineUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

namespace mlir::tt::d2m::utils {

void convertToAffineCompatibilityForm(GenericOp fusedOp, OpBuilder &builder) {
  // Use distinct non-unit prime sentinel values so affine analysis sees
  // non-trivial loop trip counts regardless of the real block-factor values.
  // Each dimension gets its own unique prime so the analysis can distinguish
  // the loop bounds of different nesting levels.
  static constexpr int64_t kSentinelPrimes[] = {5,  7,  11, 13, 17,
                                                19, 23, 29, 31, 37};
  auto blockFactorsAttr = fusedOp.getBlockFactors();
  TT_assertv(blockFactorsAttr.size() <= std::size(kSentinelPrimes),
             "more block-factor dimensions ({}) than available sentinel "
             "primes ({})",
             blockFactorsAttr.size(), std::size(kSentinelPrimes));

  // Create shared constants at the start of the fused block
  Block &fusedBlock = fusedOp.getRegion(0).front();
  builder.setInsertionPointToStart(&fusedBlock);
  SmallVector<Value> sharedConstants;
  for (size_t dim = 0; dim < blockFactorsAttr.size(); ++dim) {
    int64_t sentinel = kSentinelPrimes[dim];
    auto constOp =
        builder.create<arith::ConstantIndexOp>(fusedOp.getLoc(), sentinel);
    constOp->setAttr("d2m.block_factor_constant",
                     builder.getI64IntegerAttr(dim));
    sharedConstants.push_back(constOp.getResult());
  }

  // Replace all get_block_factor ops with the shared constants
  SmallVector<GetBlockFactorOp> blockFactorOps;
  fusedOp.getRegion(0).walk(
      [&](GetBlockFactorOp op) { blockFactorOps.push_back(op); });

  for (GetBlockFactorOp op : blockFactorOps) {
    int64_t dim = op.getDim();
    op.replaceAllUsesWith(sharedConstants[dim]);
    op.erase();
  }

  // CRITICAL: Update loop upper bounds to use shared constants
  // The cloned loops may reference get_block_factor values from original
  // contexts
  SmallVector<affine::AffineForOp> allLoops;
  fusedOp.getRegion(0).walk(
      [&](affine::AffineForOp loop) { allLoops.push_back(loop); });

  for (affine::AffineForOp loop : allLoops) {
    // Check if this is a blocking loop with symbolic upper bound
    if (!loop->hasAttr("d2m.blocking_loop")) {
      continue;
    }

    auto upperBoundMap = loop.getUpperBoundMap();
    if (upperBoundMap.getNumResults() != 1 ||
        upperBoundMap.getNumSymbols() != 1) {
      continue;
    }

    // The loop should have one upper bound operand (the symbol)
    auto ubOperands = loop.getUpperBoundOperands();
    if (ubOperands.size() != 1) {
      continue;
    }

    // Determine which dimension based on the blocking_loop attribute
    int64_t dim =
        loop->getAttrOfType<IntegerAttr>("d2m.blocking_loop").getInt();

    // Replace the upper bound operand with the shared constant
    loop->setOperand(loop.getNumControlOperands() - ubOperands.size(),
                     sharedConstants[dim]);
  }
}

void convertFromAffineCompatibilityForm(GenericOp compatGeneric,
                                        OpBuilder &builder) {
  // Restore get_block_factor ops from tagged constants.
  SmallVector<arith::ConstantIndexOp> taggedConstants;
  compatGeneric.getRegion(0).walk([&](arith::ConstantIndexOp op) {
    if (op->hasAttr("d2m.block_factor_constant")) {
      taggedConstants.push_back(op);
    }
  });

  for (arith::ConstantIndexOp constOp : taggedConstants) {
    auto dimAttr =
        constOp->getAttrOfType<IntegerAttr>("d2m.block_factor_constant");
    int64_t dim = dimAttr.getInt();

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(constOp);

    auto getBlockFactorOp =
        builder.create<GetBlockFactorOp>(constOp.getLoc(), dim);
    constOp.replaceAllUsesWith(getBlockFactorOp.getResult());
    constOp.erase();
  }
}

} // namespace mlir::tt::d2m::utils
