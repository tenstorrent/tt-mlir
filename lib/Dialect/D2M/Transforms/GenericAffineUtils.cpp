// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "GenericAffineUtils.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

#include <array>
#include <cassert>
#include <cstdint>

namespace mlir::tt::d2m {
namespace {

static constexpr llvm::StringLiteral kBlockOffsetDimAttr =
    "d2m.block_offset_dim";

// A fixed prime table used to pick non-trivial placeholder constants.
static constexpr std::array<int64_t, 32> kPrimePlaceholders = {
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
    157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211,
    223, 227, 229, 233, 239, 241, 251, 257, 263, 269};

static int64_t choosePrimePlaceholder(int64_t dim) {
  // Deterministic pseudo-randomization keyed by block dimension. All
  // d2m.block_offset ops for the same dim map to the same placeholder so
  // affine equivalence relations are preserved.
  uint64_t mixed = static_cast<uint64_t>(dim + 17) * 2654435761ull;
  unsigned idx = static_cast<unsigned>(mixed % kPrimePlaceholders.size());
  return kPrimePlaceholders[idx];
}

} // namespace

void convertBlockOffsetsToTaggedConstants(Operation *scope) {
  if (!scope) {
    return;
  }

  SmallVector<BlockOffsetOp> blockOffsetOps;
  scope->walk([&](BlockOffsetOp op) { blockOffsetOps.push_back(op); });

  OpBuilder builder(scope->getContext());
  for (BlockOffsetOp blockOffsetOp : blockOffsetOps) {
    int64_t dim = blockOffsetOp.getDim();
    int64_t placeholder = choosePrimePlaceholder(dim);

    builder.setInsertionPoint(blockOffsetOp);
    auto constantOp = builder.create<arith::ConstantIndexOp>(
        blockOffsetOp.getLoc(), placeholder);
    constantOp->setAttr(kBlockOffsetDimAttr, builder.getI64IntegerAttr(dim));
    blockOffsetOp.replaceAllUsesWith(constantOp.getResult());
    blockOffsetOp.erase();
  }
}

void restoreTaggedConstantsToBlockOffsets(Operation *scope) {
  if (!scope) {
    return;
  }

  SmallVector<arith::ConstantOp> taggedConstants;
  scope->walk([&](arith::ConstantOp constantOp) {
    if (constantOp->hasAttr(kBlockOffsetDimAttr)) {
      taggedConstants.push_back(constantOp);
    }
  });

  OpBuilder builder(scope->getContext());
  for (arith::ConstantOp constantOp : taggedConstants) {
    auto dimAttr =
        mlir::dyn_cast<IntegerAttr>(constantOp->getAttr(kBlockOffsetDimAttr));
    assert(dimAttr && "expected integer block offset dim tag");
    assert(constantOp.getType().isIndex() &&
           "expected tagged constant to have index type");

    builder.setInsertionPoint(constantOp);
    Value blockOffset =
        builder.create<BlockOffsetOp>(constantOp.getLoc(), dimAttr.getInt());
    constantOp.replaceAllUsesWith(blockOffset);
    constantOp.erase();
  }
}

} // namespace mlir::tt::d2m
