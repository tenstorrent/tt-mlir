// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/TopKUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"

#include <cstdint>

namespace mlir::tt::d2m::utils {

std::pair<Value, Value> emitLeafTopk(RewriterBase &rewriter, Location loc,
                                     Value layoutedInput, int32_t k,
                                     int32_t dim, int64_t reductionDimSize,
                                     const LeafTopKBuffers &buffers) {
  MLIRContext *ctx = rewriter.getContext();
  auto layoutedType = cast<RankedTensorType>(layoutedInput.getType());
  auto metalLayout = cast<ttcore::MetalLayoutAttr>(layoutedType.getEncoding());
  auto valTileType = cast<ttcore::TileType>(layoutedType.getElementType());
  ArrayRef<int64_t> deviceShape = layoutedType.getShape();
  std::size_t physicalRank = deviceShape.size() / 2;

  auto parallel =
      ttcore::IteratorTypeAttr::get(ctx, ttcore::IteratorType::Parallel);
  llvm::SmallVector<Attribute> iteratorTypes(physicalRank, parallel);
  AffineMap identityMap = rewriter.getMultiDimIdentityMap(physicalRank);

  // TopkBlockOp sorts down tile columns; dim=1 puts the sort dim on tile rows.
  Value topkInput = layoutedInput;
  if (buffers.transpose.isPlaced()) {
    Value transposed = emitUnaryGeneric(
        rewriter, loc, layoutedInput,
        rewriter
            .create<EmptyOp>(loc, buffers.transpose.type,
                             buffers.transpose.vgmInverse,
                             buffers.transpose.vgmForward)
            .getResult(),
        [&](OpBuilder &b, Location l, ValueRange args) {
          return b.create<TileTransposeOp>(l, args[1].getType(), args[0])
              .getResult();
        },
        buffers.transpose.grid);
    topkInput =
        materializeToLayout(rewriter, loc, transposed, buffers.transpose);
  }

  auto idxTileType = ttcore::TileType::get(rewriter.getI32Type());

  // One scratch tile per core for the lane pattern; the kernel derives the
  // whole index buffer from it plus its own grid coordinate.
  Value idxScratchEmpty;
  if (buffers.scratch.isPlaced()) {
    idxScratchEmpty = rewriter
                          .create<EmptyOp>(loc, buffers.scratch.type,
                                           buffers.scratch.vgmInverse,
                                           buffers.scratch.vgmForward)
                          .getResult();
  } else {
    ArrayRef<int64_t> idxGridShape = metalLayout.getGridShape(layoutedType);
    llvm::SmallVector<int64_t> idxScratchShape(idxGridShape.begin(),
                                               idxGridShape.end());
    idxScratchShape.append({1, 1});
    auto idxScratchLayout = ttcore::MetalLayoutAttr::get(
        ctx, llvm::SmallVector<int64_t>{1, 1}, ttcore::MemorySpace::DeviceL1,
        ttcore::TensorMemoryLayout::Sharded);
    idxScratchEmpty = rewriter
                          .create<EmptyOp>(loc, idxScratchShape, idxTileType,
                                           idxScratchLayout)
                          .getResult();
  }

  Value topkValsEmpty =
      buffers.values.isPlaced()
          ? rewriter
                .create<EmptyOp>(loc, buffers.values.type,
                                 buffers.values.vgmInverse,
                                 buffers.values.vgmForward)
                .getResult()
          : rewriter.create<EmptyOp>(loc, deviceShape, valTileType, metalLayout)
                .getResult();
  Value topkIdxEmpty =
      buffers.indices.isPlaced()
          ? rewriter
                .create<EmptyOp>(loc, buffers.indices.type,
                                 buffers.indices.vgmInverse,
                                 buffers.indices.vgmForward)
                .getResult()
          : rewriter.create<EmptyOp>(loc, deviceShape, idxTileType, metalLayout)
                .getResult();

  llvm::SmallVector<Value> topkInputs = {topkInput, idxScratchEmpty};
  llvm::SmallVector<Value> topkOutputs = {topkValsEmpty, topkIdxEmpty};
  // A constant map keeps the 1x1 scratch shard out of the generic's
  // shard-extent comparison against the full-shard operands.
  llvm::SmallVector<AffineExpr> idxConstExprs(
      physicalRank, rewriter.getAffineConstantExpr(0));
  AffineMap idxConstMap = AffineMap::get(physicalRank, 0, idxConstExprs, ctx);
  auto topkGeneric = rewriter.create<GenericOp>(
      loc, topkInputs, topkOutputs, /*additionalArgs=*/ValueRange(),
      rewriter.getAffineMapArrayAttr(llvm::SmallVector<AffineMap>{
          identityMap, idxConstMap, identityMap, identityMap}),
      rewriter.getArrayAttr(iteratorTypes), ThreadType::Unified,
      buffers.values.isPlaced() ? buffers.values.grid : nullptr);

  buildParallelGenericRegion(
      rewriter, loc, topkGeneric, topkInputs, topkOutputs,
      [&](ArrayRef<Value> blockArgs) -> llvm::SmallVector<Value> {
        auto topkBlock = rewriter.create<TopkBlockOp>(
            loc, blockArgs[0], blockArgs[1], blockArgs[2], blockArgs[3], k,
            reductionDimSize, /*stableSort=*/false, dim,
            /*generateIndices=*/true);
        return {topkBlock.getResultValues(), topkBlock.getResultIndices()};
      });

  return {topkGeneric->getResult(0), topkGeneric->getResult(1)};
}

} // namespace mlir::tt::d2m::utils
