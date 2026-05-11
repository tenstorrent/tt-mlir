// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/MoeGptLayoutRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <numeric>

namespace mlir::tt::ttnn::workarounds::decomposition {

namespace {

// Mirror of moe_gpt_program_factory.cpp::ring_pos2bank_id: rank the DRAM
// banks by their owner worker core's coordinate (y desc, x desc).  The owner
// worker for each bank is chipDesc.dramBankToLogicalWorkerNoc0[bank_id].
// tt-metal sorts on translated NOC0 (physical) coords, but since translation
// is a uniform offset applied to every worker, sorting on logical coords
// yields the same order.
llvm::SmallVector<uint32_t>
computeMoeGptBankRingOrder(ttcore::ChipDescAttr chipDesc) {
  auto banks = chipDesc.getDramBankToLogicalWorkerNoc0();

  llvm::SmallVector<uint32_t> ring(banks.size());
  std::iota(ring.begin(), ring.end(), 0u);
  std::sort(ring.begin(), ring.end(), [&banks](uint32_t a, uint32_t b) {
    if (banks[a].getY() != banks[b].getY()) {
      return banks[a].getY() > banks[b].getY();
    }
    return banks[a].getX() > banks[b].getX();
  });
  return ring;
}

// Build a CoreRangeSetAttr placing one shard per DRAM bank, in ring order, in
// DRAM-coord space (bank_id, 0).  This is the placement the moe_gpt kernel
// expects.
CoreRangeSetAttr buildMoeGptBankCRS(MLIRContext *ctx,
                                    ArrayRef<uint32_t> ringOrder) {
  SmallVector<CoreRangeAttr> ranges;
  ranges.reserve(ringOrder.size());
  for (uint32_t bankId : ringOrder) {
    auto coord = CoreCoordAttr::get(ctx, bankId, 0);
    ranges.push_back(CoreRangeAttr::get(ctx, coord, coord));
  }
  return CoreRangeSetAttr::get(ctx, ranges);
}

// Per-shard (height, width) for a moe_gpt weight tensor.
//   w0_w1: (num_cores, L, E, groups, K_bias, 4*TILE);
//   w2:    (num_cores, L, E, 2,     N_bias, 4*TILE).
// Shard shape = (prod(shape[1..rank-2]), shape[-1]).
std::pair<int64_t, int64_t> moeGptWeightShardShape(ArrayRef<int64_t> shape) {
  int64_t shardH = 1;
  for (int i = 1; i + 1 < static_cast<int>(shape.size()); ++i) {
    shardH *= shape[i];
  }
  return {shardH, shape.back()};
}

// Build the DRAM HEIGHT_SHARDED MemoryConfigAttr for a moe_gpt weight, with
// the explicit ring-ordered bank CRS.
MemoryConfigAttr buildMoeGptWeightMemoryConfigAttr(MLIRContext *ctx,
                                                   ArrayRef<int64_t> shape,
                                                   CoreRangeSetAttr crs) {
  auto [shardH, shardW] = moeGptWeightShardShape(shape);
  auto shardShape = ShapeAttr::get(ctx, {shardH, shardW});
  auto orientation = ShardOrientationAttr::get(ctx, ShardOrientation::RowMajor);
  auto shardSpec = ShardSpecAttr::get(ctx, crs, shardShape, orientation);
  return MemoryConfigAttr::get(
      ctx, TensorMemoryLayoutAttr::get(ctx, TensorMemoryLayout::HeightSharded),
      BufferTypeAttr::get(ctx, BufferType::DRAM),
      std::optional<ShardSpecAttr>(shardSpec));
}

// Build the TTNNLayoutAttr encoding for a moe_gpt weight on device, carrying
// the same explicit ring-ordered CRS as memory_config so the ToDeviceOp
// result encoding matches the placement.
TTNNLayoutAttr buildMoeGptWeightEncoding(TTNNLayoutAttr seedEncoding,
                                         ArrayRef<int64_t> shape,
                                         CoreRangeSetAttr crs,
                                         int64_t numBanks) {
  return TTNNLayoutAttr::Builder(seedEncoding, shape)
      .setBufferType(BufferType::DRAM)
      .setMemoryLayout(TensorMemoryLayout::HeightSharded)
      .setGridShape({numBanks, 1})
      .setCoreRangeSet(crs)
      .build();
}

// Emit the MLIR equivalent of the runtime weight-prep pipeline:
//   from_device      : device bf16 RM DRAM INTERLEAVED → host bf16 RM
//   to_layout (TILE) : host bf16 RM → host bf16 TILE
//   typecast (bfp4)  : host bf16 TILE → host bfp4 TILE
//   to_device        : host bfp4 TILE → device bfp4 TILE DRAM HEIGHT_SHARDED,
//                       on the kernel's expected ring-ordered DRAM banks.
//
// The device→device reshard of a 6D bfloat16 tensor corrupts the data
// (tt-metal), hence the explicit host round-trip — mirrors what the runtime
// impl used to do in prepareMoeGptWeight.
Value buildPreparedWeight(PatternRewriter &rewriter, Operation *anchor,
                          Location loc, Value weight, CoreRangeSetAttr bankCRS,
                          int64_t numBanks) {
  auto weightType = mlir::cast<RankedTensorType>(weight.getType());
  auto *ctx = rewriter.getContext();

  // Step 1: from_device → host (SystemMemory).
  auto hostType = utils::RankedTensorTypeFactory::create(
      weightType, BufferType::SystemMemory);
  Value hostWeight = rewriter.create<FromDeviceOp>(loc, hostType, weight);

  // Step 2: to_layout TILE on host (no dtype, no memory_config).
  auto hostTiledType =
      utils::RankedTensorTypeFactory::create(hostType, Layout::Tile);
  Value hostTiled =
      rewriter.create<ToLayoutOp>(loc, hostTiledType, hostWeight, Layout::Tile,
                                  /*dtype=*/ttcore::DataTypeAttr(),
                                  /*memory_config=*/MemoryConfigAttr());

  // Step 3: typecast to bfp4 on host.
  auto hostBfp4Type = utils::RankedTensorTypeFactory::create(
      hostTiledType, ttcore::DataType::BFP_BFloat4);
  auto bfp4DtypeAttr =
      ttcore::DataTypeAttr::get(ctx, ttcore::DataType::BFP_BFloat4);
  Value hostBfp4 =
      rewriter.create<TypecastOp>(loc, hostBfp4Type, hostTiled, bfp4DtypeAttr);

  // Step 4: to_device with the explicit bank-permuted memory_config and a
  // matching result encoding (same CRS) so the two stay in sync.
  auto hostBfp4Encoding =
      mlir::cast<TTNNLayoutAttr>(hostBfp4Type.getEncoding());
  auto deviceEncoding = buildMoeGptWeightEncoding(
      hostBfp4Encoding, weightType.getShape(), bankCRS, numBanks);
  auto deviceType = RankedTensorType::get(
      weightType.getShape(), hostBfp4Type.getElementType(), deviceEncoding);
  auto memConfig =
      buildMoeGptWeightMemoryConfigAttr(ctx, weightType.getShape(), bankCRS);
  Value device = utils::getOrInsertDevice(rewriter, anchor);
  return rewriter.create<ToDeviceOp>(loc, deviceType, hostBfp4, device,
                                     memConfig);
}

} // namespace

LogicalResult
MoeGptLayoutRewritePattern::matchAndRewrite(MoeGptOp srcOp,
                                            PatternRewriter &rewriter) const {

  auto tilizeOutType =
      mlir::cast<RankedTensorType>(srcOp.getTilizeOut().getType());
  auto tilizeOutEncoding =
      mlir::cast<TTNNLayoutAttr>(tilizeOutType.getEncoding());

  // Idempotency: tilize_out becomes L1 HEIGHT_SHARDED after this pattern runs.
  // This also gates the weight prep below so we don't re-insert ToLayoutOp
  // chains on re-match.
  if (tilizeOutEncoding.getBufferType() == BufferType::L1 &&
      tilizeOutEncoding.getMemLayout() &&
      tilizeOutEncoding.getMemLayout().getValue() ==
          TensorMemoryLayout::HeightSharded) {
    return failure();
  }

  auto chipDesc = ttcore::getCurrentScopeSystemDesc(srcOp).getChipDescs()[0];
  auto physicalGrid = chipDesc.getGrid();
  int64_t numWorkerCores = physicalGrid[0] * physicalGrid[1];

  // Tilize outputs: L1 HEIGHT_SHARDED on a {numWorkerCores, 1} virtual grid.
  // The runtime validates the kernel's allocated memory config against the
  // compile-time encoding via insertTTNNTensorAndValidate; a default unit
  // grid would fail that check, so we set the grid explicitly. The operand-
  // workarounds framework cannot set a non-default grid, so we do it here.
  auto deviceAttr = ttcore::lookupDevice(srcOp.getOperation());

  auto buildHsEncoding = [&](TTNNLayoutAttr seed, ArrayRef<int64_t> shape) {
    return TTNNLayoutAttr::Builder(seed, shape)
        .setBufferType(BufferType::L1)
        .setMemoryLayout(TensorMemoryLayout::HeightSharded)
        .setGridShape({numWorkerCores, 1})
        .buildWithCanonicalCorePlacement(deviceAttr);
  };

  auto tilizeOutShape = tilizeOutType.getShape();
  auto newTilizeOutType =
      RankedTensorType::get(tilizeOutShape, tilizeOutType.getElementType(),
                            buildHsEncoding(tilizeOutEncoding, tilizeOutShape));

  auto tilizeOutRmType =
      mlir::cast<RankedTensorType>(srcOp.getTilizeOutRm().getType());
  auto tilizeOutRmEncoding =
      mlir::cast<TTNNLayoutAttr>(tilizeOutRmType.getEncoding());
  auto tilizeOutRmShape = tilizeOutRmType.getShape();
  auto newTilizeOutRmType = RankedTensorType::get(
      tilizeOutRmShape, tilizeOutRmType.getElementType(),
      buildHsEncoding(tilizeOutRmEncoding, tilizeOutRmShape));

  SmallVector<Type> newResultTypes(srcOp.getResultTypes());
  newResultTypes[3] = newTilizeOutType;
  newResultTypes[4] = newTilizeOutRmType;

  // Weight prep: convert bf16 DRAM INTERLEAVED → bfloat4_b TILE DRAM
  // HEIGHT_SHARDED at the kernel's expected DRAM-bank coords.  The bank CRS
  // is derived from the system descriptor (dramBankToLogicalWorkerNoc0 +
  // coordTranslationOffsets) and threaded through both the result-type
  // encoding and the MemoryConfigAttr — so the placement is expressed in IR
  // exactly once and the two cannot drift.
  auto ringOrder = computeMoeGptBankRingOrder(chipDesc);
  auto bankCRS = buildMoeGptBankCRS(rewriter.getContext(), ringOrder);
  int64_t numBanks = static_cast<int64_t>(ringOrder.size());

  Value newW0W1 =
      buildPreparedWeight(rewriter, srcOp.getOperation(), srcOp.getLoc(),
                          srcOp.getW0W1Tensor(), bankCRS, numBanks);
  Value newW2 =
      buildPreparedWeight(rewriter, srcOp.getOperation(), srcOp.getLoc(),
                          srcOp.getW2Tensor(), bankCRS, numBanks);

  auto newOp = rewriter.create<MoeGptOp>(
      srcOp.getLoc(), TypeRange(newResultTypes), srcOp.getInputTensor(),
      srcOp.getExpertIndices(), srcOp.getExpertScores(),
      srcOp.getExpertMapping(), newW0W1, newW2,
      srcOp.getOutputHeightShardDimAttr(), srcOp.getOutputWidthShardDimAttr(),
      srcOp.getHiddenSizeAttr(), srcOp.getClusterAxisAttr());

  rewriter.replaceOp(srcOp, newOp.getResults());
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
