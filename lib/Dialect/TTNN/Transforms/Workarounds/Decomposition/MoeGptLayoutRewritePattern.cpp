// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/MoeGptLayoutRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <cstdint>

namespace mlir::tt::ttnn::workarounds::decomposition {

namespace {

// WH N300 kernel ring order: sorted bank_ids that mirror
// moe_gpt_program_factory.cpp::ring_pos2bank_id — owner worker-core =
// device->get_optimal_dram_bank_to_logical_worker_assignment(
//     NOC::RISCV_0_default)[bank_id], sorted by (y desc, x desc).
// Hardcoded from a runtime dump. TODO: expose DRAM-bank → owner-worker in the
// system descriptor so this can be computed at compile time and generalized
// beyond WH N300.
constexpr std::array<uint32_t, 12> kWormholeMoeGptBankRingOrder = {
    5, 0, 7, 8, 11, 3, 10, 2, 9, 6, 4, 1};

// Builds a MemoryConfigAttr for MoeGpt weights: DRAM HEIGHT_SHARDED with an
// explicit CoreRangeSet over the WH DRAM-bank coords in kernel ring order.
// Each shard owns one DRAM bank (CoreCoord(bank_id, 0) in DRAM-coord space).
// The placement is not expressible via GridAttr affine maps, so we bypass the
// ShardSpecAttr(GridAttr) builder and construct the set directly.
MemoryConfigAttr
buildMoeGptWeightMemoryConfigAttr(MLIRContext *ctx,
                                  RankedTensorType weightType) {
  auto shape = weightType.getShape();
  // w0_w1: (num_cores, L, E, groups, K_bias, 4*TILE);
  // w2:    (num_cores, L, E, 2,     N_bias, 4*TILE).
  // Shard shape = (prod(shape[1..rank-2]), shape[-1]).
  int64_t shardH = 1;
  for (int i = 1; i + 1 < static_cast<int>(shape.size()); ++i) {
    shardH *= shape[i];
  }
  int64_t shardW = shape.back();

  SmallVector<CoreRangeAttr> ranges;
  ranges.reserve(kWormholeMoeGptBankRingOrder.size());
  for (uint32_t bankId : kWormholeMoeGptBankRingOrder) {
    auto coord = CoreCoordAttr::get(ctx, bankId, 0);
    ranges.push_back(CoreRangeAttr::get(ctx, coord, coord));
  }
  auto coreRangeSet = CoreRangeSetAttr::get(ctx, ranges);
  auto shardShape = ShapeAttr::get(ctx, {shardH, shardW});
  auto orientation = ShardOrientationAttr::get(ctx, ShardOrientation::RowMajor);
  auto shardSpec =
      ShardSpecAttr::get(ctx, coreRangeSet, shardShape, orientation);

  return MemoryConfigAttr::get(
      ctx, TensorMemoryLayoutAttr::get(ctx, TensorMemoryLayout::HeightSharded),
      BufferTypeAttr::get(ctx, BufferType::DRAM),
      std::optional<ShardSpecAttr>(shardSpec));
}

// Emit the MLIR equivalent of the runtime weight-prep pipeline:
//   from_device      : device bf16 RM DRAM INTERLEAVED → host bf16 RM
//   to_layout (TILE) : host bf16 RM → host bf16 TILE
//   typecast (bfp4)  : host bf16 TILE → host bfp4 TILE
//   to_device        : host bfp4 TILE → device bfp4 TILE DRAM HEIGHT_SHARDED,
//                       using memory_config with the explicit bank CoreRangeSet
//                       from buildMoeGptWeightMemoryConfigAttr.
//
// The device→device reshard of a 6D bfloat16 tensor corrupts the data
// (tt-metal), hence the explicit host round-trip — mirrors what the runtime
// impl used to do in prepareMoeGptWeight.
//
// Known debug-only side effect: the ToDeviceOp result's TTNNLayoutAttr
// encoding uses an affine-map-derived CoreRangeSet (identity), which cannot
// mirror the explicit permuted CoreRangeSet carried by memory_config. In
// debug builds (TT_RUNTIME_DEBUG=1) insertTTNNTensorAndValidate will fail on
// this output. Production builds are unaffected (validation is a no-op).
// Proper fix: extend TTNNLayoutAttr to carry an explicit CoreRangeSet.
Value buildPreparedWeight(PatternRewriter &rewriter, Operation *anchor,
                          Location loc, Value weight) {
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

  // Step 4: to_device with the explicit bank-permuted memory_config.
  // Result encoding uses the default (GridAttr-derived) CoreRangeSet; the
  // actual placement follows memory_config (see comment above).
  auto deviceType =
      utils::RankedTensorTypeFactory::create(hostBfp4Type, BufferType::DRAM);
  deviceType = utils::RankedTensorTypeFactory::create(
      deviceType, TensorMemoryLayout::HeightSharded);
  auto memConfig = buildMoeGptWeightMemoryConfigAttr(ctx, weightType);
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
  // compile-time encoding via insertTTNNTensorAndValidate; a {1,1} grid
  // (what withMemoryLayout would preserve) fails this check. The operand-
  // workarounds framework cannot set a non-default grid, so we do it here.
  auto [workerVirtToPhys, workerPhysToVirt] =
      optimizer_utils::createSingleDeviceVirtualToPhysicalAffineMaps(
          srcOp.getContext(), TensorMemoryLayout::HeightSharded,
          {physicalGrid[0], physicalGrid[1]});
  auto heightShardedGrid =
      ttcore::GridAttr::get(srcOp.getContext(), {numWorkerCores, 1},
                            workerVirtToPhys, workerPhysToVirt);

  auto buildHsEncoding = [&](TTNNLayoutAttr encoding, ArrayRef<int64_t> shape) {
    return encoding.withBufferType(BufferType::L1)
        .withMemoryLayout(TensorMemoryLayout::HeightSharded)
        .withGrid(shape, heightShardedGrid);
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
  // HEIGHT_SHARDED at the kernel's expected DRAM-bank coords via a
  // ToLayoutOp carrying an explicit MemoryConfigAttr. The MemoryConfigAttr's
  // CoreRangeSet encodes the permuted bank placement (see
  // buildMoeGptWeightMemoryConfigAttr); the ToLayoutOp's result-type encoding
  // cannot mirror that set because TTNNLayoutAttr/GridAttr only support
  // affine-map-derived ranges. Known side effect: in debug builds
  // (TT_RUNTIME_DEBUG=1), insertTTNNTensorAndValidate will fail on the
  // ToLayoutOp output due to the CoreRangeSet mismatch; production builds
  // are unaffected (validation is a no-op). Proper fix: extend TTNNLayoutAttr
  // to carry an explicit CoreRangeSet.
  Value newW0W1 = buildPreparedWeight(rewriter, srcOp.getOperation(),
                                      srcOp.getLoc(), srcOp.getW0W1Tensor());
  Value newW2 = buildPreparedWeight(rewriter, srcOp.getOperation(),
                                    srcOp.getLoc(), srcOp.getW2Tensor());

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
