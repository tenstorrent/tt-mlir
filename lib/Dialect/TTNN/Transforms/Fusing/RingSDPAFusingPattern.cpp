// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Fusing/RingSDPAFusingPattern.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include <algorithm>
#include <utility>

namespace mlir::tt::ttnn::fusing {

// tt-metal's joint layout string. Wan passes "rear" on the self-attention path
// (models/tt_dit/models/transformers/wan2_2/attention_wan.py).
static constexpr llvm::StringLiteral kJointStrategy = "rear";

// Ring fabric tuning. tt-metal's own defaults are 1 and 8; Wan runs 5 and 32,
// which is the configuration the ring path has actually been exercised with.
static constexpr uint32_t kNumWorkersPerLink = 5;
static constexpr uint32_t kNumBuffersPerChannel = 32;

// The ring kernel asserts on exactly two links
// (exp_ring_joint_sdpa_device_operation.cpp:228), so this is a requirement
// rather than a tuning choice.
static constexpr uint32_t kNumLinks = 2;

// Metal uses exp_ring_joint only when TP=4 and SP=32. Every other SP>1 mesh
// (including Galaxy 8x4) uses the non-experimental ring_joint kernel.
static constexpr int64_t kExpRingSP = 32;
static constexpr int64_t kExpRingTP = 4;

// The head/sequence swap on a rank-4 tensor: [B, S, H, D] <-> [B, H, S, D].
// Self-inverse, so the same array serves for peeling and for re-applying.
static constexpr int64_t kHeadSeqSwap[] = {0, 2, 1, 3};

static Value createHeadSeqTranspose(mlir::PatternRewriter &rewriter,
                                    Location loc, Value input) {
  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  llvm::SmallVector<int64_t> outputShape = ttmlir::utils::applyPermutation(
      inputType.getShape(), llvm::ArrayRef<int64_t>(kHeadSeqSwap));
  RankedTensorType outputType =
      utils::RankedTensorTypeFactory::create(inputType, outputShape);
  return rewriter.create<PermuteOp>(
      loc, outputType, input,
      rewriter.getDenseI64ArrayAttr(llvm::ArrayRef<int64_t>(kHeadSeqSwap)),
      /*pad_value=*/mlir::FloatAttr());
}

Value RingSDPAFusing::skipLayoutLike(Value v) {
  while (Operation *op = v.getDefiningOp()) {
    if (!op->hasOneUse()) {
      break;
    }
    Value input;
    if (auto toLayout = dyn_cast<ToLayoutOp>(op)) {
      input = toLayout.getInput();
    } else if (auto spec = dyn_cast<ToTensorSpecOp>(op)) {
      input = spec.getInput();
    } else if (auto memCfg = dyn_cast<ToMemoryConfigOp>(op)) {
      input = memCfg.getInput();
    } else if (auto typecast = dyn_cast<TypecastOp>(op)) {
      input = typecast.getInput();
    } else {
      break;
    }
    auto inType = dyn_cast<RankedTensorType>(input.getType());
    auto outType = dyn_cast<RankedTensorType>(v.getType());
    if (!inType || !outType || inType.getShape() != outType.getShape()) {
      break;
    }
    v = input;
  }
  return v;
}

Value RingSDPAFusing::peelHeadSeqTranspose(Value v, PermuteOp &permute) {
  permute = nullptr;
  auto candidate = v.getDefiningOp<PermuteOp>();
  if (!candidate || !candidate->hasOneUse()) {
    return v;
  }
  if (candidate.getPermutation() != llvm::ArrayRef<int64_t>(kHeadSeqSwap)) {
    return v;
  }
  permute = candidate;
  return candidate.getInput();
}

AllGatherOp RingSDPAFusing::matchPairedGather(Value v, AllGatherOp keyGather) {
  auto gather = v.getDefiningOp<AllGatherOp>();
  if (!gather || !gather->hasOneUse()) {
    return nullptr;
  }
  // Both gathers must describe the same collective, or the pair does not
  // represent one ring and the attributes cannot be lifted onto a single op.
  if (gather.getClusterAxis() != keyGather.getClusterAxis() ||
      gather.getAllGatherDim() != keyGather.getAllGatherDim() ||
      gather.getNumLinks() != keyGather.getNumLinks() ||
      gather.getTopology() != keyGather.getTopology() ||
      gather.getSubDeviceId() != keyGather.getSubDeviceId()) {
    return nullptr;
  }
  return gather;
}

Value RingSDPAFusing::peelPaddingSlice(Value v, int64_t seqDim,
                                       SliceStaticOp &slice) {
  slice = nullptr;
  auto candidate = v.getDefiningOp<SliceStaticOp>();
  if (!candidate || !candidate->hasOneUse()) {
    return v;
  }

  RankedTensorType inputType = candidate.getInput().getType();
  const int64_t rank = inputType.getRank();
  llvm::ArrayRef<mlir::Attribute> begins = candidate.getBegins().getValue();
  llvm::ArrayRef<mlir::Attribute> ends = candidate.getEnds().getValue();
  llvm::ArrayRef<mlir::Attribute> step = candidate.getStep().getValue();
  if (static_cast<int64_t>(begins.size()) != rank ||
      static_cast<int64_t>(ends.size()) != rank ||
      static_cast<int64_t>(step.size()) != rank) {
    return v;
  }

  auto asInt = [](mlir::Attribute a) {
    return mlir::cast<mlir::IntegerAttr>(a).getInt();
  };
  for (int64_t d = 0; d < rank; ++d) {
    if (asInt(step[d]) != 1 || asInt(begins[d]) != 0) {
      return v;
    }
    // Every dim other than the sequence axis must be left whole; a slice that
    // trims heads or the head dim is not a padding trim.
    if (d != seqDim && asInt(ends[d]) != inputType.getShape()[d]) {
      return v;
    }
  }
  if (asInt(ends[seqDim]) > inputType.getShape()[seqDim]) {
    return v;
  }

  slice = candidate;
  return candidate.getInput();
}

bool RingSDPAFusing::slicesAgree(SliceStaticOp a, SliceStaticOp b) {
  return a.getBegins() == b.getBegins() && a.getEnds() == b.getEnds() &&
         a.getStep() == b.getStep();
}

// Metal Wan's ring SDPA chunk table
// (models/tt_dit/models/transformers/wan2_2/attention_wan.py). These are
// empirical L1-safe sizes, not a derived formula. Unlisted (arch, SP, TP)
// triples fall back to (256, 256).
static std::pair<uint64_t, uint64_t> lookupRingChunkSizes(bool isBlackhole,
                                                          int64_t spFactor,
                                                          int64_t tpFactor) {
  if (!isBlackhole && spFactor == 2 && tpFactor == 4) {
    return {256, 256};
  }
  if (!isBlackhole && spFactor == 8 && tpFactor == 4) {
    return {256, 256};
  }
  if (isBlackhole && spFactor == 2 && tpFactor == 2) {
    return {128, 512};
  }
  if (isBlackhole && spFactor == 8 && tpFactor == 4) {
    return {288, 512};
  }
  if (isBlackhole && spFactor == 32 && tpFactor == 4) {
    return {224, 512};
  }
  return {256, 256};
}

// Shrink `preferred` until it divides `extent` and stays tile-aligned. Lit
// shapes are often shorter than the table's Wan sequences.
static uint64_t fitChunkSize(int64_t extent, uint64_t preferred) {
  uint64_t chunk = std::min(preferred, static_cast<uint64_t>(extent));
  chunk -= chunk % ttnn::TILE_WIDTH;
  if (chunk < ttnn::TILE_WIDTH) {
    return ttnn::TILE_WIDTH;
  }
  while (chunk >= ttnn::TILE_WIDTH &&
         extent % static_cast<int64_t>(chunk) != 0) {
    chunk -= ttnn::TILE_WIDTH;
  }
  return chunk >= ttnn::TILE_WIDTH ? chunk : ttnn::TILE_WIDTH;
}

SDPAProgramConfigAttr
RingSDPAFusing::buildProgramConfig(ScaledDotProductAttentionOp srcOp,
                                   int64_t localSeqLen, int64_t gatheredSeqLen,
                                   int64_t spFactor, int64_t tpFactor,
                                   bool reserveCclColumn) {
  MLIRContext *ctx = srcOp.getContext();

  // ChipDesc.grid is compute_with_storage_grid_size stored as (Y, X).
  // WorkerGrid can be taller (Galaxy dumps as 9x8) and is the wrong source.
  ttcore::ChipDescAttr chip = ttcore::getOpChipDescAttr(srcOp.getOperation());
  llvm::ArrayRef<int64_t> chipGrid = chip.getGrid();
  uint32_t gridY = static_cast<uint32_t>(chipGrid[0]);
  uint32_t gridX = static_cast<uint32_t>(chipGrid[1]);
  // Metal Wan's non-exp ring_joint uses compute grid (full.x - 1, full.y) and
  // places CCL at (sdpa_grid.x, 0). Exp uses the full grid.
  if (reserveCclColumn && gridX >= 2) {
    --gridX;
  }
  auto grid = CoreCoordAttr::get(ctx, gridX, gridY);

  const bool isBlackhole =
      chip.getArch().getValue() == ttcore::Arch::Blackhole;
  auto [qPreferred, kPreferred] =
      lookupRingChunkSizes(isBlackhole, spFactor, tpFactor);

  return SDPAProgramConfigAttr::get(
      ctx, grid, /*sub_core_grids=*/nullptr,
      /*q_chunk_size=*/fitChunkSize(localSeqLen, qPreferred),
      /*k_chunk_size=*/fitChunkSize(gatheredSeqLen, kPreferred),
      /*exp_approx_mode=*/nullptr,
      /*max_cores_per_head_batch=*/std::nullopt);
}

mlir::LogicalResult
RingSDPAFusing::matchAndRewrite(ScaledDotProductAttentionOp srcOp,
                                mlir::PatternRewriter &rewriter) const {
  // The ring kernel folds the softmax incrementally as blocks arrive, so it
  // supports neither an explicit mask nor the causal/windowed/sink variants.
  if (srcOp.getAttentionMask() || srcOp.getAttentionSink() ||
      srcOp.getSlidingWindowSize() || srcOp.getIsCausal()) {
    return rewriter.notifyMatchFailure(
        srcOp, "ring SDPA supports only unmasked, non-causal attention");
  }

  RankedTensorType queryType = srcOp.getQuery().getType();
  const int64_t rank = queryType.getRank();
  if (rank != 4) {
    return rewriter.notifyMatchFailure(srcOp, "expected a rank-4 query");
  }
  const int64_t seqDim = rank - 2;

  // Peel a head/sequence transpose, if the collective ran in [B, S, H, D] and
  // was only swapped into SDPA's [B, H, S, D] at the last moment. Everything
  // below the peel is stated in the gather's layout, so the sequence and head
  // axes move with it.
  PermuteOp keyTranspose;
  PermuteOp valueTranspose;
  Value transposedKey =
      peelHeadSeqTranspose(skipLayoutLike(srcOp.getKey()), keyTranspose);
  Value transposedValue =
      peelHeadSeqTranspose(skipLayoutLike(srcOp.getValue()), valueTranspose);
  if (static_cast<bool>(keyTranspose) != static_cast<bool>(valueTranspose)) {
    return rewriter.notifyMatchFailure(
        srcOp, "only one of key/value carries a head/sequence transpose");
  }
  // kHeadSeqSwap exchanges dims 1 and 2, so under the peel the sequence axis is
  // at seqDim - 1 and the head axis at seqDim.
  const int64_t gatherSeqDim = keyTranspose ? seqDim - 1 : seqDim;
  const int64_t gatherHeadDim = keyTranspose ? seqDim : seqDim - 1;

  // Peel the frontend's padding trim, if present, so the all-gather underneath
  // it is still matchable. Its length becomes logical_n further down.
  SliceStaticOp keySlice;
  SliceStaticOp valueSlice;
  Value gatheredKey =
      peelPaddingSlice(skipLayoutLike(transposedKey), gatherSeqDim, keySlice);
  Value gatheredValue = peelPaddingSlice(skipLayoutLike(transposedValue),
                                         gatherSeqDim, valueSlice);
  if (static_cast<bool>(keySlice) != static_cast<bool>(valueSlice)) {
    return rewriter.notifyMatchFailure(
        srcOp, "only one of key/value carries a padding slice");
  }
  if (keySlice && !slicesAgree(keySlice, valueSlice)) {
    return rewriter.notifyMatchFailure(srcOp,
                                       "key and value padding slices disagree");
  }

  auto keyGather = skipLayoutLike(gatheredKey).getDefiningOp<AllGatherOp>();
  if (!keyGather || !keyGather->hasOneUse()) {
    return rewriter.notifyMatchFailure(
        srcOp, "key is not produced by a single-use all_gather");
  }
  AllGatherOp valueGather =
      matchPairedGather(skipLayoutLike(gatheredValue), keyGather);
  if (!valueGather) {
    return rewriter.notifyMatchFailure(
        srcOp, "value is not produced by a matching single-use all_gather");
  }
  if (keyGather == valueGather) {
    return rewriter.notifyMatchFailure(srcOp,
                                       "key and value share one all_gather");
  }

  // The gather must be on the sequence axis: that is what makes this a
  // sequence-parallel attention rather than some other collective that happens
  // to feed K/V.
  if (keyGather.getAllGatherDim() != gatherSeqDim) {
    return rewriter.notifyMatchFailure(
        srcOp, "all_gather is not on the sequence axis");
  }

  // A single-device ring is a no-op; leave the plain form alone.
  ttcore::DeviceOp deviceOp = ttcore::lookupDeviceOp(srcOp.getOperation());
  if (!deviceOp) {
    return rewriter.notifyMatchFailure(srcOp, "no device in scope");
  }
  llvm::SmallVector<int64_t> meshShape{deviceOp.getDeviceAttr().getMeshShape()};
  const uint32_t clusterAxis = keyGather.getClusterAxis();
  if (clusterAxis >= meshShape.size() || meshShape[clusterAxis] < 2) {
    return rewriter.notifyMatchFailure(
        srcOp, "cluster_axis spans fewer than 2 devices");
  }

  // Q must still be sharded on the sequence axis. The gathered K/V is the whole
  // sequence, so Q being shorter by exactly the ring size is the signature; a Q
  // that is already full length means the sequence is not SP-sharded here.
  Value key = keyGather.getInput();
  RankedTensorType shardedKeyType = mlir::cast<RankedTensorType>(key.getType());
  const int64_t localSeqLen = shardedKeyType.getShape()[gatherSeqDim];
  const int64_t gatheredSeqLen =
      keyGather.getResult().getType().getShape()[gatherSeqDim];
  if (queryType.getShape()[seqDim] != localSeqLen) {
    return rewriter.notifyMatchFailure(
        srcOp, "query sequence length does not match the pre-gather K/V");
  }

  // Remaining tt-metal validate() requirements that plain SDPA does not share.
  // TT_FATAL(NQH == NKH): no GQA on the ring path.
  if (queryType.getShape()[seqDim - 1] !=
      shardedKeyType.getShape()[gatherHeadDim]) {
    return rewriter.notifyMatchFailure(
        srcOp, "ring SDPA requires equal query and key/value head counts");
  }
  // TT_FATAL(N_local % TILE_HEIGHT == 0).
  if (localSeqLen % ttnn::TILE_HEIGHT != 0) {
    return rewriter.notifyMatchFailure(
        srcOp, "per-device sequence length is not tile-aligned");
  }
  // All inputs must share the query's dtype, and the kernel only accepts the
  // bf16 family. tt-mlir carries bfp8/bfp4 as bf16 element types with the tile
  // dtype in the layout, so this check admits those too.
  if (!queryType.getElementType().isBF16() ||
      shardedKeyType.getElementType() != queryType.getElementType() ||
      mlir::cast<RankedTensorType>(valueGather.getInput().getType())
              .getElementType() != queryType.getElementType()) {
    return rewriter.notifyMatchFailure(
        srcOp, "ring SDPA requires bf16 query/key/value of a single dtype");
  }

  // The absorbed slice, if any, is the true unpadded length; otherwise the
  // whole gathered sequence is real. tt-metal also requires the padding delta
  // to fit inside one shard, so a trim that would leave some device holding
  // only padding is rejected rather than silently widened.
  const int64_t logicalN =
      keySlice ? mlir::cast<mlir::IntegerAttr>(
                     keySlice.getEnds().getValue()[gatherSeqDim])
                     .getInt()
               : gatheredSeqLen;
  if (gatheredSeqLen - logicalN >= localSeqLen) {
    return rewriter.notifyMatchFailure(
        srcOp, "padding slice would leave a device with only padded tokens");
  }

  int64_t tpSize = 1;
  for (size_t i = 0; i < meshShape.size(); ++i) {
    if (i != clusterAxis) {
      tpSize *= meshShape[i];
    }
  }
  const bool useExpKernel =
      meshShape[clusterAxis] == kExpRingSP && tpSize == kExpRingTP;

  SDPAProgramConfigAttr programConfig =
      buildProgramConfig(srcOp, localSeqLen, gatheredSeqLen,
                         /*spFactor=*/meshShape[clusterAxis], tpSize,
                         /*reserveCclColumn=*/!useExpKernel);

  // The op takes K/V in the query's layout. When the transpose was peeled the
  // shards are still in the gather's layout, so re-apply the swap here -- on one
  // shard rather than on the gathered sequence, which is what the peel bought.
  Value ringKey = key;
  Value ringValue = valueGather.getInput();
  if (keyTranspose) {
    ringKey = createHeadSeqTranspose(rewriter, srcOp.getLoc(), ringKey);
    ringValue = createHeadSeqTranspose(rewriter, srcOp.getLoc(), ringValue);
  }

  // Result shapes follow tt-metal's
  // ExpRingJointSDPADeviceOperation::compute_output_specs exactly.
  //
  // With no joint inputs the joint output is the query shape with a zero
  // sequence extent, and stats is [B, H, padded_N * 2, 1] (the kernel's running
  // max in the first half, running sum in the second). N_local is tile-aligned
  // by the guard above, so padded_N == N_local.
  llvm::SmallVector<int64_t> jointShape{queryType.getShape()};
  jointShape[seqDim] = 0;
  RankedTensorType jointResultType =
      utils::RankedTensorTypeFactory::create(queryType, jointShape);

  llvm::SmallVector<int64_t> statsShape{queryType.getShape()};
  statsShape[seqDim] = localSeqLen * 2;
  statsShape.back() = 1;
  RankedTensorType statsType =
      utils::RankedTensorTypeFactory::create(queryType, statsShape);

  auto jointStrategy = rewriter.getStringAttr(kJointStrategy);
  auto logicalNAttr = rewriter.getI64IntegerAttr(logicalN);
  auto dimAttr = rewriter.getSI32IntegerAttr(seqDim);
  auto clusterAxisAttr = rewriter.getUI32IntegerAttr(clusterAxis);
  auto numLinksAttr = rewriter.getUI32IntegerAttr(kNumLinks);
  auto topologyAttr =
      ttcore::TopologyAttr::get(rewriter.getContext(), ttcore::Topology::Ring);
  auto workersAttr = rewriter.getUI32IntegerAttr(kNumWorkersPerLink);
  auto buffersAttr = rewriter.getUI32IntegerAttr(kNumBuffersPerChannel);

  Value result;
  if (useExpKernel) {
    auto ringOp = rewriter.create<ExpRingJointScaledDotProductAttentionOp>(
        srcOp.getLoc(), srcOp.getResult().getType(), jointResultType, statsType,
        srcOp.getQuery(), ringKey, ringValue, Value(), Value(), Value(),
        Value(), Value(), ValueRange(), jointStrategy, logicalNAttr, dimAttr,
        clusterAxisAttr, programConfig, numLinksAttr, topologyAttr,
        keyGather.getSubDeviceIdAttr(), srcOp.getScaleAttr(), workersAttr,
        buffersAttr, /*compute_config=*/nullptr);
    result = ringOp.getResult();
  } else {
    auto ringOp = rewriter.create<RingJointScaledDotProductAttentionOp>(
        srcOp.getLoc(), srcOp.getResult().getType(), jointResultType, statsType,
        srcOp.getQuery(), ringKey, ringValue, Value(), Value(), Value(),
        Value(), Value(), ValueRange(), jointStrategy, logicalNAttr, dimAttr,
        clusterAxisAttr, programConfig, numLinksAttr, topologyAttr,
        keyGather.getSubDeviceIdAttr(), srcOp.getScaleAttr(), workersAttr,
        buffersAttr, /*compute_config=*/nullptr);
    result = ringOp.getResult();
  }

  rewriter.replaceOp(srcOp, result);
  return success();
}

} // namespace mlir::tt::ttnn::fusing
