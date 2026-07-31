// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Fusing/RingSDPAFusingPattern.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

namespace mlir::tt::ttnn::fusing {

// tt-metal's joint layout string. Wan passes "rear" on the self-attention path
// (models/tt_dit/models/transformers/wan2_2/attention_wan.py).
static constexpr llvm::StringLiteral kJointStrategy = "rear";

// Ring fabric tuning. tt-metal's own defaults are 1 and 8; Wan runs 5 and 32,
// which is the configuration the ring path has actually been exercised with.
static constexpr uint32_t kNumWorkersPerLink = 5;
static constexpr uint32_t kNumBuffersPerChannel = 32;

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

// Largest tile-aligned power of two in [32, 512] that divides `extent`.
static uint64_t chooseChunkSize(int64_t extent) {
  for (uint64_t chunk = 512; chunk > 32; chunk /= 2) {
    if (extent % static_cast<int64_t>(chunk) == 0) {
      return chunk;
    }
  }
  return 32;
}

SDPAProgramConfigAttr
RingSDPAFusing::buildProgramConfig(ScaledDotProductAttentionOp srcOp,
                                   int64_t localSeqLen,
                                   int64_t gatheredSeqLen) {
  MLIRContext *ctx = srcOp.getContext();

  // WorkerGrid shape is [Y, X]; CoreCoord is (x, y). Same derivation as
  // PagedScaledDotProductAttentionDecodeProgramConfigRewritePattern.
  ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(srcOp.getOperation());
  llvm::ArrayRef<int64_t> workerGridShape =
      deviceAttr.getWorkerGrid().getShape();
  auto grid = CoreCoordAttr::get(ctx, static_cast<uint32_t>(workerGridShape[1]),
                                 static_cast<uint32_t>(workerGridShape[0]));

  // tt-metal's only constraint on these is divisibility by TILE_WIDTH
  // (validate: TT_FATAL(q_chunk_size % TILE_WIDTH == 0), likewise for k), which
  // every value chooseChunkSize can return satisfies. So this is valid but
  // UNTUNED -- a performance question for bring-up, not a correctness one.
  return SDPAProgramConfigAttr::get(
      ctx, grid, /*sub_core_grids=*/nullptr,
      /*q_chunk_size=*/chooseChunkSize(localSeqLen),
      /*k_chunk_size=*/chooseChunkSize(gatheredSeqLen),
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

  // Peel the frontend's padding trim, if present, so the all-gather underneath
  // it is still matchable. Its length becomes logical_n further down.
  SliceStaticOp keySlice;
  SliceStaticOp valueSlice;
  Value gatheredKey = peelPaddingSlice(srcOp.getKey(), seqDim, keySlice);
  Value gatheredValue = peelPaddingSlice(srcOp.getValue(), seqDim, valueSlice);
  if (static_cast<bool>(keySlice) != static_cast<bool>(valueSlice)) {
    return rewriter.notifyMatchFailure(
        srcOp, "only one of key/value carries a padding slice");
  }
  if (keySlice && !slicesAgree(keySlice, valueSlice)) {
    return rewriter.notifyMatchFailure(srcOp,
                                       "key and value padding slices disagree");
  }

  auto keyGather = gatheredKey.getDefiningOp<AllGatherOp>();
  if (!keyGather || !keyGather->hasOneUse()) {
    return rewriter.notifyMatchFailure(
        srcOp, "key is not produced by a single-use all_gather");
  }
  AllGatherOp valueGather = matchPairedGather(gatheredValue, keyGather);
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
  if (keyGather.getAllGatherDim() != seqDim) {
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
  const int64_t localSeqLen = shardedKeyType.getShape()[seqDim];
  const int64_t gatheredSeqLen =
      keyGather.getResult().getType().getShape()[seqDim];
  if (queryType.getShape()[seqDim] != localSeqLen) {
    return rewriter.notifyMatchFailure(
        srcOp, "query sequence length does not match the pre-gather K/V");
  }

  // Remaining tt-metal validate() requirements that plain SDPA does not share.
  // TT_FATAL(NQH == NKH): no GQA on the ring path.
  if (queryType.getShape()[seqDim - 1] !=
      shardedKeyType.getShape()[seqDim - 1]) {
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
      keySlice
          ? mlir::cast<mlir::IntegerAttr>(keySlice.getEnds().getValue()[seqDim])
                .getInt()
          : gatheredSeqLen;
  if (gatheredSeqLen - logicalN >= localSeqLen) {
    return rewriter.notifyMatchFailure(
        srcOp, "padding slice would leave a device with only padded tokens");
  }

  SDPAProgramConfigAttr programConfig =
      buildProgramConfig(srcOp, localSeqLen, gatheredSeqLen);

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

  auto ringOp = rewriter.create<ExpRingJointScaledDotProductAttentionOp>(
      srcOp.getLoc(),
      /*result=*/srcOp.getResult().getType(), jointResultType, statsType,
      /*query=*/srcOp.getQuery(), key, /*value=*/valueGather.getInput(),
      /*joint_query=*/Value(), /*joint_key=*/Value(), /*joint_value=*/Value(),
      /*persistent_output_buffer_k=*/Value(),
      /*persistent_output_buffer_v=*/Value(),
      /*multi_device_global_semaphore=*/ValueRange(),
      /*joint_strategy=*/rewriter.getStringAttr(kJointStrategy),
      /*logical_n=*/rewriter.getI64IntegerAttr(logicalN),
      /*dim=*/rewriter.getSI32IntegerAttr(seqDim),
      /*cluster_axis=*/rewriter.getUI32IntegerAttr(clusterAxis), programConfig,
      /*num_links=*/keyGather.getNumLinksAttr(),
      /*topology=*/keyGather.getTopologyAttr(),
      /*sub_device_id=*/keyGather.getSubDeviceIdAttr(),
      /*scale=*/srcOp.getScaleAttr(),
      /*num_workers_per_link=*/rewriter.getUI32IntegerAttr(kNumWorkersPerLink),
      /*num_buffers_per_channel=*/
      rewriter.getUI32IntegerAttr(kNumBuffersPerChannel),
      /*compute_config=*/nullptr);

  // Only the attention output has users; the all-gathers were single-use and
  // die with the SDPA.
  rewriter.replaceOp(srcOp, ringOp.getResult());
  return success();
}

} // namespace mlir::tt::ttnn::fusing
