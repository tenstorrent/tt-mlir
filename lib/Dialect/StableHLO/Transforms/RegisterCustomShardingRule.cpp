// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/StableHLOUtils.h"

#include "llvm/Support/Error.h"
#include <shardy/dialect/sdy/ir/enums.h>

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_REGISTERCUSTOMSHARDINGRULEPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

static constexpr llvm::StringLiteral sdpaTargetName =
    "tt.scaled_dot_product_attention";

static constexpr llvm::StringLiteral pagedSdpaDecodeTargetName =
    "tt.paged_scaled_dot_product_attention_decode";

static constexpr llvm::StringLiteral chunkedSdpaTargetName =
    "tt.chunked_scaled_dot_product_attention";

static constexpr llvm::StringLiteral pagedUpdateCacheTargetName =
    "tt.paged_update_cache";

static constexpr llvm::StringLiteral pagedFillCacheTargetName =
    "tt.paged_fill_cache";

static constexpr llvm::StringLiteral pagedFlashMlaDecodeTargetName =
    "tt.paged_flash_mla_decode";

static constexpr llvm::StringLiteral sparseMatmulTargetName =
    "tt.sparse_matmul";

static constexpr llvm::StringLiteral allToAllDispatchTargetName =
    "tt.all_to_all_dispatch";

static constexpr llvm::StringLiteral allToAllCombineTargetName =
    "tt.all_to_all_combine";

static constexpr llvm::StringLiteral moeExpertTokenRemapTargetName =
    "tt.moe_expert_token_remap";

static constexpr llvm::StringLiteral flashMlaPrefillTargetName =
    "tt.flash_mla_prefill";

static mlir::sdy::OpShardingRuleAttr
getScatterShardingRule(mlir::stablehlo::ScatterOp scatterOp) {
  mlir::Operation::operand_range inputs = scatterOp.getInputs();
  mlir::Operation::operand_range updates = scatterOp.getUpdates();
  mlir::Value indices = scatterOp.getScatterIndices();

  if (!llvm::hasSingleElement(inputs) || !llvm::hasSingleElement(updates)) {
    scatterOp->emitError(
        "Scatter operation has multiple input or update tensors. This is not "
        "supported.");
    return mlir::sdy::OpShardingRuleAttr();
  }

  RankedTensorType inputType =
      llvm::dyn_cast<RankedTensorType>(inputs.front().getType());
  RankedTensorType updateType =
      llvm::dyn_cast<RankedTensorType>(updates.front().getType());
  RankedTensorType indicesType =
      llvm::dyn_cast<RankedTensorType>(indices.getType());

  if (!inputType || !updateType || !indicesType) {
    scatterOp->emitError(
        "Scatter operation has unranked tensor types. This is not supported.");
    return mlir::sdy::OpShardingRuleAttr();
  }

  const int64_t inputRank = inputType.getRank();
  const int64_t indicesRank = indicesType.getRank();
  const int64_t updateRank = updateType.getRank();

  auto dimNums = scatterOp.getScatterDimensionNumbers();
  ArrayRef<int64_t> scatterDimsToOperandDims =
      dimNums.getScatterDimsToOperandDims();
  ArrayRef<int64_t> insertedWindowDims = dimNums.getInsertedWindowDims();

  // Return null if input is not sharded along scatter_dims_to_operand_dims.
  // This would fallback to default sharding rule in
  // op_sharding_rule_registry.cc.
  bool isShardedAlongScatterDims = false;
  if (sdy::TensorShardingAttr inputSharding =
          mlir::sdy::getSharding(inputs.front())) {
    ArrayRef<mlir::sdy::DimensionShardingAttr> dimShardings =
        inputSharding.getDimShardings();
    for (int64_t scatterDim : scatterDimsToOperandDims) {
      if (scatterDim < static_cast<int64_t>(dimShardings.size()) &&
          !dimShardings[scatterDim].getAxes().empty()) {
        isShardedAlongScatterDims = true;
        break;
      }
    }
  }

  if (!isShardedAlongScatterDims) {
    // Return null to fallback to default sharding rule.
    return mlir::sdy::OpShardingRuleAttr();
  }

  // Check input and update have the same rank.
  if (inputRank != updateRank) {
    scatterOp->emitError(
        "Custom sharding rule not implemented for scatter "
        "operations where input and update tensors have different ranks.");
    return mlir::sdy::OpShardingRuleAttr();
  }

  sdy::OpShardingRuleBuilder builder(scatterOp);

  // Input, updates, and result need to be sharded together.
  // Replicate if dimension is in scatter_dims_to_operand_dims.
  // Replicate if dimension is in inserted_window_dims.
  // Shard otherwise.
  for (int64_t inputDim = 0; inputDim < inputRank; inputDim++) {
    bool isScatterDimsToOperandDim =
        llvm::is_contained(scatterDimsToOperandDims, inputDim);
    bool isInsertedWindowDim = llvm::is_contained(insertedWindowDims, inputDim);

    if (isScatterDimsToOperandDim || isInsertedWindowDim) {
      // Dimension is in scatter_dims_to_operand_dims or inserted_window_dims -
      // MUST REPLICATE.
      builder.addFactor({inputDim, sdy::kNullDim,
                         inputDim}, // [input_dim, indices_dim, updates_dim]
                        {inputDim}, // result_dim
                        inputType.getDimSize(inputDim),
                        mlir::sdy::FactorType::kNeedReplication);
    } else {
      // Dimension is NOT in scatter_dims_to_operand_dims or
      // inserted_window_dims - CAN SHARD. Shard input, updates, and result
      // together.
      builder.addFactor({inputDim, sdy::kNullDim,
                         inputDim}, // [input_dim, indices_dim, updates_dim]
                        {inputDim}, // result_dim
                        inputType.getDimSize(inputDim),
                        sdy::FactorType::kPassThrough // Can be sharded.
      );
    }
  }

  // Replicate all scatter_indices dimensions.
  for (int64_t indicesDim = 0; indicesDim < indicesRank; indicesDim++) {
    builder.addFactor({sdy::kNullDim, indicesDim,
                       sdy::kNullDim}, // [input_dim, indices_dim, updates_dim]
                      {sdy::kNullDim}, // Doesn't appear in result.
                      indicesType.getDimSize(indicesDim),
                      sdy::FactorType::kNeedReplication);
  }

  return builder.build();
}

static mlir::sdy::OpShardingRuleAttr
getSDPAShardingRule(mlir::stablehlo::CustomCallOp op) {
  if (op.getNumOperands() < 3 || op.getNumResults() != 1) {
    op.getOperation()->emitWarning()
        << "SDPA expects at least 3 operands (Q, K, V) and 1 result";
    return mlir::sdy::OpShardingRuleAttr();
  }

  auto qType = llvm::dyn_cast<RankedTensorType>(op.getOperand(0).getType());
  auto kType = llvm::dyn_cast<RankedTensorType>(op.getOperand(1).getType());
  auto vType = llvm::dyn_cast<RankedTensorType>(op.getOperand(2).getType());
  auto outType = llvm::dyn_cast<RankedTensorType>(op.getResult(0).getType());

  if (!qType || !kType || !vType || !outType) {
    op.getOperation()->emitWarning() << "SDPA requires ranked tensor types";
    return mlir::sdy::OpShardingRuleAttr();
  }

  // SDPA operates on 4D tensors: [B, H, S, D]
  if (qType.getRank() != 4 || kType.getRank() != 4 || vType.getRank() != 4 ||
      outType.getRank() != 4) {
    op.getOperation()->emitWarning() << "SDPA requires 4D tensors [B, H, S, D]";
    return mlir::sdy::OpShardingRuleBuilder::buildPointwise(op);
  }

  ArrayRef<int64_t> qShape = qType.getShape();
  ArrayRef<int64_t> kShape = kType.getShape();
  ArrayRef<int64_t> vShape = vType.getShape();
  ArrayRef<int64_t> outShape = outType.getShape();

  // Grouped-Query / Multi-Query Attention support:
  //   Q/Out: [B, num_q_heads,  S, D]
  //   K/V:   [B, num_kv_heads, S, D]
  // Constraints:
  //   - Q and Output must have the same shape
  //   - K and V must have the same shape
  //   - B, S, D must match across all tensors
  //   - num_q_heads must be divisible by num_kv_heads (GQA group ratio)
  if (kShape != vShape || qShape != outShape || qShape[0] != kShape[0] || // B
      qShape[2] != kShape[2] ||                                           // S
      qShape[3] != kShape[3]) {                                           // D
    op.getOperation()->emitWarning()
        << "SDPA shape validation failed: incompatible Q/K/V/Out dimensions";
    return mlir::sdy::OpShardingRuleAttr();
  }

  int64_t qHeads = qShape[1];
  int64_t kvHeads = kShape[1];

  // Supported SDPA head layouts:
  //   - MHA: qHeads == kvHeads
  //   - GQA: qHeads > kvHeads and qHeads % kvHeads == 0
  //   - MQA: kvHeads == 1 (special case of GQA)
  //
  // Branch explicitly between the identical-head MHA case and the grouped-head
  // GQA/MQA case for clarity.

  // For standard MHA (qHeads == kvHeads) with no extra operands, all tensors
  // have identical shapes so we can use the efficient addPointwise builder.
  // When an attention_mask is present its shape differs from Q (e.g.
  // [1, 1, S, S] vs [B, H, S, D]), so addPointwise cannot be used; fall through
  // to the explicit per-operand builder below, which leaves the mask unsharded.
  if (qHeads == kvHeads && op.getNumOperands() == 3) {
    auto getFactorType = [](int64_t dim) -> mlir::sdy::FactorType {
      if (dim == 0 || dim == 1) {
        return mlir::sdy::FactorType::kPassThrough;
      }
      return mlir::sdy::FactorType::kNeedReplication;
    };
    return mlir::sdy::OpShardingRuleBuilder(op)
        .addPointwise(qShape, getFactorType)
        .build();
  }

  auto isStaticPositiveDim = [](int64_t dim) {
    return !ShapedType::isDynamic(dim) && dim > 0;
  };
  if (!isStaticPositiveDim(qShape[0]) || !isStaticPositiveDim(qHeads) ||
      !isStaticPositiveDim(kvHeads) || !isStaticPositiveDim(qShape[2]) ||
      !isStaticPositiveDim(qShape[3])) {
    op.getOperation()->emitWarning()
        << "SDPA GQA/MQA path requires static, positive B/H/S/D dimensions";
    return mlir::sdy::OpShardingRuleAttr();
  }

  if (qHeads % kvHeads != 0) {
    op.getOperation()->emitWarning()
        << "SDPA: num_q_heads (" << qHeads
        << ") must be divisible by num_kv_heads (" << kvHeads << ")";
    return mlir::sdy::OpShardingRuleAttr();
  }

  // GQA/MQA path: Q/Out have more heads than K/V, so we cannot use
  // addPointwise (requires identical shapes), so build the B/H/S/D factors
  // explicitly.
  //
  // Dimension assignment:
  //   dim 0 -> Batch         (kPassThrough)
  //   dim 1 -> Head          (kPassThrough via explicit head factor)
  //   dim 2 -> Sequence len  (kNeedReplication)
  //   dim 3 -> Hidden size   (kNeedReplication)
  //
  // Head dimension: single factor with size = qHeads linking Q/K/V/Out dim 1.
  // For GQA, K/V dim 1 (kvHeads) differs from the factor size (qHeads), but
  // Shardy handles this proportionally: sharding by F gives qHeads/F Q-heads
  // and kvHeads/F KV-heads, preserving the GQA ratio.
  // This is the same pattern used by paged SDPA decode.
  int64_t numOperands = op.getNumOperands();

  sdy::OpShardingRuleBuilder builder(op);

  // Helper: build operand dim vector with Q/K/V dims set, rest kNullDim.
  auto makeOpDims = [&](int64_t qDim, int64_t kDim,
                        int64_t vDim) -> SmallVector<int64_t> {
    SmallVector<int64_t> dims(numOperands, sdy::kNullDim);
    dims[0] = qDim;
    dims[1] = kDim;
    dims[2] = vDim;
    return dims;
  };

  // Batch (dim 0): kPassThrough — can be sharded freely.
  builder.addFactor(makeOpDims(0, 0, 0), {0}, qShape[0],
                    sdy::FactorType::kPassThrough);

  // Head (dim 1): kPassThrough — can be sharded.
  // Single factor links Q/K/V/Out head dims together using Q's head count.
  builder.addFactor(makeOpDims(1, 1, 1), {1}, qHeads,
                    sdy::FactorType::kPassThrough);

  // Sequence length (dim 2): kNeedReplication — cannot shard without
  // distributed attention support.
  builder.addFactor(makeOpDims(2, 2, 2), {2}, qShape[2],
                    sdy::FactorType::kNeedReplication);

  // Hidden size (dim 3): kNeedReplication — cannot shard.
  builder.addFactor(makeOpDims(3, 3, 3), {3}, qShape[3],
                    sdy::FactorType::kNeedReplication);

  return builder.build();
}

// Sharding rule for the `tt.flash_mla_prefill` custom_call.
//
// Tensor layout (matches the StableHLO conversion at
// StableHLOToTTIRPatterns.cpp:8618):
//   Q   : [B, Hq,  S, dh_qk]      required
//   K   : [B, Hkv, S, dh_qk]      required, Hq % Hkv == 0
//   V   : [B, Hkv, S, head_dim_v] optional (`has_value`)
//   mask: [1|B, 1, S, S]          optional (`has_attention_mask`)
//   Out : [B, Hq,  S, head_dim_v]
//
// Factor design (mirrors SDPA prefill but with separate factors for the
// asymmetric Q/K vs V/Out head dims that MLA requires):
//   - Batch    (kPassThrough,    size B)        : Q/K/V/Out dim 0; mask dim 0
//                                                 only if mask.shape[0] == B.
//   - Heads    (kPassThrough,    size qHeads)   : Q/Out dim 1, K/V dim 1.
//                                                 Shardy handles the GQA ratio
//                                                 proportionally (same trick
//                                                 as getSDPAShardingRule).
//   - Sequence (kNeedReplication, size S)       : Q/K/V/Out dim 2, mask dim 2.
//   - dh_qk    (kNeedReplication, size dh_qk)   : Q/K dim 3 only.
//   - head_dim_v (kNeedReplication, size hdv)   : V/Out dim 3 only.
//   - Mask key (kNeedReplication, size S)       : mask dim 3 only.
static mlir::sdy::OpShardingRuleAttr
getFlashMlaPrefillShardingRule(mlir::stablehlo::CustomCallOp op) {
  // Recover has_value / has_attention_mask from mhlo.frontend_attributes so we
  // can map the variable-length operand list to roles.
  mlir::DictionaryAttr frontendAttrs =
      mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
          op->getDiscardableAttr("mhlo.frontend_attributes"));

  auto readBool = [&](llvm::StringRef key) -> bool {
    if (!frontendAttrs) {
      return false;
    }
    if (auto s = frontendAttrs.getAs<mlir::StringAttr>(key)) {
      return s.getValue().equals_insensitive("true");
    }
    return false;
  };
  bool hasValue = readBool("has_value");
  bool hasAttentionMask = readBool("has_attention_mask");

  int64_t expectedNumOperands =
      2 + (hasValue ? 1 : 0) + (hasAttentionMask ? 1 : 0);
  if (static_cast<int64_t>(op.getNumOperands()) != expectedNumOperands ||
      op.getNumResults() != 1) {
    op.getOperation()->emitWarning()
        << "flash_mla_prefill operand count does not match has_value / "
           "has_attention_mask flags";
    return mlir::sdy::OpShardingRuleAttr();
  }

  // Operand index layout: Q=0, K=1, then V and/or mask in declaration order.
  int64_t qIdx = 0;
  int64_t kIdx = 1;
  int64_t vIdx = hasValue ? 2 : sdy::kNullDim;
  int64_t mIdx = hasAttentionMask ? (hasValue ? 3 : 2) : sdy::kNullDim;

  auto qType = llvm::dyn_cast<RankedTensorType>(op.getOperand(qIdx).getType());
  auto kType = llvm::dyn_cast<RankedTensorType>(op.getOperand(kIdx).getType());
  auto outType = llvm::dyn_cast<RankedTensorType>(op.getResult(0).getType());
  RankedTensorType vType, mType;
  if (hasValue) {
    vType = llvm::dyn_cast<RankedTensorType>(op.getOperand(vIdx).getType());
  }
  if (hasAttentionMask) {
    mType = llvm::dyn_cast<RankedTensorType>(op.getOperand(mIdx).getType());
  }

  if (!qType || !kType || !outType || (hasValue && !vType) ||
      (hasAttentionMask && !mType)) {
    op.getOperation()->emitWarning()
        << "flash_mla_prefill requires ranked tensor types";
    return mlir::sdy::OpShardingRuleAttr();
  }

  if (qType.getRank() != 4 || kType.getRank() != 4 || outType.getRank() != 4 ||
      (hasValue && vType.getRank() != 4) ||
      (hasAttentionMask && mType.getRank() != 4)) {
    op.getOperation()->emitWarning() << "flash_mla_prefill requires 4D tensors";
    return mlir::sdy::OpShardingRuleAttr();
  }

  ArrayRef<int64_t> qShape = qType.getShape();
  ArrayRef<int64_t> kShape = kType.getShape();
  ArrayRef<int64_t> outShape = outType.getShape();

  int64_t B = qShape[0];
  int64_t qHeads = qShape[1];
  int64_t kvHeads = kShape[1];
  int64_t S = qShape[2];
  int64_t dhQK = qShape[3];
  int64_t headDimV = outShape[3];

  auto isStaticPositiveDim = [](int64_t dim) {
    return !ShapedType::isDynamic(dim) && dim > 0;
  };
  if (!isStaticPositiveDim(B) || !isStaticPositiveDim(qHeads) ||
      !isStaticPositiveDim(kvHeads) || !isStaticPositiveDim(S) ||
      !isStaticPositiveDim(dhQK) || !isStaticPositiveDim(headDimV)) {
    op.getOperation()->emitWarning()
        << "flash_mla_prefill requires static, positive dimensions";
    return mlir::sdy::OpShardingRuleAttr();
  }

  // Cross-operand shape consistency: B/S match across all tensors; Q/K share
  // dh_qk; Q/Out share Hq; K/V share Hkv and S; V's dim 3 == head_dim_v.
  if (kShape[0] != B || outShape[0] != B || kShape[2] != S ||
      outShape[2] != S || kShape[3] != dhQK || outShape[1] != qHeads) {
    op.getOperation()->emitWarning()
        << "flash_mla_prefill shape validation failed (B/S/Hq/dh_qk mismatch)";
    return mlir::sdy::OpShardingRuleAttr();
  }
  if (hasValue) {
    ArrayRef<int64_t> vShape = vType.getShape();
    if (vShape[0] != B || vShape[1] != kvHeads || vShape[2] != S ||
        vShape[3] != headDimV) {
      op.getOperation()->emitWarning()
          << "flash_mla_prefill V shape inconsistent with K/Out";
      return mlir::sdy::OpShardingRuleAttr();
    }
  }
  if (qHeads % kvHeads != 0) {
    op.getOperation()->emitWarning()
        << "flash_mla_prefill: num_q_heads (" << qHeads
        << ") must be divisible by num_kv_heads (" << kvHeads << ")";
    return mlir::sdy::OpShardingRuleAttr();
  }

  int64_t maskBatch = sdy::kNullDim;
  if (hasAttentionMask) {
    ArrayRef<int64_t> mShape = mType.getShape();
    // Mask shape: [1|B, 1, S, S]. Heads dim is always 1.
    if (!isStaticPositiveDim(mShape[0]) || mShape[1] != 1 || mShape[2] != S ||
        mShape[3] != S) {
      op.getOperation()->emitWarning()
          << "flash_mla_prefill mask must be [1|B, 1, S, S]";
      return mlir::sdy::OpShardingRuleAttr();
    }
    if (mShape[0] == B) {
      maskBatch = 0; // mask participates in Batch factor.
    } else if (mShape[0] != 1) {
      op.getOperation()->emitWarning()
          << "flash_mla_prefill mask batch dim must be 1 or " << B;
      return mlir::sdy::OpShardingRuleAttr();
    }
  }

  int64_t numOperands = op.getNumOperands();
  sdy::OpShardingRuleBuilder builder(op);

  // Helper: build operand-dim vector with the named operands set, rest
  // kNullDim.
  auto makeOpDims = [&](int64_t qDim, int64_t kDim, int64_t vDim,
                        int64_t maskDim) -> SmallVector<int64_t> {
    SmallVector<int64_t> dims(numOperands, sdy::kNullDim);
    dims[qIdx] = qDim;
    dims[kIdx] = kDim;
    if (hasValue) {
      dims[vIdx] = vDim;
    }
    if (hasAttentionMask) {
      dims[mIdx] = maskDim;
    }
    return dims;
  };

  // Batch (dim 0): kPassThrough.
  builder.addFactor(makeOpDims(0, 0, 0, maskBatch), {0}, B,
                    sdy::FactorType::kPassThrough);

  // Heads (dim 1): kPassThrough, factor size qHeads. Mask heads are always 1
  // so the mask sits out of this factor (kNullDim).
  // When MLA's compressed latent K/V is a single shared head (kvHeads == 1),
  // it is broadcast across every query head and must stay replicated when the
  // query heads are sharded.
  int64_t kvHeadDim = (kvHeads == 1) ? sdy::kNullDim : 1;
  builder.addFactor(makeOpDims(1, kvHeadDim, kvHeadDim, sdy::kNullDim), {1},
                    qHeads, sdy::FactorType::kPassThrough);

  // Sequence (dim 2): kNeedReplication, shared across Q/K/V/Out/mask.
  builder.addFactor(makeOpDims(2, 2, 2, 2), {2}, S,
                    sdy::FactorType::kNeedReplication);

  // dh_qk (dim 3 on Q/K only): kNeedReplication.
  builder.addFactor(makeOpDims(3, 3, sdy::kNullDim, sdy::kNullDim),
                    {sdy::kNullDim}, dhQK, sdy::FactorType::kNeedReplication);

  // head_dim_v (dim 3 on V/Out only): kNeedReplication.
  builder.addFactor(makeOpDims(sdy::kNullDim, sdy::kNullDim, 3, sdy::kNullDim),
                    {3}, headDimV, sdy::FactorType::kNeedReplication);

  // Mask key sequence (dim 3 on mask only): kNeedReplication.
  if (hasAttentionMask) {
    SmallVector<int64_t> maskDims(numOperands, sdy::kNullDim);
    maskDims[mIdx] = 3;
    builder.addFactor(maskDims, {sdy::kNullDim}, S,
                      sdy::FactorType::kNeedReplication);
  }

  return builder.build();
}

static mlir::sdy::OpShardingRuleAttr buildHeadShardedCustomCallRule(
    mlir::stablehlo::CustomCallOp op, llvm::ArrayRef<int64_t> operandHeadDims,
    llvm::ArrayRef<int64_t> resultHeadDims, int64_t headSize) {
  assert(static_cast<int64_t>(operandHeadDims.size()) == op.getNumOperands() &&
         "operandHeadDims size must match number of operands");
  assert(static_cast<int64_t>(resultHeadDims.size()) == op.getNumResults() &&
         "resultHeadDims size must match number of results");

  mlir::sdy::OpShardingRuleBuilder builder(op);

  SmallVector<int64_t> resolvedOperandDims(operandHeadDims.begin(),
                                           operandHeadDims.end());
  SmallVector<int64_t> resolvedResultDims(resultHeadDims.begin(),
                                          resultHeadDims.end());

  builder.addFactor(resolvedOperandDims, resolvedResultDims, headSize,
                    mlir::sdy::FactorType::kPassThrough);
  return builder.build();
}

// Dispatch function for paged attention CustomCall sharding rules.
static mlir::sdy::OpShardingRuleAttr
getChunkedSdpaShardingRule(mlir::stablehlo::CustomCallOp op) {
  // Chunked prefill SDPA over paged K/V:
  //  0: query  [num_users, num_heads, chunk_len, head_size]
  //  1: key    [num_blocks_total, num_kv_heads, block_size, head_size]
  //  2: value  [num_blocks_total, num_kv_heads, block_size, head_size]
  //  3: page_table, 4: chunk_start_idx (null-shardable)
  auto queryType = llvm::cast<RankedTensorType>(op.getOperand(0).getType());
  auto keyType = llvm::cast<RankedTensorType>(op.getOperand(1).getType());
  auto valueType = llvm::cast<RankedTensorType>(op.getOperand(2).getType());
  auto outputType = llvm::cast<RankedTensorType>(op.getResult(0).getType());

  if (queryType.getShape() != outputType.getShape()) {
    op.getOperation()->emitWarning()
        << "Chunked SDPA: query and output shapes must match.";
    return mlir::sdy::OpShardingRuleAttr();
  }

  llvm::SmallVector<RankedTensorType> qkvTypes = {queryType, keyType,
                                                  valueType};
  if (llvm::any_of(qkvTypes, [&](RankedTensorType type) {
        return type.getRank() != 4;
      })) {
    op.getOperation()->emitWarning()
        << "Chunked SDPA: unexpected Q/K/V layouts, q rank: "
        << queryType.getRank() << ", key rank: " << keyType.getRank()
        << ", value rank: " << valueType.getRank();
    return mlir::sdy::OpShardingRuleAttr();
  }

  // Query [U, H, chunk_len, D], K/V [B, H, S, D], and output all carry the
  // head dim at index 1.
  const int64_t headDim = 1;

  int64_t headSize = queryType.getShape()[headDim];

  SmallVector<int64_t> operandHeadDims(op.getNumOperands(),
                                       mlir::sdy::kNullDim);
  SmallVector<int64_t> resultHeadDims(op.getNumResults(), mlir::sdy::kNullDim);

  operandHeadDims[0] = headDim; // query
  operandHeadDims[1] = headDim; // key
  operandHeadDims[2] = headDim; // value
  resultHeadDims[0] = headDim;  // output

  return buildHeadShardedCustomCallRule(op, operandHeadDims, resultHeadDims,
                                        headSize);
}

static mlir::sdy::OpShardingRuleAttr
getPagedAttentionShardingRule(mlir::stablehlo::CustomCallOp op) {
  llvm::StringRef target = op.getCallTargetName();

  if (target == pagedSdpaDecodeTargetName) {
    // Paged SDPA decode
    //  0: query  [1, num_users, num_heads, head_size]
    //  1: key    [num_blocks_total, num_heads, block_size, head_size]
    //  2: value  [num_blocks_total, num_heads, block_size, head_size]
    //  3+: page_table, attention_mask, ...
    auto queryType = llvm::cast<RankedTensorType>(op.getOperand(0).getType());
    auto keyType = llvm::cast<RankedTensorType>(op.getOperand(1).getType());
    auto valueType = llvm::cast<RankedTensorType>(op.getOperand(2).getType());
    auto outputType = llvm::cast<RankedTensorType>(op.getResult(0).getType());

    if (queryType.getShape() != outputType.getShape()) {
      op.getOperation()->emitWarning()
          << "Paged SDPA decode: query and output shapes must match.";
      return mlir::sdy::OpShardingRuleAttr();
    }

    llvm::SmallVector<RankedTensorType> qkvTypes = {queryType, keyType,
                                                    valueType};
    if (llvm::any_of(qkvTypes, [&](RankedTensorType type) {
          return type.getRank() != 4;
        })) {
      op.getOperation()->emitWarning()
          << "Paged SDPA decode: unexpected Q/K/V layouts, q rank: "
          << queryType.getRank() << ", key rank: " << keyType.getRank()
          << ", value rank: " << valueType.getRank();
      return mlir::sdy::OpShardingRuleAttr();
    }

    const int64_t queryUsersDim = 1; // [1, U, H, D]
    const int64_t queryHeadDim = 2;  // [1, U, H, D]
    const int64_t kvHeadDim = 1;     // [B, H, S, D]
    const int64_t outputUsersDim = 1;
    const int64_t outputHeadDim = 2;

    int64_t numHeads = queryType.getShape()[queryHeadDim];
    int64_t numUsers = queryType.getShape()[queryUsersDim];

    int64_t numOperands = op.getNumOperands();
    int64_t numResults = op.getNumResults();

    mlir::sdy::OpShardingRuleBuilder builder(op);

    // Head factor (kPassThrough, size = num_heads): links Q/K/V/Out head dims.
    SmallVector<int64_t> headOperandDims(numOperands, mlir::sdy::kNullDim);
    SmallVector<int64_t> headResultDims(numResults, mlir::sdy::kNullDim);
    headOperandDims[0] = queryHeadDim; // query
    headOperandDims[1] = kvHeadDim;    // key
    headOperandDims[2] = kvHeadDim;    // value
    headResultDims[0] = outputHeadDim; // output
    builder.addFactor(headOperandDims, headResultDims, numHeads,
                      mlir::sdy::FactorType::kPassThrough);

    // Users factor (kPassThrough, size = num_users): required so the SPMD
    // partitioner can keep each DP replica's decode local. Without it, page
    // table / cur_pos sharding can't propagate and a single replica observes
    // KV cache corruption from the other replica's writes.
    SmallVector<int64_t> usersOperandDims(numOperands, mlir::sdy::kNullDim);
    SmallVector<int64_t> usersResultDims(numResults, mlir::sdy::kNullDim);
    usersOperandDims[0] = queryUsersDim; // query

    // Operand layout (see TTIR_PagedScaledDotProductAttentionDecodeOp in
    // TTIROps.td): the base operands are [query, key, value, page_table];
    // an optional attention_mask, then an optional cur_pos_tensor, then an
    // optional attention_sink follow in that order. Only page_table (fixed at
    // index 3) and cur_pos_tensor carry the users/batch dim, so we must not
    // assume cur_pos_tensor is at index 4 -- with an attention_mask present,
    // index 4 is the mask and cur_pos_tensor shifts to 5. Use the has_*
    // frontend flags to locate it.
    mlir::DictionaryAttr frontendAttrs =
        mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
            op->getDiscardableAttr("mhlo.frontend_attributes"));
    auto hasFlag = [&](llvm::StringRef name) {
      if (frontendAttrs) {
        if (auto strAttr = frontendAttrs.getAs<mlir::StringAttr>(name)) {
          return strAttr.getValue().equals_insensitive("true");
        }
      }
      return false;
    };

    constexpr int64_t pageTableIdx = 3;
    if (numOperands > pageTableIdx) {
      usersOperandDims[pageTableIdx] = 0; // page_table
    }
    if (hasFlag("has_cur_pos_tensor")) {
      const int64_t curPosIdx =
          pageTableIdx + 1 + (hasFlag("has_attention_mask") ? 1 : 0);
      if (curPosIdx < numOperands) {
        usersOperandDims[curPosIdx] = 0; // cur_pos_tensor
      }
    }
    usersResultDims[0] = outputUsersDim; // output
    builder.addFactor(usersOperandDims, usersResultDims, numUsers,
                      mlir::sdy::FactorType::kPassThrough);

    return builder.build();
  }

  if (target == chunkedSdpaTargetName) {
    return getChunkedSdpaShardingRule(op);
  }

  if (target == pagedUpdateCacheTargetName) {
    // Paged update cache
    //  0: cache        [num_pages_total, num_heads, block_size, hidden_size]
    //  1: fill_value   [1, num_users, num_heads, hidden_size]
    //  2+: update_indices, page_table, ...
    auto cacheType = llvm::cast<RankedTensorType>(op.getOperand(0).getType());
    auto outputType = llvm::cast<RankedTensorType>(op.getResult(0).getType());

    if (cacheType.getShape() != outputType.getShape()) {
      op.getOperation()->emitWarning()
          << "Paged Update Cache: cache and output shapes must match.";
      return mlir::sdy::OpShardingRuleAttr();
    }

    const int64_t cacheHeadDim = 1;
    const int64_t fillValueUsersDim = 1;
    const int64_t fillValueHeadDim = 2;
    const int64_t outputHeadDim = 1;

    auto fillValueType =
        llvm::cast<RankedTensorType>(op.getOperand(1).getType());

    int64_t numHeads = cacheType.getShape()[cacheHeadDim];
    int64_t numUsers = fillValueType.getShape()[fillValueUsersDim];

    int64_t numOperands = op.getNumOperands();
    int64_t numResults = op.getNumResults();

    mlir::sdy::OpShardingRuleBuilder builder(op);

    // Head factor (kPassThrough, size = num_heads): links cache / fill_value /
    // output head dims.
    SmallVector<int64_t> headOperandDims(numOperands, mlir::sdy::kNullDim);
    SmallVector<int64_t> headResultDims(numResults, mlir::sdy::kNullDim);
    headOperandDims[0] = cacheHeadDim;     // cache
    headOperandDims[1] = fillValueHeadDim; // fill_value
    headResultDims[0] = outputHeadDim;     // output
    builder.addFactor(headOperandDims, headResultDims, numHeads,
                      mlir::sdy::FactorType::kPassThrough);

    // Users factor (kPassThrough, size = num_users): cache and output have NO
    // users dim. This tells the SPMD partitioner each replica writes a
    // disjoint cache slice, so no cross-device sync is needed for the update.
    SmallVector<int64_t> usersOperandDims(numOperands, mlir::sdy::kNullDim);
    SmallVector<int64_t> usersResultDims(numResults, mlir::sdy::kNullDim);
    usersOperandDims[1] = fillValueUsersDim; // fill_value
    if (numOperands > 2) {
      usersOperandDims[2] = 0; // update_indices
    }
    if (numOperands > 3) {
      usersOperandDims[3] = 0; // page_table
    }
    // operand[0] (cache) and result[0] (cache) stay kNullDim — critical.
    builder.addFactor(usersOperandDims, usersResultDims, numUsers,
                      mlir::sdy::FactorType::kPassThrough);

    return builder.build();
  }

  if (target == pagedFillCacheTargetName) {
    // Paged fill cache
    //  0: cache       [num_pages_total, num_heads, block_size, hidden_size]
    //  1: fill_value  [batch, num_heads, seq_len, hidden_size]
    //  2: page_table  [batch, num_blocks_per_user]
    //  3: batch_idx   [batch]   (optional)
    auto cacheType = llvm::cast<RankedTensorType>(op.getOperand(0).getType());
    auto outputType = llvm::cast<RankedTensorType>(op.getResult(0).getType());

    if (cacheType.getShape() != outputType.getShape()) {
      op.getOperation()->emitWarning()
          << "Paged Fill Cache: cache and output shapes must match.";
      return mlir::sdy::OpShardingRuleAttr();
    }

    const int64_t numOperands = op.getNumOperands();
    const int64_t numResults = op.getNumResults();

    // Head factor (TP): num_heads, on cache/fill_value/output dim 1.
    const int64_t cacheHeadDim = 1;
    const int64_t fillValueHeadDim = 1;
    const int64_t outputHeadDim = 1;
    int64_t headSize = cacheType.getShape()[cacheHeadDim];

    // Batch factor (DP): link the batch dim of fill_value/page_table/batch_idx
    // so the indices shard with the (DP-sharded) fill_value; without it they
    // stay replicated and lowering fails. Cache/output have no batch dim.
    const int64_t fillValueBatchDim = 0;
    const int64_t pageTableBatchDim = 0;
    const int64_t batchIdxBatchDim = 0;
    auto fillValueType =
        llvm::cast<RankedTensorType>(op.getOperand(1).getType());
    int64_t batchSize = fillValueType.getShape()[fillValueBatchDim];

    mlir::sdy::OpShardingRuleBuilder builder(op);

    SmallVector<int64_t> headOperandDims(numOperands, mlir::sdy::kNullDim);
    SmallVector<int64_t> headResultDims(numResults, mlir::sdy::kNullDim);
    headOperandDims[0] = cacheHeadDim;     // cache
    headOperandDims[1] = fillValueHeadDim; // fill_value
    headResultDims[0] = outputHeadDim;     // output
    builder.addFactor(headOperandDims, headResultDims, headSize,
                      mlir::sdy::FactorType::kPassThrough);

    SmallVector<int64_t> batchOperandDims(numOperands, mlir::sdy::kNullDim);
    SmallVector<int64_t> batchResultDims(numResults, mlir::sdy::kNullDim);
    batchOperandDims[1] = fillValueBatchDim; // fill_value
    if (numOperands > 2) {
      batchOperandDims[2] = pageTableBatchDim; // page_table
    }
    if (numOperands > 3) {
      batchOperandDims[3] = batchIdxBatchDim; // batch_idx
    }
    builder.addFactor(batchOperandDims, batchResultDims, batchSize,
                      mlir::sdy::FactorType::kPassThrough);

    return builder.build();
  }

  if (target == pagedFlashMlaDecodeTargetName) {
    // Paged flash MLA decode. The StableHLO custom_call carries its operands in
    // a fixed order, with the optional operands gated by has_* frontend
    // attributes:
    //   0: query        [1, num_users, nqh, dh_qk]
    //   1: key          [max_num_blocks, nkv, block_size, dh_qk]
    //   [value]         [max_num_blocks, nkv, block_size, head_dim_v]
    //   page_table      [num_users, max_blocks_per_seq]
    //   [attention_mask][num_users, nqh, 1, seq_k]
    //   [cur_pos_tensor][num_users]
    //   [attention_sink][nqh]
    //  result: output   [1, num_users, nqh, head_dim_v]
    //
    // Unlike paged SDPA decode, MLA keeps a single compressed latent KV cache
    // (nkv is always 1) that is shared across all query heads, so the KV
    // cache cannot be head-sharded and always stays replicated. Two query
    // dimensions can be sharded, each described by its own pass-through factor:
    //
    //   - Head (nqh): shard the query head dim and the matching output head
    //     dim. Each device computes attention for its slice of query heads
    //     against the full (replicated) latent KV cache; the per-head outputs
    //     are then concatenated.
    //
    //   - Batch (num_users): shard the query num_users dim together with the
    //     user-indexed operands (page_table, cur_pos, attention_mask) and the
    //     output num_users dim. Each device handles a slice of users.
    auto queryType = llvm::cast<RankedTensorType>(op.getOperand(0).getType());
    auto outputType = llvm::cast<RankedTensorType>(op.getResult(0).getType());

    if (queryType.getRank() != 4 || outputType.getRank() != 4) {
      op.getOperation()->emitWarning()
          << "Paged flash MLA decode: query and output must be 4D, got query "
             "rank "
          << queryType.getRank() << ", output rank " << outputType.getRank();
      return mlir::sdy::OpShardingRuleAttr();
    }

    const int64_t queryBatchDim = 1;  // [1, num_users, nqh, dh_qk]
    const int64_t queryHeadDim = 2;   // [1, num_users, nqh, dh_qk]
    const int64_t outputBatchDim = 1; // [1, num_users, nqh, head_dim_v]
    const int64_t outputHeadDim = 2;  // [1, num_users, nqh, head_dim_v]

    // Query and output share the num_users and nqh head dimensions, so
    // validate just those
    if (queryType.getShape()[queryHeadDim] !=
        outputType.getShape()[outputHeadDim]) {
      op.getOperation()->emitWarning() << "Paged flash MLA decode: query and "
                                          "output head dimension must match.";
      return mlir::sdy::OpShardingRuleAttr();
    }
    if (queryType.getShape()[queryBatchDim] !=
        outputType.getShape()[outputBatchDim]) {
      op.getOperation()->emitWarning() << "Paged flash MLA decode: query and "
                                          "output num_users dimension must "
                                          "match.";
      return mlir::sdy::OpShardingRuleAttr();
    }

    const int64_t numOperands = op.getNumOperands();
    const int64_t numResults = op.getNumResults();
    const int64_t numHeads = queryType.getShape()[queryHeadDim];
    const int64_t numUsers = queryType.getShape()[queryBatchDim];

    mlir::sdy::OpShardingRuleBuilder builder(op);

    // Head factor (nqh): only the query head dim and the matching output head
    // dim shard; every other operand (including the latent KV cache) stays
    // replicated.
    {
      SmallVector<int64_t> operandDims(numOperands, mlir::sdy::kNullDim);
      SmallVector<int64_t> resultDims(numResults, mlir::sdy::kNullDim);
      operandDims[0] = queryHeadDim;
      resultDims[0] = outputHeadDim;
      builder.addFactor(operandDims, resultDims, numHeads,
                        mlir::sdy::FactorType::kPassThrough);
    }

    // Batch factor (num_users): the query and output num_users dims shard
    // together with the user-indexed operands. The optional operands appear
    // after query/key in a fixed order gated by the has_* frontend attributes,
    // so reconstruct their indices the same way the StableHLO->TTIR conversion
    // does. Without the frontend attributes we cannot locate those operands, so
    // fall back to head-only sharding.
    mlir::DictionaryAttr frontendAttrs =
        mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
            op->getDiscardableAttr("mhlo.frontend_attributes"));
    if (frontendAttrs) {
      auto readFlag = [&](llvm::StringRef name) -> bool {
        auto strAttr = frontendAttrs.getAs<mlir::StringAttr>(name);
        return strAttr && strAttr.getValue().equals_insensitive("true");
      };
      bool hasValue = readFlag("has_value");
      bool hasAttentionMask = readFlag("has_attention_mask");
      bool hasCurPosTensor = readFlag("has_cur_pos_tensor");

      // operand order: query(0), key(1), [value], page_table,
      //                [attention_mask], [cur_pos_tensor], [attention_sink]
      int64_t idx = 2;
      if (hasValue) {
        ++idx;
      }
      int64_t pageTableIdx = idx++;
      int64_t attentionMaskIdx = hasAttentionMask ? idx++ : mlir::sdy::kNullDim;
      int64_t curPosIdx = hasCurPosTensor ? idx++ : mlir::sdy::kNullDim;

      if (pageTableIdx < numOperands) {
        SmallVector<int64_t> operandDims(numOperands, mlir::sdy::kNullDim);
        SmallVector<int64_t> resultDims(numResults, mlir::sdy::kNullDim);
        operandDims[0] = queryBatchDim; // query num_users (dim 1)
        operandDims[pageTableIdx] = 0;  // page_table num_users (dim 0)
        if (curPosIdx != mlir::sdy::kNullDim) {
          operandDims[curPosIdx] = 0; // cur_pos num_users (dim 0)
        }
        if (attentionMaskIdx != mlir::sdy::kNullDim) {
          auto maskType = llvm::cast<RankedTensorType>(
              op.getOperand(attentionMaskIdx).getType());
          // Only shard the mask's leading dim when it carries one entry per
          // user; a size-1 leading dim is a batch broadcast and stays
          // replicated.
          if (maskType.getRank() > 0 && maskType.getShape()[0] == numUsers) {
            operandDims[attentionMaskIdx] = 0; // mask num_users (dim 0)
          }
        }
        resultDims[0] = outputBatchDim; // output num_users (dim 1)
        builder.addFactor(operandDims, resultDims, numUsers,
                          mlir::sdy::FactorType::kPassThrough);
      }
    }

    return builder.build();
  }

  op.getOperation()->emitWarning()
      << "Paged attention sharding rule called for unexpected target: "
      << target;
  llvm_unreachable("Unexpected target for paged attention sharding rule");

  return mlir::sdy::OpShardingRuleAttr();
}

// Sharding rule for sparse_matmul used in MoE (Mixture of Experts) models.
// Supports both Expert Parallelism (EP) and Tensor Parallelism (TP).
//
// GPT-OSS parallelism strategy:
//   - EP: Expert dimension (E) sharded across row axis
//   - TP: Weight dimension (K or N) sharded across column axis
//
// Supported modes:
// 1. is_input_b_sparse=True (gate/up projection - column parallel):
//    Input A: [A, B, M, K]      - replicated
//    Input B: [1, E, K, N]      - EP on E (dim 1), TP on N (dim 3)
//    Sparsity: [A, B, 1, E]     - EP on E (dim 3)
//    Output: [A, B, 1, E, M, N] - EP on E (dim 3), TP on N (dim 5)
//
// 2. is_input_a_sparse=True (down projection - row parallel):
//    Input A: [A, E, M, K]      - EP on E (dim 1), TP on K (dim 3)
//    Input B: [1, E, K, N]      - EP on E (dim 1), TP on K (dim 2)
//    Sparsity: [1, 1, A, E]     - EP on E (dim 3)
//    Output: [A, E, M, N]       - EP on E (dim 1), needs allreduce for TP
static mlir::sdy::OpShardingRuleAttr
getSparseMatmulShardingRule(mlir::stablehlo::CustomCallOp op) {
  // Operands: input_a, input_b, sparsity
  if (op.getNumOperands() != 3 || op.getNumResults() != 1) {
    op.getOperation()->emitWarning()
        << "sparse_matmul expects 3 operands and 1 result";
    return mlir::sdy::OpShardingRuleAttr();
  }

  auto inputAType =
      llvm::dyn_cast<RankedTensorType>(op.getOperand(0).getType());
  auto inputBType =
      llvm::dyn_cast<RankedTensorType>(op.getOperand(1).getType());
  auto sparsityType =
      llvm::dyn_cast<RankedTensorType>(op.getOperand(2).getType());
  auto outputType = llvm::dyn_cast<RankedTensorType>(op.getResult(0).getType());

  if (!inputAType || !inputBType || !sparsityType || !outputType) {
    op.getOperation()->emitWarning()
        << "sparse_matmul requires ranked tensor types";
    return mlir::sdy::OpShardingRuleAttr();
  }

  // Parse frontend attributes to determine sparse mode
  mlir::DictionaryAttr frontendAttrs =
      mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
          op->getDiscardableAttr("mhlo.frontend_attributes"));

  bool isInputASparse = false;
  bool isInputBSparse = true; // default

  if (frontendAttrs) {
    if (auto strAttr =
            frontendAttrs.getAs<mlir::StringAttr>("is_input_a_sparse")) {
      isInputASparse = strAttr.getValue().equals_insensitive("true");
    }
    if (auto strAttr =
            frontendAttrs.getAs<mlir::StringAttr>("is_input_b_sparse")) {
      isInputBSparse = strAttr.getValue().equals_insensitive("true");
    }
  }

  // Input B always has shape [1, E, K, N] - E at dim 1, K at dim 2, N at dim 3
  if (inputBType.getRank() != 4) {
    op.getOperation()->emitWarning()
        << "sparse_matmul input_b must be 4D [1, E, K, N]";
    return mlir::sdy::OpShardingRuleAttr();
  }

  int64_t numExperts = inputBType.getShape()[1];
  int64_t kDim = inputBType.getShape()[2];
  int64_t nDim = inputBType.getShape()[3];

  mlir::sdy::OpShardingRuleBuilder builder(op);

  // ===== Factor 1: Expert Parallelism (EP) =====
  // Expert dimension positions vary by sparse mode
  int64_t inputAExpertDim = mlir::sdy::kNullDim;
  int64_t inputBExpertDim = 1;
  int64_t sparsityExpertDim = sparsityType.getRank() - 1; // last dim
  int64_t outputExpertDim;

  if (!isInputASparse && isInputBSparse) {
    // Mode: [A,B,M,K] @ [1,E,K,N] -> [A,B,1,E,M,N]
    inputAExpertDim = mlir::sdy::kNullDim; // replicate
    outputExpertDim = 3;
  } else if (isInputASparse && !isInputBSparse) {
    // Mode: [A,E,M,K] @ [1,E,K,N] -> [A,E,M,N]
    inputAExpertDim = 1;
    outputExpertDim = 1;
  } else if (isInputASparse && isInputBSparse) {
    // Mode: [1,E,M,K] @ [1,E,K,N] -> [1,E,M,N]
    inputAExpertDim = 1;
    outputExpertDim = 1;
  } else {
    op.getOperation()->emitWarning()
        << "sparse_matmul: both sparse flags cannot be false";
    return mlir::sdy::OpShardingRuleAttr();
  }

  // Add EP factor - expert dimension can be sharded
  builder.addFactor({inputAExpertDim, inputBExpertDim, sparsityExpertDim},
                    {outputExpertDim}, numExperts,
                    mlir::sdy::FactorType::kPassThrough);

  // ===== Sparsity non-E dimensions: force replication =====
  // The sparsity tensor is a binary 0/1 mask from moe_expert_token_remap.
  // Its non-E dimensions (batch/dispatch metadata) must stay replicated;
  // otherwise Shardy inserts all_reduce(sum) which corrupts the mask
  // (e.g. turning {0,1} values into {0,N_devices}).
  for (int64_t i = 0; i < sparsityType.getRank() - 1; ++i) {
    int64_t dimSize = sparsityType.getShape()[i];
    if (dimSize > 1) {
      builder.addFactor({mlir::sdy::kNullDim, mlir::sdy::kNullDim, i},
                        {mlir::sdy::kNullDim}, dimSize,
                        mlir::sdy::FactorType::kNeedReplication,
                        /*isBlocked=*/true);
    }
  }

  // ===== Factor 2: Tensor Parallelism (TP) =====
  // TP strategy depends on projection type:
  // - Gate/Up (is_input_b_sparse): Column parallel on N dimension
  // - Down (is_input_a_sparse): Row parallel on K dimension (output needs
  // allreduce)

  if (!isInputASparse && isInputBSparse) {
    // Gate/Up projection: Support both K sharding (row parallel) and N sharding
    // (column parallel)

    // K dimension: contracting dimension (needs all-reduce if sharded)
    // Input A: [A,B,M,K] - K at dim 3
    // Input B: [1,E,K,N] - K at dim 2
    // Output: [A,B,1,E,M,N] - K is contracted (reduced)
    // This enables sharding hidden_states when they come from attention with 2D
    // TP
    builder.addFactor(
        {3, 2, mlir::sdy::kNullDim}, // inputA K (dim 3), inputB K (dim 2)
        {mlir::sdy::kNullDim},       // output doesn't have K dim (contracted)
        kDim, mlir::sdy::FactorType::kReduction);

    // N dimension: Column parallel on N (standard TP pattern)
    // Input A: [A,B,M,K] - no N dim (replicated)
    // Input B: [1,E,K,N] - TP on N (dim 3)
    // Output: [A,B,1,E,M,N] - TP on N (dim 5)
    builder.addFactor(
        {mlir::sdy::kNullDim, 3, mlir::sdy::kNullDim}, // inputA, inputB,
                                                       // sparsity
        {5},                                           // output N dim
        nDim, mlir::sdy::FactorType::kPassThrough);
  } else if (isInputASparse) {
    // Down projection: Row parallel on K (contracting dimension)
    // Input A: [A,E,M,K] - TP on K (dim 3)
    // Input B: [1,E,K,N] - TP on K (dim 2)
    // Output: [A,E,M,N] - K is contracted (reduced), needs all_reduce for TP
    //
    // Following the pattern from DotGeneralOp in Shardy:
    // Contracting dimensions use kReduction factor type.
    // This tells Shardy that:
    // - K dimension can be sharded (row parallel)
    // - Each device computes partial sum
    // - All-reduce is needed to combine partial sums
    builder.addFactor(
        {3, 2, mlir::sdy::kNullDim}, // inputA K (dim 3), inputB K (dim 2)
        {mlir::sdy::kNullDim},       // output doesn't have K dim (contracted)
        kDim, mlir::sdy::FactorType::kReduction);

    // N dimension: passes through from input B to output
    // Input A: [A,E,M,K] - no N dim
    // Input B: [1,E,K,N] - N at dim 3
    // Output: [A,E,M,N] - N at dim 3
    // This allows column-parallel sharding on N if needed
    builder.addFactor(
        {mlir::sdy::kNullDim, 3, mlir::sdy::kNullDim}, // inputA none, inputB N
        {3},                                           // output N dim
        nDim, mlir::sdy::FactorType::kPassThrough);
  }

  return builder.build();
}

// =====================================================================
// all_to_all_dispatch sharding rule
// =====================================================================
// Dispatch is an opaque CCL operation that handles EP communication
// along ONE mesh axis only (cluster_axis; -1 = flatten 2D mesh to 1D).
// Experts are compound-sharded across both mesh dims; input is replicated
// on one dim and all-to-all'ed on the other.
//
// All factors use kNeedReplication with isBlocked=true to prevent
// Shardy from propagating any sharding through the op. Without isBlocked,
// Shardy would insert unnecessary all_gather/AllSlice pairs.
//
// Operands: [0] input [B,S,1,H], [1] indices [B,S,1,K], [2] mapping [1,1,E,D]
// Results:  [0] dispatched [1,B*D,S,H], [1] metadata [1,B*D,S,K]
//
// Note: torch-xla may emit the custom_call with either:
//   - Variadic results: 2 separate tensor results (modern StableHLO)
//   - Tuple result: 1 result of type tuple<tensor, tensor> (legacy HLO style)
// The builder must handle both forms since OpShardingRuleBuilder(op)
// crashes on TupleType (it calls cast<ShapedType> on each result type).
static mlir::sdy::OpShardingRuleAttr
getAllToAllDispatchShardingRule(mlir::stablehlo::CustomCallOp op) {
  if (op.getNumOperands() != 3) {
    op.getOperation()->emitWarning()
        << "all_to_all_dispatch expects 3 operands, got "
        << op.getNumOperands();
    return mlir::sdy::OpShardingRuleAttr();
  }

  auto inputType = llvm::dyn_cast<RankedTensorType>(op.getOperand(0).getType());
  auto indicesType =
      llvm::dyn_cast<RankedTensorType>(op.getOperand(1).getType());
  auto mappingType =
      llvm::dyn_cast<RankedTensorType>(op.getOperand(2).getType());

  if (!inputType || inputType.getRank() != 4 || !indicesType ||
      indicesType.getRank() != 4 || !mappingType ||
      mappingType.getRank() != 4) {
    op.getOperation()->emitWarning()
        << "all_to_all_dispatch: all operands must be 4D";
    return mlir::sdy::OpShardingRuleAttr();
  }

  // Extract dimension sizes
  int64_t bDim = inputType.getShape()[0];   // B
  int64_t sDim = inputType.getShape()[1];   // S
  int64_t hDim = inputType.getShape()[3];   // H
  int64_t kDim = indicesType.getShape()[3]; // K
  int64_t eDim = mappingType.getShape()[2]; // E
  int64_t dDim = mappingType.getShape()[3]; // D

  // Resolve result types: handle both variadic (2 results) and tuple (1 result)
  SmallVector<Type> resultTypes;
  if (op.getNumResults() == 2) {
    // Variadic results: each result is a tensor
    for (auto type : op.getResultTypes()) {
      resultTypes.push_back(type);
    }
  } else if (op.getNumResults() == 1) {
    // Possibly a tuple result
    auto tupleType = llvm::dyn_cast<mlir::TupleType>(op.getResult(0).getType());
    if (tupleType && tupleType.size() == 2) {
      for (auto elemType : tupleType.getTypes()) {
        resultTypes.push_back(elemType);
      }
    } else {
      op.getOperation()->emitWarning()
          << "all_to_all_dispatch: expected 2-element tuple or 2 results";
      return mlir::sdy::OpShardingRuleAttr();
    }
  } else {
    op.getOperation()->emitWarning()
        << "all_to_all_dispatch: unexpected number of results: "
        << op.getNumResults();
    return mlir::sdy::OpShardingRuleAttr();
  }

  // Use explicit type-range constructor to bypass TupleType cast issue
  // Result 0: dispatched [1, B*D, S, H]
  // Result 1: metadata [1, B*D, S, K]
  mlir::sdy::OpShardingRuleBuilder builder(op.getOperandTypes(),
                                           TypeRange(resultTypes),
                                           op.getContext(), std::nullopt);

  // B factor: input[0]=dim0, input[1]=dim0, input[2]=kNull,
  //           result[0]=kNull (B is absorbed into B*D), result[1]=kNull
  builder.addFactor({0, 0, mlir::sdy::kNullDim},
                    {mlir::sdy::kNullDim, mlir::sdy::kNullDim}, bDim,
                    mlir::sdy::FactorType::kNeedReplication,
                    /*isBlocked=*/true);

  // S factor: input[0]=dim1, input[1]=dim1, input[2]=kNull,
  //           result[0]=dim2, result[1]=dim2
  builder.addFactor({1, 1, mlir::sdy::kNullDim}, {2, 2}, sDim,
                    mlir::sdy::FactorType::kNeedReplication,
                    /*isBlocked=*/true);

  // H factor: input[0]=dim3, input[1]=kNull, input[2]=kNull,
  //           result[0]=dim3, result[1]=kNull
  builder.addFactor({3, mlir::sdy::kNullDim, mlir::sdy::kNullDim},
                    {3, mlir::sdy::kNullDim}, hDim,
                    mlir::sdy::FactorType::kNeedReplication,
                    /*isBlocked=*/true);

  // K factor: input[0]=kNull, input[1]=dim3, input[2]=kNull,
  //           result[0]=kNull, result[1]=dim3
  builder.addFactor({mlir::sdy::kNullDim, 3, mlir::sdy::kNullDim},
                    {mlir::sdy::kNullDim, 3}, kDim,
                    mlir::sdy::FactorType::kNeedReplication,
                    /*isBlocked=*/true);

  // E factor: input[0]=kNull, input[1]=kNull, input[2]=dim2,
  //           result[0]=kNull, result[1]=kNull
  builder.addFactor({mlir::sdy::kNullDim, mlir::sdy::kNullDim, 2},
                    {mlir::sdy::kNullDim, mlir::sdy::kNullDim}, eDim,
                    mlir::sdy::FactorType::kNeedReplication,
                    /*isBlocked=*/true);

  // D factor: input[0]=kNull, input[1]=kNull, input[2]=dim3,
  //           result[0]=kNull, result[1]=kNull
  builder.addFactor({mlir::sdy::kNullDim, mlir::sdy::kNullDim, 3},
                    {mlir::sdy::kNullDim, mlir::sdy::kNullDim}, dDim,
                    mlir::sdy::FactorType::kNeedReplication,
                    /*isBlocked=*/true);

  return builder.build();
}

// =====================================================================
// all_to_all_combine sharding rule
// =====================================================================
// Combine restores tokens along cluster_axis. E uses a single kPassThrough
// factor so Shardy does NOT insert any all_reduce for the expert dimension.
// For 2D mesh with compound-sharded experts, the needed cross-column
// all_reduce(sum) is inserted explicitly in the StableHLOToTTIR conversion.
//
// H, BD, S, K: kNeedReplication with isBlocked (no sharding propagation).
//
// Operands: [0] expert_out [E,B*D,S,H], [1] metadata [1,B*D,S,K],
//           [2] mapping [1,1,E_total,D]
// Results:  [0] combined [K,B,S,H]
static mlir::sdy::OpShardingRuleAttr
getAllToAllCombineShardingRule(mlir::stablehlo::CustomCallOp op) {
  if (op.getNumOperands() != 3) {
    op.getOperation()->emitWarning()
        << "all_to_all_combine expects 3 operands, got " << op.getNumOperands();
    return mlir::sdy::OpShardingRuleAttr();
  }

  auto expertOutType =
      llvm::dyn_cast<RankedTensorType>(op.getOperand(0).getType());
  auto metadataType =
      llvm::dyn_cast<RankedTensorType>(op.getOperand(1).getType());
  auto mappingType =
      llvm::dyn_cast<RankedTensorType>(op.getOperand(2).getType());

  if (!expertOutType || expertOutType.getRank() != 4 || !metadataType ||
      metadataType.getRank() != 4 || !mappingType ||
      mappingType.getRank() != 4) {
    op.getOperation()->emitWarning()
        << "all_to_all_combine: all operands must be 4D";
    return mlir::sdy::OpShardingRuleAttr();
  }

  auto resultType = llvm::dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!resultType || resultType.getRank() != 4) {
    op.getOperation()->emitWarning() << "all_to_all_combine: result must be 4D";
    return mlir::sdy::OpShardingRuleAttr();
  }

  // Extract dimension sizes
  int64_t eDim = expertOutType.getShape()[0];  // E (global expert count)
  int64_t bdDim = expertOutType.getShape()[1]; // B*D
  int64_t sDim = expertOutType.getShape()[2];  // S
  int64_t hDim = expertOutType.getShape()[3];  // H
  int64_t kDim = metadataType.getShape()[3];   // K
  int64_t dDim = mappingType.getShape()[3];    // D (num dispatch devices)

  mlir::sdy::OpShardingRuleBuilder builder(op);

  // E factor: single kPassThrough — combine handles dispatch-axis
  // communication internally. Cross-column reduction (for 2D compound
  // sharding) is inserted explicitly in the StableHLOToTTIR conversion pass.
  // Using kReduction here causes Shardy to insert all_reduce on BOTH axes
  // (kPassThrough with kNullDim in the result also triggers reduce), which
  // corrupts the combine output by summing unrelated batches.
  builder.addFactor({0, mlir::sdy::kNullDim, mlir::sdy::kNullDim},
                    {mlir::sdy::kNullDim}, eDim,
                    mlir::sdy::FactorType::kPassThrough);

  // E factor (mapping): mapping dim 2 is the global expert count and must
  // stay replicated — combine reads it to determine expert-to-device routing.
  int64_t eTotalDim = mappingType.getShape()[2];
  builder.addFactor({mlir::sdy::kNullDim, mlir::sdy::kNullDim, 2},
                    {mlir::sdy::kNullDim}, eTotalDim,
                    mlir::sdy::FactorType::kNeedReplication,
                    /*isBlocked=*/true);

  // H factor: kNeedReplication with isBlocked.
  // H is replicated through combine (not sharded on any axis).
  builder.addFactor({3, mlir::sdy::kNullDim, mlir::sdy::kNullDim}, {3}, hDim,
                    mlir::sdy::FactorType::kNeedReplication,
                    /*isBlocked=*/true);

  // K factor: expertOut=kNull, metadata[1]=dim3, mapping=kNull, result=dim0
  builder.addFactor({mlir::sdy::kNullDim, 3, mlir::sdy::kNullDim}, {0}, kDim,
                    mlir::sdy::FactorType::kNeedReplication,
                    /*isBlocked=*/true);

  // S factor: expertOut[0]=dim2, metadata[1]=dim2, mapping=kNull, result=dim2
  builder.addFactor({2, 2, mlir::sdy::kNullDim}, {2}, sDim,
                    mlir::sdy::FactorType::kNeedReplication,
                    /*isBlocked=*/true);

  // BD factor: expertOut[0]=dim1, metadata[1]=dim1, mapping=kNull, result=dim1
  builder.addFactor({1, 1, mlir::sdy::kNullDim}, {1}, bdDim,
                    mlir::sdy::FactorType::kNeedReplication,
                    /*isBlocked=*/true);

  // D factor: expertOut=kNull, metadata=kNull, mapping[2]=dim3, result=kNull
  builder.addFactor({mlir::sdy::kNullDim, mlir::sdy::kNullDim, 3},
                    {mlir::sdy::kNullDim}, dDim,
                    mlir::sdy::FactorType::kNeedReplication,
                    /*isBlocked=*/true);

  return builder.build();
}

// Sharding rules for stablehlo.batch_norm_{training,grad}.
// (batch_norm_inference is handled natively by Shardy's
// op_sharding_rule_registry.)
//
// Feature dimension (feature_index): kPassThrough — the only dim that can be
// sharded. Non-feature dimensions: kNeedReplication — batch norm reduces over
// these to compute statistics, so they must stay replicated for correct
// results.

// batch_norm_training(input[ND], scale[1D], bias[1D])
//   -> (output[ND], mean[1D], variance[1D])
static mlir::sdy::OpShardingRuleAttr
getBatchNormTrainingShardingRule(mlir::stablehlo::BatchNormTrainingOp bnOp) {
  sdy::OpShardingRuleBuilder builder(bnOp);
  uint64_t featureIndex = bnOp.getFeatureIndex();
  for (auto [dim, dimSize] :
       llvm::enumerate(bnOp.getOperand().getType().getShape())) {
    // Operands: input[dim], scale[?], bias[?]
    // Results:  output[dim], mean[?], variance[?]
    // 1-D operands/results participate only on the feature dimension (at dim
    // 0).
    int64_t oneDimIdx = (dim == featureIndex) ? 0 : sdy::kNullDim;
    builder.addFactor({static_cast<int64_t>(dim), oneDimIdx, oneDimIdx},
                      {static_cast<int64_t>(dim), oneDimIdx, oneDimIdx},
                      dimSize,
                      dim == featureIndex ? sdy::FactorType::kPassThrough
                                          : sdy::FactorType::kNeedReplication);
  }
  return builder.build();
}

// batch_norm_grad(input[ND], scale[1D], mean[1D], var[1D], grad_output[ND])
//   -> (grad_input[ND], grad_scale[1D], grad_bias[1D])
static mlir::sdy::OpShardingRuleAttr
getBatchNormGradShardingRule(mlir::stablehlo::BatchNormGradOp bnOp) {
  sdy::OpShardingRuleBuilder builder(bnOp);
  uint64_t featureIndex = bnOp.getFeatureIndex();
  for (auto [dim, dimSize] :
       llvm::enumerate(bnOp.getOperand().getType().getShape())) {
    // Operands: input[dim], scale[?], mean[?], var[?], grad_output[dim]
    // Results:  grad_input[dim], grad_scale[?], grad_bias[?]
    int64_t oneDimIdx = (dim == featureIndex) ? 0 : sdy::kNullDim;
    builder.addFactor({static_cast<int64_t>(dim), oneDimIdx, oneDimIdx,
                       oneDimIdx, static_cast<int64_t>(dim)},
                      {static_cast<int64_t>(dim), oneDimIdx, oneDimIdx},
                      dimSize,
                      dim == featureIndex ? sdy::FactorType::kPassThrough
                                          : sdy::FactorType::kNeedReplication);
  }
  return builder.build();
}

template <typename OpTy>
struct StablehloShardingModel
    : public mlir::sdy::ShardingRuleOpInterface::ExternalModel<
          StablehloShardingModel<OpTy>, OpTy> {

  mlir::sdy::OpShardingRuleAttr getShardingRule(mlir::Operation *op) const {
    if (auto scatterOp = llvm::dyn_cast<mlir::stablehlo::ScatterOp>(op)) {
      return getScatterShardingRule(scatterOp);
    }
    if (auto bnOp = llvm::dyn_cast<mlir::stablehlo::BatchNormTrainingOp>(op)) {
      return getBatchNormTrainingShardingRule(bnOp);
    }
    if (auto bnOp = llvm::dyn_cast<mlir::stablehlo::BatchNormGradOp>(op)) {
      return getBatchNormGradShardingRule(bnOp);
    }
    return mlir::sdy::OpShardingRuleBuilder::buildPointwise(op);
  }

  bool shouldKeepOutputShardingsDivisible(mlir::Operation *) const {
    return true;
  }
};

// =====================================================================
// moe_expert_token_remap sharding rule
// =====================================================================
// Device-local data movement op that remaps global expert indices to local
// per-device experts. D/B/S/K factors are opaque (kNeedReplication + blocked).
// The E factor uses kPassThrough so compound sharding propagates through:
// outputs use global E shape, and UpdateGlobalToLocalShapes divides E by
// the compound mesh factor to produce E_local on each device.
//
// Operands: [0] topk [D,B,S,E], [1] mapping [1,1,E,D], [2] metadata [D,B,S,K]
// Results:  [0] mapping_out [1,B,S,E], [1] reduced [1,1,ceil(BS/R),E]
static mlir::sdy::OpShardingRuleAttr
getMoeExpertTokenRemapShardingRule(mlir::stablehlo::CustomCallOp op) {
  if (op.getNumOperands() != 3) {
    op.getOperation()->emitWarning()
        << "moe_expert_token_remap expects 3 operands, got "
        << op.getNumOperands();
    return mlir::sdy::OpShardingRuleAttr();
  }

  auto topkType = llvm::dyn_cast<RankedTensorType>(op.getOperand(0).getType());
  auto mappingInputType =
      llvm::dyn_cast<RankedTensorType>(op.getOperand(1).getType());
  auto metadataType =
      llvm::dyn_cast<RankedTensorType>(op.getOperand(2).getType());

  if (!topkType || topkType.getRank() != 4 || !mappingInputType ||
      mappingInputType.getRank() != 4 || !metadataType ||
      metadataType.getRank() != 4) {
    op.getOperation()->emitWarning()
        << "moe_expert_token_remap: all operands must be 4D";
    return mlir::sdy::OpShardingRuleAttr();
  }

  // Resolve result types (handle tuple and variadic)
  SmallVector<Type> resultTypes;
  if (op.getNumResults() == 2) {
    for (auto type : op.getResultTypes()) {
      resultTypes.push_back(type);
    }
  } else if (op.getNumResults() == 1) {
    auto tupleType = llvm::dyn_cast<mlir::TupleType>(op.getResult(0).getType());
    if (tupleType && tupleType.size() == 2) {
      for (auto elemType : tupleType.getTypes()) {
        resultTypes.push_back(elemType);
      }
    } else {
      return mlir::sdy::OpShardingRuleAttr();
    }
  } else {
    return mlir::sdy::OpShardingRuleAttr();
  }

  int64_t dDim = topkType.getShape()[0];
  int64_t bDim = topkType.getShape()[1];
  int64_t sDim = topkType.getShape()[2];
  int64_t eDim = topkType.getShape()[3];
  int64_t kDim = metadataType.getShape()[3];

  mlir::sdy::OpShardingRuleBuilder builder(op.getOperandTypes(),
                                           TypeRange(resultTypes),
                                           op.getContext(), std::nullopt);

  // D factor: topk[0], mapping_in=kNull, metadata[0],
  //           mapping_out=kNull, reduced=kNull
  builder.addFactor({0, mlir::sdy::kNullDim, 0},
                    {mlir::sdy::kNullDim, mlir::sdy::kNullDim}, dDim,
                    mlir::sdy::FactorType::kNeedReplication,
                    /*isBlocked=*/true);
  // B factor: topk[1], mapping_in=kNull, metadata[1],
  //           mapping_out[1], reduced=kNull
  builder.addFactor({1, mlir::sdy::kNullDim, 1}, {1, mlir::sdy::kNullDim}, bDim,
                    mlir::sdy::FactorType::kNeedReplication,
                    /*isBlocked=*/true);
  // S factor: topk[2], mapping_in=kNull, metadata[2],
  //           mapping_out[2], reduced=kNull
  builder.addFactor({2, mlir::sdy::kNullDim, 2}, {2, mlir::sdy::kNullDim}, sDim,
                    mlir::sdy::FactorType::kNeedReplication,
                    /*isBlocked=*/true);
  // E factor (replicated inputs): topk[3] and mapping_in[2] must keep E_global.
  // The kernel needs the full expert-to-device table and all expert routing
  // weights to perform global-to-local remapping internally.
  builder.addFactor({3, 2, mlir::sdy::kNullDim},
                    {mlir::sdy::kNullDim, mlir::sdy::kNullDim}, eDim,
                    mlir::sdy::FactorType::kNeedReplication,
                    /*isBlocked=*/true);
  // E factor (passthrough outputs): mapping_out[3] and reduced[3] use global E
  // shape but receive compound sharding from downstream sparse_matmul.
  // UpdateGlobalToLocalShapes divides E by the mesh factor to produce E_local.
  builder.addFactor(
      {mlir::sdy::kNullDim, mlir::sdy::kNullDim, mlir::sdy::kNullDim}, {3, 3},
      eDim, mlir::sdy::FactorType::kPassThrough);
  // K factor: topk=kNull, mapping_in=kNull, metadata[3],
  //           mapping_out=kNull, reduced=kNull
  builder.addFactor({mlir::sdy::kNullDim, mlir::sdy::kNullDim, 3},
                    {mlir::sdy::kNullDim, mlir::sdy::kNullDim}, kDim,
                    mlir::sdy::FactorType::kNeedReplication,
                    /*isBlocked=*/true);

  return builder.build();
}

// Sharding rule for RMS norm custom_call (converted from composite).
//
// Operands:
//   operand 0: input  [batch dims..., normalized dims...]
//   operand 1: weight [normalized dims...] (optional operand)
//   operand 2: bias   [normalized dims...] (optional operand)
// Result: same shape as input.
//
// Batch dimensions can be freely sharded. Normalized dimensions require
// replication because RMS norm reduces over them and Shardy needs to
// insert collectives to handle cross-device reduction ops.
static mlir::sdy::OpShardingRuleAttr
getRMSNormShardingRule(mlir::stablehlo::CustomCallOp op) {
  auto inputType = llvm::dyn_cast<RankedTensorType>(op.getOperand(0).getType());
  if (!inputType) {
    return mlir::sdy::OpShardingRuleAttr();
  }

  auto compositeAttrs = mlir::dyn_cast_or_null<DictionaryAttr>(
      op->getDiscardableAttr(utils::kCustomCallCompositeAttrsKey));
  if (!compositeAttrs) {
    return mlir::sdy::OpShardingRuleAttr();
  }
  auto normalizedShapeAttr = compositeAttrs.get("normalized_shape");
  int64_t numNormalizedDims = 0;

  if (auto denseAttr =
          mlir::dyn_cast_or_null<DenseIntElementsAttr>(normalizedShapeAttr)) {
    numNormalizedDims = denseAttr.getNumElements();
  } else if (auto arrayAttr =
                 mlir::dyn_cast_or_null<ArrayAttr>(normalizedShapeAttr)) {
    numNormalizedDims = arrayAttr.size();
  } else {
    return mlir::sdy::OpShardingRuleAttr();
  }

  const int64_t inputRank = inputType.getRank();
  const int64_t numBatchDims = inputRank - numNormalizedDims;
  if (numBatchDims < 0) {
    return mlir::sdy::OpShardingRuleAttr();
  }

  const int64_t numOperands = op.getNumOperands();
  mlir::sdy::OpShardingRuleBuilder builder(op);

  // Batch dimensions: pass-through for input and result.
  for (int64_t dim = 0; dim < numBatchDims; dim++) {
    SmallVector<int64_t> operandDims(numOperands, mlir::sdy::kNullDim);
    operandDims[0] = dim;
    builder.addFactor(operandDims, {dim}, inputType.getDimSize(dim),
                      mlir::sdy::FactorType::kPassThrough);
  }

  // Normalized dimensions: need replication because RMS norm reduces over them.
  for (int64_t i = 0; i < numNormalizedDims; i++) {
    int64_t inputDim = numBatchDims + i;
    SmallVector<int64_t> operandDims(numOperands, mlir::sdy::kNullDim);
    operandDims[0] = inputDim;
    if (numOperands > 1) {
      operandDims[1] = i;
    }
    if (numOperands > 2) {
      operandDims[2] = i;
    }
    builder.addFactor(operandDims, {inputDim}, inputType.getDimSize(inputDim),
                      mlir::sdy::FactorType::kNeedReplication);
  }

  return builder.build();
}

// Sharding rule for torch.gather-style custom_call (tenstorrent.gather /
// tenstorrent.gather_dim).
//
// Operands:
//   operand 0: input  [d0, ..., dim_K, ..., dN]   (rank N+1)
//   operand 1: index  [d0, ..., dim_J, ..., dN]   (same rank as input)
// Result: same shape as index (torch.gather semantics).
//
// Non-gather dims align across input/index/result and can be sharded freely.
// The input's gather dim must be replicated because indices can reference any
// position along it. The index/result gather dim is independent of the input
// gather dim and can be sharded freely.
static mlir::sdy::OpShardingRuleAttr
getGatherDimShardingRule(mlir::stablehlo::CustomCallOp op) {
  if (op.getNumOperands() != 2 || op.getNumResults() != 1) {
    op.getOperation()->emitWarning()
        << "gather sharding rule expects 2 operands and 1 result, got "
        << op.getNumOperands() << " operands and " << op.getNumResults()
        << " results";
    return mlir::sdy::OpShardingRuleAttr();
  }

  auto inputType = llvm::dyn_cast<RankedTensorType>(op.getOperand(0).getType());
  auto indexType = llvm::dyn_cast<RankedTensorType>(op.getOperand(1).getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!inputType || !indexType || !resultType) {
    op.getOperation()->emitWarning()
        << "gather sharding rule requires ranked tensor types";
    return mlir::sdy::OpShardingRuleAttr();
  }

  const int64_t rank = inputType.getRank();
  if (indexType.getRank() != rank || resultType.getRank() != rank) {
    op.getOperation()->emitWarning()
        << "gather sharding rule requires input, index, and result of equal "
           "rank; got input rank "
        << rank << ", index rank " << indexType.getRank() << ", result rank "
        << resultType.getRank();
    return mlir::sdy::OpShardingRuleAttr();
  }

  auto compositeAttrs = mlir::dyn_cast_or_null<DictionaryAttr>(
      op->getDiscardableAttr(utils::kCustomCallCompositeAttrsKey));
  if (!compositeAttrs) {
    op.getOperation()->emitWarning()
        << "gather sharding rule: missing tt.composite_attributes";
    return mlir::sdy::OpShardingRuleAttr();
  }
  auto dimAttr = compositeAttrs.getAs<IntegerAttr>("dim");
  if (!dimAttr) {
    op.getOperation()->emitWarning()
        << "gather sharding rule: missing or non-integer 'dim' attribute";
    return mlir::sdy::OpShardingRuleAttr();
  }
  int64_t dim = dimAttr.getInt();
  if (dim < 0) {
    dim += rank;
  }
  if (dim < 0 || dim >= rank) {
    op.getOperation()->emitWarning()
        << "gather sharding rule: dim " << dimAttr.getInt()
        << " out of range for rank " << rank;
    return mlir::sdy::OpShardingRuleAttr();
  }

  sdy::OpShardingRuleBuilder builder(op);

  // Non-gather dims: single passthrough factor linking input[d], index[d],
  // and result[d].
  for (int64_t d = 0; d < rank; ++d) {
    if (d == dim) {
      continue;
    }
    builder.addFactor({d, d}, {d}, inputType.getDimSize(d),
                      sdy::FactorType::kPassThrough);
  }

  // Input gather dim: replication required, appears only on the input.
  builder.addFactor({dim, sdy::kNullDim}, {sdy::kNullDim},
                    inputType.getDimSize(dim),
                    sdy::FactorType::kNeedReplication);

  // Index/result gather dim: passthrough, links index and result.
  builder.addFactor({sdy::kNullDim, dim}, {dim}, indexType.getDimSize(dim),
                    sdy::FactorType::kPassThrough);

  return builder.build();
}

struct StablehloCustomCallShardingModel
    : public mlir::sdy::ShardingRuleOpInterface::ExternalModel<
          StablehloCustomCallShardingModel, ::mlir::stablehlo::CustomCallOp> {

  mlir::sdy::OpShardingRuleAttr getShardingRule(mlir::Operation *op) const {
    auto custom = llvm::cast<mlir::stablehlo::CustomCallOp>(op);
    assert(custom && "StablehloCustomCallShardingModel must be attached to "
                     "CustomCallOp only");
    return getCustomCallShardingRule(custom);
  }

  bool shouldKeepOutputShardingsDivisible(mlir::Operation *) const {
    return true;
  }

private:
  mlir::sdy::OpShardingRuleAttr
  getCustomCallShardingRule(mlir::stablehlo::CustomCallOp op) const {
    llvm::StringRef target = op.getCallTargetName();

    auto shardOpFunc = customCallShardingRules.lookup(target);
    if (shardOpFunc) {
      return shardOpFunc(op);
    }

    op.getOperation()->emitWarning()
        << "StableHLO CustomCallOp sharding rule is not defined for target '"
        << target << "'";
    return mlir::sdy::OpShardingRuleAttr();
  }

  // Map from CustomCall target names to their corresponding sharding rule
  // functions.
  llvm::DenseMap<llvm::StringRef, std::function<mlir::sdy::OpShardingRuleAttr(
                                      mlir::stablehlo::CustomCallOp)>>
      customCallShardingRules = {
          {sdpaTargetName, getSDPAShardingRule},
          // Composite SDPA (frontend
          // "tenstorrent.scaled_dot_product_attention") is converted to a
          // custom_call keeping its composite name as the target, so map that
          // name to the same head-sharding rule.
          {utils::kTTSDPACompositeName, getSDPAShardingRule},
          {pagedSdpaDecodeTargetName, getPagedAttentionShardingRule},
          {chunkedSdpaTargetName, getPagedAttentionShardingRule},
          {pagedUpdateCacheTargetName, getPagedAttentionShardingRule},
          {pagedFillCacheTargetName, getPagedAttentionShardingRule},
          {pagedFlashMlaDecodeTargetName, getPagedAttentionShardingRule},
          {sparseMatmulTargetName, getSparseMatmulShardingRule},
          {allToAllDispatchTargetName, getAllToAllDispatchShardingRule},
          {allToAllCombineTargetName, getAllToAllCombineShardingRule},
          {moeExpertTokenRemapTargetName, getMoeExpertTokenRemapShardingRule},
          {utils::kTTRMSNormCustomCallTargetName, getRMSNormShardingRule},
          {flashMlaPrefillTargetName, getFlashMlaPrefillShardingRule},
          {utils::kTTGatherDimCustomCallTargetName, getGatherDimShardingRule},
          {utils::kTTGatherCustomCallTargetName, getGatherDimShardingRule},
      };
};

class RegisterCustomShardingRulePass
    : public impl::RegisterCustomShardingRulePassBase<
          RegisterCustomShardingRulePass> {
public:
  using impl::RegisterCustomShardingRulePassBase<
      RegisterCustomShardingRulePass>::RegisterCustomShardingRulePassBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    MLIRContext *context = rootModule.getContext();

    context->loadDialect<mlir::stablehlo::StablehloDialect>();
    // Register for stablehlo.CustomCallOp
    mlir::stablehlo::CustomCallOp::attachInterface<
        StablehloCustomCallShardingModel>(*context);
    mlir::stablehlo::ScatterOp::attachInterface<
        StablehloShardingModel<mlir::stablehlo::ScatterOp>>(*context);
    mlir::stablehlo::BatchNormTrainingOp::attachInterface<
        StablehloShardingModel<mlir::stablehlo::BatchNormTrainingOp>>(*context);
    mlir::stablehlo::BatchNormGradOp::attachInterface<
        StablehloShardingModel<mlir::stablehlo::BatchNormGradOp>>(*context);
  }
};

} // namespace mlir::tt::stablehlo
