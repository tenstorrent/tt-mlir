// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/ConvRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/DataMovementRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/EmbeddingRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/MatmulRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/NormalizationRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/TransformerRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/TypecastRules.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Utils.h"

#include "llvm/ADT/DenseMap.h"

#include <mutex>

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// Default OpRuleBook implementation
//===----------------------------------------------------------------------===//

OutputHints
OpRuleBook::getOutputHints(Operation * /*op*/,
                           const std::vector<OpConfig> &legalConfigs) const {
  // Primary: NULL hint only -- let the backend decide output from inputs.
  // Fallback: sharded configs -- tried only when NULL yields non-sharded
  // output.
  OutputHints result;
  result.hints.push_back(OpConfig(TTNNLayoutAttr()));
  for (const auto &cfg : legalConfigs) {
    if (!cfg.outputLayout) {
      continue;
    }
    auto memLayout = cfg.outputLayout.getMemLayout();
    if (memLayout && isShardedMemoryLayout(memLayout.getValue())) {
      result.fallbackHints.push_back(cfg);
    }
  }
  return result;
}

bool OpRuleBook::preferCandidate(Operation * /*op*/, const BeamCandidate &a,
                                 const BeamCandidate &b) const {
  // Prefer more sharded inputs: fewer interleaved reads = less NOC traffic.
  auto countShardedInputs = [](const BeamCandidate &c) {
    unsigned count = 0;
    for (const auto &layout : c.inputLayouts) {
      auto ml = layout.getMemLayout();
      if (ml && isShardedMemoryLayout(ml.getValue())) {
        ++count;
      }
    }
    return count;
  };
  int64_t shardedA = countShardedInputs(a);
  int64_t shardedB = countShardedInputs(b);
  if (shardedA != shardedB) {
    return shardedA > shardedB;
  }
  // Tied on sharded-input count: greedily prefer the candidate whose sharded
  // inputs cover a larger total grid volume. This assumes ops benefit from
  // inputs already distributed across more cores, not just sharded outputs.
  auto shardedGridVolume = [](const BeamCandidate &c) -> int64_t {
    int64_t total = 0;
    for (const auto &layout : c.inputLayouts) {
      auto ml = layout.getMemLayout();
      if (ml && isShardedMemoryLayout(ml.getValue())) {
        total += ttmlir::utils::volume(layout.getGridShape());
      }
    }
    return total;
  };
  return shardedGridVolume(a) > shardedGridVolume(b);
}

//===----------------------------------------------------------------------===//
// Registry: maps OperationName -> OpRuleBook
//===----------------------------------------------------------------------===//

const OpRuleBook &getRuleBook(Operation *op) {
  static OpRuleBook defaultRules;
  static Conv2dRuleBook conv2d;
  static MatmulRuleBook matmul;
  static ConcatRuleBook concat;
  static SliceRuleBook slice;
  static ReshapeRuleBook reshape;
  static PadRuleBook pad;
  static ConcatenateHeadsRuleBook concatHeads;
  static SDPARuleBook sdpa;
  static SDPADecodeRuleBook sdpaDecode;
  static EmbeddingRuleBook embedding;
  static TypecastRuleBook typecast;
  static RotaryEmbeddingRuleBook rotaryEmbedding;
  static SplitQKVRuleBook splitQKV;
  static RmsNormRuleBook rmsNorm;
  static MeshPartitionRuleBook meshPartition;
  static PagedUpdateCacheRuleBook pagedUpdateCache;
  static FillCacheRuleBook fillCache;
  static PagedFillCacheRuleBook pagedFillCache;

  static llvm::DenseMap<mlir::OperationName, const OpRuleBook *> registry;
  static std::once_flag initFlag;
  std::call_once(initFlag, [&] {
    MLIRContext *ctx = op->getContext();
    auto reg = [&](StringRef name, const OpRuleBook *rb) {
      registry[OperationName(name, ctx)] = rb;
    };
    reg(Conv2dOp::getOperationName(), &conv2d);
    reg(ConvTranspose2dOp::getOperationName(), &conv2d);
    reg(MatmulOp::getOperationName(), &matmul);
    reg(LinearOp::getOperationName(), &matmul);
    reg(ConcatOp::getOperationName(), &concat);
    reg(SliceStaticOp::getOperationName(), &slice);
    reg(SliceDynamicOp::getOperationName(), &slice);
    reg(ReshapeOp::getOperationName(), &reshape);

    // TODO(rpavlovicTT): split permute's from reshape's rule book
    // https://github.com/tenstorrent/tt-mlir/issues/7988
    reg(PermuteOp::getOperationName(), &reshape);
    reg(PadOp::getOperationName(), &pad);
    reg(ConcatenateHeadsOp::getOperationName(), &concatHeads);
    reg(NLPConcatHeadsDecodeOp::getOperationName(), &sdpa);
    reg(ScaledDotProductAttentionOp::getOperationName(), &sdpa);
    reg(ScaledDotProductAttentionDecodeOp::getOperationName(), &sdpaDecode);
    reg(PagedScaledDotProductAttentionDecodeOp::getOperationName(),
        &sdpaDecode);
    reg(EmbeddingOp::getOperationName(), &embedding);
    reg(TypecastOp::getOperationName(), &typecast);
    reg(WhereOp::getOperationName(), &typecast);
    reg(RotaryEmbeddingOp::getOperationName(), &rotaryEmbedding);
    reg(RotaryEmbeddingLlamaOp::getOperationName(), &rotaryEmbedding);
    reg(SplitQueryKeyValueAndSplitHeadsOp::getOperationName(), &splitQKV);
    reg(RMSNormOp::getOperationName(), &rmsNorm);
    reg(MeshPartitionOp::getOperationName(), &meshPartition);
    reg(PagedUpdateCacheOp::getOperationName(), &pagedUpdateCache);
    reg(FillCacheOp::getOperationName(), &fillCache);
    reg(PagedFillCacheOp::getOperationName(), &pagedFillCache);
  });
  auto it = registry.find(op->getName());
  return it != registry.end() ? *it->second : defaultRules;
}

} // namespace mlir::tt::ttnn
