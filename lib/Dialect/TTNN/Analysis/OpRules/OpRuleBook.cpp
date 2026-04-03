// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/ConvRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/DataMovementRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/EmbeddingRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/MatmulRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/TransformerRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/TypecastRules.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

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
  result.attemptL1Sharding = true;
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
  static EmbeddingRuleBook embedding;
  static TypecastRuleBook typecast;
  static RotaryEmbeddingRuleBook rotaryEmbedding;

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
    reg(PermuteOp::getOperationName(), &reshape);
    reg(PadOp::getOperationName(), &pad);
    reg(ConcatenateHeadsOp::getOperationName(), &concatHeads);
    reg(NLPConcatHeadsDecodeOp::getOperationName(), &sdpa);
    reg(ScaledDotProductAttentionDecodeOp::getOperationName(), &sdpa);
    reg(PagedScaledDotProductAttentionDecodeOp::getOperationName(), &sdpa);
    reg(ScaledDotProductAttentionOp::getOperationName(), &sdpa);
    reg(EmbeddingOp::getOperationName(), &embedding);
    reg(TypecastOp::getOperationName(), &typecast);
    reg(WhereOp::getOperationName(), &typecast);
    reg(RotaryEmbeddingOp::getOperationName(), &rotaryEmbedding);
    reg(RotaryEmbeddingLlamaOp::getOperationName(), &rotaryEmbedding);
  });
  auto it = registry.find(op->getName());
  return it != registry.end() ? *it->second : defaultRules;
}

} // namespace mlir::tt::ttnn
