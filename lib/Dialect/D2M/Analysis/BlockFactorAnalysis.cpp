// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/BlockFactorAnalysis.h"

namespace mlir::tt::ttir {

SmallVector<BlockFactorAnalysis::BufferConfig>
BlockFactorAnalysis::analyzeGenericOp(GenericOp op) {
  SmallVector<BufferConfig> results;

  BufferConfig result;
  result.operand_buffer_settings.reserve(op.getOperands().size());
  result.predicted_runtime_cost = 0.0f;

  SmallVector<SmallVector<int64_t>> operandShardShapes =
      op.getOperandShardShapes(/*convertTileToScalar=*/false);
  for (auto [operandIndex, operand] : llvm::enumerate(op.getOperands())) {
    size_t num_buffers =
        (constraints.buffering_strategy ==
         BlockFactorAnalysisConstraints::BufferStrategy::DOUBLE_BUFFERED)
            ? 2
            : 1;

    BufferSetting bufferSetting;
    bufferSetting.buffer_shape = operandShardShapes[operandIndex];
    bufferSetting.num_buffers = num_buffers;
    result.operand_buffer_settings.push_back(bufferSetting);
  }

  results.push_back(result);
  return results;
}

} // namespace mlir::tt::ttir
