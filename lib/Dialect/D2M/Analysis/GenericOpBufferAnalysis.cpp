// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/GenericOpBufferAnalysis.h"

namespace mlir::tt::d2m {

SmallVector<GenericOpBufferAnalysis::OpConfig>
GenericOpBufferAnalysis::analyzeGenericOp(const Constraints &constraints,
                                          GenericOp op) const {
  SmallVector<OpConfig> results;

  OpConfig result;
  result.operandBufferSettings.reserve(op.getOperands().size());
  result.predictedRuntimeCost = 0.0f;

  SmallVector<SmallVector<int64_t>> operandShardShapes =
      op.getOperandShardShapes(/*convertTileToScalar=*/false);
  for (auto [operandIndex, operand] : llvm::enumerate(op.getOperands())) {
    size_t numBuffers = (constraints.bufferingStrategy ==
                         Constraints::BufferStrategy::DoubleBuffered)
                            ? 2
                            : 1;

    BufferSetting bufferSetting;
    bufferSetting.bufferShape = operandShardShapes[operandIndex];
    bufferSetting.numBuffers = numBuffers;
    result.operandBufferSettings.push_back(std::move(bufferSetting));
  }

  results.push_back(result);
  return results;
}

} // namespace mlir::tt::d2m
