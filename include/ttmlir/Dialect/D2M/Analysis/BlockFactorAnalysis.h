// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_ANALYSIS_BLOCKFACTORANALYSIS_H
#define TTMLIR_DIALECT_TTIR_ANALYSIS_BLOCKFACTORANALYSIS_H

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::d2m {

struct BlockFactorAnalysisConstraints {
  enum class BufferStrategy : uint32_t { SingleBuffered, DoubleBuffered };
  BufferStrategy buffering_strategy = BufferStrategy::DoubleBuffered;
};

/// Analysis for determining buffer configuration options for ttir::GenericOp
/// Multiple buffering options may be returned with different runtime tradeoffs;
/// each BufferConfig represents an implementable set of buffering decisions for
/// all operands (hopefully) representing a point on the implementation pareto
/// frontier.
class BlockFactorAnalysis {
public:
  /// Buffer settings for a single operand.
  struct BufferSetting {
    SmallVector<int64_t> buffer_shape; // in terms of elements
    size_t num_buffers;
  };

  /// Buffer configuration options for a single GenericOp.
  struct BufferConfig {
    /// Buffer settings indexed by operand index of the associated GenericOp.
    SmallVector<BufferSetting> operand_buffer_settings;
    float predicted_runtime_cost = 0.0f;
  };

  BlockFactorAnalysis() = default;
  explicit BlockFactorAnalysis(BlockFactorAnalysisConstraints constraints)
      : constraints(constraints) {}
  ~BlockFactorAnalysis() = default;

  BlockFactorAnalysis(const BlockFactorAnalysis &) = default;
  BlockFactorAnalysis &operator=(const BlockFactorAnalysis &) = default;
  BlockFactorAnalysis(BlockFactorAnalysis &&) = default;
  BlockFactorAnalysis &operator=(BlockFactorAnalysis &&) = default;

  // Analyze a ttir::GenericOp and return buffering
  SmallVector<BufferConfig> analyzeGenericOp(GenericOp op);

  const BlockFactorAnalysisConstraints &getConstraints() const {
    return constraints;
  }

private:
  BlockFactorAnalysisConstraints constraints;
};

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_TTIR_ANALYSIS_BLOCKFACTORANALYSIS_H
