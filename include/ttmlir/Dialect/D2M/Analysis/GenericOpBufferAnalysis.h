// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_GENERICOPBUFFERANALYSIS_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_GENERICOPBUFFERANALYSIS_H

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::d2m {

/// Analysis for determining buffer configuration options for ttir::GenericOp
/// Multiple buffering options may be returned with different runtime tradeoffs;
/// each BufferConfig represents an implementable set of buffering decisions for
/// all operands (hopefully) representing a point on the implementation pareto
/// frontier.
class GenericOpBufferAnalysis {
public:
  struct Constraints {
    enum class BufferStrategy : uint32_t { SingleBuffered, DoubleBuffered };
    BufferStrategy bufferingStrategy = BufferStrategy::DoubleBuffered;
  };

  /// Buffer settings for a single operand.
  struct BufferSetting {
    SmallVector<int64_t> bufferShape; // in terms of elements
    std::uint32_t numBuffers;
  };

  /// Configuration for a GenericOp with predicted cost.
  struct OpConfig {
    /// Buffer settings indexed by operand index of the associated GenericOp.
    SmallVector<BufferSetting> operandBufferSettings;
    float predictedRuntimeCost = 0.0f;
  };

  GenericOpBufferAnalysis() {}
  ~GenericOpBufferAnalysis() = default;

  // Analyze a ttir::GenericOp and return multiple buffering configurations.
  SmallVector<OpConfig>
  analyzeGenericOp(const GenericOpBufferAnalysis::Constraints &constraints,
                   GenericOp op) const;
};

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_GENERICOPBUFFERANALYSIS_H
