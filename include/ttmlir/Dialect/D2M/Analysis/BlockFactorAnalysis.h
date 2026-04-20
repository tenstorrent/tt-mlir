// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_BLOCKFACTORANALYSIS_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_BLOCKFACTORANALYSIS_H

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::d2m {

/// Analysis that determines optimal block factors for d2m.generic ops.
///
/// For each generic op with an iteration space (i.e., not DMA-only or
/// explicit-datamovement form), this analysis computes reblocked block
/// factors according to the configured buffer size policy.
struct BlockFactorAnalysis {
  enum class BufferSizePolicy { Auto, Min, Max };

  struct Options {
    BufferSizePolicy policy = BufferSizePolicy::Auto;
    uint32_t numBuffers = 2;
    bool allowAliasedEltwiseBlocking = true;
  };

  struct Result {
    SmallVector<int64_t> reblockedFactors;
  };

  BlockFactorAnalysis(Operation *op, const Options &opts);

  const Result *lookup(GenericOp genericOp) const;

private:
  llvm::DenseMap<Operation *, Result> results;
};

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_BLOCKFACTORANALYSIS_H
