// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITIONWORKAROUND_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITIONWORKAROUND_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>

namespace mlir::tt::ttnn::wa {

// Base class for all decomposition workarounds
class DecompositionWorkaround {
public:
  virtual ~DecompositionWorkaround() = default;

  // Check if workaround applies and if so, apply it
  virtual LogicalResult matchAndRewrite(Operation *op,
                                         PatternRewriter &rewriter) const = 0;
};

using DecompositionWorkaroundPtr = std::unique_ptr<DecompositionWorkaround>;
using DecompositionWorkarounds = llvm::SmallVector<DecompositionWorkaroundPtr>;

wa::DecompositionWorkarounds
getRopeDecompositionWorkarounds();

} // namespace mlir::tt::ttnn::wa

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITIONWORKAROUND_H