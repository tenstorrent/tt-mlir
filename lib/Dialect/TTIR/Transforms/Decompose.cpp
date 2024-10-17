// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRDECOMPOSE
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Decompose pass
//===----------------------------------------------------------------------===//

class TTIRDecompose : public impl::TTIRDecomposeBase<TTIRDecompose> {
public:
  void runOnOperation() final {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
  }
};

} // namespace mlir::tt::ttir
