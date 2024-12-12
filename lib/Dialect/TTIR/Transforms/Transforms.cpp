// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRSLIDINGWINDOW2DFIXSHAPES
#define GEN_PASS_DEF_TTIRFUSERELU
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

class TTIRFuseRelu : public impl::TTIRFuseReluBase<TTIRFuseRelu> {
public:
  using impl::TTIRFuseReluBase<TTIRFuseRelu>::TTIRFuseReluBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseRelu>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace mlir::tt::ttir
