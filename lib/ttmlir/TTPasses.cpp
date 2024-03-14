// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "ttmlir/TTPasses.h"

namespace mlir::tt {
#define GEN_PASS_DEF_TTSWITCHBARFOO
#include "ttmlir/TTPasses.h.inc"

namespace {
class TTSwitchBarFooRewriter : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    std::cout << std::string("asdf ") << op->getName().getStringRef().str()
              << std::endl;
    op->dump();
    if (op->getName().getStringRef() == "bar") {
      // rewriter.modifyOpInPlace(op, [&op]() { op.setSymName("foo"); });
      return success();
    }
    return failure();
  }
};

class TTSwitchBarFoo
    : public impl::TTSwitchBarFooBase<TTSwitchBarFoo> {
public:
  using impl::TTSwitchBarFooBase<
      TTSwitchBarFoo>::TTSwitchBarFooBase;
  void runOnOperation() final {
    std::cout << std::string("asdfasdf") << std::endl;
    RewritePatternSet patterns(&getContext());
    patterns.add<TTSwitchBarFooRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::tt
