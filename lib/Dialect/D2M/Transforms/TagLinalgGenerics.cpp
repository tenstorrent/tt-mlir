// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/Analysis/DestRegisterAnalysis.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MTAGLINALGGENERICS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MTagLinalgGenerics
    : public impl::D2MTagLinalgGenericsBase<D2MTagLinalgGenerics> {
public:
  using impl::D2MTagLinalgGenericsBase<
      D2MTagLinalgGenerics>::D2MTagLinalgGenericsBase;

  void runOnOperation() final {

    // Run the analysis once before tagging any operations.
    DestRegisterAnalysis analysis = getAnalysis<DestRegisterAnalysis>();

    // Tag each generic op with its counter.
    getOperation()->walk([&](linalg::GenericOp genericOp) {
      auto it = analysis.opToGenericOpCounter.find(genericOp.getOperation());
      if (it != analysis.opToGenericOpCounter.end()) {
        int counter = it->second;
        // Add the counter as an attribute.
        genericOp->setAttr(
            "generic_op_counter",
            IntegerAttr::get(IntegerType::get(genericOp->getContext(), 32),
                             counter));
      }
    });

    // Mark the analysis as preserved for use by downstream passes.
    markAnalysesPreserved<DestRegisterAnalysis>();
  }
};
} // namespace

} // namespace mlir::tt::d2m
