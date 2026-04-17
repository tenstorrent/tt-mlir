// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRSTRIPINTERMEDIATETTNNLAYOUTS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

class TTIRStripIntermediateTTNNLayouts
    : public impl::TTIRStripIntermediateTTNNLayoutsBase<
          TTIRStripIntermediateTTNNLayouts> {
public:
  using impl::TTIRStripIntermediateTTNNLayoutsBase<
      TTIRStripIntermediateTTNNLayouts>::TTIRStripIntermediateTTNNLayoutsBase;

  void runOnOperation() final {
    // TODO:
    //   1. For each func::FuncOp in the module:
    //        a. collectPreservedValues(funcOp, preserved)
    //        b. funcOp.walk([&](Operation *op) {
    //             stripIntermediateResultTypes(op, preserved);
    //           });
    //   2. signalPassFailure() on any error.

    ModuleOp module = getOperation();
    module->walk([&](func::FuncOp func) {

    });
  };
}

} // namespace
} // namespace mlir::tt::ttir
