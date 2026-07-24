// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Pass/PassManager.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MSPLITUNIFIEDTHREAD
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Umbrella pass that splits unified thread regions. It runs the three stages of
// the split back-to-back:
//   1. d2m-assign-threads:   annotate each op with its destination thread and
//                            lower data-movement ops to explicit-CB form.
//   2. d2m-split-threads:    mechanically split into datamovement + compute
//                            regions by the thread annotations.
//   3. d2m-insert-compute-cb: insert the compute-side CB synchronization ops.
class D2MSplitUnifiedThread
    : public impl::D2MSplitUnifiedThreadBase<D2MSplitUnifiedThread> {
public:
  using impl::D2MSplitUnifiedThreadBase<
      D2MSplitUnifiedThread>::D2MSplitUnifiedThreadBase;

  void runOnOperation() final {
    OpPassManager pm(ModuleOp::getOperationName());
    pm.addPass(createD2MAssignThreads());
    pm.addPass(createD2MSplitThreads());
    pm.addPass(createD2MInsertComputeCB());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
