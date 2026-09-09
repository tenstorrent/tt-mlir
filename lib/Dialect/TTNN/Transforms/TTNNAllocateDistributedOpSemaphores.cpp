// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Interfaces/TTNNDistributedOpInterface.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNALLOCATEDISTRIBUTEDOPSEMAPHORES
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Walks every DistributedOpInterface op and invokes allocateSemaphores. The
// interface implementation owns the allocation logic and is idempotent.
class TTNNAllocateDistributedOpSemaphores
    : public impl::TTNNAllocateDistributedOpSemaphoresBase<
          TTNNAllocateDistributedOpSemaphores> {
public:
  using impl::TTNNAllocateDistributedOpSemaphoresBase<
      TTNNAllocateDistributedOpSemaphores>::
      TTNNAllocateDistributedOpSemaphoresBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    moduleOp.walk(
        [&](DistributedOpInterface op) { op.allocateSemaphores(rewriter); });
  }
};

} // namespace
} // namespace mlir::tt::ttnn
