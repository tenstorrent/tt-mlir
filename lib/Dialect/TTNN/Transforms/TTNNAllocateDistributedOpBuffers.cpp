// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Interfaces/TTNNDistributedOpInterface.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNALLOCATEDISTRIBUTEDOPBUFFERS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Walks every DistributedOpInterface op and invokes allocateBuffers. The
// interface implementation owns the allocation logic and is idempotent.
class TTNNAllocateDistributedOpBuffers
    : public impl::TTNNAllocateDistributedOpBuffersBase<
          TTNNAllocateDistributedOpBuffers> {
public:
  using impl::TTNNAllocateDistributedOpBuffersBase<
      TTNNAllocateDistributedOpBuffers>::TTNNAllocateDistributedOpBuffersBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    moduleOp.walk(
        [&](DistributedOpInterface op) { op.allocateBuffers(rewriter); });
  }
};

} // namespace
} // namespace mlir::tt::ttnn
