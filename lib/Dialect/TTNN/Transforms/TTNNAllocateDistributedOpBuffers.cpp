// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Interfaces/TTNNDistributedOpInterface.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/OpModel/TTNN/TTNNOutputTensorInference.h"

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

#ifdef TTMLIR_ENABLE_OPMODEL
    // Align each a2a's persistent metadata drain core to the core its consumer
    // moe_compute reads from, else the layout pipeline reshards the persistent
    // buffers and deadlocks. Stash it as ttnn.moe_metadata_drain_core (read by
    // AllToAllDispatchMetadataOp::allocateBuffers); done here so no pass drops it.
    {
      op_model::ScopedSingletonDeviceGuard deviceGuard(moduleOp);
      moduleOp.walk([&](ttnn::MoeComputeOp mc) {
        CoreRangeSetAttr drainCrs =
            op_model::getMoeTilizeDrainCoreRangeSet(&mc);
        // The persistent metadata (dispatched/indices/scores) all originate at
        // the producer a2a; find it via moe_compute's inputs.
        for (mlir::Value operand : mc->getOperands()) {
          if (auto a2a =
                  operand.getDefiningOp<ttnn::AllToAllDispatchMetadataOp>()) {
            a2a->setAttr("ttnn.moe_metadata_drain_core", drainCrs);
            break;
          }
        }
      });
    }
#endif // TTMLIR_ENABLE_OPMODEL

    moduleOp.walk(
        [&](DistributedOpInterface op) { op.allocateBuffers(rewriter); });
  }
};

} // namespace
} // namespace mlir::tt::ttnn
