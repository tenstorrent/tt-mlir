// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNPREPARED2MSUBGRAPHSFORTRACE
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// D2M subgraph lowering emits each subgraph as a get_device -> empty -> generic
// triple, leaving the (non-hoistable) get_device and empty ops interleaved
// between hoistable ops. This pass pulls those non-hoistable ops out of the way
// so TTNNTraceHoistTransform sees a contiguous run of hoistable ops:
//   - all get_device ops are merged into one at the top of the function, and
//   - all ttnn.empty scratch buffers are moved into the prelude, ahead of the
//     first hoistable op.
class TTNNPrepareD2MSubgraphsForTrace
    : public impl::TTNNPrepareD2MSubgraphsForTraceBase<
          TTNNPrepareD2MSubgraphsForTrace> {
public:
  using impl::TTNNPrepareD2MSubgraphsForTraceBase<
      TTNNPrepareD2MSubgraphsForTrace>::TTNNPrepareD2MSubgraphsForTraceBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    moduleOp.walk([&](func::FuncOp funcOp) {
      if (!ttmlir::utils::isForwardDeviceFunc(funcOp)) {
        return;
      }
      if (funcOp.getBlocks().size() != 1) {
        return;
      }
      prepareFuncOp(rewriter, funcOp);
    });
  }

private:
  // Merge every ttnn.get_device in `block` into a single op placed at the top
  // of the block and return it. Device handles are interchangeable, so all uses
  // are redirected to the one canonical device.
  GetDeviceOp mergeGetDeviceOps(IRRewriter &rewriter, Block &block) {
    // getOrInsertDevice returns the first existing get_device, or inserts one
    // at the top of the block if none exist.
    GetDeviceOp canonicalDevice = utils::getOrInsertDevice(rewriter, &block);

    // The existing device op may sit mid-block (D2M emits it next to each
    // subgraph); move the canonical one into the prelude so it dominates the
    // sunk creation ops.
    if (&block.front() != canonicalDevice.getOperation()) {
      canonicalDevice->moveBefore(&block.front());
    }

    // Redirect and erase the remaining (duplicate) device ops.
    llvm::SmallVector<GetDeviceOp> duplicateDevices;
    for (GetDeviceOp deviceOp : block.getOps<GetDeviceOp>()) {
      if (deviceOp != canonicalDevice) {
        duplicateDevices.push_back(deviceOp);
      }
    }
    for (GetDeviceOp duplicateDevice : duplicateDevices) {
      rewriter.replaceOp(duplicateDevice, canonicalDevice.getResult());
    }

    return canonicalDevice;
  }

  // Move all ttnn.empty scratch buffers into the prelude, immediately after
  // `deviceOp`, preserving their relative order. Their only operand is the
  // device, which now dominates the insertion point. Only empty ops are handled
  // for now; other creation ops can be added here if a need arises.
  void moveEmptyOpsToPrelude(GetDeviceOp deviceOp, Block &block) {
    llvm::SmallVector<EmptyOp> emptyOps(block.getOps<EmptyOp>());

    Operation *insertAfter = deviceOp;
    for (EmptyOp emptyOp : emptyOps) {
      emptyOp->moveAfter(insertAfter);
      insertAfter = emptyOp;
    }
  }

  void prepareFuncOp(IRRewriter &rewriter, func::FuncOp funcOp) {
    Block &block = funcOp.getBlocks().front();

    // Nothing to prepare unless the function has a device (and therefore the
    // D2M subgraph prelude ops we care about).
    if (block.getOps<GetDeviceOp>().empty()) {
      return;
    }

    GetDeviceOp deviceOp = mergeGetDeviceOps(rewriter, block);
    moveEmptyOpsToPrelude(deviceOp, block);
  }
};

} // namespace
} // namespace mlir::tt::ttnn
