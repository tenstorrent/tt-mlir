// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Conversion/TTIRToTTNN/TTIRToTTNN.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNFORCEDRAMINTERLEAVED
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

static Value getOrInsertDevice(PatternRewriter &rewriter, Operation *op) {
  Block *block = op->getBlock();
  for (auto &op : block->getOperations()) {
    if (auto deviceOp = dyn_cast<ttnn::GetDeviceOp>(op)) {
      return deviceOp.getResult();
    }
  }

  DeviceAttr deviceAttr = getCurrentScopeDevice(op);
  auto currentInsertionPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(block, block->begin());
  auto deviceOp = rewriter.create<ttnn::GetDeviceOp>(
      op->getLoc(), rewriter.getType<DeviceType>(deviceAttr),
      ttnn::MeshShapeAttr::get(op->getContext(), 1, 1));
  rewriter.restoreInsertionPoint(currentInsertionPoint);
  return deviceOp.getResult();
}

class TTNNForceDRAMInterleavedRewriter : public RewritePattern {
public:
  TTNNForceDRAMInterleavedRewriter(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {

    if (isa<ttnn::EmptyOp>(op) || isa<ttnn::ToMemoryConfigOp>(op) ||
        isa<ttnn::FullOp>(op)) {
      return failure();
    }

    if (op->getResults().size() == 0) {
      return failure();
    }
    if (!isa<RankedTensorType>(op->getResult(0).getType())) {
      return failure();
    }

    RankedTensorType result_type =
        mlir::cast<RankedTensorType>(op->getResult(0).getType());
    if (!isa<mlir::tt::LayoutAttr>(result_type.getEncoding())) {
      return failure();
    }

    mlir::tt::LayoutAttr layout_attr =
        mlir::cast<mlir::tt::LayoutAttr>(result_type.getEncoding());
    if (layout_attr.getMemLayout() ==
            mlir::tt::TensorMemoryLayout::Interleaved &&
        layout_attr.getMemorySpace() == mlir::tt::MemorySpace::DeviceDRAM) {
      return failure();
    }
    auto device = getOrInsertDevice(rewriter, op);

    // Now we know that this op does NOT leave its output on device dram -
    // interleaved If any of the ops that consumes this output are not
    // to_memory_config, insert it on the output.
    bool need_to_memcfg = false;
    for (Operation *user : op->getUsers()) {
      if (!isa<ToMemoryConfigOp>(user)) {
        need_to_memcfg = true;
      }
    }
    if (!need_to_memcfg) {
      return failure();
    }

    layout_attr = layout_attr.withMemoryLayout(
        rewriter.getContext(), mlir::tt::TensorMemoryLayout::Interleaved);
    layout_attr = layout_attr.withMemorySpace(
        rewriter.getContext(), mlir::tt::MemorySpace::DeviceDRAM);
    auto new_type = RankedTensorType::get(
        result_type.getShape(), result_type.getElementType(), layout_attr);

    auto to_memory_config = rewriter.create<ToMemoryConfigOp>(
        op->getLoc(), new_type, op->getResult(0), device);

    // Want to replace all uses of the Op output with users of the
    // to_memory_config output However, one of the Op users IS the
    // to_memory_config, so we don't want to replace that with itself.
    rewriter.replaceAllUsesExcept(op->getResult(0), to_memory_config,
                                  to_memory_config);

    // Move to_memory_config sequentially after the op to keep the IR properly
    // ordered
    to_memory_config->moveAfter(op);

    return success();
  }
};

class TTNNForceDRAMInterleaved
    : public impl::TTNNForceDRAMInterleavedBase<TTNNForceDRAMInterleaved> {
public:
  using impl::TTNNForceDRAMInterleavedBase<
      TTNNForceDRAMInterleaved>::TTNNForceDRAMInterleavedBase;

  void runOnOperation() final {
    auto device = getCurrentScopeDevice(getOperation());
    assert(device && "Device not found");
    RewritePatternSet patterns(&getContext());
    patterns.add<TTNNForceDRAMInterleavedRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      getOperation()->dump();
      signalPassFailure();
      return;
    }
  }
};

} // namespace mlir::tt::ttnn
