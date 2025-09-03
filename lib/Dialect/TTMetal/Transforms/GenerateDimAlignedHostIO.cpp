// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"
#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::tt::ttmetal {
#define GEN_PASS_DEF_TTMETALGENERATEDIMALIGNEDHOSTIO
#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h.inc"

namespace {
// Extract the dimAlignments attached in the ttcore.host_layout attr of the
// enququeRead/Write op, and create a HostLayoutAttr to pass the info to the
// bounce tensor's memref.alloc.
MemRefType getHostAlignedMemref(Operation *op, MemRefType hostMemref) {
  auto hostLayout =
      op->getAttrOfType<ttcore::MetalLayoutAttr>(ttcore::HostLayoutAttr::name);
  const auto dimAlignments = llvm::to_vector(hostLayout.getDimAlignments());
  op->removeAttr(ttcore::HostLayoutAttr::name);

  auto hostLayoutAttr = ttcore::HostLayoutAttr::get(
      op->getContext(), hostMemref.getShape(), dimAlignments);
  return MemRefType::get(hostMemref.getShape(), hostMemref.getElementType(),
                         hostLayoutAttr, hostMemref.getMemorySpace());
}

// Determine whether we actually need to create a bounce tensor with the
// aligned shape and do strided copy or not. Since the logical & device shapes
// might not have the same rank, compare volumes instead.
bool matchingHostDeviceShapes(MemRefType hostMemref, MemRefType deviceMemref) {
  const auto hostShape = hostMemref.getShape();
  const auto deviceShape = deviceMemref.getShape();
  const auto hostVolume =
      ttmlir::utils::product(hostShape.begin(), hostShape.end());
  const auto deviceVolume =
      ttmlir::utils::product(deviceShape.begin(), deviceShape.end());
  return hostVolume == deviceVolume;
}
} // namespace

namespace {
struct AlignedHostInputRewriter
    : public OpRewritePattern<ttmetal::EnqueueWriteBufferOp> {
  using OpRewritePattern<ttmetal::EnqueueWriteBufferOp>::OpRewritePattern;

  // For enqueue write (to device), copy the input tensor to a fully-aligned
  // bounce tensor, and pass the latter to the enqueue write instead.
  LogicalResult matchAndRewrite(ttmetal::EnqueueWriteBufferOp op,
                                PatternRewriter &rewriter) const final {
    auto hostMemref = op.getInput().getType();
    auto deviceMemref = op.getOutput().getType();
    assert(hostMemref.hasStaticShape());
    assert(deviceMemref.hasStaticShape());
    // Either a block's input or a global constant.
    assert(op.getInput().getDefiningOp() == nullptr ||
           mlir::isa<memref::GetGlobalOp>(op.getInput().getDefiningOp()));
    assert(mlir::isa<BlockArgument>(op.getInput()) ||
           mlir::isa<MemRefType>(op.getInput().getType()));
    assert(mlir::isa<ttmetal::CreateBufferOp>(op.getOutput().getDefiningOp()));
    if (matchingHostDeviceShapes(hostMemref, deviceMemref)) {
      op->removeAttr(ttcore::HostLayoutAttr::name);
      return success();
    }

    auto hostAlignedMemref = getHostAlignedMemref(op, hostMemref);

    rewriter.setInsertionPoint(op);
    auto bounceAllocOp =
        rewriter.create<memref::AllocOp>(op.getLoc(), hostAlignedMemref);

    rewriter.create<memref::CopyOp>(op.getLoc(), op.getInput(), bounceAllocOp);
    op.getInputMutable().assign(bounceAllocOp.getResult());

    return success();
  }
};
} // namespace

namespace {
struct AlignedHostOutputRewriter
    : public OpRewritePattern<ttmetal::EnqueueReadBufferOp> {
  using OpRewritePattern<ttmetal::EnqueueReadBufferOp>::OpRewritePattern;

  // For enqueue read (from device), let enqueue read store data into the
  // fully-aligned bounce tensor, and copy the data to the original output
  // tensor.
  LogicalResult matchAndRewrite(ttmetal::EnqueueReadBufferOp op,
                                PatternRewriter &rewriter) const final {
    auto hostMemref = op.getOutput().getType();
    auto deviceMemref = op.getInput().getType();
    assert(hostMemref.hasStaticShape());
    assert(deviceMemref.hasStaticShape());
    assert(mlir::isa<ttmetal::CreateBufferOp>(op.getInput().getDefiningOp()));
    assert(mlir::isa<memref::AllocOp>(op.getOutput().getDefiningOp()));
    if (matchingHostDeviceShapes(hostMemref, deviceMemref)) {
      op->removeAttr(ttcore::HostLayoutAttr::name);
      return success();
    }

    auto hostAlignedMemref = getHostAlignedMemref(op, hostMemref);

    rewriter.setInsertionPoint(op);
    auto bounceAllocOp =
        rewriter.create<memref::AllocOp>(op.getLoc(), hostAlignedMemref);

    auto allocOp =
        mlir::dyn_cast<memref::AllocOp>(op.getOutput().getDefiningOp());
    op.getOutputMutable().assign(bounceAllocOp.getResult());

    rewriter.setInsertionPointAfter(op);
    rewriter.create<memref::CopyOp>(op.getLoc(), bounceAllocOp, allocOp);

    return success();
  }
};
} // namespace

namespace {
class TTMetalGenerateDimAlignedHostIO
    : public impl::TTMetalGenerateDimAlignedHostIOBase<
          TTMetalGenerateDimAlignedHostIO> {
public:
  using impl::TTMetalGenerateDimAlignedHostIOBase<
      TTMetalGenerateDimAlignedHostIO>::TTMetalGenerateDimAlignedHostIOBase;

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<AlignedHostInputRewriter, AlignedHostOutputRewriter>(ctx);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::ttmetal
