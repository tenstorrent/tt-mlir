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

namespace mlir::tt::ttmetal {
#define GEN_PASS_DEF_TTMETALGENERATEDIMALIGNEDHOSTIO
#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h.inc"

namespace {
class TTMetalGenerateDimAlignedHostIO
    : public impl::TTMetalGenerateDimAlignedHostIOBase<
          TTMetalGenerateDimAlignedHostIO> {

  static constexpr llvm::StringLiteral hostInfoAttr = "host_info";

  // Extract the dimAlignments attached in the host_info attr of the
  // enququeRead/Write op, and create a HostLayoutAttr to pass the info to the
  // bounce tensor's memref.alloc.
  MemRefType getHostAlignedMemref(ModuleOp module, Operation *op,
                                  MemRefType hostMemref) {
    auto hostInfo = op->getAttrOfType<ttcore::MetalLayoutAttr>(hostInfoAttr);
    const auto dimAlignments = llvm::to_vector(hostInfo.getDimAlignments());
    op->removeAttr(hostInfoAttr);

    auto hostLayoutAttr = ttcore::HostLayoutAttr::get(
        module.getContext(), hostMemref.getShape(), dimAlignments);
    return MemRefType::get(hostMemref.getShape(), hostMemref.getElementType(),
                           hostLayoutAttr, hostMemref.getMemorySpace());
  };

  // Determine whether we actually need to create a bounce tensor with the
  // aligned shape and do strided copy or not. Since the logical & device shapes
  // might not have the same rank, compare volumes instead.
  bool matchingHostDeviceShapes(MemRefType hostMemref,
                                MemRefType deviceMemref) const {
    const auto hostShape = hostMemref.getShape();
    const auto deviceShape = deviceMemref.getShape();
    const auto hostVolume =
        ttmlir::utils::product(hostShape.begin(), hostShape.end());
    const auto deviceVolume =
        ttmlir::utils::product(deviceShape.begin(), deviceShape.end());
    return hostVolume == deviceVolume;
  };

  // For enqueue write (to device), copy the input tensor to a fully-aligned
  // bounce tensor, and pass the latter to the enqueue write instead.
  void rewriteEnqueueWrites(ModuleOp module, IRRewriter &rewriter,
                            ttmetal::EnqueueWriteBufferOp op) {
    auto hostMemref = op.getInput().getType();
    auto deviceMemref = op.getOutput().getType();
    assert(hostMemref.hasStaticShape());
    assert(deviceMemref.hasStaticShape());
    assert(op.getInput().getDefiningOp() == nullptr);
    assert(mlir::isa<BlockArgument>(op.getInput()));
    assert(mlir::isa<ttmetal::CreateBufferOp>(op.getOutput().getDefiningOp()));
    if (matchingHostDeviceShapes(hostMemref, deviceMemref)) {
      return;
    }

    auto hostAlignedMemref = getHostAlignedMemref(module, op, hostMemref);

    rewriter.setInsertionPoint(op);
    auto bounceAllocOp =
        rewriter.create<memref::AllocOp>(op.getLoc(), hostAlignedMemref);

    rewriter.create<memref::CopyOp>(op.getLoc(), op.getInput(), bounceAllocOp);
    op.getInputMutable().assign(bounceAllocOp.getResult());
  }

  // For enqueue read (from device), let enqueue read store data into the
  // fully-aligned bounce tensor, and copy the data to the original output
  // tensor.
  void rewriteEnqueueReads(ModuleOp module, IRRewriter &rewriter,
                           ttmetal::EnqueueReadBufferOp op) {
    auto hostMemref = op.getOutput().getType();
    auto deviceMemref = op.getInput().getType();
    assert(hostMemref.hasStaticShape());
    assert(deviceMemref.hasStaticShape());
    assert(mlir::isa<ttmetal::CreateBufferOp>(op.getInput().getDefiningOp()));
    assert(mlir::isa<memref::AllocOp>(op.getOutput().getDefiningOp()));
    if (matchingHostDeviceShapes(hostMemref, deviceMemref)) {
      return;
    }

    auto hostAlignedMemref = getHostAlignedMemref(module, op, hostMemref);

    rewriter.setInsertionPoint(op);
    auto bounceAllocOp =
        rewriter.create<memref::AllocOp>(op.getLoc(), hostAlignedMemref);

    auto allocOp =
        mlir::dyn_cast<memref::AllocOp>(op.getOutput().getDefiningOp());
    op.getOutputMutable().assign(bounceAllocOp.getResult());

    rewriter.setInsertionPointAfter(op);
    rewriter.create<memref::CopyOp>(op.getLoc(), bounceAllocOp, allocOp);
  }

public:
  using impl::TTMetalGenerateDimAlignedHostIOBase<
      TTMetalGenerateDimAlignedHostIO>::TTMetalGenerateDimAlignedHostIOBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    module->walk([&](Operation *op) {
      if (auto w = mlir::dyn_cast<ttmetal::EnqueueWriteBufferOp>(op)) {
        rewriteEnqueueWrites(module, rewriter, w);
      } else if (auto r = mlir::dyn_cast<ttmetal::EnqueueReadBufferOp>(op)) {
        rewriteEnqueueReads(module, rewriter, r);
      }
    });
  }
};
} // namespace

} // namespace mlir::tt::ttmetal
