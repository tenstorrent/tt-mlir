// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <dbg.h>

namespace mlir::tt::ttmetal {
#define GEN_PASS_DEF_TTMETALGENERATESTRIDEDHOSTIO
#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h.inc"

namespace {

class TTMetalFooRewriter
    : public OpRewritePattern<ttmetal::EnqueueWriteBufferOp> {
public:
  using OpRewritePattern<ttmetal::EnqueueWriteBufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttmetal::EnqueueWriteBufferOp op,
                                PatternRewriter &rewriter) const final {
    return success();
  }
};

} // namespace

namespace {
class TTMetalGenerateStridedHostIO
    : public impl::TTMetalGenerateStridedHostIOBase<
          TTMetalGenerateStridedHostIO> {
public:
  using impl::TTMetalGenerateStridedHostIOBase<
      TTMetalGenerateStridedHostIO>::TTMetalGenerateStridedHostIOBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<ttmetal::EnqueueWriteBufferOp> enqueueWrites;
    SmallVector<ttmetal::EnqueueReadBufferOp> enqueueReads;

    module->walk([&](Operation *op) {
      if (auto w = mlir::dyn_cast<ttmetal::EnqueueWriteBufferOp>(op)) {
        enqueueWrites.push_back(w);
      } else if (auto r = mlir::dyn_cast<ttmetal::EnqueueReadBufferOp>(op)) {
        enqueueReads.push_back(r);
      }
    });

    // Dedup this?
    static constexpr llvm::StringLiteral hostInfoAttr("host_info");

    for (auto op : enqueueWrites) {
      fprintf(stderr, "-- WriteOp: ");
      op.dump();
      auto hostMemref = op.getInput().getType();
      auto deviceMemref = op.getOutput().getType();
      assert(hostMemref.hasStaticShape());
      assert(deviceMemref.hasStaticShape());
      assert(op.getInput().getDefiningOp() == nullptr);
      assert(mlir::isa<BlockArgument>(op.getInput()));
      assert(
          mlir::isa<ttmetal::CreateBufferOp>(op.getOutput().getDefiningOp()));
      auto hostShape = hostMemref.getShape();
      auto deviceShape = deviceMemref.getShape();
      fprintf(stderr, "---- Shapes: ");
      dbg(hostShape, deviceShape);

      auto hostInfo = op->getAttrOfType<ttcore::MetalLayoutAttr>(hostInfoAttr);
      llvm::SmallVector<int64_t> dimAlignments =
          llvm::to_vector(hostInfo.getDimAlignments());
      op->removeAttr(hostInfoAttr);
      fprintf(stderr, "---- dimAlignments: ");
      dbg(dimAlignments);
    }

    for (auto op : enqueueReads) {
      fprintf(stderr, "-- ReadOp: ");
      op.dump();
      auto hostMemref = op.getOutput().getType();
      auto deviceMemref = op.getInput().getType();
      assert(hostMemref.hasStaticShape());
      assert(deviceMemref.hasStaticShape());
      assert(mlir::isa<memref::AllocOp>(op.getOutput().getDefiningOp()));
      auto hostShape = hostMemref.getShape();
      auto deviceShape = deviceMemref.getShape();
      fprintf(stderr, "---- Shapes: ");
      dbg(hostShape, deviceShape);

      auto hostInfo = op->getAttrOfType<ttcore::MetalLayoutAttr>(hostInfoAttr);
      llvm::SmallVector<int64_t> dimAlignments =
          llvm::to_vector(hostInfo.getDimAlignments());
      op->removeAttr(hostInfoAttr);
      fprintf(stderr, "---- dimAlignments: ");
      dbg(dimAlignments);

      IRRewriter rewriter(&getContext());

      auto allocOp =
          mlir::dyn_cast<memref::AllocOp>(op.getOutput().getDefiningOp());
      auto hostStridesAttr = ttcore::HostStridesLayoutAttr::get(
          module.getContext(), hostShape, dimAlignments);
      auto hostStridesMemref =
          MemRefType::get(hostShape, hostMemref.getElementType(),
                          hostStridesAttr, hostMemref.getMemorySpace());

      // Removed due to WA
      // assert(hostStridesAttr.getStride()[0] *
      //            ttmlir::utils::roundUp(hostShape[0], dimAlignments[0]) ==
      //        ttmlir::utils::product(deviceShape.begin(), deviceShape.end()));

      rewriter.setInsertionPoint(op);
      auto newAllocOp =
          rewriter.create<memref::AllocOp>(op.getLoc(), hostStridesMemref);

      op.getOutputMutable().assign(newAllocOp.getResult());

      // Is the inserte locaiton OK?
      rewriter.setInsertionPointAfter(op);
      rewriter.create<memref::CopyOp>(op.getLoc(), newAllocOp, allocOp);

      // It's too late to insert DeallocOp here...
      // rewriter.setInsertionPointAfter(copyOp);
      // rewriter.create<memref::DeallocOp>(copyOp.getLoc(), newAllocOp);
    }

    // signalPassFailure();
  }
};
} // namespace

} // namespace mlir::tt::ttmetal
