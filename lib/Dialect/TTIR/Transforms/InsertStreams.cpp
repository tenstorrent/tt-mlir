// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRALWAYSINSERTSTREAMS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
class TTIRInsertStreams final : public OpRewritePattern<ttir::GenericOp> {
public:
  TTIRInsertStreams(MLIRContext *context)
      : OpRewritePattern<ttir::GenericOp>(context) {}

  LogicalResult matchAndRewrite(ttir::GenericOp op,
                                PatternRewriter &rewriter) const final {
    // unsigned inputCount = op.getInputs().size();
    // unsigned outputCount = op.getOutputs().size();
    // llvm::outs() << "TTIRAlwaysInsertStreams: Inputs: " << inputCount << ",
    // Outputs: " << outputCount << "\n";

    bool modified = false;
    for (OpOperand &operand : op->getOpOperands()) {
      // unsigned operandNum = operand.getOperandNumber();
      // bool isOutput = operandNum >= inputCount;

      if (!needsStream(operand.get())) {
        continue;
      }
      insertStream(rewriter, operand, op);
      modified = true;
    }

    return success(modified);
  }

  static bool needsStream(Value operand) {
    auto *definingOp = operand.getDefiningOp();
    // operand is already a stream, abort
    if (mlir::isa_and_nonnull<ttir::StreamLayoutOp>(definingOp)) {
      return false;
    }

    // skip special DMA-only form of generic; stream insertion will break
    // semantics
    if (auto genericOp = mlir::dyn_cast_or_null<ttir::GenericOp>(definingOp)) {
      // Lack of a compute thread implies this is in special DMA-only form,
      // which should not have streams inferred.
      if (not llvm::any_of(genericOp.getThreads(), [](Attribute attr) {
            return mlir::cast<ttir::ThreadAttr>(attr).getThreadType() ==
                   ttir::ThreadType::Compute;
          })) {
        return false;
      }
    }

    // for everything else, it needs a stream!
    return true;
  }

  void insertStream(PatternRewriter &rewriter, OpOperand &operand,
                    ttir::GenericOp op) const {
    auto memref = mlir::cast<MemRefType>(operand.get().getType());
    auto streamAttr = rewriter.getAttr<ttcore::ViewLayoutAttr>(
        rewriter.getMultiDimIdentityMap(memref.getRank()));
    auto streamMemref =
        MemRefType::get(memref.getShape(), memref.getElementType(), streamAttr,
                        memref.getMemorySpace());
    auto storageAttr = ttcore::ShardLayoutAttr::get(memref, /*buffers=*/1);
    auto storageMemref =
        MemRefType::get(memref.getShape(), memref.getElementType(), storageAttr,
                        memref.getMemorySpace());
    auto storage = rewriter.create<memref::AllocOp>(op.getLoc(), storageMemref);
    auto streamLayout = rewriter.create<ttir::StreamLayoutOp>(
        op.getLoc(), streamMemref, operand.get(), storage);
    rewriter.modifyOpInPlace(
        op, [&]() { operand.assign(streamLayout.getResult()); });
  }
};
} // namespace

namespace {
class TTIRAlwaysInsertStreams final
    : public impl::TTIRAlwaysInsertStreamsBase<TTIRAlwaysInsertStreams> {
public:
  using impl::TTIRAlwaysInsertStreamsBase<
      TTIRAlwaysInsertStreams>::TTIRAlwaysInsertStreamsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRInsertStreams>(&getContext());
    if (failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::ttir
