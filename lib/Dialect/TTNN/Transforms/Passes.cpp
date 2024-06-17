// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Passes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNOPENDEVICE
#define GEN_PASS_DEF_CONVERTTTIRTOTTNN
#define GEN_PASS_DEF_TTNNSERIALIZETOBINARY
#include "ttmlir/Dialect/TTNN/Passes.h.inc"

static Value findDevice(Operation *op) {
  Block *block = op->getBlock();
  for (auto &op : block->getOperations()) {
    if (auto deviceOp = dyn_cast<OpenDeviceOp>(op)) {
      return deviceOp.getResult();
    }
  }
  assert(false && "No device found");
  return nullptr;
}

class TTNNOpenDevice : public impl::TTNNOpenDeviceBase<TTNNOpenDevice> {
public:
  using impl::TTNNOpenDeviceBase<TTNNOpenDevice>::TTNNOpenDeviceBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    OpBuilder builder(module);
    auto systemDesc = module->getAttr(tt::SystemDescAttr::name).cast<tt::SystemDescAttr>();
    auto chipDescIndices = systemDesc.getChipDescIndices();
    assert(chipDescIndices.size() == 1 && "Multiple chips not supported yet");

    module->walk([&](func::FuncOp func) {
      // For now just push the open and close device ops to the beginning and
      // end of the function
      assert(func.getBody().hasOneBlock());
      auto *block = &func.getBody().front();
      auto opRange = block->without_terminator();

      builder.setInsertionPoint(block, opRange.begin());
      auto openDevice = builder.create<OpenDeviceOp>(
          func.getLoc(), builder.getType<tt::DeviceType>(
                             builder.getAttr<tt::GridAttr>(), chipDescIndices));

      builder.setInsertionPoint(block, opRange.end());
      builder.create<CloseDeviceOp>(func.getLoc(), openDevice.getResult());
    });
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttnn::TTNNDialect>();
  }
};

class TTIRToTTNNLayoutRewriter : public OpRewritePattern<ttir::LayoutOp> {
public:
  using OpRewritePattern<ttir::LayoutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::LayoutOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ToMemoryConfigOp>(
        op, op->getResultTypes(), op.getInput(), op.getOutput());
    return success();
  }
};

template <typename TTIROp, typename TTNNOp>
class TTIRToTTNNOpRewriter : public OpRewritePattern<TTIROp> {
public:
  using OpRewritePattern<TTIROp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TTIROp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<TTNNOp>(op, op.getResultTypes(), op.getInputs(),
                                        op.getOutputs());
    return success();
  }
};

class TensorEmptyToFullRewriter : public OpRewritePattern<tensor::EmptyOp> {
public:
  using OpRewritePattern<tensor::EmptyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::EmptyOp op,
                                PatternRewriter &rewriter) const final {
    auto device = findDevice(op);
    rewriter.replaceOpWithNewOp<FullOp>(op, op.getType(), device,
                                        rewriter.getF32FloatAttr(0.0));
    return success();
  }
};

class ConvertTTIRToTTNN
    : public impl::ConvertTTIRToTTNNBase<ConvertTTIRToTTNN> {
public:
  using impl::ConvertTTIRToTTNNBase<ConvertTTIRToTTNN>::ConvertTTIRToTTNNBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns
        .add<TTIRToTTNNLayoutRewriter, TTIRToTTNNOpRewriter<ttir::AddOp, AddOp>,
             TTIRToTTNNOpRewriter<ttir::MultiplyOp, MultiplyOp>,
             TensorEmptyToFullRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::ttnn::TTNNDialect>();
  }
};

class TTNNSerializeToBinary
    : public impl::TTNNSerializeToBinaryBase<TTNNSerializeToBinary> {
public:
  using impl::TTNNSerializeToBinaryBase<
      TTNNSerializeToBinary>::TTNNSerializeToBinaryBase;

  void runOnOperation() final { assert(false); }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttnn::TTNNDialect>();
  }
};

void createTTIRToTTNNBackendPipeline(OpPassManager &pm) {
  pm.addPass(mlir::tt::ttir::createTTIRLayout());
  pm.addPass(createTTNNOpenDevice());
  pm.addPass(createConvertTTIRToTTNN());
  // pm.addPass(createTTNNSerializeToBinary());
}

} // namespace mlir::tt::ttnn
