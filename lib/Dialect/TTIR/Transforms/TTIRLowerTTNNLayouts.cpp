// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include <iostream>

using namespace mlir;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRLOWERTTNNLAYOUTS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
static bool isTTNNTensor(Type type) {
  auto maybeTensor = mlir::dyn_cast<RankedTensorType>(type);
  if (!maybeTensor) {
    return false;
  }
  auto enc = maybeTensor.getEncoding();
  return enc && mlir::isa<ttnn::TTNNLayoutAttr>(enc);
}

class LowerTTNNLayoutsPattern : public RewritePattern {
public:
  LowerTTNNLayoutsPattern(MLIRContext *context,
                          llvm::SmallVector<int64_t> targetSquareGridShape)
      : RewritePattern(Pattern::MatchAnyOpTypeTag(), /*benefit=*/1, context),
        targetSquareGridShape(targetSquareGridShape) {}

  Type getMetalTensorType(PatternRewriter &rewriter, Value value) const {
    auto tensorType = mlir::cast<mlir::RankedTensorType>(value.getType());
    auto ttnnLayout =
        mlir::cast<ttnn::TTNNLayoutAttr>(tensorType.getEncoding());

    // Get memory space from the TTNN memref
    assert(ttnnLayout.isDeviceBufferType() && "Must be a device tensor");
    ttcore::MemorySpace memSpace =
        ttnnLayout.getBufferType() == ttnn::BufferType::DRAM
            ? ttcore::MemorySpace::DeviceDRAM
            : ttcore::MemorySpace::DeviceL1;
    // With these assumptions we can use the default alignment and dim
    // collapsing behaviour in the MetalLayoutAttr
    assert(ttnnLayout.isTiled() &&
           "Row major TTNN layouts are not supported yet");
    assert(
        mlir::cast<ttcore::TileType>(ttnnLayout.getElementType()).getHeight() ==
            ttcore::TileType::getDefaultShape()[0] &&
        mlir::cast<ttcore::TileType>(ttnnLayout.getElementType()).getWidth() ==
            ttcore::TileType::getDefaultShape()[1] &&
        "Only default tile shape is supported");
    assert(ttnnLayout.hasL1BufferType() &&
           ttnnLayout.getMemLayout().getValue() ==
               ttnn::TensorMemoryLayout::BlockSharded &&
           "Only block sharded L1 tensor memory layout is supported");

    Type elementType = ttnnLayout.getElementType();
    auto i64Ty = IntegerType::get(rewriter.getContext(), 64);
    auto intervalTy = RankedTensorType::get({1, 2}, i64Ty);
    DenseIntElementsAttr collapsedIntervals =
        DenseIntElementsAttr::get(intervalTy, llvm::ArrayRef<int64_t>({0, -1}));

    // The index map in TTNNLayoutAttr is for collapsing an N-D tensor on to
    // the grid. It has no relevance to the index map in MetalLayoutAttr.
    // Hardcode collapse intervals to [[0, -1]].
    // MetalLayoutAttr takes the grid shape of the device, not the grid on which
    // the tensor is sharded
    auto metalLayout = ttcore::MetalLayoutAttr::get(
        rewriter.getContext(), tensorType.getShape(), targetSquareGridShape,
        ttcore::OOBVal::Undef, memSpace, collapsedIntervals);
    // Get raw, unsharded physical shape.
    llvm::SmallVector<int64_t> unshardedShape =
        metalLayout.getPhysicalShape(ttcore::TileType::getDefaultShape());

    std::cout << "unshardedShape: " << unshardedShape[0] << " "
              << unshardedShape[1] << std::endl;
    // Calculate optimal grid for given physical shape.
    llvm::SmallVector<int64_t> optimalGrid(ttnnLayout.getGrid().getShape());
    std::cout << "optimalGrid: " << optimalGrid[0] << " " << optimalGrid[1]
              << std::endl;

    llvm::SmallVector<int64_t> shardedShape = metalLayout.getDeviceShape(
        optimalGrid, ttcore::TileType::getDefaultShape());
    std::cout << "shardedShape: " << shardedShape[0] << " " << shardedShape[1]
              << std::endl;

    return mlir::RankedTensorType::get(shardedShape, elementType, metalLayout);
  }

  Value insertTTNNMetalCast(PatternRewriter &rewriter, Value fromValue,
                            Type toResultType) const {
    auto castOp = rewriter.create<ttir::TTNNMetalLayoutCastOp>(
        fromValue.getLoc(), toResultType, fromValue);
    return castOp.getResult();
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getDialect()->getNamespace() != "ttir") {
      return failure();
    }

    if (isa<ttir::TTNNMetalLayoutCastOp>(op)) {
      return failure();
    }

    // Avoid infinite rewrite loops: only match ops that actually have TTNN
    // operands or results. If nothing to change, report no match.
    bool hasTTNNOperand = false;
    for (Value operand : op->getOperands()) {
      if (isTTNNTensor(operand.getType())) {
        hasTTNNOperand = true;
        break;
      }
    }
    bool hasTTNNResult = false;
    for (Value result : op->getResults()) {
      if (isTTNNTensor(result.getType())) {
        hasTTNNResult = true;
        break;
      }
    }
    if (!hasTTNNOperand && !hasTTNNResult) {
      return failure();
    }

    SmallVector<Value> newOperands;
    SmallVector<Type> newResultTypes;
    for (Value operand : op->getOperands()) {
      if (isTTNNTensor(operand.getType())) {
        newOperands.push_back(insertTTNNMetalCast(
            rewriter, operand, getMetalTensorType(rewriter, operand)));
      } else {
        newOperands.push_back(operand);
      }
    }
    for (Value result : op->getResults()) {
      if (isTTNNTensor(result.getType())) {
        newResultTypes.push_back(getMetalTensorType(rewriter, result));
      } else {
        newResultTypes.push_back(result.getType());
      }
    }

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(newOperands);
    state.addTypes(newResultTypes);
    state.addAttributes(op->getAttrs());
    state.addSuccessors(op->getSuccessors());
    for (Region &r : op->getRegions()) {
      Region *newRegion = state.addRegion();
      rewriter.cloneRegionBefore(r, *newRegion, newRegion->begin());
    }

    Operation *newOp = rewriter.create(state);

    SmallVector<Value> finalResults;
    finalResults.reserve(op->getNumResults());
    for (size_t i = 0; i < op->getNumResults(); i++) {
      Value newResult = newOp->getResult(i);
      Value originalResult = op->getResult(i);
      if (newResult.getType() != originalResult.getType()) {
        finalResults.push_back(
            insertTTNNMetalCast(rewriter, newResult, originalResult.getType()));
      } else {
        finalResults.push_back(newResult);
      }
    }
    rewriter.replaceOp(op, finalResults);

    return success();
  }

protected:
  llvm::SmallVector<int64_t> targetSquareGridShape;
};

class TTIRLowerTTNNLayoutsPass
    : public impl::TTIRLowerTTNNLayoutsBase<TTIRLowerTTNNLayoutsPass> {
  using impl::TTIRLowerTTNNLayoutsBase<
      TTIRLowerTTNNLayoutsPass>::TTIRLowerTTNNLayoutsBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerTTNNLayoutsPattern>(
        &getContext(),
        tt::ttir::utils::getSquareTargetGrid(getTargetGridShape()));
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }

private:
  // Helper to get defined device shape if an override is not provided.
  SmallVector<int64_t> getTargetGridShape() {
    // Get from device if no override given.
    mlir::ModuleOp moduleOp = getOperation();
    ttcore::DeviceAttr device = ttcore::lookupDevice(moduleOp);
    assert(device && "Device not found");
    return llvm::to_vector(device.getWorkerGrid().getShape());
  }
};
} // namespace

} // namespace mlir::tt::ttir
