// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
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
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNOPENDEVICE
#define GEN_PASS_DEF_TTNNGENERIC
#define GEN_PASS_DEF_CONVERTTTIRTOTTNN
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNOpenDevice : public impl::TTNNOpenDeviceBase<TTNNOpenDevice> {
public:
  using impl::TTNNOpenDeviceBase<TTNNOpenDevice>::TTNNOpenDeviceBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    OpBuilder builder(module);
    auto systemDesc = llvm::cast<tt::SystemDescAttr>(
        module->getAttr(tt::SystemDescAttr::name));
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
                             builder.getAttr<tt::DeviceAttr>(systemDesc)));

      builder.setInsertionPoint(block, opRange.end());
      builder.create<CloseDeviceOp>(func.getLoc(), openDevice.getResult());
    });
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttnn::TTNNDialect>();
  }
};

// Rewrites `ttnn.add` or `ttnn.multiply` call to `ttnn.kernel`.
template <typename TTNNOpType>
class TTNNNamedToKernelRewriter : public OpRewritePattern<TTNNOpType> {
public:
  using OpRewritePattern<TTNNOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(TTNNOpType op,
                                PatternRewriter &rewriter) const final {
    StringRef kernelName;
    StringRef kernelKind;

    if constexpr (std::is_same<TTNNOpType, ttnn::MultiplyOp>::value) {
      kernelName = "mulitply";
      kernelKind = "eltwise";
    } else if constexpr (std::is_same<TTNNOpType, ttnn::AddOp>::value) {
      kernelName = "add";
      kernelKind = "eltwise";
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "Unsupported Tosa operation for TTNN");
    }

    assert(kernelName.size() > 0);

    auto kernel = rewriter.create<ttnn::KernelOp>(
        op.getLoc(), op.getResultTypes(), kernelName, kernelKind,
        op.getInputs(), op.getOutputs());

    rewriter.replaceOp(op, kernel);

    return success();
  }
};

class TTNNKernelGenericRewriter : public OpRewritePattern<KernelOp> {
public:
  using OpRewritePattern<KernelOp>::OpRewritePattern;

  static bool sameRank(mlir::OperandRange operands) {
    if (operands.empty()) {
      return false;
    }
    auto rank = mlir::cast<RankedTensorType>(operands[0].getType()).getRank();
    for (auto operand : operands) {
      if (mlir::cast<RankedTensorType>(operand.getType()).getRank() != rank) {
        return false;
      }
    }
    return true;
  }

  static std::pair<ArrayAttr, ArrayAttr>
  createEltwiseIndexingMaps(PatternRewriter &rewriter,
                            mlir::OperandRange operands) {
    assert(sameRank(operands) &&
           "For now all operands must have the same rank");
    auto rank = mlir::cast<RankedTensorType>(operands[0].getType()).getRank();
    SmallVector<AffineMap> indexingMaps(operands.size(),
                                        rewriter.getMultiDimIdentityMap(rank));
    SmallVector<Attribute> iteratorTypes(
        rank, rewriter.getAttr<IteratorTypeAttr>(IteratorType::Parallel));
    return {rewriter.getAffineMapArrayAttr(indexingMaps),
            rewriter.getArrayAttr(iteratorTypes)};
  }

  static std::pair<ArrayAttr, ArrayAttr>
  createMatmulIndexingMaps(PatternRewriter &rewriter,
                           mlir::OperandRange operands) {
    assert(sameRank(operands) &&
           "For now all operands must have the same rank");
    auto rank = mlir::cast<RankedTensorType>(operands[0].getType()).getRank();
    assert(rank >= 2 && "Matmul requires rank >= 2");
    auto rank_plus_inner_dim = rank + 1;

    // (d0, d1, d2, d3) -> (d0, d1, d2, d3)
    // lhs (d0, d1, d2, d3) -> (d0, d1, d3) drop d2
    // rhs (d0, d1, d2, d3) -> (d0, d3, d2) drop d1 and swap d2 and d3
    // out (d0, d1, d2, d3) -> (d0, d1, d2) drop d3
    auto id = rewriter.getMultiDimIdentityMap(rank_plus_inner_dim);
    auto lhs = id.dropResult(rank_plus_inner_dim - 2);
    auto rhs = id.dropResult(rank_plus_inner_dim - 3);
    auto rhs_outer = rhs.getResult(rank - 2);
    rhs = rhs.insertResult(rhs_outer, rank);
    rhs = rhs.dropResult(rank - 2);
    auto out = id.dropResult(rank_plus_inner_dim - 1);

    SmallVector<AffineMap> indexingMaps = {lhs, rhs, out};
    SmallVector<Attribute> iteratorTypes(
        rank, rewriter.getAttr<IteratorTypeAttr>(IteratorType::Parallel));
    iteratorTypes.push_back(
        rewriter.getAttr<IteratorTypeAttr>(IteratorType::Systolic));
    return {rewriter.getAffineMapArrayAttr(indexingMaps),
            rewriter.getArrayAttr(iteratorTypes)};
  }

  static std::pair<ArrayAttr, ArrayAttr>
  createIndexingMaps(PatternRewriter &rewriter, StringRef kind,
                     mlir::OperandRange operands) {
    if (kind == "eltwise") {
      return createEltwiseIndexingMaps(rewriter, operands);
    }
    if (kind == "matmul") {
      return createMatmulIndexingMaps(rewriter, operands);
    }
    llvm_unreachable("Unsupported kernel kind");
  }

  static ArrayAttr createOperandConstraints(PatternRewriter &rewriter,
                                            StringRef kind,
                                            mlir::OperandRange operands) {
    auto numOperands = operands.size();
    if (kind == "eltwise") {
      return rewriter.getArrayAttr(SmallVector<Attribute>(
          numOperands, rewriter.getAttr<OperandConstraintAttr>(
                           OperandConstraint::AnyDevice)));
    }
    if (kind == "matmul") {
      return rewriter.getArrayAttr(SmallVector<Attribute>(
          numOperands, rewriter.getAttr<OperandConstraintAttr>(
                           OperandConstraint::AnyDeviceTile)));
    }
    llvm_unreachable("Unsupported kernel kind");
  }

  LogicalResult matchAndRewrite(KernelOp op,
                                PatternRewriter &rewriter) const final {
    // Test if this generic op has already been lowered, todo find a better way
    if (op.getOperation()->getParentOp()->getName() ==
        OperationName("ttnn.generic", rewriter.getContext())) {
      return failure();
    }

    // Create a dispatch op
    auto [indexingMaps, iteratorTypes] =
        createIndexingMaps(rewriter, op.getKind(), op.getOperands());
    auto constraints =
        createOperandConstraints(rewriter, op.getKind(), op.getOperands());
    auto dispatch = rewriter.create<ttnn::GenericOp>(
        op.getLoc(), op.getResults().getTypes(), op.getInputs(),
        op.getOutputs(), rewriter.getAttr<GridAttr>(), indexingMaps,
        iteratorTypes, constraints);

    // Create a new basic block for the dispatch op and create block arguments
    Block *block = rewriter.createBlock(&dispatch.getRegion());
    SmallVector<Location> blockArgumentLocs(dispatch.getOperands().size(),
                                            dispatch.getLoc());
    block->addArguments(TypeRange(dispatch.getOperandTypes()),
                        blockArgumentLocs);

    // Update the operands of the original op to use the block arguments
    op.getOperation()->setOperands(block->getArguments());

    // Move the original op into the dispatch block
    Operation *operation = op.getOperation()->clone();
    block->push_back(operation);
    rewriter.setInsertionPoint(block, block->end());
    rewriter.create<ttnn::YieldOp>(dispatch.getLoc(),
                                   ValueRange({operation->getResult(0)}));
    rewriter.replaceOp(op, dispatch);
    return success();
  }
};

class TTNNGeneric : public impl::TTNNGenericBase<TTNNGeneric> {
public:
  using impl::TTNNGenericBase<TTNNGeneric>::TTNNGenericBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    
    patterns.add<TTNNKernelGenericRewriter,
                 TTNNNamedToKernelRewriter<AddOp>,
                 TTNNNamedToKernelRewriter<MultiplyOp>>(&getContext());
    
    FrozenRewritePatternSet patternSet(std::move(patterns));
    
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttnn::TTNNDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};

} // namespace mlir::tt::ttnn
