// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TT/TTOpsTypes.h"

#include "ttmlir/Dialect/TTIR/TTIRPasses.h"

template <typename T> T div_up(T n, T d) { return (n + d - 1) / d; }

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRDISPATCH
#define GEN_PASS_DEF_TTIRLAYOUT
#define GEN_PASS_DEF_TTIRSHARD
#define GEN_PASS_DEF_TTIRLOWER
#include "ttmlir/Dialect/TTIR/TTIRPasses.h.inc"

class TTIRLinalgGenericToDispatchRewriter
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  template <typename TensorTyT>
  static TensorTyT unparallelize(TensorTyT tensorTy) {
    SmallVector<int64_t> shape(tensorTy.getShape());
    assert(shape.size() > 2);
    SmallVector<int64_t> newShape(shape.begin() + 2, shape.end());
    return tensorTy.clone(newShape);
  }

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    // Test if this generic op has already been lowered, todo remove
    if (op.getOperation()->getParentOp()->getName() ==
        OperationName("tt.dispatch", getContext()))
      return failure();

    // Create a new dispatch op, inherit the ins/outs from generic op
    auto resultTypesRange = ValueRange(op.getResults()).getTypes();
    SmallVector<Type> resultTypes(resultTypesRange.begin(),
                                  resultTypesRange.end());
    auto dispatch = rewriter.create<ttir::DispatchOp>(
        op.getLoc(), resultTypes, op.getInputs(), op.getOutputs(),
        AffineMap::getMultiDimIdentityMap(2, getContext()));

    // Create a new basic block for the dispatch op, create arguments
    Block *block = rewriter.createBlock(&dispatch.getRegion());
    std::size_t inputsSize = op.getInputs().size();
    std::size_t outputsSize = op.getOutputs().size();
    SmallVector<Type> blockArgumentTypes;
    for (auto inputTy : ValueRange(op.getInputs()).getTypes()) {
      blockArgumentTypes.push_back(
          unparallelize(dyn_cast<TensorType>(inputTy)));
    }
    for (auto outputTy : ValueRange(op.getOutputs()).getTypes()) {
      blockArgumentTypes.push_back(
          unparallelize(dyn_cast<TensorType>(outputTy)));
    }
    for (Value result : op.getResults()) {
      result.setType(unparallelize(dyn_cast<TensorType>(result.getType())));
    }

    auto shiftDown = [](AffineMapAttr attr, unsigned shift,
                        MLIRContext *ctx) -> AffineMapAttr {
      auto affineMap = cast<AffineMapAttr>(attr).getAffineMap();
      assert(affineMap.getNumDims() > shift);
      assert(affineMap.getNumResults() > shift);
      SmallVector<AffineExpr> exprs;
      for (AffineExpr affineExpr :
           affineMap.getMinorSubMap(affineMap.getNumResults() - shift)
               .getResults()) {
        auto position = dyn_cast<AffineDimExpr>(affineExpr).getPosition();
        assert(position >= shift);
        exprs.push_back(mlir::getAffineDimExpr(position - shift, ctx));
      }
      return AffineMapAttr::get(
          AffineMap::get(affineMap.getNumDims() - shift, 0, exprs, ctx));
    };

    // Fixup the affine maps, i.e. drop the dispatch par dims
    SmallVector<Attribute> parAffineMaps;
    for (Attribute attr : op.getIndexingMaps()) {
      auto affineMap = cast<AffineMapAttr>(attr);
      parAffineMaps.push_back(shiftDown(affineMap, 2, getContext()));
    }
    op.setIndexingMapsAttr(ArrayAttr::get(getContext(), parAffineMaps));

    SmallVector<Attribute> parIteratorTypes(op.getIteratorTypes().getValue());
    assert(parIteratorTypes.size() > 2);
    SmallVector<Attribute> unparIteratorTypes(parIteratorTypes.begin() + 2,
                                              parIteratorTypes.end());
    op.setIteratorTypesAttr(ArrayAttr::get(getContext(), unparIteratorTypes));

    // Rewire the generic op arguments to reference the dispatch block arguments
    SmallVector<Location> blockArgumentLocs(inputsSize + outputsSize,
                                            dispatch.getLoc());
    (void)block->addArguments(blockArgumentTypes, blockArgumentLocs);
    auto blockArguments = block->getArguments();
    op.getInputsMutable().assign(blockArguments.slice(0, inputsSize));
    op.getOutputsMutable().assign(blockArguments.slice(inputsSize));

    // Move the generic op into the dispatch block
    Operation *generic = op.getOperation()->clone();
    block->push_back(generic);
    rewriter.setInsertionPoint(block, block->end());
    rewriter.create<ttir::YieldOp>(dispatch.getLoc(),
                                   ValueRange({generic->getResult(0)}));
    rewriter.replaceOp(op, dispatch);
    return success();
  }
};

template <typename TosaEltwiseOp>
class TTIRTosaElementwiseToDispatchRewriter
    : public OpRewritePattern<TosaEltwiseOp> {
public:
  using OpRewritePattern<TosaEltwiseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TosaEltwiseOp op,
                                PatternRewriter &rewriter) const final {
    assert(op.getShift() == 0);
    op.dump();
    return failure();
  }
};

class TTIRDispatch : public impl::TTIRDispatchBase<TTIRDispatch> {
public:
  using impl::TTIRDispatchBase<TTIRDispatch>::TTIRDispatchBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRLinalgGenericToDispatchRewriter,
                 TTIRTosaElementwiseToDispatchRewriter<tosa::MulOp>>(
        &getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
  }
};

class TTIRLayout : public impl::TTIRLayoutBase<TTIRLayout> {
public:
  using impl::TTIRLayoutBase<TTIRLayout>::TTIRLayoutBase;

  void runOnOperation() final { assert(false); }
};

class TTIRShard : public impl::TTIRShardBase<TTIRShard> {
public:
  using impl::TTIRShardBase<TTIRShard>::TTIRShardBase;

  void runOnOperation() final { assert(false); }
};

class TTIRLower : public impl::TTIRLowerBase<TTIRLower> {
public:
  using impl::TTIRLowerBase<TTIRLower>::TTIRLowerBase;

  void runOnOperation() final { assert(false); }
};

} // namespace mlir::tt::ttir
