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
#include "ttmlir/Dialect/TT/TTPasses.h"

#include "ttmlir/Target/TTTarget.h"

template <typename T> T div_up(T n, T d) { return (n + d - 1) / d; }

namespace mlir::tt {
#define GEN_PASS_DEF_TTTILIZE
#define GEN_PASS_DEF_TTPARALLELIZE
#define GEN_PASS_DEF_TTCODEGEN
#define GEN_PASS_DEF_TTLOWER
#include "ttmlir/Dialect/TT/TTPasses.h.inc"

class TTMatmulToGenericRewriter : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const final {
    return generalizeNamedOp(rewriter, op);
  }
};

class TTTilizeRewriter : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    // for (auto attr : op.getIndexingMapsAttr()) {
    //   AffineMap map = dyn_cast<AffineMapAttr>(attr).getValue();
    //   if (not map.isIdentity()) {
    //     return op.emitError("Unsupported affine map access pattern for
    //     tilization");
    //   }
    // }

    auto tileTy = rewriter.getType<TileType>(32, 32, DataType::Float32);
    SmallVector<Value> tilizeOps;
    SmallVector<Value> results;
    SmallVector<Type> resultTypes;
    SmallVector<Value> untilizeOps;

    auto createTilizedEmptyTensor = [&rewriter, &op,
                                     tileTy](Value tensor) -> Value {
      auto tensorTy = dyn_cast<TensorType>(tensor.getType());
      SmallVector<int64_t> shape(tensorTy.getShape());
      // assert(shape.size() >= 2);
      if (shape.size() >= 2)
        shape[shape.size() - 2] = div_up(shape[shape.size() - 2], 32l);
      if (shape.size() >= 1)
        shape[shape.size() - 1] = div_up(shape[shape.size() - 1], 32l);
      return rewriter.create<tensor::EmptyOp>(op.getLoc(), shape, tileTy);
    };

    for (Value input : op.getInputs()) {
      if (dyn_cast<TileType>(dyn_cast<TensorType>(input.getType()).getElementType()))
        return failure(); // Already lowered to tile
      auto dest = createTilizedEmptyTensor(input);
      Value padding = nullptr; // TODO
      auto tilize = rewriter.create<tt::TensorTilizeOp>(op.getLoc(), input,
                                                        dest, padding);
      tilizeOps.push_back(tilize);
    }

    for (Value output : op.getOutputs()) {
      results.push_back(createTilizedEmptyTensor(output));
      resultTypes.push_back(results.back().getType());
    }

    SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();
    SmallVector<utils::IteratorType> iteratorTypes = op.getIteratorTypesArray();

    auto tilizedLinalgOp = rewriter.create<linalg::GenericOp>(
        op.getLoc(), resultTypes, tilizeOps, results, indexingMaps,
        iteratorTypes);
    tilizedLinalgOp.getRegion().takeBody(op->getRegion(0));

    auto ip = rewriter.saveInsertionPoint();
    for (BlockArgument arg : tilizedLinalgOp.getRegion().getArguments()) {
      Block *block = arg.getOwner();
      Type originalTy = arg.getType();
      // FIXME: Actually derive the type from the linalgOp input
      arg.setType(tileTy);
      rewriter.setInsertionPoint(block, block->begin());
      auto unpack = rewriter.create<tt::UnpackOp>(block->front().getLoc(),
                                                  originalTy, arg);
      rewriter.replaceAllUsesExcept(arg, unpack, unpack);
    }

    tilizedLinalgOp.getRegion().walk([&rewriter, tileTy](Operation *op) {
      if (dyn_cast<linalg::YieldOp>(op)) {
        Value operand = op->getOperand(0);
        rewriter.setInsertionPoint(op);
        auto pack = rewriter.create<tt::PackOp>(op->getLoc(), tileTy, operand);
        op->setOperand(0, pack);
      }
    });
    rewriter.restoreInsertionPoint(ip);

    auto originalOutputs = op.getOutputs();
    for (OpResult result : tilizedLinalgOp.getResults()) {
      untilizeOps.push_back(rewriter.create<tt::TensorUntilizeOp>(
          tilizedLinalgOp.getLoc(), result,
          originalOutputs[result.getResultNumber()]));
    }

    rewriter.replaceOp(op, untilizeOps);

    return success();
  }
};

class TTTilize : public impl::TTTilizeBase<TTTilize> {
public:
  using impl::TTTilizeBase<TTTilize>::TTTilizeBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTMatmulToGenericRewriter, TTTilizeRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::TTDialect>();
    registry.insert<mlir::vector::VectorDialect>();
  }
};

class TTParallelize : public impl::TTParallelizeBase<TTParallelize> {
public:
  using impl::TTParallelizeBase<TTParallelize>::TTParallelizeBase;

  template <typename TensorTyT>
  static TensorTyT parallelize(TensorTyT tensorTy, int64_t y = 1,
                               int64_t x = 1) {
    SmallVector<int64_t> shape(tensorTy.getShape());
    shape.insert(shape.begin(), 1);
    shape.insert(shape.begin(), 1);
    return tensorTy.clone(shape);
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext &ctx = getContext();

    llvm::SmallDenseSet<Value, 32> processed;
    module->walk([&processed, &ctx](linalg::GenericOp op) {
      for (Value input : op.getInputs()) {
        if (processed.contains(input))
          continue;
        input.setType(parallelize(dyn_cast<TensorType>(input.getType())));
        if (auto tilizeOp = input.getDefiningOp<tt::TensorTilizeOp>()) {
          auto dest = tilizeOp.getDest();
          dest.setType(parallelize(dyn_cast<RankedTensorType>(dest.getType())));
        }
      }

      for (Value output : op.getOutputs()) {
        if (processed.contains(output))
          continue;
        output.setType(parallelize(dyn_cast<TensorType>(output.getType())));
      }

      for (Value result : op.getResults()) {
        if (processed.contains(result))
          continue;
        result.setType(parallelize(dyn_cast<TensorType>(result.getType())));
      }

      SmallVector<Attribute> parAffineMaps;
      for (Attribute attr : op.getIndexingMaps()) {
        auto affineMapAttr = cast<AffineMapAttr>(attr);
        parAffineMaps.push_back(AffineMapAttr::get(
            affineMapAttr.getAffineMap()
                .shiftDims(2)
                .insertResult(mlir::getAffineDimExpr(1, &ctx), 0)
                .insertResult(mlir::getAffineDimExpr(0, &ctx), 0)));
      }
      op.setIndexingMapsAttr(ArrayAttr::get(&ctx, parAffineMaps));

      SmallVector<Attribute> parIteratorTypes(op.getIteratorTypes().getValue());
      auto iteratorTypeAttr = linalg::IteratorTypeAttr::get(&ctx, utils::IteratorType::parallel);
      parIteratorTypes.insert(parIteratorTypes.begin(), iteratorTypeAttr);
      parIteratorTypes.insert(parIteratorTypes.begin(), iteratorTypeAttr);
      op.setIteratorTypesAttr(ArrayAttr::get(&ctx, parIteratorTypes));
    });
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::TTDialect>();
  }
};

class TTLowerLinalgGenericToDispatchRewriter
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
    auto dispatch = rewriter.create<tt::DispatchOp>(
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
    rewriter.create<tt::YieldOp>(dispatch.getLoc(),
                                 ValueRange({generic->getResult(0)}));
    rewriter.replaceOp(op, dispatch);
    return success();
  }
};

class TTLowerTosaElementwiseToDispatchRewriter
    : public OpRewritePattern<tosa::MulOp> {
public:
  using OpRewritePattern<tosa::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MulOp op,
                                PatternRewriter &rewriter) const final {
    assert(op.getShift() == 0);
    op.dump();
    return failure();
  }
};

class TTLower : public impl::TTLowerBase<TTLower> {
public:
  using impl::TTLowerBase<TTLower>::TTLowerBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTLowerLinalgGenericToDispatchRewriter,
                 TTLowerTosaElementwiseToDispatchRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::TTDialect>();
  }
};

class TTCodeGen : public impl::TTCodeGenBase<TTCodeGen> {
public:
  using impl::TTCodeGenBase<TTCodeGen>::TTCodeGenBase;

  void runOnOperation() final {
    // ModuleOp module = getOperation();
    ::tt::WorkloadT workload;
    ::flatbuffers::FlatBufferBuilder fbb;
    ::tt::FinishSizePrefixedWorkloadBuffer(
        fbb, ::tt::CreateWorkload(fbb, &workload));
  }
};
} // namespace mlir::tt
