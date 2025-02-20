// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <numeric>

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Memref/IR/Memref.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRGENERICREGION
#define GEN_PASS_DEF_TTIRGENERICLINEARIZEMEMREF
#define GEN_PASS_DEF_TTIRGENERICDATAMOVEMENT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Generic - Region pass
//===----------------------------------------------------------------------===//

static MemRefType getGenericMemrefBlockArgType(Type type) {
  RankedTensorType tensorType = mlir::cast<RankedTensorType>(type);
  tt::MetalLayoutAttr layout =
      mlir::cast<tt::MetalLayoutAttr>(tensorType.getEncoding());
  MemRefType memrefType = layout.getMemref();
  assert(memrefType.getShape().size() % 2 == 0 &&
         "Memref shape must be even, e.g. GxGxSxS, grid outer dims and shard "
         "inner dims of respectively equal rank");
  size_t shardRank = memrefType.getShape().size() / 2;
  StreamLayoutAttr streamLayout =
      mlir::cast<StreamLayoutAttr>(memrefType.getLayout());

  // Key the affine map based on the stream layout,
  //   - if mode alias, the compute kernel needs to manage the indexing of the
  //   memory (i.e. in place), so inherit the affine map
  //   - if mode stream, the compute kernel can rely on the memory being
  //   gathered and delivered in row major order, i.e. identity map
  AffineMap affineMap =
      streamLayout.getStreamMode() == StreamMode::Alias
          ? streamLayout.getAffineMap().getMinorSubMap(shardRank).compose(
                ttmlir::utils::getMinorReductionMap(shardRank,
                                                    type.getContext()))
          : AffineMap{};
  MemRefType shardMemrefType = MemRefType::get(
      memrefType.getShape().take_back(shardRank), memrefType.getElementType(),
      affineMap, memrefType.getMemorySpace());
  return shardMemrefType;
}

static bool
indexingMapParticipatesInReduction(AffineMap indexingMap,
                                   SmallVector<unsigned> reductionDims) {
  if (reductionDims.empty()) {
    return false;
  }

  for (unsigned i = 0; i < indexingMap.getNumResults(); i++) {
    AffineExpr expr = indexingMap.getResult(i);
    if (llvm::any_of(reductionDims,
                     [&](unsigned dim) { return expr.isFunctionOfDim(dim); })) {
      return true;
    }
  }

  return false;
}

static SmallVector<Value> reductionViews(PatternRewriter &rewriter,
                                         Location loc, ValueRange dpsInputs,
                                         ArrayAttr indexingMaps,
                                         ArrayAttr iteratorTypes) {
  SmallVector<Value> inputs(dpsInputs);
  assert(
      inputs.size() <= indexingMaps.size() &&
      "Number of inputs must be less or equal to the number of indexing maps");
  SmallVector<unsigned> reductionDims = llvm::to_vector(llvm::make_filter_range(
      llvm::seq<unsigned>(0, iteratorTypes.size()), [&](unsigned i) {
        return mlir::cast<IteratorTypeAttr>(iteratorTypes[i]).getValue() ==
               IteratorType::Reduction;
      }));
  for (unsigned i = 0; i < inputs.size(); i++) {
    if (indexingMapParticipatesInReduction(
            mlir::cast<AffineMapAttr>(indexingMaps[i]).getValue(),
            reductionDims)) {
      auto inputTy = mlir::cast<RankedTensorType>(inputs[i].getType());
      auto inputLayout = mlir::cast<MetalLayoutAttr>(inputTy.getEncoding());
      auto streamLayout = rewriter.getAttr<StreamLayoutAttr>(
          mlir::cast<StreamLayoutAttr>(inputLayout.getMemref().getLayout())
              .getAffineMap(),
          StreamMode::Stream, 1);
      auto resultType = RankedTensorType::get(
          inputTy.getShape(), inputTy.getElementType(),
          inputLayout.withStreamLayout(rewriter.getContext(), streamLayout));

      inputs[i] =
          rewriter.create<ttir::ViewLayoutOp>(loc, resultType, inputs[i])
              ->getResult(0);
    }
  }
  return inputs;
}

std::pair<ttir::GenericOp, Block *> buildGenericOp(GenericRegionOp op,
                                                   PatternRewriter &rewriter) {
  auto dps = cast<DestinationStyleOpInterface>(op.getOperation());

  // Create a generic op.
  auto [indexingMaps, iteratorTypes] = op.getIndexingMaps(rewriter);

  // For testing purposes try getting grid of the resulting tensor and put the
  // op in the grid.
  // TODO(radenko) add a proper debug/test flag.
  auto gridAttr = rewriter.getAttr<GridAttr>();
  auto resEncoding =
      mlir::cast<RankedTensorType>(op->getResult(0).getType()).getEncoding();
  if (resEncoding) {
    auto resLayout = mlir::cast<MetalLayoutAttr>(resEncoding);
    gridAttr = resLayout.getGrid();
  }

  SmallVector<Value> inputs = reductionViews(
      rewriter, op.getLoc(), dps.getDpsInputs(), indexingMaps, iteratorTypes);

  auto genericOp = rewriter.create<ttir::GenericOp>(
      op.getLoc(), op->getResults().getTypes(), inputs, ValueRange() /* cbs */,
      dps.getDpsInits(), gridAttr, indexingMaps, iteratorTypes,
      ::llvm::ArrayRef<int64_t>(), 1 /*numRegions*/);

  // Create a new basic block for the generic op and create block arguments.
  Block *block = rewriter.createBlock(&genericOp.getRegion(0));
  SmallVector<Location> blockArgumentLocs(genericOp.getOperands().size(),
                                          genericOp.getLoc());
  SmallVector<Type> blockArgTypes(llvm::map_range(
      genericOp.getOperands().getTypes(), getGenericMemrefBlockArgType));
  block->addArguments(blockArgTypes, blockArgumentLocs);

  return std::make_pair(genericOp, block);
}

template <typename TileOpTy>
void buildLinalgGeneric(::mlir::Location loc, ::mlir::Block *block,
                        mlir::OpBuilder &opBuilder) {
  auto lhs = block->getArgument(0);
  auto rhs = block->getArgument(1);
  auto out = block->getArgument(2);

  using IteratorType = mlir::utils::IteratorType;
  auto parallel = IteratorType::parallel;
  auto parMap =
      mlir::AffineMap::getMultiDimIdentityMap(2, opBuilder.getContext());
  mlir::SmallVector<IteratorType> genericIterators = {parallel, parallel};
  mlir::SmallVector<mlir::AffineMap> parMaps = {parMap, parMap, parMap};
  opBuilder.create<mlir::linalg::GenericOp>(
      loc, mlir::ValueRange({lhs, rhs}), mlir::ValueRange({out}), parMaps,
      genericIterators,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
          mlir::ValueRange args) {
        mlir::Value result = nestedBuilder.create<TileOpTy>(
            loc, args[0].getType(), args[0], args[1]);
        nestedBuilder.create<mlir::linalg::YieldOp>(nestedLoc, result);
      });
}

class TTIRGenericMaximumRewriter
    : public OpRewritePattern<MaximumOp> {
public:
  using OpRewritePattern<MaximumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MaximumOp op,
                                PatternRewriter &rewriter) const final {
    if (mlir::isa<GenericOp>(op.getOperation()->getParentOp())) {
      return failure();
    }

    auto [genericOp, block] = buildGenericOp(op, rewriter);
    OpBuilder blockBuilder = OpBuilder::atBlockEnd(block);
    buildLinalgGeneric<TileMaximumOp>(op->getLoc(), block, blockBuilder);
    rewriter.replaceOp(op, genericOp);
    return success();
  }
};

class TTIRGenericMatmulRewriter
    : public OpRewritePattern<MatmulOp> {
public:
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp op,
                                PatternRewriter &rewriter) const final {
    if (mlir::isa<GenericOp>(op.getOperation()->getParentOp())) {
      return failure();
    }

    auto [genericOp, block] = buildGenericOp(op, rewriter);
    OpBuilder blockBuilder = OpBuilder::atBlockEnd(block);
    blockBuilder.create<ttir::TileMatmulBlockOp>(
        op->getLoc(), block->getArgument(0), block->getArgument(1),
        block->getArgument(2));
    rewriter.replaceOp(op, genericOp);
    return success();
  }
};

class TTIRGenericRegion
    : public impl::TTIRGenericRegionBase<TTIRGenericRegion> {
public:
  using impl::TTIRGenericRegionBase<TTIRGenericRegion>::TTIRGenericRegionBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericMaximumRewriter>(&getContext());
    patterns.add<TTIRGenericMatmulRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
    registry.insert<mlir::arith::ArithDialect>();
  }
};

namespace {
class TTIRGenericLinearizeMemrefRewriter
    : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  static bool isLinearizedMemref(BlockArgument arg) {
    auto memref = mlir::cast<MemRefType>(arg.getType());
    if (memref.getShape().size() == 1) {
      return true;
    }

    if (std::all_of(arg.user_begin(), arg.user_end(), [](Operation *user) {
          return mlir::isa<memref::CollapseShapeOp>(user);
        })) {
      return true;
    }

    return false;
  }

  static mlir::AffineMap linearizeAffineMap(::mlir::MLIRContext *context,
                                             mlir::AffineMap map,
                                             ArrayRef<int64_t> shape) {
    auto evaledShape = ttmlir::utils::evalShape(map, shape);
    mlir::AffineExpr indexing = getAffineConstantExpr(0, context);
    mlir::AffineExpr volumeExpr = getAffineConstantExpr(1, context);

    for (int i = map.getNumResults() - 1; i >= 0; i--) {
      mlir::AffineExpr linearIdx = getAffineDimExpr(i, context);
      mlir::AffineExpr dim = getAffineConstantExpr(evaledShape[i], context);
      indexing = linearIdx * volumeExpr + indexing;
      volumeExpr = volumeExpr * dim;
    }

    mlir::AffineMap linearResult =
        mlir::AffineMap::get(map.getNumResults(), 0, indexing, context);
    return linearResult.compose(map);
  }

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const final {
    unsigned numRegions = op.getNumRegions();
    // Only linearize the last region. i.e. compute
    Block *entry = &op.getRegion(numRegions - 1).front();
    rewriter.setInsertionPointToStart(entry);
    auto args = entry->getArguments();
    if (llvm::all_of(args, isLinearizedMemref)) {
      return failure();
    }

    rewriter.modifyOpInPlace(op, [&]() {
      for (auto arg : args) {
        if (isLinearizedMemref(arg)) {
          continue;
        }
        auto memref = mlir::cast<MemRefType>(arg.getType());
        auto shape = memref.getShape();
        auto linearMap = linearizeAffineMap(
            rewriter.getContext(), memref.getLayout().getAffineMap(), shape);
        SmallVector<ReassociationIndices, 4> collapsedDims = {
            llvm::to_vector(llvm::seq<int64_t>(0, shape.size()))};
        auto linearizedArg = rewriter.create<memref::CollapseShapeOp>(
            arg.getLoc(), arg, collapsedDims);
        rewriter.replaceAllUsesExcept(arg, linearizedArg->getResult(0),
                                      linearizedArg);
        for (auto user : linearizedArg->getUsers()) {
          if (auto load = mlir::dyn_cast<affine::AffineLoadOp>(user)) {
            load.setMap(linearMap.compose(load.getMap()));
          } else if (auto store = mlir::dyn_cast<affine::AffineStoreOp>(user)) {
            store.setMap(linearMap.compose(store.getMap()));
          }
        }
      }
    });

    return success();
  }
};
} // namespace

namespace {
class TTIRGenericLinearizeMemref
    : public impl::TTIRGenericLinearizeMemrefBase<TTIRGenericLinearizeMemref> {
public:
  using impl::TTIRGenericLinearizeMemrefBase<
      TTIRGenericLinearizeMemref>::TTIRGenericLinearizeMemrefBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericLinearizeMemrefRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
    registry.insert<mlir::arith::ArithDialect>();
  }
};
} // namespace

namespace {
class TTIRGenericDatamovementRewriter : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  static StreamMode getStreamMode(Type ty) {
    auto memref = mlir::cast<MemRefType>(ty);
    if (auto streamLayout = mlir::dyn_cast<StreamLayoutAttr>(memref.getLayout()); streamLayout) {
      return streamLayout.getStreamMode();
    }
    return StreamMode::Alias;
  }

  static bool compatibleDeviceGrid(DeviceAttr device, GridAttr grid) {
    if (grid.getShape().size() != device.getWorkerGrid().getShape().size()) {
      return false;
    }
    return true;
  }

  static SmallVector<IteratorType>
  calculateMcastIterators(GridAttr grid, DeviceAttr device,
                          AffineMap indexingMap, ArrayAttr iteratorTypes) {
    assert(grid.getShape().size() == 2 && "Currently only support 2D grid");
    assert(grid.getShape().size() == indexingMap.getNumResults());
    assert(compatibleDeviceGrid(device, grid));

    bool allParallel = true;
    SmallVector<IteratorType> mcastIterators;
    mcastIterators.reserve(grid.getShape().size());
    for (unsigned dim = 0; dim < grid.getShape().size(); dim++) {
      unsigned dimPosition = indexingMap.getDimPosition(dim);

      IteratorType iteratorType =
          mlir::cast<IteratorTypeAttr>(iteratorTypes[dimPosition]).getValue();
      mcastIterators.push_back(iteratorType);
      allParallel &= iteratorType == IteratorType::Parallel;
    }

    return allParallel ? SmallVector<IteratorType>() : mcastIterators;
  }

  static std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
  calculateMcastArguments(OpBuilder &blockBuilder, Location loc, GridAttr grid,
                          DeviceAttr device, AffineMap indexingMap,
                          ArrayAttr iteratorTypes) {
    SmallVector<IteratorType> mcastIterators =
        calculateMcastIterators(grid, device, indexingMap, iteratorTypes);
    if (mcastIterators.empty()) {
      return std::make_tuple(SmallVector<Value>(), SmallVector<Value>(),
                             SmallVector<Value>());
    }

    Value zero = blockBuilder.create<arith::ConstantOp>(
        loc, blockBuilder.getIndexType(), blockBuilder.getIndexAttr(0));
    Value one = blockBuilder.create<arith::ConstantOp>(
        loc, blockBuilder.getIndexType(), blockBuilder.getIndexAttr(1));

    SmallVector<Value> mcastOffset;
    SmallVector<Value> mcastShape;
    SmallVector<Value> conditions;
    mcastOffset.reserve(grid.getShape().size());
    mcastShape.reserve(grid.getShape().size());

    for (auto [dim, iteratorType] : llvm::enumerate(mcastIterators)) {
      Value gridDim = blockBuilder.create<arith::ConstantOp>(
          loc, blockBuilder.getIndexType(),
          blockBuilder.getIndexAttr(grid.getShape()[dim]));
      Value gridIndex =
          blockBuilder.create<GridIndexOp>(loc, blockBuilder.getIndexType(),
                                           blockBuilder.getI64IntegerAttr(dim));
      if (iteratorType == IteratorType::Parallel) {
        mcastOffset.push_back(Value(gridIndex));
        mcastShape.push_back(Value(one));
      } else {
        assert(iteratorType == IteratorType::Reduction);
        mcastOffset.push_back(zero);
        mcastShape.push_back(gridDim);

        Value iterIndex = blockBuilder.create<IterIndexOp>(
            loc, blockBuilder.getIndexType(),
            blockBuilder.getI64IntegerAttr(dim));
        Value condition = blockBuilder.create<arith::CmpIOp>(
            loc, blockBuilder.getI1Type(), mlir::arith::CmpIPredicate::eq,
            gridIndex, iterIndex);
        conditions.push_back(condition);
      }
    }

    return std::make_tuple(mcastOffset, mcastShape, conditions);
  }

  static Value createDMA(OpBuilder &builder, Location loc, Value src, Value dst,
                         SmallVector<Value> mcastOffset = {},
                         SmallVector<Value> mcastShape = {}) {
    MemRefType memrefType = mlir::cast<MemRefType>(dst.getType());
    auto zero = builder.create<arith::ConstantOp>(
        loc, builder.getIndexType(), builder.getIndexAttr(0));
    SmallVector<Value> zeros(memrefType.getShape().size(), zero);
    auto numElements = builder.create<arith::ConstantOp>(
        loc, builder.getIndexType(),
        builder.getIndexAttr(memrefType.getShape().size()));
    return builder
        .create<ttir::DMAOp>(loc, builder.getType<MemTxType>(), src, zeros,
                             zeros, dst, zeros, zeros, numElements, mcastOffset,
                             mcastShape)
        .getResult();
  }

  static LogicalResult buildDatamovementBlock(
      OpBuilder &blockBuilder, Location loc, Value genericOperand,
      Value blockOperand, GridAttr grid, DeviceAttr device,
      AffineMap indexingMap, ArrayAttr iteratorTypes, bool isOutput) {

    if (isOutput) {
      // Wait for compute
      blockBuilder.create<ttir::AwaitOp>(loc, ValueRange(blockOperand));
    }

    auto streamMode = getStreamMode(genericOperand.getType());
    if (streamMode == StreamMode::Stream) {
      Value memTx =
          isOutput ? createDMA(blockBuilder, loc, blockOperand, genericOperand)
                   : createDMA(blockBuilder, loc, genericOperand, blockOperand);
      blockBuilder.create<ttir::DMAWaitOp>(loc, memTx);

      // Multicast
      SmallVector<Value> mcastOffset, mcastShape, conditions;
      std::tie(mcastOffset, mcastShape, conditions) = calculateMcastArguments(
          blockBuilder, loc, grid, device, indexingMap, iteratorTypes);
      assert(mcastOffset.size() == mcastShape.size());
      if (!mcastOffset.empty()) {
        assert(conditions.size() == 1 && "Exactly one condition supported");
        Value mcastSrc = blockOperand;
        Value mcastDst = blockOperand;
        blockBuilder.create<scf::IfOp>(
            loc, conditions[0], [&](OpBuilder &builder, Location loc) {
              Value mcastMemTx = createDMA(builder, loc, mcastSrc, mcastDst,
                                           mcastOffset, mcastShape);
              builder.create<ttir::DMAWaitOp>(loc, mcastMemTx);
              builder.create<scf::YieldOp>(loc);
            });
        blockBuilder.create<ttir::SynchronizeOp>(loc);
      }
    }

    if (!isOutput) {
      // Wait for compute
      blockBuilder.create<ttir::YieldOp>(loc, ValueRange(blockOperand));
    }

    return success();
  }

  LogicalResult matchAndRewrite(GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (generic.getNumRegions() > 1) {
      // Already inserted, skip.
      return failure();
    }

    // One per operand
    auto numDataMovementRegions = generic.getNumOperands();
    auto newGeneric = rewriter.create<GenericOp>(
        generic->getLoc(), generic.getResultTypes(), generic.getInputs(),
        generic.getCbs(), generic.getOutputs(), generic.getGrid(),
        generic.getIndexingMaps(), generic.getIteratorTypes(),
        generic.getOperandCbMapping(),
        generic.getNumRegions() + numDataMovementRegions);

    // Insert the new data movement regions.
    auto [outputOperandsIndex, outputOperandsLength] =
        generic.getODSOperandIndexAndLength(2);
    auto device = getCurrentScopeDevice(generic);
    for (OpOperand &operand : generic->getOpOperands()) {
      Block *datamovementBlock =
          &newGeneric.getRegion(operand.getOperandNumber()).emplaceBlock();
      datamovementBlock->addArguments(
          generic.getRegion(0).getArgumentTypes(),
          SmallVector<mlir::Location>(
              generic.getRegion(0).getArgumentTypes().size(),
              generic.getLoc()));

      OpBuilder blockBuilder = OpBuilder::atBlockEnd(datamovementBlock);
      bool isOutput = operand.getOperandNumber() >= outputOperandsIndex;
      AffineMap indexingMap =
          mlir::cast<AffineMapAttr>(
              generic.getIndexingMaps()[operand.getOperandNumber()])
              .getValue();
      auto result = buildDatamovementBlock(
          blockBuilder, generic->getLoc(),
          generic->getOperand(operand.getOperandNumber()),
          datamovementBlock->getArgument(operand.getOperandNumber()),
          generic.getGrid(), device, indexingMap, generic.getIteratorTypes(),
          isOutput);
      if (failed(result)) {
        return result;
      }
    }

    // Copy over the original compute region.
    unsigned computeRegionIndex = numDataMovementRegions;
    auto &newRegion = newGeneric.getRegion(computeRegionIndex);
    auto &oldRegion = generic.getRegion(0);
    newRegion.takeBody(oldRegion);

    // Await / Yield insertion to compute region.
    {
      Block *computeBlock = &newGeneric.getRegion(computeRegionIndex).front();
      OpBuilder blockBuilder = OpBuilder::atBlockBegin(computeBlock);
      auto [inputOperandsIndex, inputOperandsLength] =
          generic.getODSOperandIndexAndLength(0);

      // Await the inputs
      blockBuilder.create<ttir::AwaitOp>(
          generic->getLoc(), ValueRange(computeBlock->getArguments().take_front(
                                 inputOperandsLength)));

      blockBuilder.setInsertionPointToEnd(computeBlock);

      // Yield the outputs
      blockBuilder.create<ttir::YieldOp>(
          generic->getLoc(), ValueRange(computeBlock->getArguments().take_back(
                                 outputOperandsLength)));
    }

    rewriter.replaceOp(generic, newGeneric);
    return success();
  }
};
} // namespace

namespace {
class TTIRGenericDatamovement : public impl::TTIRGenericDatamovementBase<TTIRGenericDatamovement> {
public:
  using impl::TTIRGenericDatamovementBase<TTIRGenericDatamovement>::TTIRGenericDatamovementBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericDatamovementRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::ttir
