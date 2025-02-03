// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <numeric>

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Memref/IR/Memref.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRGENERICREGION
#define GEN_PASS_DEF_TTIRGENERICLINEARIZEMEMREF
#define GEN_PASS_DEF_TTIRGENERICREGIONOPERANDSTOMEMREF
#define GEN_PASS_DEF_TTIRGENERICOPCBS
#define GEN_PASS_DEF_TTIRGENERICDATAMOVEMENT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Generic - Region pass
//===----------------------------------------------------------------------===//

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

  auto genericOp = rewriter.create<ttir::GenericOp>(
      op.getLoc(), op->getResults().getTypes(), dps.getDpsInputs(),
      ValueRange() /* cbs */, dps.getDpsInits(), gridAttr, indexingMaps,
      iteratorTypes, ::llvm::ArrayRef<int64_t>(), 1 /*numRegions*/);

  // Create a new basic block for the generic op and create block arguments.
  Block *block = rewriter.createBlock(&genericOp.getRegion(0));
  SmallVector<Location> blockArgumentLocs(genericOp.getOperands().size(),
                                          genericOp.getLoc());

  SmallVector<Type> blockArgTypes(
      llvm::map_range(genericOp.getOperands().getTypes(), [&](Type type) {
        RankedTensorType tensorType = mlir::cast<RankedTensorType>(type);
        tt::MetalLayoutAttr layout =
            mlir::cast<tt::MetalLayoutAttr>(tensorType.getEncoding());
        return layout.getMemref();
      }));

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
  opBuilder.create<mlir::tt::ttir::YieldOp>(loc, mlir::ValueRange());
}

class TTIRGenericRegionRewriter
    : public OpInterfaceRewritePattern<GenericRegionOp> {
public:
  using OpInterfaceRewritePattern<GenericRegionOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(GenericRegionOp op,
                                PatternRewriter &rewriter) const final {
    if (mlir::isa<GenericOp>(op.getOperation()->getParentOp())) {
      return failure();
    }

    auto [genericOp, block] = buildGenericOp(op, rewriter);
    block->eraseArguments(0, block->getNumArguments());
    SmallVector<Location> blockArgumentLocs(genericOp.getOperands().size(),
                                            genericOp.getLoc());
    block->addArguments(TypeRange(genericOp.getOperandTypes()),
                        blockArgumentLocs);

    // Convert the original op into arith/math and into the generic block.
    OpBuilder blockBuilder = OpBuilder::atBlockEnd(block);
    op.buildGenericRegion(blockBuilder, block);
    rewriter.replaceOp(op, genericOp);
    return success();
  }
};

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

class TTIRGenericRegion
    : public impl::TTIRGenericRegionBase<TTIRGenericRegion> {
public:
  using impl::TTIRGenericRegionBase<TTIRGenericRegion>::TTIRGenericRegionBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    if (newLowering) {
      patterns.add<TTIRGenericMaximumRewriter>(&getContext());
    } else {
      patterns.add<TTIRGenericRegionRewriter>(&getContext());
    }
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
    bool allConverted = true;
    for (Region &region : op.getRegions()) {
      Block *entry = &region.front();
      auto args = entry->getArguments();
      allConverted &= llvm::all_of(args, isLinearizedMemref);
    }
    if (allConverted) {
      return failure();
    }

    rewriter.modifyOpInPlace(op, [&]() {
      for (Region &region : op.getRegions()) {
        Block *entry = &region.front();
        auto args = entry->getArguments();
        rewriter.setInsertionPointToStart(entry);
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
            } else if (auto store =
                           mlir::dyn_cast<affine::AffineStoreOp>(user)) {
              store.setMap(linearMap.compose(store.getMap()));
            }
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

//===----------------------------------------------------------------------===//
// Generic - Operands to memref pass
//===----------------------------------------------------------------------===//

struct TTIRGenericOperandsToMemrefRewriter
    : public OpConversionPattern<GenericOp> {
  using OpConversionPattern<GenericOp>::OpConversionPattern;

  template <typename ValueRange>
  void convertTypes(ValueRange valueRange,
                    DenseMap<Type, Type> const &typeMap) const {
    for (auto operand : valueRange) {
      if (typeMap.count(operand.getType()) == 0) {
        continue;
      }
      operand.setType(typeMap.at(operand.getType()));
    }
  }

  LogicalResult
  matchAndRewrite(GenericOp generic, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Block *entry = &generic.getRegion(0).front();
    auto firstEntryArgType = entry->getArguments()[0].getType();
    auto encoding =
        mlir::cast<RankedTensorType>(firstEntryArgType).getEncoding();
    if (mlir::isa_and_nonnull<BufferAttr>(encoding)) {
      // Already converted.
      return failure();
    }

    rewriter.modifyOpInPlace(generic, [&]() {
      DenseMap<Type, Type> typeMap;

      for (auto blockArg : entry->getArguments()) {
        uint32_t blockArgNumber = blockArg.getArgNumber();
        auto matchingOperand = generic.getMatchingOperand(blockArgNumber);
        auto operandType = matchingOperand.getType();

        auto bufferLayout = mlir::cast<MetalLayoutAttr>(
            mlir::cast<RankedTensorType>(operandType).getEncoding());
        auto bufferType = operandType;

        int64_t cbIndex = generic.getOperandCbMapping()[blockArgNumber];

        if (cbIndex >= 0) {
          assert(static_cast<size_t>(cbIndex) < generic.getCbs().size());
          auto cb = generic.getCbs()[cbIndex];
          auto cbType = cb.getType();
          auto cbLayout = mlir::cast<MetalLayoutAttr>(
              mlir::cast<RankedTensorType>(cbType).getEncoding());
          bufferLayout = cbLayout;
          bufferType = cbType;
        }

        // TODO(rpavlovic): introduce multiplier for buffer.
        auto buffer = BufferAttr::get(
            getContext(), bufferLayout.getMemref(),
            (cbIndex >= 0 ? BufferAccess::Stream : BufferAccess::Alias));

        auto ty = RankedTensorType::get(
            buffer.getShape(),
            mlir::cast<RankedTensorType>(bufferType).getElementType(), buffer);

        typeMap[blockArg.getType()] = ty;
        blockArg.setType(ty);
      }
      for (Operation &op : generic.getRegion(0).getOps()) {
        convertTypes(op.getOperands(), typeMap);
        convertTypes(op.getResults(), typeMap);
      }
    });

    return success();
  }
};

class TTIRGenericRegionMemrefTypeConverter : public TypeConverter {
public:
  TTIRGenericRegionMemrefTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](RankedTensorType type) -> Type {
      auto encoding = type.getEncoding();
      assert(encoding);
      if (mlir::isa<BufferAttr>(encoding)) {
        return type;
      }
      auto layout = mlir::cast<MetalLayoutAttr>(type.getEncoding());
      auto buffer =
          BufferAttr::get(ctx, layout.getMemref(), BufferAccess::Alias);
      return RankedTensorType::get(buffer.getShape(), type.getElementType(),
                                   buffer);
    });
  }
};

class TTIRGenericRegionOperandsToMemref
    : public impl::TTIRGenericRegionOperandsToMemrefBase<
          TTIRGenericRegionOperandsToMemref> {
public:
  using impl::TTIRGenericRegionOperandsToMemrefBase<
      TTIRGenericRegionOperandsToMemref>::TTIRGenericRegionOperandsToMemrefBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    TTIRGenericRegionMemrefTypeConverter typeConverter(&getContext());
    patterns.add<TTIRGenericOperandsToMemrefRewriter>(typeConverter,
                                                      &getContext());

    mlir::ConversionTarget target(getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};

//===----------------------------------------------------------------------===//
// Generic - CBs pass
//===----------------------------------------------------------------------===//

class TTIRGenericOpCBsRewriter : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (!generic.getOperandCbMapping().empty()) {
      // Already inserted CBs, therefore skip.
      return failure();
    }

    rewriter.setInsertionPointToStart(generic->getBlock());

    SmallVector<Value> cbValues;
    SmallVector<int64_t> operandCBMapping;

    for (auto operand : generic->getOperands()) {
      auto ty = mlir::cast<RankedTensorType>(operand.getType());

      // Enforcing tiled layout as in kernel we always want to work with tiles.
      auto desiredElementType = rewriter.getType<TileType>(ty.getElementType());
      auto desiredLayout = rewriter.getAttr<MetalLayoutAttr>(
          ty, MemorySpace::DeviceL1, generic.getGrid(), desiredElementType);

      auto operandTy = operand.getType();
      auto operandLayout = mlir::cast<MetalLayoutAttr>(
          mlir::cast<RankedTensorType>(operandTy).getEncoding());

      if (desiredLayout.getGrid() == operandLayout.getGrid()) {
        // TODO(rpavlovic): should we check other layout features such as
        // linear?
        operandCBMapping.push_back(-1);
        continue;
      }

      // Creating a CB for the operand. It takes the same type as the operand,
      // but changes its grid. This may result in overly large CBs at the
      // moment.
      auto emptyOp = rewriter.create<tensor::EmptyOp>(
          generic->getLoc(), ty.getShape(), ty.getElementType(), desiredLayout);
      cbValues.push_back(emptyOp.getResult());
      operandCBMapping.push_back(cbValues.size() - 1);
    }

    rewriter.setInsertionPointAfter(generic);
    auto newGenericOp = rewriter.create<ttir::GenericOp>(
        generic->getLoc(), generic.getResultTypes(), generic.getInputs(),
        cbValues, generic.getOutputs(), generic.getGrid(),
        generic.getIndexingMaps(), generic.getIteratorTypes(), operandCBMapping,
        1 /*numRegions*/);

    auto &oldRegion = generic.getRegion(0);
    newGenericOp->getRegion(0).takeBody(oldRegion);

    rewriter.replaceOp(generic, newGenericOp);
    return success();
  }
};

class TTIRGenericOpCBs : public impl::TTIRGenericOpCBsBase<TTIRGenericOpCBs> {
public:
  using impl::TTIRGenericOpCBsBase<TTIRGenericOpCBs>::TTIRGenericOpCBsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericOpCBsRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }
};

namespace {
class TTIRGenericDatamovementRewriter : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult
  buildDatamovementBlock(OpBuilder blockBuilder, Location loc,
                         ArrayRef<OpOperand> genericOperands,
                         ArrayRef<BlockArgument> blockOperands) const {
    blockBuilder.create<ttir::YieldOp>(loc, ValueRange());
    return success();
  }

  LogicalResult matchAndRewrite(GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (generic.getNumRegions() > 1) {
      // Already inserted, skip.
      return failure();
    }

    auto newGeneric = rewriter.create<GenericOp>(
        generic->getLoc(), generic.getResultTypes(), generic.getInputs(),
        generic.getCbs(), generic.getOutputs(), generic.getGrid(),
        generic.getIndexingMaps(), generic.getIteratorTypes(),
        generic.getOperandCbMapping(), generic.getNumRegions() + 1);

    Block *datamovementBlock = &newGeneric.getRegion(0).emplaceBlock();
    datamovementBlock->addArguments(
        generic.getRegion(0).getArgumentTypes(),
        SmallVector<mlir::Location>(
            generic.getRegion(0).getArgumentTypes().size(), generic.getLoc()));

    OpBuilder blockBuilder = OpBuilder::atBlockEnd(datamovementBlock);
    auto result = buildDatamovementBlock(blockBuilder, generic->getLoc(),
                                         generic->getOpOperands(),
                                         datamovementBlock->getArguments());
    if (failed(result)) {
      return result;
    }

    for (unsigned regionNum = 1; regionNum < newGeneric.getNumRegions();
         ++regionNum) {
      auto &newRegion = newGeneric.getRegion(regionNum);
      auto &oldRegion = generic.getRegion(regionNum - 1);
      newRegion.takeBody(oldRegion);
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
