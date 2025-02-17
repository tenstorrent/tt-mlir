// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <numeric>

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRGENERICREGION
#define GEN_PASS_DEF_TTIRGENERICLINEARIZEMEMREF
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
      iteratorTypes);

  // Create a new basic block for the generic op and create block arguments.
  Block *block = rewriter.createBlock(&genericOp.getRegion());
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
  opBuilder.create<mlir::tt::ttir::YieldOp>(loc, mlir::ValueRange());
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

class TTIRGenericRegion
    : public impl::TTIRGenericRegionBase<TTIRGenericRegion> {
public:
  using impl::TTIRGenericRegionBase<TTIRGenericRegion>::TTIRGenericRegionBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericMaximumRewriter>(&getContext());
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
    Block *entry = &op.getRegion().front();
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

} // namespace mlir::tt::ttir
