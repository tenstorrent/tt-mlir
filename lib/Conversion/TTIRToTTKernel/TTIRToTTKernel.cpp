// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTKernel/TTIRToTTKernel.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTIR/Analysis/CBProducerConsumer.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRTraits.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include <type_traits>

#include <utility>

namespace mlir::tt::ttkernel {

namespace {

static Value i32(OpBuilder &rewriter, Location loc, int32_t value) {
  return rewriter
      .create<arith::ConstantOp>(loc, rewriter.getI32Type(),
                                 rewriter.getI32IntegerAttr(value))
      .getResult();
}

static Value index(OpBuilder &rewriter, Location loc, int64_t value) {
  return rewriter
      .create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                 rewriter.getIndexAttr(value))
      .getResult();
}

static std::pair<Value, Value>
getVirtualCoordsFromLogicalCoords(OpBuilder &rewriter, Location loc,
                                  ttcore::ChipDescAttr chipDesc,
                                  ValueRange dstCoreIndex) {
  auto offset = chipDesc.getCoordTranslationOffsets();

  return {rewriter
              .create<arith::AddIOp>(dstCoreIndex[0].getLoc(), dstCoreIndex[0],
                                     index(rewriter, loc, offset[0]))
              .getResult(),
          rewriter
              .create<arith::AddIOp>(dstCoreIndex[1].getLoc(), dstCoreIndex[1],
                                     index(rewriter, loc, offset[1]))
              .getResult()};
}

static std::pair<Value, Value> getMcastEndCoords(PatternRewriter &rewriter,
                                                 Location loc, Value &nocStartY,
                                                 Value &nocStartX,
                                                 OperandRange mcastShape) {
  return {rewriter.create<arith::SubIOp>(
              nocStartY.getLoc(),
              rewriter.create<arith::AddIOp>(nocStartY.getLoc(), nocStartY,
                                             mcastShape[0]),
              index(rewriter, loc, 1)),
          rewriter.create<arith::SubIOp>(
              nocStartX.getLoc(),
              rewriter.create<arith::AddIOp>(nocStartX.getLoc(), nocStartX,
                                             mcastShape[1]),
              index(rewriter, loc, 1))};
}

static Value getTileIndexFromBlockView(RewriterBase &rewriter, Location loc,
                                       Value inputView) {
  if (auto collapseOp =
          mlir::dyn_cast<memref::CollapseShapeOp>(inputView.getDefiningOp())) {
    return getTileIndexFromBlockView(rewriter, loc, collapseOp.getSrc());
  }
  if (auto subViewOp =
          mlir::dyn_cast<memref::SubViewOp>(inputView.getDefiningOp())) {
    // We have blocked this input. We need to get the indicies for the first
    // tile in the subview.
    SmallVector<Value> indices = {index(rewriter, loc, 0),
                                  index(rewriter, loc, 0)};
    SmallVector<Value> sourceIndices;

    // TODO(#4717): This call alone should be enough to get the tile indices,
    // but currently it returns block index instead. Once fixed, we can remove
    // all the other calculations below.
    affine::resolveIndicesIntoOpWithOffsetsAndStrides(
        rewriter, loc, subViewOp.getMixedOffsets(), subViewOp.getMixedStrides(),
        subViewOp.getDroppedDims(), indices, sourceIndices);

    auto resultTy = mlir::cast<MemRefType>(subViewOp.getResult().getType());
    Value rtIdx = index(rewriter, loc, resultTy.getShape()[0]);
    Value ktIdx = index(rewriter, loc, resultTy.getShape()[1]);
    Value tilesPerBlock = rewriter.create<arith::MulIOp>(loc, rtIdx, ktIdx);

    // Convert the resolved source row offset to a block-row index.
    Value rowBlockIdx =
        rewriter.create<arith::DivSIOp>(loc, sourceIndices[0], rtIdx);
    Value rowBase =
        rewriter.create<arith::MulIOp>(loc, rowBlockIdx, tilesPerBlock);
    return rewriter.create<arith::AddIOp>(loc, rowBase, sourceIndices[1]);
  }

  if (mlir::isa<memref::CastOp>(inputView.getDefiningOp())) {
    // We have not blocked this input. Ignore the cast and start from index 0 of
    // the input.
    return index(rewriter, loc, 0);
  }
  llvm_unreachable("Expected subview or cast op");
}

static Value getCB(ConversionPatternRewriter &rewriter, Value cb) {
  if (auto collapseOp =
          mlir::dyn_cast<memref::CollapseShapeOp>(cb.getDefiningOp())) {
    return getCB(rewriter, collapseOp.getSrc());
  }

  if (memref::LoadOp loadOp =
          mlir::dyn_cast<memref::LoadOp>(cb.getDefiningOp());
      loadOp) {
    assert(loadOp.getIndices().size() == 1 &&
           "Expected single index in load op, failing.");
    return rewriter.getRemappedValue(loadOp.getMemref());
  }

  if (mlir::isa<memref::SubViewOp>(cb.getDefiningOp())) {
    memref::SubViewOp subViewOp =
        mlir::cast<memref::SubViewOp>(cb.getDefiningOp());
    return rewriter.getRemappedValue(subViewOp.getSource());
  }

  if (mlir::isa<memref::CastOp>(cb.getDefiningOp())) {
    memref::CastOp castOp = mlir::cast<memref::CastOp>(cb.getDefiningOp());
    return rewriter.getRemappedValue(castOp.getSource());
  }
  llvm_unreachable("Expected load or subview op");
}

static Value getDstIdxFromResult(Value ttirOpResult) {
  memref::StoreOp storeOp;
  for (Operation *op : ttirOpResult.getUsers()) {
    auto maybeStore = mlir::dyn_cast<memref::StoreOp>(op);
    if (maybeStore && ttcore::getMemorySpace(maybeStore.getMemRef()) ==
                          ttcore::MemorySpace::RegisterDst) {
      storeOp = mlir::cast<memref::StoreOp>(op);
      break;
    }
  }
  assert(storeOp && "Expected store op.");
  assert(storeOp.getIndices().size() == 1 &&
         "Expected single index in store op");
  return storeOp.getIndices().front();
}

// This is a workaround special case for getting an in/out CB. This whole
// routine should go away with issue:
// https://github.com/tenstorrent/tt-mlir/issues/3602
template <typename LoadOrStoreOp>
static Value getInOrOutCB(ConversionPatternRewriter &rewriter, Operation *op) {
  static_assert(std::is_same_v<LoadOrStoreOp, memref::LoadOp> ||
                std::is_same_v<LoadOrStoreOp, memref::StoreOp>);
  func::FuncOp func = op->getParentOfType<func::FuncOp>();
  assert(func && "Expected func op.");
  Value cb = nullptr;
  func.walk([&](LoadOrStoreOp loadStore) {
    if (ttcore::getMemorySpace(loadStore.getMemRef()) ==
        ttcore::MemorySpace::DeviceL1) {
      cb = loadStore.getMemRef();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  assert(cb && "CB not found.");
  return rewriter.getRemappedValue(cb);
}

// This is a workaround special case for getting an input CB. This whole
// routine should go away with issue:
// https://github.com/tenstorrent/tt-mlir/issues/3602
static Value getInCB(ConversionPatternRewriter &rewriter, Operation *op) {
  // Search for a load from L1.
  return getInOrOutCB<memref::LoadOp>(rewriter, op);
}

// This is a workaround special case for getting an output CB. This whole
// routine should go away with issue:
// https://github.com/tenstorrent/tt-mlir/issues/3602
static Value getOutCB(ConversionPatternRewriter &rewriter, Operation *op) {
  // Search for a store to L1.
  return getInOrOutCB<memref::StoreOp>(rewriter, op);
}

static void setInsertionPointAfterOperands(OpBuilder &rewriter,
                                           llvm::ArrayRef<Value> operands) {
  Operation *latestDefOp = nullptr;
  for (Value operand : operands) {
    Operation *definingOp = operand.getDefiningOp();
    if (!latestDefOp ||
        (definingOp && !definingOp->isBeforeInBlock(latestDefOp))) {
      latestDefOp = definingOp;
    }
  }

  assert(latestDefOp != nullptr);
  rewriter.setInsertionPointAfter(latestDefOp);
}

} // namespace

namespace {
class AcquireDstRewriter : public OpConversionPattern<ttir::AcquireDstOp> {
public:
  using OpConversionPattern<ttir::AcquireDstOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::AcquireDstOp op, ttir::AcquireDstOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.create<ttkernel::TileRegsAcquireOp>(op.getLoc());
    // Dst is an implicit resource in TTKernel, so we can just erase it.
    rewriter.eraseOp(op);
    return success();
  };
};
} // namespace

namespace {
class MemrefLoadRewriter : public OpConversionPattern<memref::LoadOp> {
public:
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, memref::LoadOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    assert(adaptor.getIndices().size() == 1 &&
           "Expected single index in load op, failing.");
    rewriter.replaceOp(op, adaptor.getIndices().front());
    return success();
  };
};
} // namespace

namespace {
class MemrefStoreRewriter : public OpConversionPattern<memref::StoreOp> {
public:
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  static LogicalResult lowerCopyTile(memref::LoadOp load, memref::StoreOp store,
                                     memref::StoreOpAdaptor adaptor,
                                     ConversionPatternRewriter &rewriter) {
    assert(adaptor.getIndices().size() == 1);
    auto cb = rewriter.getRemappedValue(load.getMemref());
    auto cbIndex = adaptor.getValue();
    auto dstIndex = adaptor.getIndices().front();
    rewriter.create<ttkernel::CopyTileInitOp>(store.getLoc(), cb);
    rewriter.replaceOpWithNewOp<ttkernel::CopyTileOp>(store, cb, cbIndex,
                                                      dstIndex);
    return success();
  }

  static LogicalResult lowerPackTile(memref::StoreOp store,
                                     memref::StoreOpAdaptor adaptor,
                                     ConversionPatternRewriter &rewriter) {
    assert(adaptor.getIndices().size() == 1);
    auto dst = adaptor.getValue();
    auto cb = adaptor.getMemref();
    auto storeIdx = adaptor.getIndices().front();
    rewriter.replaceOpWithNewOp<ttkernel::PackTileOp>(
        store, dst, cb, storeIdx, rewriter.getBoolAttr(true));
    return success();
  }

  LogicalResult
  matchAndRewrite(memref::StoreOp op, memref::StoreOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto load = mlir::dyn_cast<memref::LoadOp>(op.getValue().getDefiningOp());
    bool storeToDst = ttcore::getMemorySpace(op.getMemRef()) ==
                      ttcore::MemorySpace::RegisterDst;

    if (load && storeToDst) {
      // If we are coming from a load, then we are a copy tile. Pattern:
      //    %0 = memref.load %arg0, %c0 : memref<1x!tt.tile, l1>
      //    tt.store %0, %arg1, %c0 : memref<1x!tt.tile, dst>
      return lowerCopyTile(load, op, adaptor, rewriter);
    }

    if (storeToDst) {
      // Otherwise we're storing the result of an op:
      //    %0 = ttir.tile_sigmoid %arg0
      //    tt.store %0, %arg2, %c0 : memref<1x!tt.tile, dst>
      rewriter.eraseOp(op);
      return success();
    }

    // Otherwise we're packing the result from dst:
    //    %0 = memref.load %arg0, %c0 : memref<1x!tt.tile, dst>
    //    tt.store %0, %arg2, %c0 : memref<1x!tt.tile, l1>
    return lowerPackTile(op, adaptor, rewriter);
  };
};
} // namespace

template <typename... Pairs>
struct OpMap {};

// clang-format off
using ComputeOpMap = OpMap<
  // Elementwise FPU.
  std::pair<ttir::TileAddOp,         std::pair<ttkernel::AddTilesInitOp,            ttkernel::AddTilesOp>>,
  std::pair<ttir::TileMatmulOp,      std::pair<ttkernel::MatmulInitOp,              ttkernel::MatmulTilesOp>>,
  std::pair<ttir::TileMatmulBlockOp, std::pair<ttkernel::MatmulBlockInitOp,         ttkernel::ExperimentalMatmulBlockOp>>,
  std::pair<ttir::TileMulOp,         std::pair<ttkernel::MulTilesInitOp,            ttkernel::MulTilesOp>>,
  std::pair<ttir::TileSubOp,         std::pair<ttkernel::SubTilesInitOp,            ttkernel::SubTilesOp>>,

  // Elementwise SFPU Unary.
  std::pair<ttir::TileAbsOp,         std::pair<ttkernel::AbsTileInitOp,             ttkernel::AbsTileOp>>,
  std::pair<ttir::TileCeilOp,        std::pair<ttkernel::RoundingTileInitOp,        ttkernel::CeilTileOp>>,
  std::pair<ttir::TileCosOp,         std::pair<ttkernel::CosTileInitOp,             ttkernel::CosTileOp>>,
  std::pair<ttir::TileExpOp,         std::pair<ttkernel::ExpTileInitOp,             ttkernel::ExpTileOp>>,
  std::pair<ttir::TileFloorOp,       std::pair<ttkernel::RoundingTileInitOp,        ttkernel::FloorTileOp>>,
  std::pair<ttir::TileGeluOp,        std::pair<ttkernel::GeluTileInitOp,            ttkernel::GeluTileOp>>,
  std::pair<ttir::TileLogOp,         std::pair<ttkernel::LogTileInitOp,             ttkernel::LogTileOp>>,
  std::pair<ttir::TileLogicalNotOp,  std::pair<ttkernel::LogicalNotUnaryTileInitOp, ttkernel::LogicalNotUnaryTileOp>>,
  std::pair<ttir::TileNegativeOp,    std::pair<ttkernel::NegativeTileInitOp,        ttkernel::NegativeTileOp>>,
  std::pair<ttir::TileRecipOp,       std::pair<ttkernel::RecipTileInitOp,           ttkernel::RecipTileOp>>,
  std::pair<ttir::TileRsqrtOp,       std::pair<ttkernel::RsqrtTileInitOp,           ttkernel::RsqrtTileOp>>,
  std::pair<ttir::TileSqrtOp,        std::pair<ttkernel::SqrtTileInitOp,            ttkernel::SqrtTileOp>>,
  std::pair<ttir::TileSigmoidOp,     std::pair<ttkernel::SigmoidTileInitOp,         ttkernel::SigmoidTileOp>>,
  std::pair<ttir::TileSinOp,         std::pair<ttkernel::SinTileInitOp,             ttkernel::SinTileOp>>,
  std::pair<ttir::TileTanOp,         std::pair<ttkernel::TanTileInitOp,             ttkernel::TanTileOp>>,
  std::pair<ttir::TileEqzOp,         std::pair<ttkernel::EqzTileInitOp,             ttkernel::EqzTileOp>>,
  std::pair<ttir::TileNezOp,         std::pair<ttkernel::NezTileInitOp,             ttkernel::NezTileOp>>,
  std::pair<ttir::TileGtzOp,         std::pair<ttkernel::GtzTileInitOp,             ttkernel::GtzTileOp>>,
  std::pair<ttir::TileGezOp,         std::pair<ttkernel::GezTileInitOp,             ttkernel::GezTileOp>>,
  std::pair<ttir::TileLtzOp,         std::pair<ttkernel::LtzTileInitOp,             ttkernel::LtzTileOp>>,
  std::pair<ttir::TileLezOp,         std::pair<ttkernel::LezTileInitOp,             ttkernel::LezTileOp>>,

  // Elementwise SFPU Binary.
  std::pair<ttir::TileSubBinaryOp,   std::pair<ttkernel::SubBinaryTilesInitOp,      ttkernel::SubBinaryTilesOp>>,
  std::pair<ttir::TileDivOp,         std::pair<ttkernel::DivBinaryTilesInitOp,      ttkernel::DivBinaryTilesOp>>,
  std::pair<ttir::TileMaximumOp,     std::pair<ttkernel::MaxTilesInitOp,            ttkernel::MaxTilesOp>>,
  std::pair<ttir::TilePowOp,         std::pair<ttkernel::PowBinaryTilesInitOp,      ttkernel::PowBinaryTilesOp>>
>;
// clang-format on

template <typename SrcOp, typename OpMap>
struct OpMapLookup;

template <typename SrcOp>
struct OpMapLookup<SrcOp, OpMap<>> {
  using type = std::pair<void, void>;
};

template <typename SrcOp, typename Key, typename Value, typename... Rest>
struct OpMapLookup<SrcOp, OpMap<std::pair<Key, Value>, Rest...>> {
  using type =
      std::conditional_t<std::is_same_v<SrcOp, Key>, Value,
                         typename OpMapLookup<SrcOp, OpMap<Rest...>>::type>;
};

template <typename SrcOp, typename OpMap>
using TTKernelOpPair = typename OpMapLookup<SrcOp, OpMap>::type;

namespace {

template <typename ConcreteOp>
class TTIRFPUOpsRewriter : public OpConversionPattern<ConcreteOp> {
public:
  using OpConversionPattern<ConcreteOp>::OpConversionPattern;
  using KernelOpPair = TTKernelOpPair<ConcreteOp, ComputeOpMap>;
  using InitOp = typename KernelOpPair::first_type;
  using FPUOp = typename KernelOpPair::second_type;

  static_assert(FPUOp::template hasTrait<TTKernelFPUOpTrait>(),
                "FPUOp must have TTKernelFPUOpTrait");

  static constexpr int arity = FPUOp::arity;

  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    assert(op->hasOneUse() || op->use_empty());
    if constexpr (arity == 1) {
      assert(op->getNumOperands() == 1u);
    } else if constexpr (arity == 2) {
      assert(op->getNumOperands() == 2u);
      auto insertionPoint = rewriter.getInsertionPoint();
      auto cbA = getCB(rewriter, op.getLhs());
      auto cbB = getCB(rewriter, op.getRhs());
      auto outCB = getOutCB(rewriter, op);
      setInsertionPointAfterOperands(rewriter, {cbA, cbB, outCB});
      rewriter.create<ttkernel::BinaryOpInitCommonOp>(op->getLoc(), cbA, cbB,
                                                      outCB);
      rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);
    } else {
      static_assert(arity == 3 && !ttmlir::utils::always_false<ConcreteOp>(),
                    "FPUOp must be unary, binary or ternary");
      assert(op->getNumOperands() == 3u);
    }

    if constexpr (std::is_same_v<ConcreteOp, ttir::TileMatmulOp>) {
      auto insertionPoint = rewriter.getInsertionPoint();
      auto cbA = getCB(rewriter, op.getA());
      auto cbB = getCB(rewriter, op.getB());
      auto outCB = getOutCB(rewriter, op);
      setInsertionPointAfterOperands(rewriter, {cbA, cbB, outCB});
      auto transpose = i32(rewriter, op->getLoc(), 0);
      rewriter.create<ttkernel::MatmulInitOp>(op->getLoc(), cbA, cbB, outCB,
                                              transpose);
      rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);
      rewriter.create<ttkernel::MatmulInitShortOp>(op->getLoc(), cbA, cbB,
                                                   transpose);
      rewriter.create<ttkernel::MatmulTilesOp>(op->getLoc(), cbA, cbB,
                                               adaptor.getA(), adaptor.getB(),
                                               adaptor.getC(), transpose);
    } else if constexpr (std::is_same_v<ConcreteOp, ttir::TileMatmulBlockOp>) {
      auto insertionPoint = rewriter.getInsertionPoint();
      auto cbA = getCB(rewriter, op.getA());
      auto cbB = getCB(rewriter, op.getB());
      auto outCB = getCB(rewriter, op.getOutput());
      setInsertionPointAfterOperands(rewriter, {cbA, cbB, outCB});

      // destIndex is always 0 because we call an experimental LLK that fills up
      // the entire dest in a loop.
      Value destIndex = index(rewriter, op->getLoc(), 0);

      assert(op.hasBlockDims() &&
             "MatmulBlockOp must have all block dimensions to be lowered");
      auto rt_i32 =
          i32(rewriter, op->getLoc(), static_cast<int32_t>(*op.getBlockM()));
      auto kt_i32 =
          i32(rewriter, op->getLoc(), static_cast<int32_t>(*op.getBlockK()));
      auto ct_i32 =
          i32(rewriter, op->getLoc(), static_cast<int32_t>(*op.getBlockN()));
      auto nt_i32 = i32(rewriter, op->getLoc(),
                        static_cast<int32_t>(*op.getBBlockStride()));

      auto transpose = i32(rewriter, op->getLoc(), 0);

      rewriter.create<ttkernel::MatmulBlockInitOp>(
          op->getLoc(), cbA, cbB, outCB, transpose, ct_i32, rt_i32, kt_i32);
      rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);
      rewriter.create<ttkernel::MatmulBlockInitShortOp>(
          op->getLoc(), cbA, cbB, transpose, ct_i32, rt_i32, kt_i32);

      // Get the tile index for each input in the global memref. This is done by
      // resolving tile (0,0) from the subview, representing a block, into the
      // address space of the source memref.
      Value aTileIndex =
          getTileIndexFromBlockView(rewriter, op->getLoc(), op.getA());
      Value bTileIndex =
          getTileIndexFromBlockView(rewriter, op->getLoc(), op.getB());

      rewriter.create<ttkernel::ExperimentalMatmulBlockOp>(
          op->getLoc(), cbA, cbB, aTileIndex, bTileIndex, destIndex, transpose,
          ct_i32, rt_i32, kt_i32, nt_i32);

      // Operands are linearized, so this pass creates a
      // ttkernel.cb_reinterpret_shape from each memref.collapse_shape. However,
      // ttkernel.matmul_block does *not* use the cb_reinterpret_shape so we
      // need to clean it up. Cannonicalize does *not* do this automatically.
      auto tryEraseDeadCBReinterpret = [&](Value v) {
        Value remapped = rewriter.getRemappedValue(v);
        if (!remapped) {
          return;
        }
        if (auto *def = remapped.getDefiningOp()) {
          if (auto cbri = dyn_cast<ttkernel::CBReinterpretShapeOp>(def)) {
            if (cbri->use_empty()) {
              rewriter.eraseOp(cbri);
            }
          }
        }
      };
      tryEraseDeadCBReinterpret(op.getA());
      tryEraseDeadCBReinterpret(op.getB());
      tryEraseDeadCBReinterpret(op.getOutput());

    } else if constexpr (arity == 2) {
      auto dstIdx = getDstIdxFromResult(op.getResult());
      rewriter.create<InitOp>(op->getLoc(), getCB(rewriter, op.getLhs()),
                              getCB(rewriter, op.getRhs()));
      rewriter.create<FPUOp>(op->getLoc(), getCB(rewriter, op.getLhs()),
                             getCB(rewriter, op.getRhs()), adaptor.getLhs(),
                             adaptor.getRhs(), dstIdx);
    } else {
      return llvm::failure();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

namespace {

template <typename ConcreteOp>
class TTIRSFPUOpsRewriter : public OpConversionPattern<ConcreteOp> {
public:
  using OpConversionPattern<ConcreteOp>::OpConversionPattern;
  using KernelOpPair = TTKernelOpPair<ConcreteOp, ComputeOpMap>;
  using InitOp = typename KernelOpPair::first_type;
  using SFPUOp = typename KernelOpPair::second_type;

  static_assert(SFPUOp::template hasTrait<TTKernelSFPUOpTrait>(),
                "SFPUOp must have TTKernelSFPUOpTrait");

  static constexpr int arity = SFPUOp::arity;

  static_assert(arity == 1 || arity == 2,
                "Only unary and binary SFPUOps are supported");

  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto insertionPoint = rewriter.getInsertionPoint();
    auto inCB = getInCB(rewriter, op);
    auto outCB = getOutCB(rewriter, op);
    setInsertionPointAfterOperands(rewriter, {inCB, outCB});
    // Don't init_sfpu for tile sub binary op because it's only used in
    // conjuction with a comparison op that calls init_sfpu and we only need one
    // per compute kernel.
    if constexpr (!std::is_same_v<ConcreteOp, ttir::TileSubBinaryOp>) {
      rewriter.create<ttkernel::InitSFPUOp>(op->getLoc(), inCB, outCB);
    }
    rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);

    rewriter.create<InitOp>(op->getLoc());
    if constexpr (std::is_same_v<SFPUOp, ttkernel::CeilTileOp> ||
                  std::is_same_v<SFPUOp, ttkernel::FloorTileOp>) {
      const auto elemType =
          mlir::cast<ttcore::TileType>(op.getInput().getType())
              .getElementType();
      const bool isCBF32 = llvm::isa<Float32Type>(elemType);
      if (isCBF32) {
        if (std::is_same_v<SFPUOp, ttkernel::CeilTileOp>) {
          rewriter.create<ttkernel::CeilTileF32Op>(op->getLoc(),
                                                   adaptor.getInput());
        } else {
          rewriter.create<ttkernel::FloorTileF32Op>(op->getLoc(),
                                                    adaptor.getInput());
        }
      } else {
        rewriter.create<SFPUOp>(op->getLoc(), adaptor.getInput());
      }
    } else if constexpr (std::is_same_v<SFPUOp, ttkernel::AbsTileOp> ||
                         std::is_same_v<SFPUOp,
                                        ttkernel::LogicalNotUnaryTileOp>) {
      const auto elemType =
          mlir::cast<ttcore::TileType>(op.getInput().getType())
              .getElementType();
      bool isCBI32 = false;
      if (llvm::isa<IntegerType>(elemType)) {
        isCBI32 = mlir::cast<IntegerType>(elemType).isSigned() &&
                  mlir::cast<IntegerType>(elemType).getWidth() == 32;
      }
      if (isCBI32) {
        if (std::is_same_v<SFPUOp, ttkernel::AbsTileOp>) {
          rewriter.create<ttkernel::AbsTileI32Op>(op->getLoc(),
                                                  adaptor.getInput());
        } else {
          rewriter.create<ttkernel::LogicalNotUnaryTileI32Op>(
              op->getLoc(), adaptor.getInput());
        }
      } else {
        rewriter.create<SFPUOp>(op->getLoc(), adaptor.getInput());
      }
    } else if constexpr (std::is_same_v<SFPUOp, ttkernel::EqzTileOp> ||
                         std::is_same_v<SFPUOp, ttkernel::NezTileOp> ||
                         std::is_same_v<SFPUOp, ttkernel::GtzTileOp> ||
                         std::is_same_v<SFPUOp, ttkernel::GezTileOp> ||
                         std::is_same_v<SFPUOp, ttkernel::LtzTileOp> ||
                         std::is_same_v<SFPUOp, ttkernel::LezTileOp>) {
      const auto elemType =
          mlir::cast<ttcore::TileType>(op.getInput().getType())
              .getElementType();
      bool isCBI32 = false;
      if (llvm::isa<IntegerType>(elemType)) {
        isCBI32 = mlir::cast<IntegerType>(elemType).isSigned() &&
                  mlir::cast<IntegerType>(elemType).getWidth() == 32;
      }
      if (isCBI32) {
        if constexpr (std::is_same_v<SFPUOp, ttkernel::EqzTileOp>) {
          rewriter.create<ttkernel::EqzTileI32Op>(op->getLoc(),
                                                  adaptor.getInput());
        } else if constexpr (std::is_same_v<SFPUOp, ttkernel::NezTileOp>) {
          rewriter.create<ttkernel::NezTileI32Op>(op->getLoc(),
                                                  adaptor.getInput());
        } else if constexpr (std::is_same_v<SFPUOp, ttkernel::GtzTileOp>) {
          rewriter.create<ttkernel::GtzTileI32Op>(op->getLoc(),
                                                  adaptor.getInput());
        } else if constexpr (std::is_same_v<SFPUOp, ttkernel::GezTileOp>) {
          rewriter.create<ttkernel::GezTileI32Op>(op->getLoc(),
                                                  adaptor.getInput());
        } else if constexpr (std::is_same_v<SFPUOp, ttkernel::LtzTileOp>) {
          rewriter.create<ttkernel::LtzTileI32Op>(op->getLoc(),
                                                  adaptor.getInput());
        } else if constexpr (std::is_same_v<SFPUOp, ttkernel::LezTileOp>) {
          rewriter.create<ttkernel::LezTileI32Op>(op->getLoc(),
                                                  adaptor.getInput());
        }
      } else {
        rewriter.create<SFPUOp>(op->getLoc(), adaptor.getInput());
      }
    } else if constexpr (arity == 1) {
      rewriter.create<SFPUOp>(op->getLoc(), adaptor.getInput());
    } else if constexpr (std::is_same_v<SFPUOp, ttkernel::MaxTilesOp>) {
      rewriter.create<SFPUOp>(op->getLoc(), adaptor.getLhs(), adaptor.getRhs());
    } else {
      const auto dstIdx = getDstIdxFromResult(op.getResult());
      rewriter.create<SFPUOp>(op->getLoc(), adaptor.getLhs(), adaptor.getRhs(),
                              dstIdx);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

namespace {

class TTIRTilizeUntilizeRewriter
    : public OpTraitConversionPattern<
          mlir::tt::ttir::TTIRGenericRegionComputeOpTrait> {
public:
  using OpTraitConversionPattern<
      mlir::tt::ttir::TTIRGenericRegionComputeOpTrait>::
      OpTraitConversionPattern;

  static Value findUncollapsedMemref(Value memref) {
    if (auto funcArg = mlir::dyn_cast<BlockArgument>(memref)) {
      return funcArg;
    }
    if (auto collapseOp =
            mlir::dyn_cast<memref::CollapseShapeOp>(memref.getDefiningOp())) {
      return findUncollapsedMemref(collapseOp.getSrc());
    }
    llvm_unreachable("Expected BlockArgument or CollapseShapeOp");
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (auto tilizeOp = mlir::dyn_cast<ttir::TileTilizeBlockOp>(op)) {
      assert(operands.size() == 2);
      Value src = operands[0];
      Value dst = operands[1];
      auto uncollapsedMemrefType = mlir::cast<MemRefType>(
          findUncollapsedMemref(tilizeOp.getOutput()).getType());
      auto blockR =
          i32(rewriter, op->getLoc(), uncollapsedMemrefType.getShape()[0]);
      auto blockC =
          i32(rewriter, op->getLoc(), uncollapsedMemrefType.getShape()[1]);
      rewriter.create<ttkernel::ComputeKernelHWStartupOp>(op->getLoc(), src,
                                                          nullptr, dst);
      rewriter.create<ttkernel::TilizeInitOp>(op->getLoc(), src, blockC, dst);
      rewriter.create<ttkernel::ExperimentalTilizeBlockOp>(op->getLoc(), src,
                                                           dst, blockR, blockC);
    } else if (auto untilizeOp =
                   mlir::dyn_cast<ttir::TileUntilizeBlockOp>(op)) {
      assert(operands.size() == 2);
      Value src = operands[0];
      Value dst = operands[1];
      auto uncollapsedMemrefType = mlir::cast<MemRefType>(
          findUncollapsedMemref(untilizeOp.getInput()).getType());
      auto blockR =
          i32(rewriter, op->getLoc(), uncollapsedMemrefType.getShape()[0]);
      auto blockC =
          i32(rewriter, op->getLoc(), uncollapsedMemrefType.getShape()[1]);
      rewriter.create<ttkernel::ComputeKernelHWStartupOp>(op->getLoc(), src,
                                                          nullptr, dst);
      rewriter.create<ttkernel::UntilizeInitOp>(op->getLoc(), src);
      rewriter.create<ttkernel::ExperimentalUntilizeBlockOp>(
          op->getLoc(), src, dst, blockR, blockC);
    } else {
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  };
};
} // namespace

namespace {
class TTIRTileTransposeRewriter
    : public OpConversionPattern<ttir::TileTransposeOp> {
public:
  using OpConversionPattern<ttir::TileTransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::TileTransposeOp op,
                  ttir::TileTransposeOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // TileTransposeOp is a unary op that takes an input tile and produces
    // an output tile.

    Value inCB = getInCB(rewriter, op);

    Value outCB = getOutCB(rewriter, op);

    auto insertionPoint = rewriter.getInsertionPoint();
    setInsertionPointAfterOperands(rewriter, {inCB, outCB});
    rewriter.create<ttkernel::TransposeInitOp>(op->getLoc(), inCB, outCB);
    rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);

    // Get the tile index from the input operand.
    Value tileIndex = adaptor.getInput();

    // Get the destination index where the result will be stored.
    Value dstIdx = getDstIdxFromResult(op.getResult());

    rewriter.create<ttkernel::TransposeTileOp>(op->getLoc(), inCB, tileIndex,
                                               dstIdx);

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {

class TTIRTypecastRewriter
    : public OpTraitConversionPattern<
          mlir::tt::ttir::TTIRGenericRegionComputeOpTrait> {
public:
  using OpTraitConversionPattern<
      mlir::tt::ttir::TTIRGenericRegionComputeOpTrait>::
      OpTraitConversionPattern;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (mlir::isa<ttir::TileTypecastOp>(op)) {
      rewriter.create<ttkernel::TypecastTileInitOp>(op->getLoc());

      auto inDtype =
          mlir::cast<ttcore::TileType>(operands[0].getType()).getDataType();
      auto outDtype = mlir::cast<ttcore::TileType>(op->getResult(0).getType())
                          .getDataType();
      rewriter.create<ttkernel::TypecastTileOp>(
          op->getLoc(), i32(rewriter, op->getLoc(), 0), inDtype, outDtype);
    } else {
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  };
};
} // namespace

namespace {

template <typename ConcreteOp>
class TTIRAwaitYieldRewriter : public OpConversionPattern<ConcreteOp> {
public:
  using OpConversionPattern<ConcreteOp>::OpConversionPattern;

  static_assert(std::is_same_v<ConcreteOp, ttir::AwaitOp> ||
                    std::is_same_v<ConcreteOp, ttir::YieldOp>,
                "Expected Await or Yield op passed to TTIRAwaitYieldRewriter.");

  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto device = ttcore::lookupDevice(op);
    for (Value input : adaptor.getValues()) {
      auto cb = mlir::dyn_cast<ttkernel::CBType>(input.getType());
      assert(cb && "Expected CB input type to await/yield, failing.");
      auto memref = cb.getMemref();
      auto cbNumPages = device.getMemrefCBNumPages(memref);
      auto numPages = i32(rewriter, op->getLoc(), cbNumPages);
      Block *block = op->getBlock();
      if (mlir::isa<ttir::AwaitOp>(op)) {
        rewriter.create<ttkernel::CBWaitFrontOp>(op.getLoc(), input, numPages);
        auto popFront = rewriter.create<ttkernel::CBPopFrontOp>(
            op.getLoc(), input, numPages);
        rewriter.moveOpBefore(popFront, block->getTerminator());
      } else if (mlir::isa<ttir::YieldOp>(op)) {
        auto reserveBack = rewriter.create<ttkernel::CBReserveBackOp>(
            op.getLoc(), input, numPages);
        if (mlir::isa<func::FuncOp>(block->getParentOp())) {
          rewriter.moveOpAfter(reserveBack, input.getDefiningOp());
        } else {
          rewriter.moveOpBefore(reserveBack, &block->front());
        }
        rewriter.moveOpBefore(numPages.getDefiningOp(), reserveBack);
        rewriter.create<ttkernel::CBPushBackOp>(op.getLoc(), input, numPages);
      }
    }

    rewriter.eraseOp(op);
    return success();
  };
};

} // namespace

namespace {

static Value castCBTypeAsAddress(OpBuilder &rewriter, Location loc, Value cb) {
  // This is required because we blanket convert Memrefs into CBs with a type
  // converter, however there are actually two paths a memref can take:
  // 1. It can be a CBType, which is the case for local memrefs
  // 2. It can represent remote data, which we need to lower to a compile time
  // address (I32 type)
  // More information on ticket #3172
  return rewriter
      .create<UnrealizedConversionCastOp>(loc, rewriter.getI32Type(), cb)
      ->getResult(0);
}

static Value buildNocAddress(OpBuilder &rewriter, Location loc, Value cb,
                             ValueRange index, ttcore::ChipDescAttr chipDesc,
                             ttcore::MemorySpace memspace) {
  assert(memspace == ttcore::MemorySpace::DeviceL1 ||
         memspace == ttcore::MemorySpace::DeviceDRAM);
  auto baseAddr = castCBTypeAsAddress(rewriter, loc, cb);
  assert(index.size() == 3);
  Value noc_addr_op;
  if (memspace == ttcore::MemorySpace::DeviceL1) {
    auto gridY = index[0];
    auto gridX = index[1];
    auto offset = index[2];
    auto offsetInt =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), offset);
    auto addr = rewriter.create<arith::AddIOp>(loc, baseAddr, offsetInt);
    // Translate the src coordinates to virtual coordinates.
    auto [virtY, virtX] = getVirtualCoordsFromLogicalCoords(
        rewriter, loc, chipDesc, ValueRange{gridY, gridX});
    noc_addr_op =
        rewriter.create<ttkernel::GetNocAddrOp>(loc, virtX, virtY, addr);
  } else {
    auto bankID = index[1];
    auto bankIDInt =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), bankID);
    auto offset = index[2];
    auto offsetInt =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), offset);
    auto addr = rewriter.create<arith::AddIOp>(loc, baseAddr, offsetInt);

    return rewriter.create<ttkernel::GetNocAddrFromBankIDOp>(loc, bankIDInt,
                                                             addr);
  }
  return noc_addr_op;
}

template <typename ReadWritePtrOp>
static Value buildL1Address(OpBuilder &rewriter, Location loc, Value cb,
                            ValueRange index) {
  // Use the cb addr as the write address since it is local.
  Value baseAddr = rewriter.create<ReadWritePtrOp>(loc, cb);
  auto offset =
      rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), index[0]);
  return rewriter.create<arith::AddIOp>(loc, baseAddr, offset);
}

class TTIRDMAReadRewriter : public OpConversionPattern<ttir::DMAReadOp> {
public:
  TTIRDMAReadRewriter(TypeConverter &typeConverter, MLIRContext *context,
                      const ttir::AssociatedDMAWaits *associatedDMAWaits,
                      const ttir::CBProducerConsumer *cbProducerConsumer)
      : OpConversionPattern<ttir::DMAReadOp>(typeConverter, context),
        associatedDMAWaits(associatedDMAWaits),
        cbProducerConsumer(cbProducerConsumer) {}

  LogicalResult
  matchAndRewrite(ttir::DMAReadOp op, ttir::DMAReadOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto device = ttcore::lookupDevice(op);
    auto systemDesc = ttcore::getCurrentScopeSystemDesc(op);
    auto chipIds = device.getChipIds();
    assert(chipIds.size() == 1);
    auto chipDesc = systemDesc.getChipDesc(chipIds[0]);

    // NOTE: All reads must be from remote locations in DMAReadOp
    // local->local transfers are lowered as nocAsyncWrites, which require
    // write barriers.
    auto srcNocAddr =
        buildNocAddress(rewriter, op.getLoc(), adaptor.getSrc(),
                        op.getSrcIndices(), chipDesc, op.getSrcMemorySpace());
    auto dstCBMapping = cbProducerConsumer->get(op.getDst());
    TT_assertv((dstCBMapping == ttir::ThreadCBOrientation::Producer ||
                dstCBMapping == ttir::ThreadCBOrientation::Default),
               "Expected dst cb of a read op to have a producer or default "
               "orientation, failing.");
    Value dstL1Addr = buildL1Address<ttkernel::GetWritePtrOp>(
        rewriter, op.getLoc(), adaptor.getDst(), op.getDstIndices());

    auto size = i32(rewriter, op->getLoc(), op.getSizeBytes());
    rewriter.create<ttkernel::NocAsyncReadOp>(op.getLoc(), srcNocAddr,
                                              dstL1Addr, size);

    // Add attribute marking whether the DMA wait is for a read or write
    // operation This will be used when loweing the wait ops because the current
    // DMA op will be replaced with a NullTx.
    auto dmaWaitOps = associatedDMAWaits->get(op);
    for (auto dmaWaitOp : dmaWaitOps) {
      rewriter.modifyOpInPlace(dmaWaitOp, [&]() {
        dmaWaitOp->setDiscardableAttr("ttkernel.lowering.associated_noc_read",
                                      rewriter.getUnitAttr());
      });
    }

    rewriter.replaceOpWithNewOp<ttir::NullTxOp>(op);

    return success();
  }

private:
  const ttir::AssociatedDMAWaits *associatedDMAWaits;
  const ttir::CBProducerConsumer *cbProducerConsumer;
};

class TTIRDMAWriteRewriter : public OpConversionPattern<ttir::DMAWriteOp> {
public:
  TTIRDMAWriteRewriter(TypeConverter &typeConverter, MLIRContext *context,
                       const ttir::AssociatedDMAWaits *associatedDMAWaits,
                       const ttir::CBProducerConsumer *cbProducerConsumer)
      : OpConversionPattern<ttir::DMAWriteOp>(typeConverter, context),
        associatedDMAWaits(associatedDMAWaits),
        cbProducerConsumer(cbProducerConsumer) {}

  LogicalResult
  matchAndRewrite(ttir::DMAWriteOp op, ttir::DMAWriteOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto device = ttcore::lookupDevice(op);
    auto systemDesc = ttcore::getCurrentScopeSystemDesc(op);
    auto chipIds = device.getChipIds();
    assert(chipIds.size() == 1);
    auto chipDesc = systemDesc.getChipDesc(chipIds[0]);

    if (op.isDstLocal()) {
      // Local to Local Datamovement & Multicast

      // Both src and dst are local, use the metal cb pointers to determine
      // addressing
      Value srcL1Start;
      auto srcCBMapping = cbProducerConsumer->get(op.getSrc());
      if (srcCBMapping == ttir::ThreadCBOrientation::Producer) {
        srcL1Start = rewriter.create<ttkernel::GetWritePtrOp>(op.getLoc(),
                                                              adaptor.getSrc());
      } else {
        srcL1Start = rewriter.create<ttkernel::GetReadPtrOp>(op.getLoc(),
                                                             adaptor.getSrc());
      }
      auto dstCBMapping = cbProducerConsumer->get(op.getDst());
      TT_assertv((dstCBMapping == ttir::ThreadCBOrientation::Producer ||
                  dstCBMapping == ttir::ThreadCBOrientation::ProducerConsumer ||
                  dstCBMapping == ttir::ThreadCBOrientation::Default),
                 "Expected dst cb of a write op to have a producer, "
                 "producer-consumer or default orientation, failing.");
      Value dstL1Start = rewriter.create<ttkernel::GetWritePtrOp>(
          op.getLoc(), adaptor.getDst());

      Value transferSize = i32(rewriter, op->getLoc(), op.getSizeBytes());
      if (op.isMcast()) {
        // Multicast lowering
        // Get virtual start coordinates from DMA op logical coordinates
        auto [virtY, virtX] = getVirtualCoordsFromLogicalCoords(
            rewriter, op.getLoc(), chipDesc, op.getMcastStartIndex());
        // Get the multicast end coordinates from the virtual start coordinates
        // and mcast shape
        auto [mcastEndY, mcastEndX] = getMcastEndCoords(
            rewriter, op.getLoc(), virtY, virtX, op.getMcastShape());
        auto numDestsIdx = rewriter.create<arith::MulIOp>(
            op.getLoc(), op.getMcastShape()[0], op.getMcastShape()[1]);
        auto numDests = rewriter.create<arith::IndexCastOp>(
            op.getLoc(), rewriter.getI32Type(), numDestsIdx);
        auto mcastAddr =
            rewriter.create<ttkernel::ExperimentalGetNocMulticastAddrOp>(
                op.getLoc(), virtX, virtY, mcastEndX, mcastEndY, dstL1Start,
                nullptr);
        if (adaptor.getSrc() == adaptor.getDst()) {
          // If src and dst refer to the same memref, we do not loopback mcast
          // Dests are one less because the sender core is not included
          rewriter.create<ttkernel::NocAsyncWriteMulticastOp>(
              op.getLoc(), srcL1Start, mcastAddr, transferSize, numDests,
              nullptr, nullptr, nullptr);
        } else {
          // If src != dst, we loopback mcast
          rewriter.create<ttkernel::NocAsyncWriteMulticastLoopbackSrcOp>(
              op.getLoc(), srcL1Start, mcastAddr, transferSize, numDests,
              nullptr, nullptr, nullptr);
        }
      } else {
        // Local L1 to Local L1 local data movement lowering
        // Get local coordinates using myY and myX ops
        auto myY = rewriter.create<ttir::CoreIndexOp>(
            op.getLoc(), rewriter.getIndexType(),
            rewriter.getI64IntegerAttr(0));
        auto myX = rewriter.create<ttir::CoreIndexOp>(
            op.getLoc(), rewriter.getIndexType(),
            rewriter.getI64IntegerAttr(1));
        // Convert local coordinates to virtual coordinates
        auto [virtY, virtX] = getVirtualCoordsFromLogicalCoords(
            rewriter, op.getLoc(), chipDesc, ValueRange{myY, myX});
        auto nocAddr = rewriter.create<ttkernel::GetNocAddrOp>(
            op.getLoc(), virtX, virtY, dstL1Start);
        rewriter.create<ttkernel::NocAsyncWriteOp>(op.getLoc(), srcL1Start,
                                                   nocAddr, transferSize);
      }
    } else if (op.isDstRemote()) {
      auto srcL1Addr = buildL1Address<ttkernel::GetReadPtrOp>(
          rewriter, op.getLoc(), adaptor.getSrc(), op.getSrcIndices());
      auto dstNocAddr =
          buildNocAddress(rewriter, op.getLoc(), adaptor.getDst(),
                          op.getDstIndices(), chipDesc, op.getDstMemorySpace());
      auto size = i32(rewriter, op->getLoc(), op.getSizeBytes());
      rewriter.create<ttkernel::NocAsyncWriteOp>(op.getLoc(), srcL1Addr,
                                                 dstNocAddr, size);
    }

    // Add attribute marking whether the DMA wait is for a read or write
    // operation This will be used when loweing the wait ops because the current
    // DMA op will be replaced with a NullTx.
    auto dmaWaitOps = associatedDMAWaits->get(op);
    for (auto dmaWaitOp : dmaWaitOps) {
      rewriter.modifyOpInPlace(dmaWaitOp, [&]() {
        dmaWaitOp->setDiscardableAttr("ttkernel.lowering.associated_noc_write",
                                      rewriter.getUnitAttr());
      });
    }

    rewriter.replaceOpWithNewOp<ttir::NullTxOp>(op);
    return success();
  }

private:
  const ttir::AssociatedDMAWaits *associatedDMAWaits;
  const ttir::CBProducerConsumer *cbProducerConsumer;
};
} // namespace

namespace {
class TTIRCoreIndexRewriter : public OpConversionPattern<ttir::CoreIndexOp> {
public:
  using OpConversionPattern<ttir::CoreIndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::CoreIndexOp op, ttir::CoreIndexOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto device = ttcore::lookupDevice(op);
    auto systemDesc = ttcore::getCurrentScopeSystemDesc(op);
    auto chipIds = device.getChipIds();
    assert(chipIds.size() == 1);
    auto chipDesc = systemDesc.getChipDesc(chipIds[0]);

    assert(op.getDim() == 0 ||
           op.getDim() == 1 &&
               "Expected core index dim to be in range 0-1, failing.");
    if (op.getDim()) {
      auto coreIndex = rewriter.create<ttkernel::MyXOp>(op.getLoc(), nullptr);
      auto normalizedCoreIndex = rewriter.create<arith::SubIOp>(
          op.getLoc(), coreIndex,
          index(rewriter, op->getLoc(),
                chipDesc.getCoordTranslationOffsets()[1]));
      rewriter.replaceOp(op, normalizedCoreIndex);
    } else {
      auto coreIndex = rewriter.create<ttkernel::MyYOp>(op.getLoc(), nullptr);
      auto normalizedCoreIndex = rewriter.create<arith::SubIOp>(
          op.getLoc(), coreIndex,
          index(rewriter, op->getLoc(),
                chipDesc.getCoordTranslationOffsets()[0]));
      rewriter.replaceOp(op, normalizedCoreIndex);
    }
    return success();
  }
};
} // namespace

namespace {
class TTIRDMAWaitRewriter : public OpConversionPattern<ttir::DMAWaitOp> {
public:
  using OpConversionPattern<ttir::DMAWaitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::DMAWaitOp op, ttir::DMAWaitOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto isRead =
        op->getDiscardableAttr("ttkernel.lowering.associated_noc_read");
    auto isWrite =
        op->getDiscardableAttr("ttkernel.lowering.associated_noc_write");
    assert(isRead || isWrite);

    if (isRead) {
      rewriter.create<ttkernel::NocAsyncReadBarrierOp>(op.getLoc());
    }

    if (isWrite) {
      rewriter.create<ttkernel::NocAsyncWriteBarrierOp>(op.getLoc());
    }

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {
class TTIRGetGlobalOperandRewriter
    : public OpConversionPattern<ttir::GetGlobalOperandOp> {
public:
  using OpConversionPattern<ttir::GetGlobalOperandOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::GetGlobalOperandOp op,
                  ttir::GetGlobalOperandOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    func::FuncOp entry = op->getParentOfType<func::FuncOp>();
    auto arg =
        rewriter.getAttr<ArgAttr>(ArgType::BufferAddress, op.getOperandIndex());
    size_t argIndex;
    rewriter.modifyOpInPlace(entry, [&]() {
      argIndex = ArgSpecAttr::appendCompileTimeArg(entry, arg);
    });

    rewriter.replaceOpWithNewOp<ttkernel::GetCompileArgValOp>(
        op, rewriter.getI32Type(), argIndex);

    return success();
  }
};
} // namespace

namespace {
class TTIRNullTxRewriter : public OpConversionPattern<ttir::NullTxOp> {
public:
  using OpConversionPattern<ttir::NullTxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::NullTxOp op, ttir::NullTxOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, rewriter.getIndexType(),
                                                   rewriter.getIndexAttr(0));
    return success();
  }
};
} // namespace

namespace {
class MemRefCollapseRewriter
    : public OpConversionPattern<memref::CollapseShapeOp> {
public:
  using OpConversionPattern<memref::CollapseShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CollapseShapeOp op,
                  memref::CollapseShapeOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (ttcore::getMemorySpace(op.getSrc()) ==
        ttcore::MemorySpace::RegisterDst) {
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }
    rewriter.replaceOpWithNewOp<CBReinterpretShapeOp>(
        op, getTypeConverter()->convertType(op.getResult().getType()),
        adaptor.getSrc());
    return success();
  }
};
} // namespace

namespace {
class TTIRKernelFunctionArgsRewriter
    : public OpConversionPattern<func::FuncOp> {
public:
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  static ThreadType getTTKernelThreadType(func::FuncOp op) {
    ttir::ThreadAttr threadAttr =
        op->getAttrOfType<ttir::ThreadAttr>(ttir::ThreadAttr::name);
    switch (threadAttr.getThreadType()) {
    case ttir::ThreadType::Compute: {
      return ThreadType::Compute;
    }
    case ttir::ThreadType::Datamovement: {
      return ThreadType::Noc;
    }
    }
  }

  static void convertFunctionAttrs(Builder &builder, func::FuncOp op,
                                   ArrayRef<ArgAttr> rtArgs,
                                   ArrayRef<ArgAttr> ctArgs) {

    // Get the TTKernel thread type and replace TTIR thread type with TTKernel
    // thread type
    ThreadType threadType = getTTKernelThreadType(op);
    op->removeAttr(ttir::ThreadAttr::name);
    op->setAttr(ThreadTypeAttr::name,
                builder.getAttr<ThreadTypeAttr>(threadType));
    ArgSpecAttr::setArgSpec(op, builder.getAttr<ArgSpecAttr>(rtArgs, ctArgs));
  }

  LogicalResult
  matchAndRewrite(func::FuncOp op, func::FuncOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!op->hasAttr(ttir::ThreadAttr::name) ||
        (op.getFunctionType().getNumInputs() == 0)) {
      return failure();
    }

    Block *block = &op.getCallableRegion()->front();
    auto blockArgs = block->getArguments();
    assert(!blockArgs.empty());

    SmallVector<ArgAttr> rtArgSpecVector;
    SmallVector<ArgAttr> ctArgSpecVector;
    size_t currentSemaphoreIndex = 0;
    TypeConverter::SignatureConversion signatureConverter(op.getNumArguments());
    OpBuilder::InsertionGuard funcInsertionGuard(rewriter);
    rewriter.setInsertionPointToStart(block);
    for (auto arg : blockArgs) {
      Type argType = getTypeConverter()->convertType(arg.getType());
      if (mlir::isa<CBType>(argType)) {
        auto cb = rewriter.create<GetCompileArgValOp>(
            op.getLoc(), getTypeConverter()->convertType(arg.getType()),
            rewriter.getI32IntegerAttr(arg.getArgNumber()));
        signatureConverter.remapInput(arg.getArgNumber(), {cb});
        ctArgSpecVector.push_back(
            rewriter.getAttr<ArgAttr>(ArgType::CBPort, arg.getArgNumber()));
      } else if (mlir::isa<SemaphoreType>(argType)) {
        if (getTTKernelThreadType(op) != ThreadType::Noc) {
          continue;
        }
        size_t ctArgIndex = ctArgSpecVector.size();
        auto semaphoreIndex = rewriter.create<GetCompileArgValOp>(
            op.getLoc(), rewriter.getI32Type(),
            rewriter.getI32IntegerAttr(ctArgIndex));
        auto semaphore =
            rewriter.create<GetSemaphoreOp>(op.getLoc(), semaphoreIndex);
        signatureConverter.remapInput(arg.getArgNumber(),
                                      semaphore.getResult());
        ctArgSpecVector.push_back(rewriter.getAttr<ArgAttr>(
            ArgType::Semaphore, currentSemaphoreIndex++));
      } else {
        llvm_unreachable("unexpected block argument type");
      }
    }

    rewriter.applySignatureConversion(block, signatureConverter,
                                      getTypeConverter());
    rewriter.modifyOpInPlace(op, [&]() {
      op.setType(rewriter.getFunctionType(TypeRange(), TypeRange()));
      convertFunctionAttrs(rewriter, op, rtArgSpecVector, ctArgSpecVector);
    });
    return success();
  }
};
} // namespace

namespace {
template <typename ConcreteOp>
class TTIRSemaphoreUpdateRewriter : public OpConversionPattern<ConcreteOp> {
public:
  using OpConversionPattern<ConcreteOp>::OpConversionPattern;

  static_assert(std::is_same_v<ConcreteOp, ttir::SemaphoreSetOp> ||
                    std::is_same_v<ConcreteOp, ttir::SemaphoreIncOp>,
                "Expected SemaphoreSet or SemaphoreInc op passed to "
                "TTIRSemaphoreUpdateRewriter.");

  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto device = ttcore::lookupDevice(op);
    auto systemDesc = ttcore::getCurrentScopeSystemDesc(op);
    auto chipIds = device.getChipIds();
    assert(chipIds.size() == 1);
    auto chipDesc = systemDesc.getChipDesc(chipIds[0]);

    Value value = op.getValue();
    Value semaphoreAddr = adaptor.getSemaphore();

    if (op.getDstCoreIndex().empty()) {
      assert(!mlir::isa<ttir::SemaphoreIncOp>(op) &&
             "ttir.semaphore_inc to local core is illegal.");

      // Local semaphore set
      auto semaphorePtr =
          rewriter.create<ttkernel::CastToL1PtrOp>(op.getLoc(), semaphoreAddr);

      rewriter.replaceOpWithNewOp<ttkernel::NocSemaphoreSetOp>(op, semaphorePtr,
                                                               value);
    } else if (op.getMcastShape().empty()) {
      assert(!mlir::isa<ttir::SemaphoreSetOp>(op) &&
             "ttir.semaphore_set to single remote core is illegal.");
      auto [virtY, virtX] = getVirtualCoordsFromLogicalCoords(
          rewriter, op.getLoc(), chipDesc, op.getDstCoreIndex());
      auto nocAddr = rewriter.create<ttkernel::GetNocAddrOp>(
          op.getLoc(), virtX, virtY, semaphoreAddr);
      rewriter.replaceOpWithNewOp<ttkernel::NocSemaphoreIncOp>(op, nocAddr,
                                                               value, nullptr);
    } else {
      assert(!mlir::isa<ttir::SemaphoreIncOp>(op) &&
             "ttir.semaphore_inc multicast is illegal.");

      auto [virtY, virtX] = getVirtualCoordsFromLogicalCoords(
          rewriter, op.getLoc(), chipDesc, op.getDstCoreIndex());
      auto [mcastEndY, mcastEndX] = getMcastEndCoords(
          rewriter, op.getLoc(), virtY, virtX, op.getMcastShape());
      Value numDestsIdx = rewriter.create<arith::MulIOp>(
          op.getLoc(), op.getMcastShape()[0], op.getMcastShape()[1]);
      Value numDests = rewriter.create<arith::IndexCastOp>(
          op.getLoc(), rewriter.getI32Type(), numDestsIdx);
      auto mcastAddr =
          rewriter.create<ttkernel::ExperimentalGetNocMulticastAddrOp>(
              op.getLoc(), virtX, virtY, mcastEndX, mcastEndY, semaphoreAddr,
              nullptr);

      auto semaphorePtr =
          rewriter.create<ttkernel::CastToL1PtrOp>(op.getLoc(), semaphoreAddr);
      rewriter.create<ttkernel::NocSemaphoreSetOp>(op.getLoc(), semaphorePtr,
                                                   value);
      rewriter.replaceOpWithNewOp<ttkernel::NocSemaphoreSetMulticastOp>(
          op, semaphoreAddr, mcastAddr, numDests, nullptr, nullptr);
    }

    return success();
  }
};
} // namespace

namespace {
class TTIRSemaphoreWaitRewriter
    : public OpConversionPattern<ttir::SemaphoreWaitOp> {
public:
  using OpConversionPattern<ttir::SemaphoreWaitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SemaphoreWaitOp op,
                  ttir::SemaphoreWaitOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    Value semaphoreAddr = adaptor.getSemaphore();
    auto semaphorePtr =
        rewriter.create<ttkernel::CastToL1PtrOp>(op.getLoc(), semaphoreAddr);

    rewriter.replaceOpWithNewOp<ttkernel::NocSemaphoreWaitOp>(op, semaphorePtr,
                                                              op.getValue());
    if (op.getResetValue()) {
      rewriter.create<ttkernel::NocSemaphoreSetOp>(op.getLoc(), semaphorePtr,
                                                   op.getResetValue());
    }

    return success();
  }
};
} // namespace

} // namespace mlir::tt::ttkernel

namespace mlir::tt {

void populateTTIRToTTKernelPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns, TypeConverter &typeConverter,
    const ttir::AssociatedDMAWaits &associatedDMAWaits,
    const ttir::CBProducerConsumer &cbProducerConsumer) {
  // clang-format off
  patterns.add<ttkernel::TTIRKernelFunctionArgsRewriter,

               // Elementwise FPU.
               ttkernel::TTIRFPUOpsRewriter<ttir::TileAddOp>,
               ttkernel::TTIRFPUOpsRewriter<ttir::TileMatmulOp>,
               ttkernel::TTIRFPUOpsRewriter<ttir::TileMatmulBlockOp>,
               ttkernel::TTIRFPUOpsRewriter<ttir::TileMulOp>,
               ttkernel::TTIRFPUOpsRewriter<ttir::TileSubOp>,

               // Elementwise SFPU Unary.
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileAbsOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileCeilOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileCosOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileExpOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileFloorOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileGeluOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileLogOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileLogicalNotOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileNegativeOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileRecipOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileRsqrtOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileSqrtOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileSigmoidOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileSinOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileTanOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileEqzOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileNezOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileGtzOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileGezOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileLtzOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileLezOp>,

               // Elementwise SFPU Binary.
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileSubBinaryOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileDivOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TileMaximumOp>,
               ttkernel::TTIRSFPUOpsRewriter<ttir::TilePowOp>,

               ttkernel::TTIRTilizeUntilizeRewriter,
               ttkernel::TTIRTileTransposeRewriter,
               ttkernel::TTIRTypecastRewriter,
               ttkernel::AcquireDstRewriter,
               ttkernel::MemrefLoadRewriter,
               ttkernel::MemrefStoreRewriter,
               ttkernel::TTIRAwaitYieldRewriter<ttir::AwaitOp>,
               ttkernel::TTIRAwaitYieldRewriter<ttir::YieldOp>,
               ttkernel::TTIRDMAWaitRewriter,
               ttkernel::TTIRCoreIndexRewriter,
               ttkernel::TTIRGetGlobalOperandRewriter,
               ttkernel::TTIRNullTxRewriter,
               ttkernel::MemRefCollapseRewriter,
               ttkernel::TTIRSemaphoreUpdateRewriter<ttir::SemaphoreSetOp>,
               ttkernel::TTIRSemaphoreUpdateRewriter<ttir::SemaphoreIncOp>,
               ttkernel::TTIRSemaphoreWaitRewriter>(typeConverter, ctx);

  patterns.add<ttkernel::TTIRDMAReadRewriter>(typeConverter, ctx, &associatedDMAWaits, &cbProducerConsumer);
  patterns.add<ttkernel::TTIRDMAWriteRewriter>(typeConverter, ctx, &associatedDMAWaits, &cbProducerConsumer);
  // clang-format on
}

} // namespace mlir::tt
