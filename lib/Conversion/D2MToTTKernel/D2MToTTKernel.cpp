// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/D2MToTTKernel/D2MToTTKernel.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Analysis/CBProducerConsumer.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Utils.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
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
  Value virtY = rewriter.create<ttkernel::ConvertLogicalYToTranslatedOp>(
      dstCoreIndex[0].getLoc(), dstCoreIndex[0].getType(), dstCoreIndex[0]);
  Value virtX = rewriter.create<ttkernel::ConvertLogicalXToTranslatedOp>(
      dstCoreIndex[1].getLoc(), dstCoreIndex[1].getType(), dstCoreIndex[1]);
  return {virtY, virtX};
}

static std::pair<Value, Value> getMcastEndCoords(PatternRewriter &rewriter,
                                                 Location loc,
                                                 const Value &nocStartY,
                                                 const Value &nocStartX,
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

static Value getCB(ConversionPatternRewriter &rewriter, Value cb) {
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

// Get DST index from where a compute op result is stored.
// Handles both affine.store and memref.store.
// When a value has multiple stores to different DST memrefs (due to value
// reuse across DST regions after LICM), we prefer stores in the same block
// as the defining op. This ensures we get the DST index for the correct
// allocation context.
static Value getDstIdxFromResult(Value d2mOpResult) {
  memref::StoreOp storeOp;
  for (Operation *op : d2mOpResult.getUsers()) {
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

// Check if an operand comes from DST.
static bool operandFromDst(Value operand) {
  if (auto affineLoad = operand.getDefiningOp<affine::AffineLoadOp>()) {
    return ttcore::getMemorySpace(affineLoad.getMemRef()) ==
           ttcore::MemorySpace::RegisterDst;
  }
  if (auto memrefLoad = operand.getDefiningOp<memref::LoadOp>()) {
    return ttcore::getMemorySpace(memrefLoad.getMemRef()) ==
           ttcore::MemorySpace::RegisterDst;
  }
  return false;
}

static void setInsertionPointAfterOperands(OpBuilder &rewriter,
                                           llvm::ArrayRef<Value> operands,
                                           bool allowHoisting) {
  Operation *latestDefOp = nullptr;
  Block *currentBlock = rewriter.getInsertionBlock();

  for (Value operand : operands) {
    Operation *definingOp = operand.getDefiningOp();
    if (!definingOp) {
      continue;
    }

    bool inCurrentBlock = (definingOp->getBlock() == currentBlock);

    if (!allowHoisting && !inCurrentBlock) {
      continue;
    }

    if (!latestDefOp) {
      latestDefOp = definingOp;
    } else if (latestDefOp->getBlock() == definingOp->getBlock()) {
      if (!definingOp->isBeforeInBlock(latestDefOp)) {
        latestDefOp = definingOp;
      }
    } else if (inCurrentBlock) {
      latestDefOp = definingOp;
    }
  }

  if (!latestDefOp) {
    return;
  }

  if (latestDefOp->getBlock() == currentBlock) {
    auto currentInsertionPoint = rewriter.getInsertionPoint();
    if (allowHoisting || currentInsertionPoint->isBeforeInBlock(latestDefOp)) {
      rewriter.setInsertionPointAfter(latestDefOp);
    }
  } else {
    // Hoist to outer block (only reachable when allowHoisting=true)
    rewriter.setInsertionPointAfter(latestDefOp);
  }
}

static void setInsertionPointToFuncStart(OpBuilder &rewriter,
                                         func::FuncOp func) {
  Block &entry = func.getBody().front();
  Operation *firstLoop = nullptr;

  for (Operation &entryOp : entry) {
    if (isa<scf::ForOp>(entryOp)) {
      firstLoop = &entryOp;
      break;
    }
  }

  if (firstLoop) {
    rewriter.setInsertionPoint(firstLoop);
  } else {
    rewriter.setInsertionPointToEnd(&entry);
  }
}

static bool hasMatmulInit(func::FuncOp func) {
  return llvm::any_of(func.getBody().front(), [](Operation &op) {
    return isa<ttkernel::MatmulInitOp>(op);
  });
}

} // namespace

namespace {
template <typename ConcreteOp>
class PassthroughRewriter : public OpConversionPattern<ConcreteOp> {
public:
  using OpConversionPattern<ConcreteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  };
};
} // namespace

namespace {
class MemRefSubviewRewriter : public OpConversionPattern<memref::SubViewOp> {
public:
  using OpConversionPattern<memref::SubViewOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::SubViewOp op,
                  typename memref::SubViewOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // We have blocked this input. We need to get the indicies for the first
    // tile in the subview.
    SmallVector<Value> indices = {index(rewriter, op.getLoc(), 0),
                                  index(rewriter, op.getLoc(), 0)};
    SmallVector<Value> sourceIndices;

    // TODO(#4717): This call alone should be enough to get the tile indices,
    // but currently it returns block index instead. Once fixed, we can remove
    // all the other calculations below.
    affine::resolveIndicesIntoOpWithOffsetsAndStrides(
        rewriter, op.getLoc(), op.getMixedOffsets(), op.getMixedStrides(),
        op.getDroppedDims(), indices, sourceIndices);

    auto resultTy = mlir::cast<MemRefType>(op.getResult().getType());
    Value rtIdx = index(rewriter, op.getLoc(), resultTy.getShape()[0]);
    Value ktIdx = index(rewriter, op.getLoc(), resultTy.getShape()[1]);
    Value tilesPerBlock =
        rewriter.create<arith::MulIOp>(op.getLoc(), rtIdx, ktIdx);

    // Convert the resolved source row offset to a block-row index.
    Value rowBlockIdx =
        rewriter.create<arith::DivSIOp>(op.getLoc(), sourceIndices[0], rtIdx);
    Value rowBase =
        rewriter.create<arith::MulIOp>(op.getLoc(), rowBlockIdx, tilesPerBlock);
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, rowBase, sourceIndices[1]);
    return success();
  };
};
} // namespace

namespace {
class AcquireDstRewriter : public OpConversionPattern<d2m::AcquireDstOp> {
public:
  using OpConversionPattern<d2m::AcquireDstOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::AcquireDstOp op, d2m::AcquireDstOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.create<ttkernel::TileRegsAcquireOp>(op.getLoc());
    // Dst is an implicit resource in TTKernel, so we can just erase it.
    rewriter.eraseOp(op);
    return success();
  };
};
} // namespace

namespace {
class D2MSetL1AccumulateRewriter
    : public OpConversionPattern<d2m::SetL1AccumulateOp> {
public:
  using OpConversionPattern<d2m::SetL1AccumulateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::SetL1AccumulateOp op,
                  d2m::SetL1AccumulateOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttkernel::PackReconfigL1AccOp>(
        op, adaptor.getEnable());

    // Insert an unconditional disable before return.
    // Packer config state persists across program launches.
    auto func = op->getParentOfType<func::FuncOp>();
    if (func) {
      func.walk([&](func::ReturnOp returnOp) {
        OpBuilder builder(returnOp);
        Value zero = builder.create<arith::ConstantOp>(
            returnOp.getLoc(), builder.getI32Type(),
            builder.getI32IntegerAttr(0));
        builder.create<ttkernel::PackReconfigL1AccOp>(returnOp.getLoc(), zero);
      });
    }

    return success();
  };
};
} // namespace

// Helper to compute linear index from multi-dimensional indices.
// For shape <d0, d1, d2, ...> and indices [i0, i1, i2, ...]:
// linear = i0 * (d1*d2*...) + i1 * (d2*d3*...) + ... + i_last.
static Value computeLinearIndex(Location loc, ArrayRef<int64_t> shape,
                                ValueRange indices,
                                ConversionPatternRewriter &rewriter) {
  if (indices.size() == 1) {
    return indices.front();
  }

  Value linearIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  for (size_t i = 0; i < indices.size(); ++i) {
    int64_t stride = 1;
    for (size_t j = i + 1; j < shape.size(); ++j) {
      stride *= shape[j];
    }

    Value contribution = indices[i];
    if (stride != 1) {
      auto strideVal = rewriter.create<arith::ConstantIndexOp>(loc, stride);
      contribution = rewriter.create<arith::MulIOp>(loc, indices[i], strideVal);
    }
    linearIdx = rewriter.create<arith::AddIOp>(loc, linearIdx, contribution);
  }
  return linearIdx;
}

namespace {
class MemrefLoadRewriter : public OpConversionPattern<memref::LoadOp> {
public:
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, memref::LoadOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // For DST loads, the indices give us the DST slot.
    // For multi-index accesses (e.g., memref<4x1x1x...>), compute linear index.
    Value linearIdx =
        computeLinearIndex(op.getLoc(), op.getMemRefType().getShape(),
                           adaptor.getIndices(), rewriter);
    rewriter.replaceOp(op, linearIdx);
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
    auto cb = rewriter.getRemappedValue(load.getMemref());
    auto cbIndex = adaptor.getValue();
    auto dstIndex =
        computeLinearIndex(store.getLoc(), store.getMemRefType().getShape(),
                           adaptor.getIndices(), rewriter);

    auto inCB = getInCB(rewriter, store);
    auto outCB = getOutCB(rewriter, store);

    auto insertionPoint = rewriter.getInsertionPoint();
    rewriter.setInsertionPointToStart(rewriter.getInsertionBlock());
    setInsertionPointAfterOperands(rewriter, {inCB, outCB},
                                   /*allowHoisting*/ true);
    rewriter.create<ttkernel::InitSFPUOp>(store.getLoc(), inCB, outCB);
    rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);

    rewriter.create<ttkernel::CopyTileInitOp>(store.getLoc(), cb);
    rewriter.replaceOpWithNewOp<ttkernel::CopyTileOp>(store, cb, cbIndex,
                                                      dstIndex);
    return success();
  }

  static LogicalResult lowerPackTile(memref::StoreOp store,
                                     memref::StoreOpAdaptor adaptor,
                                     ConversionPatternRewriter &rewriter) {
    auto dst = adaptor.getValue();
    auto cb = adaptor.getMemref();
    auto storeIdx =
        computeLinearIndex(store.getLoc(), store.getMemRefType().getShape(),
                           adaptor.getIndices(), rewriter);
    rewriter.replaceOpWithNewOp<ttkernel::PackTileOp>(
        store, dst, cb, storeIdx, rewriter.getBoolAttr(true));
    return success();
  }

  LogicalResult
  matchAndRewrite(memref::StoreOp op, memref::StoreOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // Look for the load operation, potentially through a dst reinterpret cast
    Operation *definingOp = op.getValue().getDefiningOp();
    auto load = mlir::dyn_cast<memref::LoadOp>(definingOp);

    // If not a direct load, check if it's a cast wrapping a load
    if (!load && definingOp) {
      if (auto dstCast =
              mlir::dyn_cast<d2m::DstReinterpretCastOp>(definingOp)) {
        // Look through the dst reinterpret cast to find the actual load
        load = mlir::dyn_cast_or_null<memref::LoadOp>(
            dstCast.getInput().getDefiningOp());
      }
    }

    bool storeToDst = ttcore::getMemorySpace(op.getMemRef()) ==
                      ttcore::MemorySpace::RegisterDst;

    // Check if the load is from L1 (CB), not from DST.
    bool loadFromL1 = load && ttcore::getMemorySpace(load.getMemRef()) ==
                                  ttcore::MemorySpace::DeviceL1;

    if (loadFromL1 && storeToDst) {
      // If we are coming from a load from L1, then we are a copy tile. Pattern:
      //    %0 = memref.load %arg0, %c0 : memref<1x!tt.tile, l1>
      //    tt.store %0, %arg1, %c0 : memref<1x!tt.tile, dst>
      // OR with dst reinterpret cast:
      //    %0 = memref.load %arg0, %c0 : memref<1x!tt.tile, l1>
      //    %1 = d2m.dst_reinterpret_cast %0 : type1 -> type2
      //    tt.store %1, %arg1, %c0 : memref<1x!tt.tile, dst>
      return lowerCopyTile(load, op, adaptor, rewriter);
    }

    if (storeToDst) {
      // Otherwise we're storing the result of an op:
      //    %0 = d2m.tile_sigmoid %arg0
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
  // FPU.
  std::pair<d2m::TileBcastOp,       std::pair<ttkernel::UnaryBcastInitOp,          ttkernel::UnaryBcastTileOp>>,
  std::pair<d2m::TileMatmulOp,      std::pair<ttkernel::MatmulInitOp,              ttkernel::MatmulTilesOp>>,
  std::pair<d2m::TileMatmulBlockOp, std::pair<ttkernel::MatmulBlockInitOp,         ttkernel::ExperimentalMatmulBlockOp>>,

  // Reductions FPU
  std::pair<d2m::TileReduceSumOp,   std::pair<ttkernel::ComputeKernelHWStartupOp,  ttkernel::ReduceTileOp>>,
  std::pair<d2m::TileReduceMaxOp,   std::pair<ttkernel::ComputeKernelHWStartupOp,  ttkernel::ReduceTileOp>>,

  // Elementwise SFPU Unary.
  std::pair<d2m::TileAbsOp,         std::pair<ttkernel::AbsTileInitOp,             ttkernel::AbsTileOp>>,
  std::pair<d2m::TileBitwiseNotOp,  std::pair<ttkernel::BitwiseNotTileInitOp,      ttkernel::BitwiseNotTileOp>>,
  std::pair<d2m::TileCeilOp,        std::pair<ttkernel::RoundingTileInitOp,        ttkernel::CeilTileOp>>,
  std::pair<d2m::TileClampScalarOp, std::pair<ttkernel::ClampScalarTileInitOp,     ttkernel::ClampScalarTileOp>>,
  std::pair<d2m::TileCosOp,         std::pair<ttkernel::CosTileInitOp,             ttkernel::CosTileOp>>,
  std::pair<d2m::TileErfOp,         std::pair<ttkernel::ErfTileInitOp,             ttkernel::ErfTileOp>>,
  std::pair<d2m::TileErfcOp,        std::pair<ttkernel::ErfcTileInitOp,            ttkernel::ErfcTileOp>>,
  std::pair<d2m::TileExpOp,         std::pair<ttkernel::ExpTileInitOp,             ttkernel::ExpTileOp>>,
  std::pair<d2m::TileFloorOp,       std::pair<ttkernel::RoundingTileInitOp,        ttkernel::FloorTileOp>>,
  std::pair<d2m::TileGeluOp,        std::pair<ttkernel::GeluTileInitOp,            ttkernel::GeluTileOp>>,
  std::pair<d2m::TileHardsigmoidOp, std::pair<ttkernel::HardsigmoidTileInitOp,     ttkernel::HardsigmoidTileOp>>,
  std::pair<d2m::TileLogOp,         std::pair<ttkernel::LogTileInitOp,             ttkernel::LogTileOp>>,
  std::pair<d2m::TileLogicalNotOp,  std::pair<ttkernel::LogicalNotUnaryTileInitOp, ttkernel::LogicalNotUnaryTileOp>>,
  std::pair<d2m::TileNegativeOp,    std::pair<ttkernel::NegativeTileInitOp,        ttkernel::NegativeTileOp>>,
  std::pair<d2m::TileRecipOp,       std::pair<ttkernel::RecipTileInitOp,           ttkernel::RecipTileOp>>,
  std::pair<d2m::TileReluOp,        std::pair<ttkernel::ReluTileInitOp,            ttkernel::ReluTileOp>>,
  std::pair<d2m::TileRsqrtOp,       std::pair<ttkernel::RsqrtTileInitOp,           ttkernel::RsqrtTileOp>>,
  std::pair<d2m::TileSignOp,        std::pair<ttkernel::SignTileInitOp,            ttkernel::SignTileOp>>,
  std::pair<d2m::TileSqrtOp,        std::pair<ttkernel::SqrtTileInitOp,            ttkernel::SqrtTileOp>>,
  std::pair<d2m::TileSigmoidOp,     std::pair<ttkernel::SigmoidTileInitOp,         ttkernel::SigmoidTileOp>>,
  std::pair<d2m::TileSiluOp,        std::pair<ttkernel::SiluTileInitOp,            ttkernel::SiluTileOp>>,
  std::pair<d2m::TileSinOp,         std::pair<ttkernel::SinTileInitOp,             ttkernel::SinTileOp>>,
  std::pair<d2m::TileTanOp,         std::pair<ttkernel::TanTileInitOp,             ttkernel::TanTileOp>>,
  std::pair<d2m::TileTanhOp,        std::pair<ttkernel::TanhTileInitOp,            ttkernel::TanhTileOp>>,
  std::pair<d2m::TileEqzOp,         std::pair<ttkernel::EqzTileInitOp,             ttkernel::EqzTileOp>>,
  std::pair<d2m::TileNezOp,         std::pair<ttkernel::NezTileInitOp,             ttkernel::NezTileOp>>,
  std::pair<d2m::TileGtzOp,         std::pair<ttkernel::GtzTileInitOp,             ttkernel::GtzTileOp>>,
  std::pair<d2m::TileGezOp,         std::pair<ttkernel::GezTileInitOp,             ttkernel::GezTileOp>>,
  std::pair<d2m::TileLtzOp,         std::pair<ttkernel::LtzTileInitOp,             ttkernel::LtzTileOp>>,
  std::pair<d2m::TileLezOp,         std::pair<ttkernel::LezTileInitOp,             ttkernel::LezTileOp>>,
  std::pair<d2m::TileTypecastOp,    std::pair<ttkernel::TypecastTileInitOp,        ttkernel::TypecastTileOp>>,

  // Elementwise FPU Binary with SFPU fallback.
  std::pair<d2m::TileAddOp,         std::pair<ttkernel::AddBinaryTilesInitOp,      ttkernel::AddBinaryTilesOp>>,
  std::pair<d2m::TileSubOp,         std::pair<ttkernel::SubBinaryTilesInitOp,      ttkernel::SubBinaryTilesOp>>,
  std::pair<d2m::TileMulOp,         std::pair<ttkernel::MulBinaryTilesInitOp,      ttkernel::MulBinaryTilesOp>>,

  // Elementwise SFPU Binary.
  std::pair<d2m::TileBitwiseAndOp,  std::pair<ttkernel::BinaryBitwiseTileInitOp,   ttkernel::BitwiseAndBinaryTilesOp>>,
  std::pair<d2m::TileBitwiseOrOp,   std::pair<ttkernel::BinaryBitwiseTileInitOp,   ttkernel::BitwiseOrBinaryTilesOp>>,
  std::pair<d2m::TileBitwiseXorOp,  std::pair<ttkernel::BinaryBitwiseTileInitOp,   ttkernel::BitwiseXorBinaryTilesOp>>,
  std::pair<d2m::TileDivOp,         std::pair<ttkernel::DivBinaryTilesInitOp,      ttkernel::DivBinaryTilesOp>>,
  std::pair<d2m::TileMaximumOp,     std::pair<ttkernel::BinaryMaxTileInitOp,       ttkernel::BinaryMaxTileOp>>,
  std::pair<d2m::TileMinimumOp,     std::pair<ttkernel::BinaryMinTileInitOp,       ttkernel::BinaryMinTileOp>>,
  std::pair<d2m::TilePowOp,         std::pair<ttkernel::PowBinaryTilesInitOp,      ttkernel::PowBinaryTilesOp>>,

  // Elementwise SFPU Ternary.
  std::pair<d2m::TileWhereOp,       std::pair<ttkernel::WhereTileInitOp,           ttkernel::WhereTileOp>>
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
class D2MFPUOpsRewriter : public OpConversionPattern<ConcreteOp> {
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
      setInsertionPointAfterOperands(rewriter, {cbA, cbB, outCB},
                                     /*allowHoisting*/ true);
      rewriter.create<ttkernel::BinaryOpInitCommonOp>(op->getLoc(), cbA, cbB,
                                                      outCB);
      rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);
    } else {
      static_assert(arity == 3 && !ttmlir::utils::always_false<ConcreteOp>(),
                    "FPUOp must be unary, binary or ternary");
      assert(op->getNumOperands() == 3u);
    }

    if constexpr (std::is_same_v<ConcreteOp, d2m::TileMatmulOp>) {
      auto insertionPoint = rewriter.getInsertionPoint();
      auto cbA = getCB(rewriter, op.getA());
      auto cbB = getCB(rewriter, op.getB());
      auto outCB = getOutCB(rewriter, op);

      // Must have only 1 MatmulInit op per kernel, so we always insert at
      // beginning of the func, and only if no MatmulInit already exists.
      if (auto func = op->template getParentOfType<func::FuncOp>();
          !hasMatmulInit(func)) {
        setInsertionPointToFuncStart(rewriter, func);
        auto transpose = i32(rewriter, op->getLoc(), 0);
        rewriter.create<ttkernel::MatmulInitOp>(op->getLoc(), cbA, cbB, outCB,
                                                transpose);
      }

      auto transpose = i32(rewriter, op->getLoc(), 0);
      rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);
      rewriter.create<ttkernel::MatmulInitShortOp>(op->getLoc(), cbA, cbB,
                                                   transpose);
      rewriter.create<ttkernel::MatmulTilesOp>(op->getLoc(), cbA, cbB,
                                               adaptor.getA(), adaptor.getB(),
                                               adaptor.getC());
    } else if constexpr (std::is_same_v<ConcreteOp, d2m::TileMatmulBlockOp>) {
      auto insertionPoint = rewriter.getInsertionPoint();
      auto cbA = getCB(rewriter, op.getA());
      auto cbB = getCB(rewriter, op.getB());
      auto outCB = getCB(rewriter, op.getOutput());
      setInsertionPointAfterOperands(rewriter, {cbA, cbB, outCB},
                                     /*allowHoisting*/ true);

      // destIndex is always 0 because we call an experimental LLK that fills up
      // the entire dest in a loop.
      Value destIndex = index(rewriter, op->getLoc(), 0);

      auto typeA = llvm::cast<MemRefType>(op.getA().getType());
      auto typeB = llvm::cast<MemRefType>(op.getB().getType());
      auto rt_i32 = i32(rewriter, op->getLoc(), typeA.getShape()[0]);
      auto kt_i32 = i32(rewriter, op->getLoc(), typeA.getShape()[1]);
      auto ct_i32 = i32(rewriter, op->getLoc(), typeB.getShape()[1]);

      auto getNumColumns = [](Value view) {
        if (auto castOp =
                dyn_cast_or_null<memref::CastOp>(view.getDefiningOp())) {
          view = castOp.getSource();
        } else if (auto svOp = dyn_cast_or_null<memref::SubViewOp>(
                       view.getDefiningOp())) {
          view = svOp.getSource();
        }
        auto srcTy = cast<MemRefType>(view.getType());
        return srcTy.getShape()[1];
      };
      auto nt_i32 = i32(rewriter, op->getLoc(), getNumColumns(op.getB()));

      auto transpose = i32(rewriter, op->getLoc(), 0);

      rewriter.create<ttkernel::MatmulBlockInitOp>(
          op->getLoc(), cbA, cbB, outCB, transpose, ct_i32, rt_i32, kt_i32);
      rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);
      rewriter.create<ttkernel::MatmulBlockInitShortOp>(
          op->getLoc(), cbA, cbB, transpose, ct_i32, rt_i32, kt_i32);

      // Get the tile index for each input in the global memref. This is done by
      // resolving tile (0,0) from the subview, representing a block, into the
      // address space of the source memref.
      Value aTileIndex = adaptor.getA();
      Value bTileIndex = adaptor.getB();

      // If the input didn't come from a subview, we'll expect the CB directly
      // which implicitly comes from an unrealized conversion cast.  This is a
      // special case where we're reading from offset 0.
      if (mlir::isa_and_nonnull<UnrealizedConversionCastOp>(
              aTileIndex.getDefiningOp())) {
        aTileIndex = index(rewriter, op.getLoc(), 0);
      }
      if (mlir::isa_and_nonnull<UnrealizedConversionCastOp>(
              bTileIndex.getDefiningOp())) {
        bTileIndex = index(rewriter, op.getLoc(), 0);
      }

      rewriter.create<ttkernel::ExperimentalMatmulBlockOp>(
          op->getLoc(), cbA, cbB, aTileIndex, bTileIndex, destIndex, transpose,
          ct_i32, rt_i32, kt_i32, nt_i32);
    } else if constexpr (std::is_same_v<ConcreteOp, d2m::TileReduceSumOp> ||
                         std::is_same_v<ConcreteOp, d2m::TileReduceMaxOp>) {
      ttkernel::ReduceType reduce_type;
      d2m::ReduceDim reduce_dim = op.getReduceDim();
      if constexpr (std::is_same_v<ConcreteOp, d2m::TileReduceSumOp>) {
        reduce_type = ttkernel::ReduceType::Sum;
      } else {
        reduce_type = ttkernel::ReduceType::Max;
      }
      ttkernel::ReduceDim kernel_reduce_dim;
      switch (reduce_dim) {
      case d2m::ReduceDim::C:
        kernel_reduce_dim = ttkernel::ReduceDim::Col;
        break;
      case d2m::ReduceDim::R:
        kernel_reduce_dim = ttkernel::ReduceDim::Row;
        break;
      case d2m::ReduceDim::RC:
        kernel_reduce_dim = ttkernel::ReduceDim::Scalar;
        break;
      }

      auto insertionPoint = rewriter.getInsertionPoint();
      auto cbA = getCB(rewriter, op.getA());
      auto cbB = getCB(rewriter, op.getB());
      auto outCB = getOutCB(rewriter, op);
      setInsertionPointAfterOperands(rewriter, {cbA, cbB, outCB},
                                     /*allowHoisting*/ true);
      rewriter.create<ttkernel::ComputeKernelHWStartupOp>(op->getLoc(), cbA,
                                                          cbB, outCB);
      rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);
      rewriter.create<ttkernel::ReduceInitOp>(op->getLoc(), cbA, cbB, outCB,
                                              reduce_type, kernel_reduce_dim);
      rewriter.create<ttkernel::ReduceTileOp>(
          op->getLoc(), cbA, cbB, adaptor.getA(), adaptor.getB(),
          adaptor.getC(), reduce_type, kernel_reduce_dim);
      rewriter.create<ttkernel::ReduceUninitOp>(op->getLoc());
    } else if constexpr (std::is_same_v<ConcreteOp, d2m::TileBcastOp>) {
      ttkernel::BcastType bcastType = ttkernel::BcastType::None;
      switch (op.getBcastType()) {
      case d2m::TileBcastType::Col:
        bcastType = ttkernel::BcastType::Col;
        break;
      case d2m::TileBcastType::Row:
        bcastType = ttkernel::BcastType::Row;
        break;
      case d2m::TileBcastType::Scalar:
        bcastType = ttkernel::BcastType::Scalar;
        break;
      case d2m::TileBcastType::None:
        bcastType = ttkernel::BcastType::None;
        break;
      }
      auto cb = getCB(rewriter, op.getInput());
      auto dstIdx = getDstIdxFromResult(op.getResult());
      rewriter.create<ttkernel::UnaryBcastInitOp>(op->getLoc(), cb, cb,
                                                  bcastType);
      rewriter.create<ttkernel::UnaryBcastTileOp>(
          op->getLoc(), cb, adaptor.getInput(), dstIdx, bcastType);
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
class D2MSFPUOpsRewriter : public OpConversionPattern<ConcreteOp> {
public:
  using OpConversionPattern<ConcreteOp>::OpConversionPattern;
  using KernelOpPair = TTKernelOpPair<ConcreteOp, ComputeOpMap>;
  using InitOp = typename KernelOpPair::first_type;
  using SFPUOp = typename KernelOpPair::second_type;

  static_assert(SFPUOp::template hasTrait<TTKernelSFPUOpTrait>(),
                "SFPUOp must have TTKernelSFPUOpTrait");

  static constexpr int arity = SFPUOp::arity;

  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto insertionPoint = rewriter.getInsertionPoint();
    auto inCB = getInCB(rewriter, op);
    auto outCB = getOutCB(rewriter, op);
    rewriter.setInsertionPointToStart(rewriter.getInsertionBlock());
    setInsertionPointAfterOperands(rewriter, {inCB, outCB},
                                   /*allowHoisting*/ true);
    rewriter.create<ttkernel::InitSFPUOp>(op->getLoc(), inCB, outCB);
    rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);

    // For binary ops (arity == 2), check if rhs is a scalar to create the right
    // init op
    if constexpr (arity == 2) {
      auto rhsType = adaptor.getRhs().getType();
      bool isScalarRhs = rhsType.isIntOrFloat();

      if (isScalarRhs) {
        // Use scalar-specific init ops
        if constexpr (std::is_same_v<ConcreteOp, d2m::TilePowOp>) {
          rewriter.create<ttkernel::PowerTileInitOp>(op->getLoc());
        } else {
          rewriter.create<ttkernel::BinopWithScalarTileInitOp>(op->getLoc());
        }
      } else {
        rewriter.create<InitOp>(op->getLoc());
      }
    } else if constexpr (std::is_same_v<InitOp, ttkernel::TypecastTileInitOp>) {
      const auto inDtype =
          mlir::cast<ttcore::TileType>(op.getInput().getType()).getDataType();
      const auto outDtype =
          mlir::cast<ttcore::TileType>(op.getResult().getType()).getDataType();
      rewriter.create<ttkernel::TypecastTileInitOp>(op->getLoc(), inDtype,
                                                    outDtype);
    } else {
      rewriter.create<InitOp>(op->getLoc());
    }

    if constexpr (std::is_same_v<SFPUOp, ttkernel::AbsTileOp> ||
                  std::is_same_v<SFPUOp, ttkernel::LogicalNotUnaryTileOp> ||
                  std::is_same_v<SFPUOp, ttkernel::ReluTileOp>) {
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
        } else if (std::is_same_v<SFPUOp, ttkernel::LogicalNotUnaryTileOp>) {
          rewriter.create<ttkernel::LogicalNotUnaryTileI32Op>(
              op->getLoc(), adaptor.getInput());
        } else if (std::is_same_v<SFPUOp, ttkernel::ReluTileOp>) {
          rewriter.create<ttkernel::ReluTileI32Op>(op->getLoc(),
                                                   adaptor.getInput());
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
    } else if constexpr (std::is_same_v<SFPUOp, ttkernel::TypecastTileOp>) {
      const auto inDtype =
          mlir::cast<ttcore::TileType>(op.getInput().getType()).getDataType();
      const auto outDtype =
          mlir::cast<ttcore::TileType>(op.getResult().getType()).getDataType();
      rewriter.create<ttkernel::TypecastTileOp>(
          op->getLoc(), adaptor.getInput(), inDtype, outDtype);
    } else if constexpr (std::is_same_v<SFPUOp, ttkernel::ClampScalarTileOp>) {
      // ClampScalarOp has min/max F32 attributes that need to be bitcast to i32
      auto loc = op->getLoc();
      auto minFloat = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF32FloatAttr(op.getMin().convertToFloat()));
      auto maxFloat = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF32FloatAttr(op.getMax().convertToFloat()));
      auto minParam = rewriter.create<arith::BitcastOp>(
          loc, rewriter.getI32Type(), minFloat);
      auto maxParam = rewriter.create<arith::BitcastOp>(
          loc, rewriter.getI32Type(), maxFloat);
      rewriter.create<ttkernel::ClampScalarTileOp>(loc, adaptor.getInput(),
                                                   minParam, maxParam);
    } else if constexpr (arity == 1) {
      rewriter.create<SFPUOp>(op->getLoc(), adaptor.getInput());
    } else if constexpr (arity == 2) {
      // Check if rhs is a scalar (float or integer) at runtime
      auto rhsType = adaptor.getRhs().getType();
      bool isScalarRhs = rhsType.isIntOrFloat();

      if (isScalarRhs) {
        // Handle scalar operand - need to use unary scalar ops
        const auto dstIdx = adaptor.getLhs();
        auto loc = op->getLoc();

        // Create the appropriate unary scalar op based on the D2M op type
        if constexpr (std::is_same_v<ConcreteOp, d2m::TileAddOp>) {
          // Bitcast the scalar value to i32 to pass as parameter
          rewriter.create<ttkernel::BinopWithScalarTileInitOp>(loc);
          auto scalarParam = rewriter.create<arith::BitcastOp>(
              loc, rewriter.getI32Type(), adaptor.getRhs());
          rewriter.create<ttkernel::AddUnaryTileOp>(loc, dstIdx, scalarParam);
        } else if constexpr (std::is_same_v<ConcreteOp, d2m::TileSubOp>) {
          rewriter.create<ttkernel::BinopWithScalarTileInitOp>(loc);
          auto scalarParam = rewriter.create<arith::BitcastOp>(
              loc, rewriter.getI32Type(), adaptor.getRhs());
          rewriter.create<ttkernel::SubUnaryTileOp>(loc, dstIdx, scalarParam);
        } else if constexpr (std::is_same_v<ConcreteOp, d2m::TileMulOp>) {
          rewriter.create<ttkernel::BinopWithScalarTileInitOp>(loc);
          auto scalarParam = rewriter.create<arith::BitcastOp>(
              loc, rewriter.getI32Type(), adaptor.getRhs());
          rewriter.create<ttkernel::MulUnaryTileOp>(loc, dstIdx, scalarParam);
        } else if constexpr (std::is_same_v<ConcreteOp, d2m::TileDivOp>) {
          auto scalarParam = rewriter.create<arith::BitcastOp>(
              loc, rewriter.getI32Type(), adaptor.getRhs());
          rewriter.create<ttkernel::DivUnaryTileOp>(loc, dstIdx, scalarParam);
        } else if constexpr (std::is_same_v<ConcreteOp, d2m::TilePowOp>) {
          // For power, convert float value to integer (not bitcast)
          auto scalarParam = rewriter.create<arith::FPToSIOp>(
              loc, rewriter.getI32Type(), adaptor.getRhs());
          rewriter.create<ttkernel::PowUnaryTileOp>(loc, dstIdx, scalarParam);
        }
        // Scalar ops operate in-place on DST slot - replace with the same
        // dstIdx.
        rewriter.replaceOp(op, dstIdx);
        return success();
      }
      // Otherwise, this is a binary tile operation.
      OpBuilder::InsertionGuard guard(rewriter);
      const auto dstIdx = getDstIdxFromResult(op.getResult());
      setInsertionPointAfterOperands(
          rewriter, {adaptor.getLhs(), adaptor.getRhs(), dstIdx},
          /*allowHoisting*/ false);
      if constexpr (std::is_same_v<SFPUOp, ttkernel::BitwiseAndBinaryTilesOp> ||
                    std::is_same_v<SFPUOp, ttkernel::BitwiseOrBinaryTilesOp> ||
                    std::is_same_v<SFPUOp, ttkernel::BitwiseXorBinaryTilesOp>) {
        const auto dtype =
            mlir::cast<ttcore::TileType>(op.getLhs().getType()).getDataType();
        rewriter.create<SFPUOp>(op->getLoc(), adaptor.getLhs(),
                                adaptor.getRhs(), dstIdx, dtype);
      } else {
        rewriter.create<SFPUOp>(op->getLoc(), adaptor.getLhs(),
                                adaptor.getRhs(), dstIdx);
      }
    } else {
      // Ternary tile operation (arity == 3)
      OpBuilder::InsertionGuard guard(rewriter);
      const auto dstIdx = getDstIdxFromResult(op.getResult());
      setInsertionPointAfterOperands(rewriter,
                                     {adaptor.getCondition(),
                                      adaptor.getTrueValue(),
                                      adaptor.getFalseValue(), dstIdx},
                                     /*allowHoisting*/ false);
      if constexpr (std::is_same_v<ConcreteOp, d2m::TileWhereOp>) {
        const auto dtype =
            mlir::cast<ttcore::TileType>(op.getTrueValue().getType())
                .getDataType();
        rewriter.create<ttkernel::WhereTileOp>(
            op->getLoc(), adaptor.getCondition(), adaptor.getTrueValue(),
            adaptor.getFalseValue(), dstIdx, dtype);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

namespace {

// FPU rewriter for binary ops (Add, Sub, Mul).
template <typename ConcreteOp>
class D2MFPUBinaryRewriter : public OpConversionPattern<ConcreteOp> {
public:
  // Higher benefit than D2MSFPUOpsRewriter to ensure FPU path is tried first.
  D2MFPUBinaryRewriter(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<ConcreteOp>(typeConverter, context,
                                        /*benefit=*/2) {}

  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto rhsType = adaptor.getRhs().getType();
    bool isScalarRhs = rhsType.isIntOrFloat();

    // Scalar operand: fallback to SFPU
    if (isScalarRhs) {
      return failure();
    }

    auto loc = op->getLoc();
    bool lhsFromDst = operandFromDst(op.getLhs());
    bool rhsFromDst = operandFromDst(op.getRhs());

    if (lhsFromDst && rhsFromDst) {
      // Both from DST: fallback to SFPU
      return failure();
    }

    if (!lhsFromDst && !rhsFromDst) {
      // Both from CB: standard FPU path
      return emitFPUTiles(rewriter, op, adaptor, loc);
    }

    if (lhsFromDst) {
      // LHS from Dst, RHS from Cb (FPU - DEST_TO_SRCA)
      return emitFPUBinaryDestReuse(rewriter, op, adaptor, loc,
                                    BinaryDestReuseType::DestToSrcA);
    }

    return emitFPUBinaryDestReuse(rewriter, op, adaptor, loc,
                                  BinaryDestReuseType::DestToSrcB);
  }

private:
  // Get the EltwiseBinaryType for this op
  static constexpr ttkernel::EltwiseBinaryType getEltwiseBinaryType() {
    if constexpr (std::is_same_v<ConcreteOp, d2m::TileAddOp>) {
      return ttkernel::EltwiseBinaryType::Add;
    } else if constexpr (std::is_same_v<ConcreteOp, d2m::TileSubOp>) {
      return ttkernel::EltwiseBinaryType::Sub;
    } else if constexpr (std::is_same_v<ConcreteOp, d2m::TileMulOp>) {
      return ttkernel::EltwiseBinaryType::Mul;
    }
  }

  // Standard FPU path
  LogicalResult emitFPUTiles(ConversionPatternRewriter &rewriter, ConcreteOp op,
                             typename ConcreteOp::Adaptor adaptor,
                             Location loc) const {
    auto cbA = getCB(rewriter, op.getLhs());
    auto cbB = getCB(rewriter, op.getRhs());
    auto outCB = getOutCB(rewriter, op);

    auto insertionPoint = rewriter.getInsertionPoint();
    setInsertionPointAfterOperands(rewriter, {cbA, cbB, outCB},
                                   /*allowHoisting*/ true);
    rewriter.create<ttkernel::BinaryOpInitCommonOp>(loc, cbA, cbB, outCB);
    rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);

    auto dstIdx = getDstIdxFromResult(op.getResult());

    if constexpr (std::is_same_v<ConcreteOp, d2m::TileAddOp>) {
      rewriter.create<ttkernel::AddTilesInitOp>(loc, cbA, cbB);
      rewriter.create<ttkernel::AddTilesOp>(loc, cbA, cbB, adaptor.getLhs(),
                                            adaptor.getRhs(), dstIdx);
    } else if constexpr (std::is_same_v<ConcreteOp, d2m::TileSubOp>) {
      rewriter.create<ttkernel::SubTilesInitOp>(loc, cbA, cbB);
      rewriter.create<ttkernel::SubTilesOp>(loc, cbA, cbB, adaptor.getLhs(),
                                            adaptor.getRhs(), dstIdx);
    } else if constexpr (std::is_same_v<ConcreteOp, d2m::TileMulOp>) {
      rewriter.create<ttkernel::MulTilesInitOp>(loc, cbA, cbB);
      rewriter.create<ttkernel::MulTilesOp>(loc, cbA, cbB, adaptor.getLhs(),
                                            adaptor.getRhs(), dstIdx);
    }

    rewriter.eraseOp(op);
    return success();
  }

  // One operand from Dst, use binary_dest_reuse
  LogicalResult emitFPUBinaryDestReuse(ConversionPatternRewriter &rewriter,
                                       ConcreteOp op,
                                       typename ConcreteOp::Adaptor adaptor,
                                       Location loc,
                                       BinaryDestReuseType reuseType) const {
    // Get the Cb for the non-Dst operand
    Value cb;
    Value cbTileIdx;
    Value dstOperandIdx;
    if (reuseType == BinaryDestReuseType::DestToSrcA) {
      // LHS from Dst, RHS from Cb
      cb = getCB(rewriter, op.getRhs());
      cbTileIdx = adaptor.getRhs();
      dstOperandIdx = adaptor.getLhs();
    } else {
      cb = getCB(rewriter, op.getLhs());
      cbTileIdx = adaptor.getLhs();
      dstOperandIdx = adaptor.getRhs();
    }

    auto outCB = getOutCB(rewriter, op);

    // Dst index is the same for input and output
    auto dstIdx = getDstIdxFromResult(op.getResult());

    auto insertionPoint = rewriter.getInsertionPoint();
    setInsertionPointAfterOperands(rewriter, {cb, outCB},
                                   /*allowHoisting*/ true);
    rewriter.create<ttkernel::BinaryOpInitCommonOp>(loc, cb, cb, outCB);
    rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);

    auto eltwiseType = getEltwiseBinaryType();

    // binary_dest_reuse is an in-place operation. If the DST
    // operand comes from a different slot, copy it first to the output slot.
    if (dstOperandIdx != dstIdx) {
      rewriter.create<ttkernel::CopyDestValuesInitOp>(loc);
      rewriter.create<ttkernel::CopyDestValuesOp>(loc, dstOperandIdx, dstIdx);
    }

    rewriter.create<ttkernel::BinaryDestReuseTilesInitOp>(loc, cb, eltwiseType,
                                                          reuseType);
    rewriter.create<ttkernel::BinaryDestReuseTilesOp>(
        loc, cb, cbTileIdx, dstIdx, eltwiseType, reuseType);

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {
template <typename ConcreteOp, typename BlockOp>
class D2MTilizeUntilizeRewriter : public OpConversionPattern<ConcreteOp> {
public:
  using OpConversionPattern<ConcreteOp>::OpConversionPattern;

  static Value findPreLinearizedMemref(Value memref) {
    if (mlir::isa_and_nonnull<d2m::WaitOp, d2m::ReserveOp>(
            memref.getDefiningOp())) {
      return memref;
    }
    if (auto collapseOp = mlir::dyn_cast_if_present<memref::CollapseShapeOp>(
            memref.getDefiningOp())) {
      return findPreLinearizedMemref(collapseOp.getSrc());
    }
    llvm_unreachable("Expected BlockArgument or CollapseShapeOp");
  }

  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value src = adaptor.getInput();
    Value dst = adaptor.getOutput();
    bool constexpr tilize =
        std::is_same_v<BlockOp, ttkernel::ExperimentalTilizeBlockOp>;
    auto preLinearizedMemrefType = mlir::cast<MemRefType>(
        findPreLinearizedMemref(tilize ? op.getOutput() : op.getInput())
            .getType());
    auto collapsed2DShape =
        ttcore::collapseGridTo2D(preLinearizedMemrefType.getShape());

    auto blockR = i32(rewriter, op->getLoc(), collapsed2DShape[0]);
    auto blockC = i32(rewriter, op->getLoc(), collapsed2DShape[1]);
    rewriter.create<ttkernel::ComputeKernelHWStartupOp>(op->getLoc(), src,
                                                        nullptr, dst);

    if constexpr (std::is_same_v<BlockOp,
                                 ttkernel::ExperimentalTilizeBlockOp>) {
      rewriter.create<ttkernel::TilizeInitOp>(op->getLoc(), src, blockC, dst);
    } else if constexpr (std::is_same_v<
                             BlockOp, ttkernel::ExperimentalUntilizeBlockOp>) {
      rewriter.create<ttkernel::UntilizeInitOp>(op->getLoc(), src);
    } else {
      llvm_unreachable("unsupported tilize/untilize op");
    }

    rewriter.create<BlockOp>(op->getLoc(), src, dst, blockR, blockC);

    rewriter.eraseOp(op);

    return success();
  };
};

class D2MTileFillRewriter : public OpConversionPattern<d2m::TileFillOp> {
public:
  using OpConversionPattern<d2m::TileFillOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::TileFillOp op, d2m::TileFillOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value dstIdx = getDstIdxFromResult(op.getResult());

    Value fillValue = adaptor.getValue();
    Location loc = op->getLoc();

    rewriter.create<ttkernel::ExperimentalTileFillOp>(loc, dstIdx, fillValue);

    // Replace the op with its DST index so users (like TileWhereOp) get the
    // correct operand value.
    rewriter.replaceOp(op, dstIdx);
    return success();
  }
};

class D2MWriteRowMaskTileRewriter
    : public OpConversionPattern<d2m::WriteRowMaskTileOp> {
public:
  using OpConversionPattern<d2m::WriteRowMaskTileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::WriteRowMaskTileOp op,
                  d2m::WriteRowMaskTileOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    Value validRows = adaptor.getValidRows();
    if (!validRows.getType().isInteger(32)) {
      validRows = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), validRows);
    }
    rewriter.create<ttkernel::ExperimentalWriteRowMaskTileOp>(
        loc, validRows, adaptor.getOutput());
    rewriter.eraseOp(op);
    return success();
  }
};

class D2MWriteColMaskTileRewriter
    : public OpConversionPattern<d2m::WriteColMaskTileOp> {
public:
  using OpConversionPattern<d2m::WriteColMaskTileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::WriteColMaskTileOp op,
                  d2m::WriteColMaskTileOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    Value validCols = adaptor.getValidCols();
    if (!validCols.getType().isInteger(32)) {
      validCols = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), validCols);
    }
    rewriter.create<ttkernel::ExperimentalWriteColMaskTileOp>(
        loc, validCols, adaptor.getOutput());
    rewriter.eraseOp(op);
    return success();
  }
};
class D2MExperimentalFillArangeTileRewriter
    : public OpConversionPattern<d2m::FillArangeTileOp> {
public:
  using OpConversionPattern<d2m::FillArangeTileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::FillArangeTileOp op,
                  d2m::FillArangeTileOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.create<ttkernel::ExperimentalFillArangeTileOp>(
        op->getLoc(), adaptor.getOutput());
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {
class D2MTileTransposeRewriter
    : public OpConversionPattern<d2m::TileTransposeOp> {
public:
  using OpConversionPattern<d2m::TileTransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::TileTransposeOp op, d2m::TileTransposeOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // TileTransposeOp is a unary op that takes an input tile and produces
    // an output tile.

    Value inCB = getInCB(rewriter, op);

    Value outCB = getOutCB(rewriter, op);

    auto insertionPoint = rewriter.getInsertionPoint();
    setInsertionPointAfterOperands(rewriter, {inCB, outCB},
                                   /*allowHoisting*/ true);
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

class D2MDstReinterpretCastRewriter
    : public OpConversionPattern<d2m::DstReinterpretCastOp> {
public:
  using OpConversionPattern<d2m::DstReinterpretCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::DstReinterpretCastOp op,
                  d2m::DstReinterpretCastOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  };
};
} // namespace

namespace {
template <typename D2MCBOp, typename TTKernelAcquireOp,
          typename TTKernelReleaseOp>
class D2MCBOpRewriter : public OpConversionPattern<D2MCBOp> {
public:
  using OpConversionPattern<D2MCBOp>::OpConversionPattern;

  static_assert(std::is_same_v<D2MCBOp, d2m::WaitOp> ||
                std::is_same_v<D2MCBOp, d2m::ReserveOp>);

  // Check if there's an explicit push/pop for this CB in the same block
  static bool hasExplicitRelease(D2MCBOp op) {
    Block *block = op->getBlock();
    Value cb = op.getCb();

    // Check for explicit d2m.push (for reserve) or d2m.pop (for wait)
    for (Operation &blockOp : *block) {
      if constexpr (std::is_same_v<D2MCBOp, d2m::ReserveOp>) {
        if (auto pushOp = dyn_cast<d2m::PushOp>(&blockOp)) {
          if (pushOp.getCb() == cb) {
            return true;
          }
        }
      } else if constexpr (std::is_same_v<D2MCBOp, d2m::WaitOp>) {
        if (auto popOp = dyn_cast<d2m::PopOp>(&blockOp)) {
          if (popOp.getCb() == cb) {
            return true;
          }
        }
      }
    }
    return false;
  }

  LogicalResult
  matchAndRewrite(D2MCBOp op, typename D2MCBOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // For ops in merged DMA regions, each original thread is scoped in a
    // separate scf.execute_region to ensure CB release ops for
    // reader-writer kernels do not conflict. Now that the release ops are being
    // inserted, remove the no_inline attribute so the scf.execute_region ops
    // can be canonicalized.
    if (auto executeRegionOp =
            dyn_cast<scf::ExecuteRegionOp>(op->getParentOp())) {
      if (executeRegionOp->hasAttr("no_inline")) {
        rewriter.modifyOpInPlace(executeRegionOp, [&]() {
          executeRegionOp->removeAttr("no_inline");
        });
      }
    }

    auto device = ttcore::lookupDevice(op);

    auto cbNumPages = device.getMemrefCBNumPages(
        op.getCb().getType().template getUnderlyingAs<MemRefType>());
    auto numPages = i32(rewriter, op->getLoc(), cbNumPages);

    rewriter.create<TTKernelAcquireOp>(op.getLoc(), adaptor.getCb(), numPages);

    // Only insert automatic release if there's no explicit push/pop
    if (!hasExplicitRelease(op)) {
      Block *block = op->getBlock();
      auto release = rewriter.create<TTKernelReleaseOp>(
          op.getLoc(), adaptor.getCb(), numPages);
      if (block->mightHaveTerminator()) {
        rewriter.moveOpBefore(release, block->getTerminator());
      } else {
        rewriter.moveOpAfter(release, &block->back());
      }
    }

    rewriter.replaceOp(op, adaptor.getCb());

    return success();
  };
};
} // namespace

namespace {
// Template rewriter for d2m.push/pop → ttkernel.cb_push_back/cb_pop_front
template <typename D2MReleaseOp, typename TTKernelReleaseOp>
class D2MCBReleaseOpRewriter : public OpConversionPattern<D2MReleaseOp> {
public:
  using OpConversionPattern<D2MReleaseOp>::OpConversionPattern;

  static_assert(std::is_same_v<D2MReleaseOp, d2m::PushOp> ||
                std::is_same_v<D2MReleaseOp, d2m::PopOp>);

  LogicalResult
  matchAndRewrite(D2MReleaseOp op, typename D2MReleaseOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto device = ttcore::lookupDevice(op);

    auto cbNumPages = device.getMemrefCBNumPages(
        op.getCb().getType().template getUnderlyingAs<MemRefType>());
    auto numPages = i32(rewriter, op->getLoc(), cbNumPages);

    rewriter.replaceOpWithNewOp<TTKernelReleaseOp>(op, adaptor.getCb(),
                                                   numPages);
    return success();
  }
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

class D2MDMAReadRewriter : public OpConversionPattern<d2m::DMAReadOp> {
public:
  D2MDMAReadRewriter(TypeConverter &typeConverter, MLIRContext *context,
                     const d2m::AssociatedDMAWaits *associatedDMAWaits,
                     const d2m::CBProducerConsumer *cbProducerConsumer)
      : OpConversionPattern<d2m::DMAReadOp>(typeConverter, context),
        associatedDMAWaits(associatedDMAWaits),
        cbProducerConsumer(cbProducerConsumer) {}

  LogicalResult
  matchAndRewrite(d2m::DMAReadOp op, d2m::DMAReadOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto chipDesc = ttcore::getOpChipDescAttr(op);

    // NOTE: All reads must be from remote locations in DMAReadOp
    // local->local transfers are lowered as nocAsyncWrites, which require
    // write barriers.
    auto srcNocAddr =
        buildNocAddress(rewriter, op.getLoc(), adaptor.getSrc(),
                        op.getSrcIndices(), chipDesc, op.getSrcMemorySpace());
    auto dstCBMapping = cbProducerConsumer->get(op.getDst());
    TT_assertv((dstCBMapping == d2m::ThreadCBOrientation::Producer ||
                dstCBMapping == d2m::ThreadCBOrientation::Default),
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

    rewriter.replaceOpWithNewOp<d2m::NullTxOp>(op);

    return success();
  }

private:
  const d2m::AssociatedDMAWaits *associatedDMAWaits;
  const d2m::CBProducerConsumer *cbProducerConsumer;
};

class D2MDMAWriteRewriter : public OpConversionPattern<d2m::DMAWriteOp> {
public:
  D2MDMAWriteRewriter(TypeConverter &typeConverter, MLIRContext *context,
                      const d2m::AssociatedDMAWaits *associatedDMAWaits,
                      const d2m::CBProducerConsumer *cbProducerConsumer)
      : OpConversionPattern<d2m::DMAWriteOp>(typeConverter, context),
        associatedDMAWaits(associatedDMAWaits),
        cbProducerConsumer(cbProducerConsumer) {}

  LogicalResult
  matchAndRewrite(d2m::DMAWriteOp op, d2m::DMAWriteOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto chipDesc = ttcore::getOpChipDescAttr(op);

    if (op.isDstLocal()) {
      // Local to Local Datamovement & Multicast

      // Both src and dst are local, use the metal cb pointers to determine
      // addressing
      Value srcL1Start;
      auto srcCBMapping = cbProducerConsumer->get(op.getSrc());
      if (srcCBMapping == d2m::ThreadCBOrientation::Producer) {
        srcL1Start = rewriter.create<ttkernel::GetWritePtrOp>(op.getLoc(),
                                                              adaptor.getSrc());
      } else {
        srcL1Start = rewriter.create<ttkernel::GetReadPtrOp>(op.getLoc(),
                                                             adaptor.getSrc());
      }
      auto dstCBMapping = cbProducerConsumer->get(op.getDst());
      TT_assertv((dstCBMapping == d2m::ThreadCBOrientation::Producer ||
                  dstCBMapping == d2m::ThreadCBOrientation::ProducerConsumer ||
                  dstCBMapping == d2m::ThreadCBOrientation::Default),
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
        // Get the logical multicast end coordinates from the logical start
        // coordinates and mcast shape then convert to virtual coordinates
        auto [mcastEndY, mcastEndX] =
            getMcastEndCoords(rewriter, op.getLoc(), op.getMcastStartIndex()[0],
                              op.getMcastStartIndex()[1], op.getMcastShape());
        auto [virtMcastEndY, virtMcastEndX] = getVirtualCoordsFromLogicalCoords(
            rewriter, op.getLoc(), chipDesc, {mcastEndY, mcastEndX});
        auto numDestsIdx = rewriter.create<arith::MulIOp>(
            op.getLoc(), op.getMcastShape()[0], op.getMcastShape()[1]);
        auto numDests = rewriter.create<arith::IndexCastOp>(
            op.getLoc(), rewriter.getI32Type(), numDestsIdx);
        auto numDestsMinusOne = rewriter.create<arith::SubIOp>(
            op.getLoc(), numDests,
            rewriter.create<arith::ConstantOp>(op.getLoc(),
                                               rewriter.getI32Type(),
                                               rewriter.getI32IntegerAttr(1)));
        auto mcastAddr =
            rewriter.create<ttkernel::ExperimentalGetNocMulticastAddrOp>(
                op.getLoc(), virtX, virtY, virtMcastEndX, virtMcastEndY,
                dstL1Start, nullptr);
        if (adaptor.getSrc() == adaptor.getDst()) {
          // If src and dst refer to the same memref, we do not loopback mcast
          // Dests are one less because the sender core is not included
          rewriter.create<ttkernel::NocAsyncWriteMulticastOp>(
              op.getLoc(), srcL1Start, mcastAddr, transferSize,
              numDestsMinusOne, rewriter.getBoolAttr(true), nullptr, nullptr);
        } else {
          // If src != dst, we loopback mcast
          rewriter.create<ttkernel::NocAsyncWriteMulticastLoopbackSrcOp>(
              op.getLoc(), srcL1Start, mcastAddr, transferSize, numDests,
              rewriter.getBoolAttr(true), nullptr, nullptr);
        }
      } else {
        // Local L1 to Local L1 local data movement lowering
        // Get local coordinates using myY and myX ops
        auto myY = rewriter.create<ttkernel::MyLogicalYOp>(op.getLoc());
        auto myX = rewriter.create<ttkernel::MyLogicalXOp>(op.getLoc());
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
    // Mcast writes do not require a write barrier.
    auto dmaWaitOps = associatedDMAWaits->get(op);
    StringRef waitAttr = op.isMcast()
                             ? "ttkernel.lowering.associated_noc_mcast_write"
                             : "ttkernel.lowering.associated_noc_write";
    for (auto dmaWaitOp : dmaWaitOps) {
      rewriter.modifyOpInPlace(dmaWaitOp, [&]() {
        dmaWaitOp->setDiscardableAttr(waitAttr, rewriter.getUnitAttr());
      });
    }

    rewriter.replaceOpWithNewOp<d2m::NullTxOp>(op);
    return success();
  }

private:
  const d2m::AssociatedDMAWaits *associatedDMAWaits;
  const d2m::CBProducerConsumer *cbProducerConsumer;
};
} // namespace

namespace {
class D2MCoreIndexRewriter : public OpConversionPattern<d2m::CoreIndexOp> {
public:
  using OpConversionPattern<d2m::CoreIndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::CoreIndexOp op, d2m::CoreIndexOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    Value logicalY = rewriter.create<ttkernel::MyLogicalYOp>(op.getLoc());
    Value logicalX = rewriter.create<ttkernel::MyLogicalXOp>(op.getLoc());

    // If no virtualization mapping, preserve legacy behavior.
    // Note: phys_to_virt_map is optional on the op.
    auto mapAttr = op.getPhysToVirtMapAttr();
    if (!mapAttr || mapAttr.getValue().isEmpty()) {
      TT_assertv((op.getDim() == 0 || op.getDim() == 1),
                 "Expected core index dim to be in range 0-1 with no "
                 "virtualization mapping, failing.");
      rewriter.replaceOp(op, op.getDim() ? logicalX : logicalY);
      return success();
    }

    mlir::AffineMap map = mapAttr.getValue();

    // The existing grid mapping includes a leading device index result, so we
    // select (dim + 1).
    const unsigned resultIdx = static_cast<unsigned>(op.getDim() + 1);
    TT_assertv(resultIdx < map.getNumResults(),
               "Expected result index to be less than the number of results, "
               "failing.");
    // The resultIdx is the index of the result that corresponds to the dim of
    // the core index op, this new map has one result, required for use with
    // affine apply op.
    mlir::AffineMap selectedMap =
        mlir::AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                             {map.getResult(resultIdx)}, rewriter.getContext());

    Value virtDim = rewriter.create<mlir::affine::AffineApplyOp>(
        op.getLoc(), selectedMap, ValueRange{logicalY, logicalX});
    rewriter.replaceOp(op, virtDim);
    return success();
  }
};
} // namespace

namespace {
class D2MDMAWaitRewriter : public OpConversionPattern<d2m::DMAWaitOp> {
public:
  using OpConversionPattern<d2m::DMAWaitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::DMAWaitOp op, d2m::DMAWaitOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto isRead =
        op->getDiscardableAttr("ttkernel.lowering.associated_noc_read");
    auto isWrite =
        op->getDiscardableAttr("ttkernel.lowering.associated_noc_write");
    auto isMcastWrite =
        op->getDiscardableAttr("ttkernel.lowering.associated_noc_mcast_write");
    assert(isRead || isWrite || isMcastWrite);

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
class D2MGetGlobalOperandRewriter
    : public OpConversionPattern<d2m::GetGlobalOperandOp> {
public:
  D2MGetGlobalOperandRewriter(TypeConverter &typeConverter,
                              MLIRContext *context, bool ttnnMode)
      : OpConversionPattern<d2m::GetGlobalOperandOp>(typeConverter, context),
        ttnnMode(ttnnMode) {}

  LogicalResult
  matchAndRewrite(d2m::GetGlobalOperandOp op,
                  d2m::GetGlobalOperandOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    func::FuncOp entry = op->getParentOfType<func::FuncOp>();
    ArgAttr arg;
    size_t argIndex;
    Type arg_result_type;

    // There should be exactly one generic operation that uses this kernel
    // function
    auto uses =
        SymbolTable::getSymbolUses(entry, op->getParentOfType<ModuleOp>());
    assert(llvm::range_size(*uses) == 1);
    auto genericOp = mlir::dyn_cast<d2m::GenericOp>(uses->begin()->getUser());
    assert(genericOp);

    // Build mapping between operand index and index for specific type arg to
    // use next
    llvm::SmallDenseMap<unsigned, unsigned> global_semaphore_index_mapping;
    llvm::SmallDenseMap<unsigned, unsigned> buffer_address_index_mapping;
    for (auto [i, operand] : llvm::enumerate(genericOp.getOperands())) {
      if (mlir::isa<d2m::GlobalSemaphoreType>(operand.getType())) {
        global_semaphore_index_mapping[i] =
            global_semaphore_index_mapping.size();
      } else if (mlir::isa<MemRefType>(operand.getType())) {
        buffer_address_index_mapping[i] = buffer_address_index_mapping.size();
      } else {
        llvm_unreachable("unexpected operand type");
      }
    }

    if (mlir::isa<MemRefType>(op.getResult().getType())) {
      arg = rewriter.getAttr<ArgAttr>(
          ArgType::BufferAddress,
          buffer_address_index_mapping[op.getOperandIndex()]);
      arg_result_type = rewriter.getI32Type();
    } else if (mlir::isa<d2m::GlobalSemaphoreType>(op.getResult().getType())) {
      arg = rewriter.getAttr<ArgAttr>(
          ArgType::GlobalSemaphore,
          global_semaphore_index_mapping[op.getOperandIndex()]);
      arg_result_type = ttkernel::L1AddrType::get(rewriter.getContext());
    } else {
      assert(false && "unexpected arg type to GetGlobalOperandOp");
    }

    if (ttnnMode) {
      // to do: just erase if arg already in list, and get index
      rewriter.modifyOpInPlace(entry, [&]() {
        argIndex = ArgSpecAttr::appendRuntimeArg(entry, arg);
      });
      rewriter.replaceOpWithNewOp<ttkernel::GetCommonArgValOp>(
          op, arg_result_type, index(rewriter, op->getLoc(), argIndex));

    } else {
      // to do: just erase if arg already in list
      rewriter.modifyOpInPlace(entry, [&]() {
        argIndex = ArgSpecAttr::appendCompileTimeArg(entry, arg);
      });
      rewriter.replaceOpWithNewOp<ttkernel::GetCompileArgValOp>(
          op, arg_result_type, argIndex);
    }
    return success();
  }

private:
  bool ttnnMode;
};
} // namespace

namespace {
class D2MNullTxRewriter : public OpConversionPattern<d2m::NullTxOp> {
public:
  using OpConversionPattern<d2m::NullTxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::NullTxOp op, d2m::NullTxOpAdaptor adaptor,
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
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};
} // namespace

namespace {
class D2MKernelFunctionArgsRewriter : public OpConversionPattern<func::FuncOp> {
public:
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  static ThreadType getTTKernelThreadType(func::FuncOp op) {
    d2m::ThreadAttr threadAttr =
        op->getAttrOfType<d2m::ThreadAttr>(d2m::ThreadAttr::name);
    switch (threadAttr.getThreadType()) {
    case d2m::ThreadType::Compute: {
      return ThreadType::Compute;
    }
    case d2m::ThreadType::Datamovement: {
      return ThreadType::Noc;
    }
    case d2m::ThreadType::Unified: {
      // Unified threads should have been split by SplitUnifiedThread before
      // reaching this pass.
      llvm_unreachable("Unexpected thread type in backend conversion");
    }
    }
  }

  static void convertFunctionAttrs(Builder &builder, func::FuncOp op,
                                   ArrayRef<ArgAttr> rtArgs,
                                   ArrayRef<ArgAttr> ctArgs) {

    // Get the TTKernel thread type and replace D2M thread type with TTKernel
    // thread type
    ThreadType threadType = getTTKernelThreadType(op);
    op->removeAttr(d2m::ThreadAttr::name);
    op->setAttr(ThreadTypeAttr::name,
                builder.getAttr<ThreadTypeAttr>(threadType));
    ArgSpecAttr::setArgSpec(op, builder.getAttr<ArgSpecAttr>(rtArgs, ctArgs));
  }

  LogicalResult
  matchAndRewrite(func::FuncOp op, func::FuncOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!op->hasAttr(d2m::ThreadAttr::name) ||
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
            op.getLoc(), argType,
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
class D2MSemaphoreUpdateRewriter : public OpConversionPattern<ConcreteOp> {
public:
  using OpConversionPattern<ConcreteOp>::OpConversionPattern;

  static_assert(std::is_same_v<ConcreteOp, d2m::SemaphoreSetOp> ||
                    std::is_same_v<ConcreteOp, d2m::SemaphoreIncOp>,
                "Expected SemaphoreSet or SemaphoreInc op passed to "
                "D2MSemaphoreUpdateRewriter.");

  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto chipDesc = ttcore::getOpChipDescAttr(op);

    Value value = op.getValue();
    Value semaphoreAddr = adaptor.getSemaphore();

    if (op.getDstCoreIndex().empty()) {
      assert(!mlir::isa<d2m::SemaphoreIncOp>(op) &&
             "d2m.semaphore_inc to local core is illegal.");

      // Local semaphore set
      auto semaphorePtr =
          rewriter.create<ttkernel::CastToL1PtrOp>(op.getLoc(), semaphoreAddr);

      rewriter.replaceOpWithNewOp<ttkernel::NocSemaphoreSetOp>(op, semaphorePtr,
                                                               value);
    } else if (op.getMcastShape().empty()) {
      assert(!mlir::isa<d2m::SemaphoreSetOp>(op) &&
             "d2m.semaphore_set to single remote core is illegal.");
      auto [virtY, virtX] = getVirtualCoordsFromLogicalCoords(
          rewriter, op.getLoc(), chipDesc, op.getDstCoreIndex());
      auto nocAddr = rewriter.create<ttkernel::GetNocAddrOp>(
          op.getLoc(), virtX, virtY, semaphoreAddr);
      rewriter.replaceOpWithNewOp<ttkernel::NocSemaphoreIncOp>(op, nocAddr,
                                                               value, nullptr);
    } else {
      assert(!mlir::isa<d2m::SemaphoreIncOp>(op) &&
             "d2m.semaphore_inc multicast is illegal.");

      auto [virtY, virtX] = getVirtualCoordsFromLogicalCoords(
          rewriter, op.getLoc(), chipDesc, op.getDstCoreIndex());
      // Get the logical multicast end coordinates from the logical start
      // coordinates and mcast shape then convert to virtual coordinates
      auto [mcastEndY, mcastEndX] =
          getMcastEndCoords(rewriter, op.getLoc(), op.getDstCoreIndex()[0],
                            op.getDstCoreIndex()[1], op.getMcastShape());
      auto [virtMcastEndY, virtMcastEndX] = getVirtualCoordsFromLogicalCoords(
          rewriter, op.getLoc(), chipDesc, {mcastEndY, mcastEndX});
      Value numDestsIdx = rewriter.create<arith::MulIOp>(
          op.getLoc(), op.getMcastShape()[0], op.getMcastShape()[1]);
      Value numDests = rewriter.create<arith::IndexCastOp>(
          op.getLoc(), rewriter.getI32Type(), numDestsIdx);
      Value numDestsMinusOne = rewriter.create<arith::SubIOp>(
          op.getLoc(), numDests,
          rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getI32Type(),
                                             rewriter.getI32IntegerAttr(1)));
      auto mcastAddr =
          rewriter.create<ttkernel::ExperimentalGetNocMulticastAddrOp>(
              op.getLoc(), virtX, virtY, virtMcastEndX, virtMcastEndY,
              semaphoreAddr, nullptr);

      auto semaphorePtr =
          rewriter.create<ttkernel::CastToL1PtrOp>(op.getLoc(), semaphoreAddr);
      rewriter.create<ttkernel::NocSemaphoreSetOp>(op.getLoc(), semaphorePtr,
                                                   value);
      rewriter.replaceOpWithNewOp<ttkernel::NocSemaphoreSetMulticastOp>(
          op, semaphoreAddr, mcastAddr, numDestsMinusOne, nullptr, nullptr);
    }

    return success();
  }
};
} // namespace

namespace {
class D2MSemaphoreWaitRewriter
    : public OpConversionPattern<d2m::SemaphoreWaitOp> {
public:
  using OpConversionPattern<d2m::SemaphoreWaitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::SemaphoreWaitOp op, d2m::SemaphoreWaitOpAdaptor adaptor,
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

void populateD2MToTTKernelPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns, TypeConverter &typeConverter,
    const d2m::AssociatedDMAWaits &associatedDMAWaits,
    const d2m::CBProducerConsumer &cbProducerConsumer, bool ttnnMode) {
  // clang-format off
  patterns.add<ttkernel::D2MKernelFunctionArgsRewriter,
               ttkernel::PassthroughRewriter<memref::CastOp>,
               ttkernel::MemRefSubviewRewriter,

               // FPU.
               ttkernel::D2MFPUOpsRewriter<d2m::TileBcastOp>,
               ttkernel::D2MFPUOpsRewriter<d2m::TileMatmulOp>,
               ttkernel::D2MFPUOpsRewriter<d2m::TileMatmulBlockOp>,

               // Reductions FPU.
               ttkernel::D2MFPUOpsRewriter<d2m::TileReduceSumOp>,
               ttkernel::D2MFPUOpsRewriter<d2m::TileReduceMaxOp>,

               // Elementwise SFPU Unary.
               ttkernel::D2MSFPUOpsRewriter<d2m::TileAbsOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileBitwiseNotOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileCeilOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileClampScalarOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileCosOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileErfOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileErfcOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileExpOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileFloorOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileGeluOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileHardsigmoidOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileLogOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileLogicalNotOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileNegativeOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileRecipOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileReluOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileRsqrtOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileSignOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileSqrtOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileSigmoidOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileSiluOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileSinOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileTanOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileTanhOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileEqzOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileNezOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileGtzOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileGezOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileLtzOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileLezOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileTypecastOp>,

               // FPU Binary (precedes SFPU fallback patterns).
               ttkernel::D2MFPUBinaryRewriter<d2m::TileAddOp>,
               ttkernel::D2MFPUBinaryRewriter<d2m::TileSubOp>,
               ttkernel::D2MFPUBinaryRewriter<d2m::TileMulOp>,

               // Elementwise SFPU Binary (also handles scalar operands).
               // Add/Sub/Mul are fallback for when FPU rewriter fails.
               ttkernel::D2MSFPUOpsRewriter<d2m::TileAddOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileSubOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileMulOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileBitwiseAndOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileBitwiseOrOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileBitwiseXorOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileDivOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileMaximumOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TileMinimumOp>,
               ttkernel::D2MSFPUOpsRewriter<d2m::TilePowOp>,

               // Elementwise SFPU Ternary.
               ttkernel::D2MSFPUOpsRewriter<d2m::TileWhereOp>,

               ttkernel::D2MTilizeUntilizeRewriter<d2m::TileTilizeBlockOp, ttkernel::ExperimentalTilizeBlockOp>,
               ttkernel::D2MTilizeUntilizeRewriter<d2m::TileUntilizeBlockOp, ttkernel::ExperimentalUntilizeBlockOp>,
               ttkernel::D2MTileFillRewriter,
               ttkernel::D2MWriteRowMaskTileRewriter,
               ttkernel::D2MWriteColMaskTileRewriter,
               ttkernel::D2MExperimentalFillArangeTileRewriter,
               ttkernel::D2MTileTransposeRewriter,
               ttkernel::D2MDstReinterpretCastRewriter,
               ttkernel::AcquireDstRewriter,
               ttkernel::D2MSetL1AccumulateRewriter,
               ttkernel::MemrefLoadRewriter,
               ttkernel::MemrefStoreRewriter,
               ttkernel::D2MCBOpRewriter<d2m::WaitOp, ttkernel::CBWaitFrontOp, ttkernel::CBPopFrontOp>,
               ttkernel::D2MCBOpRewriter<d2m::ReserveOp, ttkernel::CBReserveBackOp, ttkernel::CBPushBackOp>,
               ttkernel::D2MCBReleaseOpRewriter<d2m::PushOp, ttkernel::CBPushBackOp>,
               ttkernel::D2MCBReleaseOpRewriter<d2m::PopOp, ttkernel::CBPopFrontOp>,
               ttkernel::D2MDMAWaitRewriter,
               ttkernel::D2MCoreIndexRewriter,
               ttkernel::D2MNullTxRewriter,
               ttkernel::MemRefCollapseRewriter,
               ttkernel::D2MSemaphoreUpdateRewriter<d2m::SemaphoreSetOp>,
               ttkernel::D2MSemaphoreUpdateRewriter<d2m::SemaphoreIncOp>,
               ttkernel::D2MSemaphoreWaitRewriter>(typeConverter, ctx);

  patterns.add<ttkernel::D2MGetGlobalOperandRewriter>(typeConverter, ctx, ttnnMode);
  patterns.add<ttkernel::D2MDMAReadRewriter>(typeConverter, ctx, &associatedDMAWaits, &cbProducerConsumer);
  patterns.add<ttkernel::D2MDMAWriteRewriter>(typeConverter, ctx, &associatedDMAWaits, &cbProducerConsumer);

  // This is needed to lower affine apply ops that may be generated when
  // `d2m.core_index` is used with a `phys_to_virt_map`.
  populateAffineToStdConversionPatterns(patterns);
  // clang-format on
}

} // namespace mlir::tt
