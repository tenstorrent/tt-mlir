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
  return arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                   rewriter.getI32IntegerAttr(value))
      .getResult();
}

static Value index(OpBuilder &rewriter, Location loc, int64_t value) {
  return arith::ConstantOp::create(rewriter, loc, rewriter.getIndexType(),
                                   rewriter.getIndexAttr(value))
      .getResult();
}

static std::pair<Value, Value>
getVirtualCoordsFromLogicalCoords(OpBuilder &rewriter, Location loc,
                                  ttcore::ChipDescAttr chipDesc,
                                  ValueRange dstCoreIndex) {
  Value virtY = ttkernel::ConvertLogicalYToTranslatedOp::create(
      rewriter, dstCoreIndex[0].getLoc(), dstCoreIndex[0].getType(),
      dstCoreIndex[0]);
  Value virtX = ttkernel::ConvertLogicalXToTranslatedOp::create(
      rewriter, dstCoreIndex[1].getLoc(), dstCoreIndex[1].getType(),
      dstCoreIndex[1]);
  return {virtY, virtX};
}

static std::pair<Value, Value> getMcastEndCoords(PatternRewriter &rewriter,
                                                 Location loc,
                                                 const Value &nocStartY,
                                                 const Value &nocStartX,
                                                 OperandRange mcastShape) {
  return {
      arith::SubIOp::create(rewriter, nocStartY.getLoc(),
                            arith::AddIOp::create(rewriter, nocStartY.getLoc(),
                                                  nocStartY, mcastShape[0]),
                            index(rewriter, loc, 1)),
      arith::SubIOp::create(rewriter, nocStartX.getLoc(),
                            arith::AddIOp::create(rewriter, nocStartX.getLoc(),
                                                  nocStartX, mcastShape[1]),
                            index(rewriter, loc, 1))};
}

static Value getCB(ConversionPatternRewriter &rewriter, Value cb) {
  if (auto loadOp = cb.getDefiningOp<memref::LoadOp>()) {
    assert(loadOp.getIndices().size() == 1 &&
           "Expected single index in load op, failing.");
    return rewriter.getRemappedValue(loadOp.getMemRef());
  }

  if (auto affLoad = cb.getDefiningOp<affine::AffineLoadOp>()) {
    return rewriter.getRemappedValue(affLoad.getMemRef());
  }

  if (auto subViewOp = cb.getDefiningOp<memref::SubViewOp>()) {
    return rewriter.getRemappedValue(subViewOp.getSource());
  }

  if (auto castOp = cb.getDefiningOp<memref::CastOp>()) {
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

// Ensure that `value` (and any same-block transitive dependencies) dominates
// the rewriter's current insertion point by moving defining ops upward if
// needed. This is necessary because getDstIdxFromResult grabs the store index
// from a memref.store that may appear *after* the compute op being lowered.
// When the new TTKernel op is created at the compute op's position, the index
// value must already be defined.
static void ensureDominatesInsertionPoint(OpBuilder &rewriter, Value value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return; // Block argument - always dominates.
  }

  Block *insertBlock = rewriter.getInsertionBlock();
  if (defOp->getBlock() != insertBlock) {
    // Two cases here:
    // 1. defOp is in a parent block -> so it dominates the current block
    // 2. defOp is in an actual non-dominating block, but this can't happen
    // since the op we are pulling the value from already uses the value (in the
    // same block), so the IR is already invalid if this is the case. In either
    // case, we can return since we already know the value dominates the
    // insertion point.
    return;
  }

  Block::iterator ip = rewriter.getInsertionPoint();
  if (ip == insertBlock->end()) {
    return; // Inserting at end - everything in the block dominates.
  }

  if (defOp->isBeforeInBlock(&*ip)) {
    return; // Already dominates the insertion point.
  }

  // Recursively ensure operands dominate first.
  for (Value operand : defOp->getOperands()) {
    ensureDominatesInsertionPoint(rewriter, operand);
  }

  defOp->moveBefore(insertBlock, rewriter.getInsertionPoint());
}

// Remapped L1 CB for a tile loaded from L1; null if the value is not an L1 load
// (including DST-only producers such as fill_tile).
static Value tryGetL1CbFromTensorShardLoad(ConversionPatternRewriter &rewriter,
                                           Value v) {
  Value memref;
  if (auto load = v.getDefiningOp<memref::LoadOp>()) {
    memref = load.getMemRef();
  } else if (auto load = v.getDefiningOp<affine::AffineLoadOp>()) {
    memref = load.getMemRef();
  } else {
    return nullptr;
  }
  if (ttcore::getMemorySpace(memref) != ttcore::MemorySpace::DeviceL1) {
    return nullptr;
  }
  return rewriter.getRemappedValue(memref);
}

// Walk `func` for the first L1 memref touched by `OpTy` (load or store).
template <typename OpTy>
static Value findFirstL1CbForOpKind(func::FuncOp func) {
  Value cb;
  func.walk([&](OpTy accessOp) {
    Value m = accessOp.getMemRef();
    if (ttcore::getMemorySpace(m) == ttcore::MemorySpace::DeviceL1) {
      cb = m;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return cb;
}

// First memref load/store wins; memref dialect ops are scanned before affine.
// Used when operand-local CB resolution is insufficient.
static Value findFirstL1CbMemref(func::FuncOp func, bool forInput) {
  if (forInput) {
    if (Value cb = findFirstL1CbForOpKind<memref::LoadOp>(func)) {
      return cb;
    }
    return findFirstL1CbForOpKind<affine::AffineLoadOp>(func);
  }
  if (Value cb = findFirstL1CbForOpKind<memref::StoreOp>(func)) {
    return cb;
  }
  return findFirstL1CbForOpKind<affine::AffineStoreOp>(func);
}

static Value getRemappedFirstL1Cb(ConversionPatternRewriter &rewriter,
                                  Operation *op, bool forInput) {
  func::FuncOp func = op->getParentOfType<func::FuncOp>();
  assert(func && "Expected func op.");
  Value cb = findFirstL1CbMemref(func, forInput);
  assert(cb && "CB not found.");
  return rewriter.getRemappedValue(cb);
}

static Value getInCB(ConversionPatternRewriter &rewriter, Operation *op) {
  return getRemappedFirstL1Cb(rewriter, op, /*forInput=*/true);
}

static Value getOutCB(ConversionPatternRewriter &rewriter, Operation *op) {
  return getRemappedFirstL1Cb(rewriter, op, /*forInput=*/false);
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
    // We have blocked this input. We need to get the indices for the first
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
        arith::MulIOp::create(rewriter, op.getLoc(), rtIdx, ktIdx);

    // Convert the resolved source row offset to a block-row index.
    Value rowBlockIdx =
        arith::DivSIOp::create(rewriter, op.getLoc(), sourceIndices[0], rtIdx);
    Value rowBase = arith::MulIOp::create(rewriter, op.getLoc(), rowBlockIdx,
                                          tilesPerBlock);
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
    ttkernel::TileRegsAcquireOp::create(rewriter, op.getLoc());
    // Dst is an implicit resource in TTKernel, so we can just erase it.
    rewriter.eraseOp(op);
    return success();
  };
};
} // namespace

namespace {
class UnpackStallOnPackRewriter
    : public OpConversionPattern<d2m::UnpackStallOnPackOp> {
public:
  using OpConversionPattern<d2m::UnpackStallOnPackOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::UnpackStallOnPackOp op, d2m::UnpackStallOnPackOpAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttkernel::UnpackStallOnPackOp>(op);
    return success();
  }
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
        Value zero = arith::ConstantOp::create(builder, returnOp.getLoc(),
                                               builder.getI32Type(),
                                               builder.getI32IntegerAttr(0));
        ttkernel::PackReconfigL1AccOp::create(builder, returnOp.getLoc(), zero);
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

  Value linearIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
  for (size_t i = 0; i < indices.size(); ++i) {
    int64_t stride = 1;
    for (size_t j = i + 1; j < shape.size(); ++j) {
      stride *= shape[j];
    }

    Value contribution = indices[i];
    if (stride != 1) {
      auto strideVal = arith::ConstantIndexOp::create(rewriter, loc, stride);
      contribution =
          arith::MulIOp::create(rewriter, loc, indices[i], strideVal);
    }
    linearIdx = arith::AddIOp::create(rewriter, loc, linearIdx, contribution);
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
    ttkernel::InitSFPUOp::create(rewriter, store.getLoc(), inCB, outCB);
    rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);

    ttkernel::CopyTileInitOp::create(rewriter, store.getLoc(), cb);
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
  std::pair<d2m::TileReduceMeanOp,  std::pair<ttkernel::ComputeKernelHWStartupOp,  ttkernel::ReduceTileOp>>,

  // Elementwise SFPU Unary.
  std::pair<d2m::TileAbsOp,         std::pair<ttkernel::AbsTileInitOp,             ttkernel::AbsTileOp>>,
  std::pair<d2m::TileAcosOp,        std::pair<ttkernel::AcosTileInitOp,            ttkernel::AcosTileOp>>,
  std::pair<d2m::TileAsinOp,        std::pair<ttkernel::AsinTileInitOp,            ttkernel::AsinTileOp>>,
  std::pair<d2m::TileAtanOp,        std::pair<ttkernel::AtanTileInitOp,            ttkernel::AtanTileOp>>,
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
  std::pair<d2m::TileLogicalNotOp,  std::pair<ttkernel::LogicalNotTileInitOp,      ttkernel::LogicalNotTileOp>>,
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
// Int32 variants of compute ops. Ops not listed here have no int32 variant.
using IntComputeOpMap = OpMap<
  // Unary SFPU (init unchanged, only the tile op differs).
  std::pair<d2m::TileAbsOp,        std::pair<ttkernel::AbsTileInitOp,             ttkernel::AbsTileI32Op>>,
  std::pair<d2m::TileNegativeOp,   std::pair<ttkernel::NegativeTileInitOp,        ttkernel::NegativeTileInt32Op>>,
  std::pair<d2m::TileReluOp,       std::pair<ttkernel::ReluTileInitOp,            ttkernel::ReluTileI32Op>>,

  // Compare-to-zero SFPU (init unchanged, only the tile op differs).
  std::pair<d2m::TileEqzOp,        std::pair<ttkernel::EqzTileInitOp,             ttkernel::EqzTileI32Op>>,
  std::pair<d2m::TileNezOp,        std::pair<ttkernel::NezTileInitOp,             ttkernel::NezTileI32Op>>,
  std::pair<d2m::TileGtzOp,        std::pair<ttkernel::GtzTileInitOp,             ttkernel::GtzTileI32Op>>,
  std::pair<d2m::TileGezOp,        std::pair<ttkernel::GezTileInitOp,             ttkernel::GezTileI32Op>>,
  std::pair<d2m::TileLtzOp,        std::pair<ttkernel::LtzTileInitOp,             ttkernel::LtzTileI32Op>>,
  std::pair<d2m::TileLezOp,        std::pair<ttkernel::LezTileInitOp,             ttkernel::LezTileI32Op>>,

  // Binary SFPU (both init and tile op differ).
  std::pair<d2m::TileAddOp,        std::pair<ttkernel::AddIntTileInitOp,          ttkernel::AddIntTileOp>>,
  std::pair<d2m::TileSubOp,        std::pair<ttkernel::SubIntTileInitOp,          ttkernel::SubIntTileOp>>,
  std::pair<d2m::TileMulOp,        std::pair<ttkernel::MulIntTileInitOp,          ttkernel::MulIntTileOp>>,
  std::pair<d2m::TileMaximumOp,    std::pair<ttkernel::BinaryMaxInt32TileInitOp,  ttkernel::BinaryMaxInt32TileOp>>,
  std::pair<d2m::TileMinimumOp,    std::pair<ttkernel::BinaryMinInt32TileInitOp,  ttkernel::BinaryMinInt32TileOp>>
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

template <typename SrcOp, typename Map>
constexpr bool hasMapping =
    !std::is_same_v<TTKernelOpPair<SrcOp, Map>, std::pair<void, void>>;

// Some int32 TTKernel ops require an explicit dtype argument.
template <typename Op>
constexpr bool needsDtypeArg = std::is_same_v<Op, ttkernel::MulIntTileInitOp> ||
                               std::is_same_v<Op, ttkernel::AddIntTileOp> ||
                               std::is_same_v<Op, ttkernel::SubIntTileOp> ||
                               std::is_same_v<Op, ttkernel::MulIntTileOp>;

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
      ttkernel::BinaryOpInitCommonOp::create(rewriter, op->getLoc(), cbA, cbB,
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
        ttkernel::MatmulInitOp::create(rewriter, op->getLoc(), cbA, cbB, outCB,
                                       transpose);
      }

      auto transpose = i32(rewriter, op->getLoc(), 0);
      rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);
      ttkernel::MatmulInitShortOp::create(rewriter, op->getLoc(), cbA, cbB,
                                          transpose);
      ttkernel::MatmulTilesOp::create(rewriter, op->getLoc(), cbA, cbB,
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

      ttkernel::MatmulBlockInitOp::create(rewriter, op->getLoc(), cbA, cbB,
                                          outCB, transpose, ct_i32, rt_i32,
                                          kt_i32);
      rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);
      ttkernel::MatmulBlockInitShortOp::create(
          rewriter, op->getLoc(), cbA, cbB, transpose, ct_i32, rt_i32, kt_i32);

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

      ttkernel::ExperimentalMatmulBlockOp::create(
          rewriter, op->getLoc(), cbA, cbB, aTileIndex, bTileIndex, destIndex,
          transpose, ct_i32, rt_i32, kt_i32, nt_i32);
    } else if constexpr (std::is_same_v<ConcreteOp, d2m::TileReduceSumOp> ||
                         std::is_same_v<ConcreteOp, d2m::TileReduceMaxOp> ||
                         std::is_same_v<ConcreteOp, d2m::TileReduceMeanOp>) {
      ttkernel::ReduceType reduce_type;
      d2m::ReduceDim reduce_dim = op.getReduceDim();
      if constexpr (std::is_same_v<ConcreteOp, d2m::TileReduceSumOp>) {
        reduce_type = ttkernel::ReduceType::Sum;
      } else if constexpr (std::is_same_v<ConcreteOp, d2m::TileReduceMaxOp>) {
        reduce_type = ttkernel::ReduceType::Max;
      } else {
        reduce_type = ttkernel::ReduceType::Avg;
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
      ttkernel::ComputeKernelHWStartupOp::create(rewriter, op->getLoc(), cbA,
                                                 cbB, outCB);
      rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);
      ttkernel::ReduceInitOp::create(rewriter, op->getLoc(), cbA, cbB, outCB,
                                     reduce_type, kernel_reduce_dim);
      ttkernel::ReduceTileOp::create(
          rewriter, op->getLoc(), cbA, cbB, adaptor.getA(), adaptor.getB(),
          adaptor.getC(), reduce_type, kernel_reduce_dim);
      ttkernel::ReduceUninitOp::create(rewriter, op->getLoc());
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
      ensureDominatesInsertionPoint(rewriter, dstIdx);
      rewriter.create<ttkernel::UnaryBcastInitOp>(op->getLoc(), cb, cb,
                                                  bcastType);
      rewriter.create<ttkernel::UnaryBcastTileOp>(
          op->getLoc(), cb, adaptor.getInput(), dstIdx, bcastType);
    } else if constexpr (arity == 2) {
      auto dstIdx = getDstIdxFromResult(op.getResult());
      ensureDominatesInsertionPoint(rewriter, dstIdx);
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
    Value outCB = getOutCB(rewriter, op);
    Value inCB;
    if constexpr (arity == 1) {
      inCB = tryGetL1CbFromTensorShardLoad(rewriter, op.getInput());
      if (!inCB) {
        // DST-only tile (e.g. after fill_tile): no L1 CB; reuse output CB for
        // init_sfpu.
        inCB = outCB;
      }
    } else {
      inCB = getInCB(rewriter, op);
    }
    rewriter.setInsertionPointToStart(rewriter.getInsertionBlock());
    setInsertionPointAfterOperands(rewriter, {inCB, outCB},
                                   /*allowHoisting*/ true);
    ttkernel::InitSFPUOp::create(rewriter, op->getLoc(), inCB, outCB);
    rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);

    // For binary ops (arity == 2), check if rhs is a scalar to create the right
    // init op
    if constexpr (arity == 2) {
      auto rhsType = adaptor.getRhs().getType();
      bool isScalarRhs = rhsType.isIntOrFloat();

      if (isScalarRhs) {
        // Use scalar-specific init ops
        if constexpr (std::is_same_v<ConcreteOp, d2m::TilePowOp>) {
          ttkernel::PowerTileInitOp::create(rewriter, op->getLoc());
        } else {
          ttkernel::BinopWithScalarTileInitOp::create(rewriter, op->getLoc());
        }
      } else if constexpr (hasMapping<ConcreteOp, IntComputeOpMap>) {
        using IntInit =
            typename TTKernelOpPair<ConcreteOp, IntComputeOpMap>::first_type;
        const auto tileType =
            mlir::cast<ttcore::TileType>(op.getLhs().getType());
        if (llvm::isa<IntegerType>(tileType.getElementType())) {
          if constexpr (needsDtypeArg<IntInit>) {
            IntInit::create(rewriter, op->getLoc(), tileType.getDataType());
          } else {
            IntInit::create(rewriter, op->getLoc());
          }
        } else {
          InitOp::create(rewriter, op->getLoc());
        }
      } else {
        InitOp::create(rewriter, op->getLoc());
      }
    } else if constexpr (std::is_same_v<InitOp, ttkernel::TypecastTileInitOp>) {
      const auto inDtype =
          mlir::cast<ttcore::TileType>(op.getInput().getType()).getDataType();
      const auto outDtype =
          mlir::cast<ttcore::TileType>(op.getResult().getType()).getDataType();
      ttkernel::TypecastTileInitOp::create(rewriter, op->getLoc(), inDtype,
                                           outDtype);
    } else {
      InitOp::create(rewriter, op->getLoc());
    }

    if constexpr (std::is_same_v<SFPUOp, ttkernel::LogicalNotTileOp>) {
      const auto dtype =
          mlir::cast<ttcore::TileType>(op.getInput().getType()).getDataType();
      ttkernel::LogicalNotTileOp::create(rewriter, op->getLoc(),
                                         adaptor.getInput(), dtype);
    } else if constexpr (std::is_same_v<SFPUOp, ttkernel::TypecastTileOp>) {
      const auto inDtype =
          mlir::cast<ttcore::TileType>(op.getInput().getType()).getDataType();
      const auto outDtype =
          mlir::cast<ttcore::TileType>(op.getResult().getType()).getDataType();
      ttkernel::TypecastTileOp::create(rewriter, op->getLoc(),
                                       adaptor.getInput(), inDtype, outDtype);
    } else if constexpr (std::is_same_v<SFPUOp, ttkernel::ClampScalarTileOp>) {
      auto loc = op->getLoc();
      // The hardware clamp API takes i32 params for both int and float clamps.
      // For floats, the raw IEEE 754 bits are passed via bitcast.
      auto minAttr = op.getMinAttr();
      auto maxAttr = op.getMaxAttr();
      if (mlir::isa<IntegerAttr>(minAttr) && mlir::isa<IntegerAttr>(maxAttr)) {
        auto intToI32Param = [&](Attribute attr) -> Value {
          auto intAttr = mlir::cast<IntegerAttr>(attr);
          return arith::ConstantOp::create(
              rewriter, loc, rewriter.getI32Type(),
              rewriter.getI32IntegerAttr(intAttr.getValue().getSExtValue()));
        };
        auto minParam = intToI32Param(minAttr);
        auto maxParam = intToI32Param(maxAttr);
        ttkernel::ClampScalarTileInt32Op::create(
            rewriter, loc, adaptor.getInput(), minParam, maxParam);
      } else {
        auto floatToI32Param = [&](Attribute attr) -> Value {
          auto floatAttr = mlir::cast<FloatAttr>(attr);
          auto f32Val = arith::ConstantOp::create(
              rewriter, loc,
              rewriter.getF32FloatAttr(floatAttr.getValue().convertToDouble()));
          return arith::BitcastOp::create(rewriter, loc, rewriter.getI32Type(),
                                          f32Val);
        };
        auto minParam = floatToI32Param(minAttr);
        auto maxParam = floatToI32Param(maxAttr);
        ttkernel::ClampScalarTileOp::create(rewriter, loc, adaptor.getInput(),
                                            minParam, maxParam);
      }
    } else if constexpr (arity == 1 &&
                         hasMapping<ConcreteOp, IntComputeOpMap>) {
      using IntSFPUOp =
          typename TTKernelOpPair<ConcreteOp, IntComputeOpMap>::second_type;
      assert(!needsDtypeArg<IntSFPUOp> &&
             "Unary int32 ops should not need dtype arg");
      const auto elemType =
          mlir::cast<ttcore::TileType>(op.getInput().getType())
              .getElementType();
      if (llvm::isa<IntegerType>(elemType)) {
        IntSFPUOp::create(rewriter, op->getLoc(), adaptor.getInput());
      } else {
        SFPUOp::create(rewriter, op->getLoc(), adaptor.getInput());
      }
    } else if constexpr (arity == 1) {
      SFPUOp::create(rewriter, op->getLoc(), adaptor.getInput());
    } else if constexpr (arity == 2) {
      // Check if rhs is a scalar (float or integer) at runtime
      auto rhsType = adaptor.getRhs().getType();
      bool isScalarRhs = rhsType.isIntOrFloat();

      if (isScalarRhs) {
        // Handle scalar operand - need to use unary scalar ops
        const auto dstIdx = adaptor.getLhs();
        auto loc = op->getLoc();
        auto scalarToI32Param = [&](Value scalar) -> Value {
          auto scalarType = mlir::cast<Type>(scalar.getType());
          if (auto floatType = llvm::dyn_cast<FloatType>(scalarType)) {
            Value f32Scalar = scalar;
            if (!floatType.isF32()) {
              f32Scalar = rewriter.create<arith::ExtFOp>(
                  loc, rewriter.getF32Type(), scalar);
            }
            return rewriter.create<arith::BitcastOp>(loc, rewriter.getI32Type(),
                                                     f32Scalar);
          }

          // Integer scalars are passed as i32 numeric values.
          if (scalarType.isInteger(32)) {
            return scalar;
          }
          if (auto intType = llvm::dyn_cast<IntegerType>(scalarType)) {
            if (intType.getWidth() < 32) {
              return rewriter.create<arith::ExtSIOp>(loc, rewriter.getI32Type(),
                                                     scalar);
            }
            return rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(),
                                                    scalar);
          }

          llvm_unreachable("Expected scalar rhs to be integer or float");
        };

        // Create the appropriate unary scalar op based on the D2M op type
        if constexpr (std::is_same_v<ConcreteOp, d2m::TileAddOp>) {
          rewriter.create<ttkernel::BinopWithScalarTileInitOp>(loc);
          auto scalarParam = scalarToI32Param(adaptor.getRhs());
          rewriter.create<ttkernel::AddUnaryTileOp>(loc, dstIdx, scalarParam);
        } else if constexpr (std::is_same_v<ConcreteOp, d2m::TileSubOp>) {
          rewriter.create<ttkernel::BinopWithScalarTileInitOp>(loc);
          auto scalarParam = scalarToI32Param(adaptor.getRhs());
          rewriter.create<ttkernel::SubUnaryTileOp>(loc, dstIdx, scalarParam);
        } else if constexpr (std::is_same_v<ConcreteOp, d2m::TileMulOp>) {
          rewriter.create<ttkernel::BinopWithScalarTileInitOp>(loc);
          auto scalarParam = scalarToI32Param(adaptor.getRhs());
          rewriter.create<ttkernel::MulUnaryTileOp>(loc, dstIdx, scalarParam);
        } else if constexpr (std::is_same_v<ConcreteOp, d2m::TileDivOp>) {
          auto scalarParam = scalarToI32Param(adaptor.getRhs());
          rewriter.create<ttkernel::DivUnaryTileOp>(loc, dstIdx, scalarParam);
        } else if constexpr (std::is_same_v<ConcreteOp, d2m::TilePowOp>) {
          // For power, convert float value to integer (not bitcast)
          auto scalarParam = arith::FPToSIOp::create(
              rewriter, loc, rewriter.getI32Type(), adaptor.getRhs());
          ttkernel::PowUnaryTileOp::create(rewriter, loc, dstIdx, scalarParam);
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
        SFPUOp::create(rewriter, op->getLoc(), adaptor.getLhs(),
                       adaptor.getRhs(), dstIdx, dtype);
      } else if constexpr (hasMapping<ConcreteOp, IntComputeOpMap>) {
        using IntSFPUOp =
            typename TTKernelOpPair<ConcreteOp, IntComputeOpMap>::second_type;
        const auto tileType =
            mlir::cast<ttcore::TileType>(op.getLhs().getType());
        if (llvm::isa<IntegerType>(tileType.getElementType())) {
          if constexpr (needsDtypeArg<IntSFPUOp>) {
            IntSFPUOp::create(rewriter, op->getLoc(), adaptor.getLhs(),
                              adaptor.getRhs(), dstIdx, tileType.getDataType());
          } else {
            IntSFPUOp::create(rewriter, op->getLoc(), adaptor.getLhs(),
                              adaptor.getRhs(), dstIdx);
          }
        } else {
          SFPUOp::create(rewriter, op->getLoc(), adaptor.getLhs(),
                         adaptor.getRhs(), dstIdx);
        }
      } else {
        SFPUOp::create(rewriter, op->getLoc(), adaptor.getLhs(),
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
        ttkernel::WhereTileOp::create(
            rewriter, op->getLoc(), adaptor.getCondition(),
            adaptor.getTrueValue(), adaptor.getFalseValue(), dstIdx, dtype);
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

    // Scalar operand: fallback to SFPU.
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

    // Integer ops use SFPU with explicit copy_tile from CB to DST.
    if constexpr (hasMapping<ConcreteOp, IntComputeOpMap>) {
      const auto elemType =
          mlir::cast<ttcore::TileType>(op.getLhs().getType()).getElementType();
      if (llvm::isa<IntegerType>(elemType)) {
        return emitIntegerBinaryTiles(rewriter, op, adaptor, loc, cbA, cbB,
                                      outCB);
      }
    }

    auto insertionPoint = rewriter.getInsertionPoint();
    setInsertionPointAfterOperands(rewriter, {cbA, cbB, outCB},
                                   /*allowHoisting*/ true);
    ttkernel::BinaryOpInitCommonOp::create(rewriter, loc, cbA, cbB, outCB);
    rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);

    auto dstIdx = getDstIdxFromResult(op.getResult());
    ensureDominatesInsertionPoint(rewriter, dstIdx);

    if constexpr (std::is_same_v<ConcreteOp, d2m::TileAddOp>) {
      ttkernel::AddTilesInitOp::create(rewriter, loc, cbA, cbB);
      ttkernel::AddTilesOp::create(rewriter, loc, cbA, cbB, adaptor.getLhs(),
                                   adaptor.getRhs(), dstIdx);
    } else if constexpr (std::is_same_v<ConcreteOp, d2m::TileSubOp>) {
      ttkernel::SubTilesInitOp::create(rewriter, loc, cbA, cbB);
      ttkernel::SubTilesOp::create(rewriter, loc, cbA, cbB, adaptor.getLhs(),
                                   adaptor.getRhs(), dstIdx);
    } else if constexpr (std::is_same_v<ConcreteOp, d2m::TileMulOp>) {
      ttkernel::MulTilesInitOp::create(rewriter, loc, cbA, cbB);
      ttkernel::MulTilesOp::create(rewriter, loc, cbA, cbB, adaptor.getLhs(),
                                   adaptor.getRhs(), dstIdx);
    }

    rewriter.eraseOp(op);
    return success();
  }

  // Integer binary: copy tiles from CB to DST, then use the int32 SFPU op.
  LogicalResult emitIntegerBinaryTiles(ConversionPatternRewriter &rewriter,
                                       ConcreteOp op,
                                       typename ConcreteOp::Adaptor adaptor,
                                       Location loc, Value cbA, Value cbB,
                                       Value outCB) const {
    using IntPair = TTKernelOpPair<ConcreteOp, IntComputeOpMap>;
    using IntInit = typename IntPair::first_type;
    using IntSFPUOp = typename IntPair::second_type;

    auto dst0 = index(rewriter, loc, 0);
    auto dst1 = index(rewriter, loc, 1);

    auto insertionPoint = rewriter.getInsertionPoint();
    setInsertionPointAfterOperands(rewriter, {cbA, outCB},
                                   /*allowHoisting*/ true);
    ttkernel::InitSFPUOp::create(rewriter, loc, cbA, outCB);
    rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);

    ttkernel::CopyTileInitOp::create(rewriter, loc, cbA);
    ttkernel::CopyTileOp::create(rewriter, loc, cbA, adaptor.getLhs(), dst0);
    ttkernel::CopyTileInitOp::create(rewriter, loc, cbB);
    ttkernel::CopyTileOp::create(rewriter, loc, cbB, adaptor.getRhs(), dst1);

    const auto dtype =
        mlir::cast<ttcore::TileType>(op.getLhs().getType()).getDataType();
    if constexpr (needsDtypeArg<IntInit>) {
      IntInit::create(rewriter, loc, dtype);
    } else {
      IntInit::create(rewriter, loc);
    }
    if constexpr (needsDtypeArg<IntSFPUOp>) {
      IntSFPUOp::create(rewriter, loc, dst0, dst1, dst0, dtype);
    } else {
      IntSFPUOp::create(rewriter, loc, dst0, dst1, dst0);
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
    ensureDominatesInsertionPoint(rewriter, dstIdx);

    auto insertionPoint = rewriter.getInsertionPoint();
    setInsertionPointAfterOperands(rewriter, {cb, outCB},
                                   /*allowHoisting*/ true);
    ttkernel::BinaryOpInitCommonOp::create(rewriter, loc, cb, cb, outCB);
    rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);

    auto eltwiseType = getEltwiseBinaryType();

    // binary_dest_reuse is an in-place operation. If the DST
    // operand comes from a different slot, copy it first to the output slot.
    if (dstOperandIdx != dstIdx) {
      ttkernel::CopyDestValuesInitOp::create(rewriter, loc);
      ttkernel::CopyDestValuesOp::create(rewriter, loc, dstOperandIdx, dstIdx);
    }

    ttkernel::BinaryDestReuseTilesInitOp::create(rewriter, loc, cb, eltwiseType,
                                                 reuseType);
    ttkernel::BinaryDestReuseTilesOp::create(rewriter, loc, cb, cbTileIdx,
                                             dstIdx, eltwiseType, reuseType);

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
    ttkernel::ComputeKernelHWStartupOp::create(rewriter, op->getLoc(), src,
                                               nullptr, dst);

    if constexpr (std::is_same_v<BlockOp,
                                 ttkernel::ExperimentalTilizeBlockOp>) {
      ttkernel::TilizeInitOp::create(rewriter, op->getLoc(), src, blockC, dst);
      BlockOp::create(rewriter, op->getLoc(), src, dst, blockR, blockC);
    } else if constexpr (std::is_same_v<
                             BlockOp,
                             ttkernel::ExperimentalPackUntilizeBlockOp>) {
      const int64_t totalColTiles = collapsed2DShape[1];

      auto chipDesc = ttcore::getOpChipDescAttr(op);
      auto tileType = mlir::cast<ttcore::TileType>(
          preLinearizedMemrefType.getElementType());
      auto scalarType = ttcore::dataTypeToElementType(rewriter.getContext(),
                                                      tileType.getDataType());
      const int64_t dstCapacity =
          chipDesc.getDstLogicalSizeTiles(scalarType, /*fullSyncEn=*/false);

      // cols_per_dst_pass must divide total_col_tiles and fit in DST.
      int64_t colsPerDstPass = std::min(dstCapacity, totalColTiles);
      while (colsPerDstPass > 1 && totalColTiles % colsPerDstPass != 0) {
        colsPerDstPass--;
      }
      auto colsPerDstPassAttr =
          rewriter.getI32IntegerAttr(static_cast<int32_t>(colsPerDstPass));
      auto totalColTilesAttr =
          rewriter.getI32IntegerAttr(static_cast<int32_t>(totalColTiles));

      ttkernel::PackUntilizeInitOp::create(rewriter, op->getLoc(), src, dst,
                                           colsPerDstPassAttr,
                                           totalColTilesAttr);
      BlockOp::create(rewriter, op->getLoc(), src, dst, blockR, blockC,
                      colsPerDstPassAttr, totalColTilesAttr);
      ttkernel::PackUntilizeUninitOp::create(rewriter, op->getLoc(), dst);
    } else {
      llvm_unreachable("unsupported tilize/untilize op");
    }

    rewriter.eraseOp(op);

    return success();
  };
};

// Coerce `d2m.fill_tile` scalar to i32 (FillTileIntOp) or f32 (FillTileOp).
static LogicalResult materializeFillTileKernelValue(
    Location loc, d2m::FillTileOp op, Value &fillValue,
    ConversionPatternRewriter &rewriter, bool &useIntFill) {
  Type ty = fillValue.getType();
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    if (!intTy.isInteger(32)) {
      Type i32Ty = rewriter.getI32Type();
      fillValue = intTy.getWidth() < 32
                      ? rewriter.create<arith::ExtSIOp>(loc, i32Ty, fillValue)
                            .getResult()
                      : rewriter.create<arith::TruncIOp>(loc, i32Ty, fillValue)
                            .getResult();
    }
    useIntFill = true;
    return success();
  }
  if (!isa<FloatType>(ty)) {
    return rewriter.notifyMatchFailure(
        op, "fill_tile value must be a float or integer type");
  }
  if (!ty.isF32()) {
    fillValue =
        rewriter.create<arith::ExtFOp>(loc, rewriter.getF32Type(), fillValue)
            .getResult();
  }
  useIntFill = false;
  return success();
}

class D2MFillTileRewriter : public OpConversionPattern<d2m::FillTileOp> {
public:
  using OpConversionPattern<d2m::FillTileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::FillTileOp op, d2m::FillTileOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value dstIdx = getDstIdxFromResult(op.getResult());
    ensureDominatesInsertionPoint(rewriter, dstIdx);

    Location loc = op->getLoc();
    // No L1 input: HW startup uses the output CB for both unpack and pack.
    Value outCB = getOutCB(rewriter, op);
    auto insertionPoint = rewriter.getInsertionPoint();
    setInsertionPointAfterOperands(rewriter, {outCB}, /*allowHoisting*/ true);
    rewriter.create<ttkernel::ComputeKernelHWStartupOp>(loc, outCB, nullptr,
                                                        outCB);
    rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);

    Value fillValue = adaptor.getValue();
    bool useIntFill = false;
    if (failed(materializeFillTileKernelValue(loc, op, fillValue, rewriter,
                                              useIntFill))) {
      return failure();
    }

    rewriter.create<ttkernel::FillTileInitOp>(loc);
    if (useIntFill) {
      rewriter.create<ttkernel::FillTileIntOp>(loc, dstIdx, fillValue);
    } else {
      rewriter.create<ttkernel::FillTileOp>(loc, dstIdx, fillValue);
    }

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
      validRows = arith::IndexCastOp::create(rewriter, loc,
                                             rewriter.getI32Type(), validRows);
    }
    ttkernel::ExperimentalWriteRowMaskTileOp::create(rewriter, loc, validRows,
                                                     adaptor.getOutput());
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
      validCols = arith::IndexCastOp::create(rewriter, loc,
                                             rewriter.getI32Type(), validCols);
    }
    ttkernel::ExperimentalWriteColMaskTileOp::create(rewriter, loc, validCols,
                                                     adaptor.getOutput());
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
    ttkernel::ExperimentalFillArangeTileOp::create(rewriter, op->getLoc(),
                                                   adaptor.getOutput());
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
    ttkernel::TransposeInitOp::create(rewriter, op->getLoc(), inCB, outCB);
    rewriter.setInsertionPoint(insertionPoint->getBlock(), insertionPoint);

    // Get the tile index from the input operand.
    Value tileIndex = adaptor.getInput();

    // Get the destination index where the result will be stored.
    Value dstIdx = getDstIdxFromResult(op.getResult());
    ensureDominatesInsertionPoint(rewriter, dstIdx);

    ttkernel::TransposeTileOp::create(rewriter, op->getLoc(), inCB, tileIndex,
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

    TTKernelAcquireOp::create(rewriter, op.getLoc(), adaptor.getCb(), numPages);

    // Only insert automatic release if there's no explicit push/pop
    if (!hasExplicitRelease(op)) {
      Block *block = op->getBlock();
      auto release = TTKernelReleaseOp::create(rewriter, op.getLoc(),
                                               adaptor.getCb(), numPages);
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
  return UnrealizedConversionCastOp::create(rewriter, loc,
                                            rewriter.getI32Type(), cb)
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
    auto offsetInt = arith::IndexCastOp::create(rewriter, loc,
                                                rewriter.getI32Type(), offset);
    auto addr = arith::AddIOp::create(rewriter, loc, baseAddr, offsetInt);
    // Translate the src coordinates to virtual coordinates.
    auto [virtY, virtX] = getVirtualCoordsFromLogicalCoords(
        rewriter, loc, chipDesc, ValueRange{gridY, gridX});
    noc_addr_op =
        ttkernel::GetNocAddrOp::create(rewriter, loc, virtX, virtY, addr);
  } else {
    auto bankID = index[1];
    auto bankIDInt = arith::IndexCastOp::create(rewriter, loc,
                                                rewriter.getI32Type(), bankID);
    auto offset = index[2];
    auto offsetInt = arith::IndexCastOp::create(rewriter, loc,
                                                rewriter.getI32Type(), offset);
    auto addr = arith::AddIOp::create(rewriter, loc, baseAddr, offsetInt);

    return ttkernel::GetNocAddrFromBankIDOp::create(rewriter, loc, bankIDInt,
                                                    addr);
  }
  return noc_addr_op;
}

template <typename ReadWritePtrOp>
static Value buildL1Address(OpBuilder &rewriter, Location loc, Value cb,
                            ValueRange index) {
  // Use the cb addr as the write address since it is local.
  Value baseAddr = ReadWritePtrOp::create(rewriter, loc, cb);
  auto offset = arith::IndexCastOp::create(rewriter, loc, rewriter.getI32Type(),
                                           index[0]);
  return arith::AddIOp::create(rewriter, loc, baseAddr, offset);
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
    ttkernel::NocAsyncReadOp::create(rewriter, op.getLoc(), srcNocAddr,
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
        srcL1Start = ttkernel::GetWritePtrOp::create(rewriter, op.getLoc(),
                                                     adaptor.getSrc());
      } else {
        srcL1Start = ttkernel::GetReadPtrOp::create(rewriter, op.getLoc(),
                                                    adaptor.getSrc());
      }
      auto dstCBMapping = cbProducerConsumer->get(op.getDst());
      TT_assertv((dstCBMapping == d2m::ThreadCBOrientation::Producer ||
                  dstCBMapping == d2m::ThreadCBOrientation::ProducerConsumer ||
                  dstCBMapping == d2m::ThreadCBOrientation::Default),
                 "Expected dst cb of a write op to have a producer, "
                 "producer-consumer or default orientation, failing.");
      Value dstL1Start = ttkernel::GetWritePtrOp::create(rewriter, op.getLoc(),
                                                         adaptor.getDst());

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
        auto numDestsIdx =
            arith::MulIOp::create(rewriter, op.getLoc(), op.getMcastShape()[0],
                                  op.getMcastShape()[1]);
        auto numDests = arith::IndexCastOp::create(
            rewriter, op.getLoc(), rewriter.getI32Type(), numDestsIdx);
        auto numDestsMinusOne = arith::SubIOp::create(
            rewriter, op.getLoc(), numDests,
            arith::ConstantOp::create(rewriter, op.getLoc(),
                                      rewriter.getI32Type(),
                                      rewriter.getI32IntegerAttr(1)));
        auto mcastAddr = ttkernel::ExperimentalGetNocMulticastAddrOp::create(
            rewriter, op.getLoc(), virtX, virtY, virtMcastEndX, virtMcastEndY,
            dstL1Start, nullptr);
        if (adaptor.getSrc() == adaptor.getDst()) {
          // If src and dst refer to the same memref, we do not loopback mcast
          // Dests are one less because the sender core is not included
          ttkernel::NocAsyncWriteMulticastOp::create(
              rewriter, op.getLoc(), srcL1Start, mcastAddr, transferSize,
              numDestsMinusOne, rewriter.getBoolAttr(true), nullptr, nullptr);
        } else {
          // If src != dst, we loopback mcast
          ttkernel::NocAsyncWriteMulticastLoopbackSrcOp::create(
              rewriter, op.getLoc(), srcL1Start, mcastAddr, transferSize,
              numDests, rewriter.getBoolAttr(true), nullptr, nullptr);
        }
      } else {
        // Local L1 to Local L1 local data movement lowering
        // Get local coordinates using myY and myX ops
        auto myY = ttkernel::MyLogicalYOp::create(rewriter, op.getLoc());
        auto myX = ttkernel::MyLogicalXOp::create(rewriter, op.getLoc());
        // Convert local coordinates to virtual coordinates
        auto [virtY, virtX] = getVirtualCoordsFromLogicalCoords(
            rewriter, op.getLoc(), chipDesc, ValueRange{myY, myX});
        auto nocAddr = ttkernel::GetNocAddrOp::create(rewriter, op.getLoc(),
                                                      virtX, virtY, dstL1Start);
        ttkernel::NocAsyncWriteOp::create(rewriter, op.getLoc(), srcL1Start,
                                          nocAddr, transferSize);
      }
    } else if (op.isDstRemote()) {
      auto srcL1Addr = buildL1Address<ttkernel::GetReadPtrOp>(
          rewriter, op.getLoc(), adaptor.getSrc(), op.getSrcIndices());
      auto dstNocAddr =
          buildNocAddress(rewriter, op.getLoc(), adaptor.getDst(),
                          op.getDstIndices(), chipDesc, op.getDstMemorySpace());
      auto size = i32(rewriter, op->getLoc(), op.getSizeBytes());
      ttkernel::NocAsyncWriteOp::create(rewriter, op.getLoc(), srcL1Addr,
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

    Value logicalY = ttkernel::MyLogicalYOp::create(rewriter, op.getLoc());
    Value logicalX = ttkernel::MyLogicalXOp::create(rewriter, op.getLoc());

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

    Value virtDim = mlir::affine::AffineApplyOp::create(
        rewriter, op.getLoc(), selectedMap, ValueRange{logicalY, logicalX});
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
      ttkernel::NocAsyncReadBarrierOp::create(rewriter, op.getLoc());
    }

    if (isWrite) {
      ttkernel::NocAsyncWriteBarrierOp::create(rewriter, op.getLoc());
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

    if (mlir::isa<MemRefType>(op.getResult().getType())) {
      arg = rewriter.getAttr<ArgAttr>(ArgType::BufferAddress,
                                      op.getOperandIndex());
      arg_result_type = rewriter.getI32Type();
    } else if (mlir::isa<d2m::GlobalSemaphoreType>(op.getResult().getType())) {
      arg = rewriter.getAttr<ArgAttr>(ArgType::GlobalSemaphore,
                                      op.getOperandIndex());
      arg_result_type = ttkernel::L1AddrType::get(rewriter.getContext());
    } else {
      llvm_unreachable("unexpected arg type to GetGlobalOperandOp");
    }

    if (ttnnMode) {
      rewriter.modifyOpInPlace(entry, [&]() {
        argIndex = ArgSpecAttr::appendRuntimeArg(entry, arg);
      });
      rewriter.replaceOpWithNewOp<ttkernel::GetCommonArgValOp>(
          op, arg_result_type, index(rewriter, op->getLoc(), argIndex));

    } else {
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

class D2MGetCBRewriter : public OpConversionPattern<d2m::GetCBOp> {
public:
  using OpConversionPattern<d2m::GetCBOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::GetCBOp op, d2m::GetCBOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Type cbType = getTypeConverter()->convertType(op.getResult().getType());

    assert(op.getOperandIndex() &&
           "d2m.get_cb must have an operand_index by the time it reaches "
           "D2MToTTKernel lowering");
    int64_t operandIndex = *op.getOperandIndex();

    // Append a CBPort entry to the parent function's ArgSpec so that
    // D2MToTTNN can generate the corresponding cb_buffer_index in the
    // kernel descriptor's ct_args.  The operand index tells the runtime
    // which operand's buffer to associate with this CB.
    func::FuncOp entry = op->getParentOfType<func::FuncOp>();
    ArgAttr cbArg = rewriter.getAttr<ArgAttr>(ArgType::CBPort, operandIndex);
    size_t ctArgIndex;
    rewriter.modifyOpInPlace(entry, [&]() {
      ctArgIndex = ArgSpecAttr::appendCompileTimeArg(entry, cbArg);
    });

    // Emit a get_compile_time_arg_val that reads the port from ct_args at
    // runtime. This allows the spatial op to remap CB ports per grid range
    // by overriding compile-time arguments.
    rewriter.replaceOpWithNewOp<ttkernel::GetCompileArgValOp>(
        op, cbType, static_cast<int32_t>(ctArgIndex));
    return success();
  }
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
    if (!op->hasAttr(d2m::ThreadAttr::name)) {
      return failure();
    }

    SmallVector<ArgAttr> rtArgSpecVector;
    SmallVector<ArgAttr> ctArgSpecVector;

    // Zero-input functions: just convert attrs and set function type.
    // The D2MGetCBRewriter will append CB entries to the ArgSpec as it
    // processes get_cb ops within the function body.
    if (op.getFunctionType().getNumInputs() == 0) {
      rewriter.modifyOpInPlace(op, [&]() {
        op.setType(rewriter.getFunctionType(TypeRange(), TypeRange()));
        convertFunctionAttrs(rewriter, op, rtArgSpecVector, ctArgSpecVector);
      });
      return success();
    }

    Block *block = &op.getCallableRegion()->front();
    auto blockArgs = block->getArguments();
    assert(!blockArgs.empty());

    size_t currentSemaphoreIndex = 0;
    TypeConverter::SignatureConversion signatureConverter(op.getNumArguments());
    OpBuilder::InsertionGuard funcInsertionGuard(rewriter);
    rewriter.setInsertionPointToStart(block);
    // Block arguments are semaphores only. CB args have been replaced by
    // d2m.get_cb ops, which are lowered by D2MGetCBRewriter.
    for (auto arg : blockArgs) {
      Type argType = getTypeConverter()->convertType(arg.getType());
      if (mlir::isa<SemaphoreType>(argType)) {
        if (getTTKernelThreadType(op) != ThreadType::Noc) {
          continue;
        }
        size_t ctArgIndex = ctArgSpecVector.size();
        auto semaphoreIndex = GetCompileArgValOp::create(
            rewriter, op.getLoc(), rewriter.getI32Type(),
            rewriter.getI32IntegerAttr(ctArgIndex));
        auto semaphore =
            GetSemaphoreOp::create(rewriter, op.getLoc(), semaphoreIndex);
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
          ttkernel::CastToL1PtrOp::create(rewriter, op.getLoc(), semaphoreAddr);

      rewriter.replaceOpWithNewOp<ttkernel::NocSemaphoreSetOp>(op, semaphorePtr,
                                                               value);
    } else if (op.getMcastShape().empty()) {
      assert(!mlir::isa<d2m::SemaphoreSetOp>(op) &&
             "d2m.semaphore_set to single remote core is illegal.");
      auto [virtY, virtX] = getVirtualCoordsFromLogicalCoords(
          rewriter, op.getLoc(), chipDesc, op.getDstCoreIndex());
      auto nocAddr = ttkernel::GetNocAddrOp::create(
          rewriter, op.getLoc(), virtX, virtY, semaphoreAddr);
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
      Value numDestsIdx = arith::MulIOp::create(
          rewriter, op.getLoc(), op.getMcastShape()[0], op.getMcastShape()[1]);
      Value numDests = arith::IndexCastOp::create(
          rewriter, op.getLoc(), rewriter.getI32Type(), numDestsIdx);
      Value numDestsMinusOne = arith::SubIOp::create(
          rewriter, op.getLoc(), numDests,
          arith::ConstantOp::create(rewriter, op.getLoc(),
                                    rewriter.getI32Type(),
                                    rewriter.getI32IntegerAttr(1)));
      auto mcastAddr = ttkernel::ExperimentalGetNocMulticastAddrOp::create(
          rewriter, op.getLoc(), virtX, virtY, virtMcastEndX, virtMcastEndY,
          semaphoreAddr, nullptr);

      auto semaphorePtr =
          ttkernel::CastToL1PtrOp::create(rewriter, op.getLoc(), semaphoreAddr);
      ttkernel::NocSemaphoreSetOp::create(rewriter, op.getLoc(), semaphorePtr,
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
        ttkernel::CastToL1PtrOp::create(rewriter, op.getLoc(), semaphoreAddr);

    rewriter.replaceOpWithNewOp<ttkernel::NocSemaphoreWaitOp>(op, semaphorePtr,
                                                              op.getValue());
    if (op.getResetValue()) {
      ttkernel::NocSemaphoreSetOp::create(rewriter, op.getLoc(), semaphorePtr,
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
               ttkernel::D2MFPUOpsRewriter<d2m::TileReduceMeanOp>,

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
               ttkernel::D2MTilizeUntilizeRewriter<d2m::TileUntilizeBlockOp, ttkernel::ExperimentalPackUntilizeBlockOp>,
               ttkernel::D2MFillTileRewriter,
               ttkernel::D2MWriteRowMaskTileRewriter,
               ttkernel::D2MWriteColMaskTileRewriter,
               ttkernel::D2MExperimentalFillArangeTileRewriter,
               ttkernel::D2MTileTransposeRewriter,
               ttkernel::D2MDstReinterpretCastRewriter,
               ttkernel::AcquireDstRewriter,
               ttkernel::UnpackStallOnPackRewriter,
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

  patterns.add<ttkernel::D2MGetGlobalOperandRewriter>(typeConverter, ctx,
                                                      ttnnMode);
  patterns.add<ttkernel::D2MGetCBRewriter>(typeConverter, ctx);
  patterns.add<ttkernel::D2MDMAReadRewriter>(typeConverter, ctx, &associatedDMAWaits, &cbProducerConsumer);
  patterns.add<ttkernel::D2MDMAWriteRewriter>(typeConverter, ctx, &associatedDMAWaits, &cbProducerConsumer);

  // This is needed to lower affine apply ops that may be generated when
  // `d2m.core_index` is used with a `phys_to_virt_map`.
  populateAffineToStdConversionPatterns(patterns);
  // clang-format on
}

} // namespace mlir::tt
