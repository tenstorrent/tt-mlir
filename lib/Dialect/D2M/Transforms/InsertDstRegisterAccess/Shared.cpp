// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/InsertDstRegisterAccess/Shared.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"

#include <type_traits>

#define DEBUG_TYPE "D2MInsertDstRegisterAccess"

namespace mlir::tt::d2m {

// ---------------------------------------------------------------------------
// Preconditions
// ---------------------------------------------------------------------------

LogicalResult verifyInsertDstRegisterAccessPreconditions(ModuleOp moduleOp) {
  WalkResult walkResult = moduleOp->walk(
      [&](linalg::GenericOp op) { return WalkResult::interrupt(); });

  if (walkResult.wasInterrupted()) {
    return moduleOp.emitOpError()
           << "found linalg.generic operations that were not converted to "
              "affine loops. Please run --d2m-linalg-to-affine before the "
              "d2m-insert-dst-register-access-unscheduled / "
              "d2m-insert-dst-register-access-scheduled passes.";
  }
  return success();
}

namespace detail {

// ---------------------------------------------------------------------------
// DstSliceAllocator
// ---------------------------------------------------------------------------

namespace {

void debugDumpDstSliceAllocator(StringRef header, ArrayRef<unsigned> sliceStack,
                                ArrayRef<unsigned> inputStack,
                                ArrayRef<unsigned> scratchSlots,
                                std::optional<unsigned> action) {
  LDBG_OS([&](raw_ostream &os) {
    os << header << "\n";
    os << "  SliceStack   = ";
    llvm::interleaveComma(sliceStack, os);
    os << "\n  InputStack   = ";
    llvm::interleaveComma(inputStack, os);
    os << "\n  ScratchSlots = ";
    llvm::interleaveComma(scratchSlots, os);
    if (action) {
      os << "\n  --> " << *action;
    }
  });
}

} // namespace

unsigned DstSliceAllocator::allocateInput() {
  TT_assertv(!sliceStack.empty(), "Out of dst slices");

  unsigned id = sliceStack.pop_back_val();
  currSliceIndex = id;
  inputStack.push_back(id);

  debugDumpDstSliceAllocator("== ALLOCATE INPUT ==", sliceStack, inputStack,
                             scratchSlots, id);
  return id;
}

unsigned DstSliceAllocator::allocateOutput() {
  TT_assertv(!sliceStack.empty(), "Out of dst slices");

  unsigned id = sliceStack.pop_back_val();
  currSliceIndex = id;

  debugDumpDstSliceAllocator("== ALLOCATE OUTPUT ==", sliceStack, inputStack,
                             scratchSlots, id);
  return id;
}

unsigned DstSliceAllocator::allocateScratch() {
  TT_assertv(!sliceStack.empty(), "Out of dst slices");

  unsigned id = sliceStack.pop_back_val();
  scratchSlots.push_back(id);

  // Intentionally do NOT update `currSliceIndex` or `inputStack`.
  // Scratch is owned by the op for the lifetime of the region; it must
  // not show up as a candidate for in-place reuse by later compute ops.
  debugDumpDstSliceAllocator("== ALLOCATE SCRATCH ==", sliceStack, inputStack,
                             scratchSlots, id);
  return id;
}

unsigned DstSliceAllocator::getCurrSliceIndex() const {
  TT_assertv(currSliceIndex.has_value(),
             "No dst slice allocated yet (call allocate* first)");
  return *currSliceIndex;
}

unsigned DstSliceAllocator::getFirstInputSliceIndex() const {
  TT_assertv(!inputStack.empty(), "No input slots allocated");
  return inputStack.front();
}

void DstSliceAllocator::deallocateAllButFirstInput() {
  TT_assertv(inputStack.size() >= 1u, "Need at least one input to keep");

  unsigned firstInput = inputStack.front();
  inputStack.erase(inputStack.begin());

  while (!inputStack.empty()) {
    unsigned id = inputStack.pop_back_val();
    sliceStack.push_back(id);
    debugDumpDstSliceAllocator("== DEALLOCATE (keeping first) ==", sliceStack,
                               inputStack, scratchSlots, id);
  }

  currSliceIndex = firstInput;
}

void DstSliceAllocator::initSliceStack() {
  TT_assert((dstSliceCapacity > 0u && dstSliceCapacity <= 16u));

  for (int i = dstSliceCapacity - 1; i >= 0; --i) {
    sliceStack.push_back(static_cast<unsigned>(i));
  }
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

SmallVector<Value> collectAncestorLoopIVs(Operation *op) {
  SmallVector<Value> loopIVs;
  Operation *current = op->getParentOp();
  while (current && !mlir::isa<d2m::GenericOp>(current)) {
    if (auto scfFor = mlir::dyn_cast<scf::ForOp>(current)) {
      loopIVs.push_back(scfFor.getInductionVar());
    } else if (auto affineFor = mlir::dyn_cast<affine::AffineForOp>(current)) {
      loopIVs.push_back(affineFor.getInductionVar());
    }
    current = current->getParentOp();
  }
  std::reverse(loopIVs.begin(), loopIVs.end());
  return loopIVs;
}

static bool valueDependsOnIV(Value value, Value iv, DenseSet<Value> &visited) {
  if (value == iv) {
    return true;
  }
  if (!visited.insert(value).second) {
    return false;
  }

  Operation *definingOp = value.getDefiningOp();
  if (!definingOp) {
    return false;
  }

  for (Value operand : definingOp->getOperands()) {
    if (valueDependsOnIV(operand, iv, visited)) {
      return true;
    }
  }
  return false;
}

static bool valueDependsOnIV(Value value, Value iv) {
  DenseSet<Value> visited;
  return valueDependsOnIV(value, iv, visited);
}

static SmallVector<Value> getUsedAffineMapOperands(AffineMap map,
                                                   ValueRange mapOperands) {
  SmallVector<bool> usedOperands(mapOperands.size(), false);
  for (AffineExpr resultExpr : map.getResults()) {
    resultExpr.walk([&](AffineExpr expr) {
      if (auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr)) {
        unsigned pos = dimExpr.getPosition();
        if (pos < usedOperands.size()) {
          usedOperands[pos] = true;
        }
      } else if (auto symbolExpr = mlir::dyn_cast<AffineSymbolExpr>(expr)) {
        unsigned pos = map.getNumDims() + symbolExpr.getPosition();
        if (pos < usedOperands.size()) {
          usedOperands[pos] = true;
        }
      }
    });
  }

  SmallVector<Value> usedMapOperands;
  for (auto [idx, isUsed] : llvm::enumerate(usedOperands)) {
    if (isUsed) {
      usedMapOperands.push_back(mapOperands[idx]);
    }
  }
  return usedMapOperands;
}

static bool accessDependsOnIV(affine::AffineLoadOp loadOp, Value iv) {
  if (valueDependsOnIV(loadOp.getMemRef(), iv)) {
    return true;
  }
  SmallVector<Value> mapOperands =
      getUsedAffineMapOperands(loadOp.getAffineMap(), loadOp.getMapOperands());
  return llvm::any_of(mapOperands,
                      [&](Value v) { return valueDependsOnIV(v, iv); });
}

static bool accessDependsOnIV(affine::AffineStoreOp storeOp, Value iv) {
  if (valueDependsOnIV(storeOp.getMemRef(), iv)) {
    return true;
  }
  SmallVector<Value> mapOperands = getUsedAffineMapOperands(
      storeOp.getAffineMap(), storeOp.getMapOperands());
  return llvm::any_of(mapOperands,
                      [&](Value v) { return valueDependsOnIV(v, iv); });
}

static bool accessDependsOnIV(memref::LoadOp loadOp, Value iv) {
  if (valueDependsOnIV(loadOp.getMemRef(), iv)) {
    return true;
  }
  return llvm::any_of(loadOp.getIndices(),
                      [&](Value idx) { return valueDependsOnIV(idx, iv); });
}

static bool accessDependsOnIV(memref::StoreOp storeOp, Value iv) {
  if (valueDependsOnIV(storeOp.getMemRef(), iv)) {
    return true;
  }
  return llvm::any_of(storeOp.getIndices(),
                      [&](Value idx) { return valueDependsOnIV(idx, iv); });
}

template <typename LoadOrStoreTy>
static SmallVector<Value> getGuardLoopIVs(LoadOrStoreTy loadOrStore,
                                          Operation *contextOp) {
  SmallVector<Value> guardIVs;
  for (Value loopIV : collectAncestorLoopIVs(contextOp)) {
    if (!accessDependsOnIV(loadOrStore, loopIV)) {
      guardIVs.push_back(loopIV);
    }
  }
  return guardIVs;
}

bool hasTileMatmul(Operation *op) {
  return op->walk([](d2m::TileMatmulOp) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

bool isTileReductionOp(Operation *op) {
  return mlir::isa<d2m::TileReduceMaxOp, d2m::TileReduceSumOp,
                   d2m::TileReduceMeanOp, d2m::TileSFPUReduceMaxOp,
                   d2m::TileSFPUReduceSumOp>(op);
}

bool isPackerL1AccumulationSupportedDataType(ttcore::DataType dt) {
  // The packer L1-acc path on Wormhole/Blackhole only operates on a fixed set
  // of native formats. Block-float (bfp_*) outputs do NOT support L1
  // accumulation: enabling it for those formats produces silently incorrect
  // (catastrophic-PCC) results on hardware.
  //
  // Source of truth: tt-llk's `PACK_L1_ACC_FORMATS` in
  // `tt_metal/tt-llk/tests/python_tests/quasar/test_pack_l1_acc_quasar.py`:
  //   { Float16_b, Float16, Float32, Int32, Int8, UInt8 }
  // (we don't have an Int8 enum in TTCore, so it is absent here).
  switch (dt) {
  case ttcore::DataType::Float32:
  case ttcore::DataType::Float16:
  case ttcore::DataType::BFloat16:
  case ttcore::DataType::Int32:
  case ttcore::DataType::UInt8:
    return true;
  default:
    return false;
  }
}

bool allTileMatmulOutputsSupportPackerL1Acc(Operation *loopOp) {
  bool allSupported = true;
  loopOp->walk([&](d2m::TileMatmulOp matmul) {
    auto tileType =
        mlir::dyn_cast<ttcore::TileType>(matmul.getResult().getType());
    if (!tileType) {
      // Be conservative: if we can't determine the tile element type, do not
      // enable L1-acc.
      allSupported = false;
      return WalkResult::interrupt();
    }
    if (!isPackerL1AccumulationSupportedDataType(tileType.getDataType())) {
      allSupported = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return allSupported;
}

// Returns true iff any AffineStore (in `copyInfos.stores`) or memref::StoreOp
// (in `copyInfos.memrefStores`) recorded for this region depends on `iv`.
// "Depends on" includes transitive dependence through subview indices.
static bool anyOutputStoreDependsOnIV(const CopyInfoMap &copyInfos, Value iv) {
  for (const auto &[loopOrOp, copyInfo] : copyInfos) {
    for (const auto &record : copyInfo.stores) {
      if (record.loadStore && accessDependsOnIV(record.loadStore, iv)) {
        return true;
      }
    }
    for (const auto &record : copyInfo.memrefStores) {
      if (record.loadStore && accessDependsOnIV(record.loadStore, iv)) {
        return true;
      }
    }
  }
  return false;
}

// Compute the static trip count of an affine.for / scf.for loop body by
// using its constant lower/upper bound and step. Returns std::nullopt if the
// loop is not constant-bounded (in which case we conservatively allow L1-acc
// at runtime, matching legacy behavior for non-constant K loops).
static std::optional<int64_t> tryGetConstantTripCount(Operation *loopOp) {
  if (auto affineFor = mlir::dyn_cast<affine::AffineForOp>(loopOp)) {
    if (!affineFor.hasConstantBounds()) {
      return std::nullopt;
    }
    int64_t lb = affineFor.getConstantLowerBound();
    int64_t ub = affineFor.getConstantUpperBound();
    int64_t step = affineFor.getStepAsInt();
    if (step <= 0) {
      return std::nullopt;
    }
    return llvm::divideCeil(std::max<int64_t>(0, ub - lb), step);
  }
  if (auto scfFor = mlir::dyn_cast<scf::ForOp>(loopOp)) {
    auto lbCst = scfFor.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    auto ubCst = scfFor.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    auto stepCst = scfFor.getStep().getDefiningOp<arith::ConstantIndexOp>();
    if (!lbCst || !ubCst || !stepCst) {
      return std::nullopt;
    }
    int64_t lb = lbCst.value();
    int64_t ub = ubCst.value();
    int64_t step = stepCst.value();
    if (step <= 0) {
      return std::nullopt;
    }
    return llvm::divideCeil(std::max<int64_t>(0, ub - lb), step);
  }
  return std::nullopt;
}

Value findOutermostReductionLoopIVForL1Acc(Operation *acquireDstOp,
                                           const CopyInfoMap &copyInfos) {
  // `collectAncestorLoopIVs` returns IVs in outermost-to-innermost order
  // (it reverses the upward walk).
  SmallVector<Value> ancestorIVs = collectAncestorLoopIVs(acquireDstOp);
  for (Value iv : ancestorIVs) {
    if (anyOutputStoreDependsOnIV(copyInfos, iv)) {
      // This loop indexes the output -- it is parallel, not a reduction.
      continue;
    }
    // Found the outermost reduction loop. Only enable L1-acc if the loop
    // actually iterates more than once (otherwise the per-iteration
    // accumulation guard would never fire and we'd just emit dead code).
    Operation *loopOp = nullptr;
    if (auto blockArg = mlir::dyn_cast<BlockArgument>(iv)) {
      loopOp = blockArg.getOwner()->getParentOp();
    }
    if (!loopOp) {
      return nullptr;
    }
    std::optional<int64_t> tripCount = tryGetConstantTripCount(loopOp);
    if (tripCount.has_value() && *tripCount <= 1) {
      return nullptr;
    }
    return iv;
  }
  return nullptr;
}

void setDstScratchIndex(OperandLoadStoreRegisterOpInterface computeOp,
                        int scratchSlice) {
  TT_assertv(computeOp.getNumDstScratchSlices() == 1,
             "setDstScratchIndex supports exactly one scratch slice");
  Operation *op = computeOp.getOperation();
  op->setAttr("dst_scratch_index",
              mlir::IntegerAttr::get(
                  mlir::IntegerType::get(op->getContext(), 64), scratchSlice));
}

static Value getSecondIterationValue(PatternRewriter &rewriter, Location loc,
                                     Value loopIV) {
  auto one = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 1));

  auto ivBlockArg = mlir::dyn_cast<BlockArgument>(loopIV);
  if (!ivBlockArg) {
    return one;
  }

  auto *ownerBlock = ivBlockArg.getOwner();
  if (!ownerBlock) {
    return one;
  }

  auto *ownerOp = ownerBlock->getParentOp();
  if (!ownerOp) {
    return one;
  }

  if (auto scfFor = mlir::dyn_cast<scf::ForOp>(ownerOp)) {
    return rewriter.create<arith::AddIOp>(loc, scfFor.getLowerBound(),
                                          scfFor.getStep());
  }

  if (auto affineFor = mlir::dyn_cast<affine::AffineForOp>(ownerOp)) {
    Value lb = nullptr;
    if (affineFor.hasConstantLowerBound()) {
      lb = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(),
                                  affineFor.getConstantLowerBound()));
    } else {
      AffineMap lowerBoundMap = affineFor.getLowerBoundMap();
      if (lowerBoundMap.getNumResults() == 1) {
        lb = rewriter.create<affine::AffineApplyOp>(
            loc, lowerBoundMap, affineFor.getLowerBoundOperands());
      }
    }

    if (lb) {
      Value step = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(),
                                  affineFor.getStepAsInt()));
      return rewriter.create<arith::AddIOp>(loc, lb, step);
    }
  }

  return one;
}

// ---------------------------------------------------------------------------
// Core infrastructure shared by both passes
// ---------------------------------------------------------------------------

bool hasAcquireDstOp(Region &region) {
  return !region.getOps<AcquireDstOp>().empty();
}

DstRegionOpClassification classifyDstRegionOps(GenericOp gOp,
                                               unsigned regionIndex) {
  DstRegionOpClassification types;
  types.hasComputeOps = gOp.hasComputeOpsInRegion(regionIndex);

  Region *genericRegion = &gOp.getRegion(regionIndex);
  Block &block = genericRegion->getBlocks().front();

  block.walk([&](Operation *op) {
    if (isa<linalg::GenericOp>(op)) {
      types.hasLinalgGeneric = true;
    } else if (auto affineFor = dyn_cast<affine::AffineForOp>(op)) {
      if (affineFor->hasAttr("d2m.linalg_root")) {
        types.hasMarkedAffineLoops = true;
      }
    } else if (auto scfFor = dyn_cast<scf::ForOp>(op)) {
      if (scfFor->hasAttr("d2m.linalg_root")) {
        types.hasMarkedAffineLoops = true;
      }
    }
  });

  return types;
}

std::pair<Type, int> inferDstInfoFromAllAccesses(const CopyInfoMap &copyInfos) {
  Type elementType = nullptr;
  int maxDstSlice = -1;

  auto updateInfo = [&](MemRefType memref, int idx) {
    if (elementType == nullptr) {
      elementType = memref.getElementType();
    }
    maxDstSlice = std::max(maxDstSlice, idx);
  };

  for (auto [loopNest, copyInfo] : copyInfos) {
    for (auto &[loadOp, bcastOp, idx, guardIVs] : copyInfo.loads) {
      updateInfo(loadOp.getMemRefType(), idx);
    }
    for (auto &[storeOp, bcastOp, idx, guardIVs] : copyInfo.stores) {
      updateInfo(storeOp.getMemRefType(), idx);
    }
    for (auto &[loadOp, bcastOp, idx, guardIVs] : copyInfo.memrefLoads) {
      updateInfo(loadOp.getMemRefType(), idx);
    }
    for (auto &[storeOp, bcastOp, idx, guardIVs] : copyInfo.memrefStores) {
      updateInfo(storeOp.getMemRefType(), idx);
    }
  }
  TT_assert(elementType != nullptr);
  TT_assert(maxDstSlice >= 0);
  return {elementType, maxDstSlice};
}

AcquireDstOp insertAcquireDst(PatternRewriter &rewriter, Location loc,
                              Region &region, const CopyInfoMap &copyInfos,
                              Operation *outermostInnerComputeLoop,
                              unsigned dstCapacity, bool insertInsideLoop) {
  assert(!copyInfos.empty());
  if (outermostInnerComputeLoop) {
    if (insertInsideLoop) {
      if (auto scfFor = dyn_cast<scf::ForOp>(outermostInnerComputeLoop)) {
        rewriter.setInsertionPointToStart(scfFor.getBody());
      } else if (auto affineFor =
                     dyn_cast<affine::AffineForOp>(outermostInnerComputeLoop)) {
        rewriter.setInsertionPointToStart(affineFor.getBody());
      } else {
        rewriter.setInsertionPoint(outermostInnerComputeLoop);
      }
    } else {
      rewriter.setInsertionPoint(outermostInnerComputeLoop);
    }
  } else {
    rewriter.setInsertionPointToStart(&region.front());
  }

  auto [elementType, maxDstSlice] = inferDstInfoFromAllAccesses(copyInfos);
  TT_assertv(maxDstSlice < static_cast<int64_t>(dstCapacity),
             "Insufficient DST capacity for all operands.");
  SmallVector<int64_t> dstShape({static_cast<int64_t>(dstCapacity)});
  MemRefType dstType =
      MemRefType::get(dstShape, elementType,
                      mlir::AffineMap::getMultiDimIdentityMap(
                          dstShape.size(), rewriter.getContext()),
                      rewriter.getAttr<ttcore::MemorySpaceAttr>(
                          ttcore::MemorySpace::RegisterDst));

  return rewriter.create<AcquireDstOp>(loc, dstType);
}

Value lookThroughSubView(Value memref) {
  if (!memref) {
    return nullptr;
  }
  while (auto subView = mlir::dyn_cast_or_null<memref::SubViewOp>(
             memref.getDefiningOp())) {
    memref = subView.getSource();
  }
  if (auto *definingOp = memref.getDefiningOp()) {
    if (mlir::isa<d2m::WaitOp, d2m::ReserveOp>(definingOp)) {
      memref = definingOp->getOperand(0);
    } else if (mlir::isa<memref::AllocOp>(definingOp) ||
               mlir::isa<OperandAliasOp>(definingOp)) {
      return memref;
    }
  }
  if (mlir::isa<CBType>(memref.getType())) {
    return memref;
  }
  if (mlir::isa<BlockArgument>(memref)) {
    return memref;
  }
  return nullptr;
}

Value stripDstRegionWrappers(Value memref) {
  if (!memref) {
    return nullptr;
  }
  while (auto *definingOp = memref.getDefiningOp()) {
    if (mlir::isa<d2m::WaitOp, d2m::ReserveOp>(definingOp)) {
      memref = definingOp->getOperand(0);
      continue;
    }
    break;
  }
  return memref;
}

bool isSameLogicalMemRefRegion(Value lhs, Value rhs) {
  lhs = stripDstRegionWrappers(lhs);
  rhs = stripDstRegionWrappers(rhs);

  if (lhs == rhs) {
    return true;
  }

  auto lhsSubView = lhs.getDefiningOp<memref::SubViewOp>();
  auto rhsSubView = rhs.getDefiningOp<memref::SubViewOp>();
  if (!lhsSubView || !rhsSubView) {
    return false;
  }

  return isSameLogicalMemRefRegion(lhsSubView.getSource(),
                                   rhsSubView.getSource()) &&
         llvm::equal(lhsSubView.getStaticOffsets(),
                     rhsSubView.getStaticOffsets()) &&
         llvm::equal(lhsSubView.getStaticSizes(),
                     rhsSubView.getStaticSizes()) &&
         llvm::equal(lhsSubView.getStaticStrides(),
                     rhsSubView.getStaticStrides()) &&
         llvm::equal(lhsSubView.getOffsets(), rhsSubView.getOffsets()) &&
         llvm::equal(lhsSubView.getSizes(), rhsSubView.getSizes()) &&
         llvm::equal(lhsSubView.getStrides(), rhsSubView.getStrides());
}

SmallVector<Value>
getObviousCarriedOutputRegions(OperandLoadStoreRegisterOpInterface computeOp) {
  SmallVector<Value> outputs;

  if (auto dpsOp = mlir::dyn_cast<DestinationStyleOpInterface>(
          computeOp.getOperation())) {
    outputs.append(dpsOp.getDpsInits().begin(), dpsOp.getDpsInits().end());
  }

  for (Value result : computeOp->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (auto affineStore = mlir::dyn_cast<affine::AffineStoreOp>(user)) {
        if (affineStore.getValue() == result) {
          outputs.push_back(affineStore.getMemRef());
        }
      } else if (auto memrefStore = mlir::dyn_cast<memref::StoreOp>(user)) {
        if (memrefStore.getValue() == result) {
          outputs.push_back(memrefStore.getMemRef());
        }
      }
    }
  }

  return outputs;
}

SmallVector<int64_t>
getAccumClassificationOperandIndices(OperandLoadStoreRegisterOpInterface op) {
  SmallVector<int64_t> operandIndices;
  auto dpsOp = mlir::dyn_cast<DestinationStyleOpInterface>(op.getOperation());

  for (OpOperand &operand : op->getOpOperands()) {
    if (op.isScalarOperand(operand.getOperandNumber())) {
      continue;
    }
    if (dpsOp && dpsOp.isDpsInit(&operand)) {
      continue;
    }
    operandIndices.push_back(operand.getOperandNumber());
  }

  return operandIndices;
}

// Core recording logic: record a load/store with an optional guard.
template <typename LoadOrStoreTy>
static void recordDstAccessImpl(LoadOrStoreTy loadOrStore,
                                CopyInfoMap &copyInfos, int dstSlice,
                                Operation *outermostInnerComputeLoop,
                                bool emitGuard) {
  if (!outermostInnerComputeLoop) {
    outermostInnerComputeLoop = loadOrStore;
  }

  auto [iter, _] = copyInfos.try_emplace(outermostInnerComputeLoop);
  Value assocCB = lookThroughSubView(loadOrStore.getMemRef());

  SmallVector<Value> guardIVs;
  if (assocCB && emitGuard) {
    guardIVs = getGuardLoopIVs(loadOrStore, outermostInnerComputeLoop);
  }

  iter->second.record(loadOrStore, dstSlice, guardIVs);
}

void recordDstAccess(affine::AffineLoadOp op, CopyInfoMap &copyInfos,
                     int dstSlice, Operation *outermostInnerComputeLoop,
                     bool emitGuard) {
  recordDstAccessImpl(op, copyInfos, dstSlice, outermostInnerComputeLoop,
                      emitGuard);
}

void recordDstAccess(affine::AffineStoreOp op, CopyInfoMap &copyInfos,
                     int dstSlice, Operation *outermostInnerComputeLoop,
                     bool emitGuard) {
  recordDstAccessImpl(op, copyInfos, dstSlice, outermostInnerComputeLoop,
                      emitGuard);
}

void recordDstAccess(memref::LoadOp op, CopyInfoMap &copyInfos, int dstSlice,
                     Operation *outermostInnerComputeLoop, bool emitGuard) {
  recordDstAccessImpl(op, copyInfos, dstSlice, outermostInnerComputeLoop,
                      emitGuard);
}

void recordDstAccess(memref::StoreOp op, CopyInfoMap &copyInfos, int dstSlice,
                     Operation *outermostInnerComputeLoop, bool emitGuard) {
  recordDstAccessImpl(op, copyInfos, dstSlice, outermostInnerComputeLoop,
                      emitGuard);
}

void recordDstAccess(affine::AffineLoadOp loadOp, d2m::TileBcastOp bcastOp,
                     CopyInfoMap &copyInfos, int dstSlice,
                     Operation *outermostInnerComputeLoop, bool emitGuard) {
  if (!outermostInnerComputeLoop) {
    outermostInnerComputeLoop = loadOp;
  }

  auto [iter, _] = copyInfos.try_emplace(outermostInnerComputeLoop);
  Value assocCB = lookThroughSubView(loadOp.getMemRef());

  SmallVector<Value> guardIVs;
  if (assocCB && emitGuard) {
    guardIVs = getGuardLoopIVs(loadOp, outermostInnerComputeLoop);
  }

  iter->second.record(loadOp, bcastOp, dstSlice, guardIVs);
}

// Heuristically identify CB loads that feed a loop-carried accumulator tile.
template <typename LoadTy>
static bool
isObviousLoopCarriedAccumulationLoad(LoadTy loadOp, int64_t operandIdx,
                                     ValueRange carriedOutputRegions,
                                     ArrayRef<int64_t> accumOperandIndices) {
  if (accumOperandIndices.size() <= 1 ||
      !llvm::is_contained(accumOperandIndices, operandIdx)) {
    return false;
  }

  for (Value outputRegion : carriedOutputRegions) {
    if (isSameLogicalMemRefRegion(loadOp.getMemRef(), outputRegion)) {
      return true;
    }
  }

  return false;
}

// Decide whether this load should preserve the DST tile across outer-loop
// iterations instead of reloading every time.
template <typename LoadTy>
static bool shouldGuardDstLoadForAccumulation(
    LoadTy loadOp, int64_t operandIdx, ValueRange carriedOutputRegions,
    ArrayRef<int64_t> accumOperandIndices, bool noAccumGuard = false) {
  if (noAccumGuard || !lookThroughSubView(loadOp.getMemRef())) {
    return false;
  }

  return isObviousLoopCarriedAccumulationLoad(
      loadOp, operandIdx, carriedOutputRegions, accumOperandIndices);
}

// Record a store that drains a computed DST tile back to memory.
template <typename StoreTy>
static void collectDstStoreAccessImpl(StoreTy storeOp, CopyInfoMap &copyInfos,
                                      int dstSlice,
                                      Operation *outermostInnerComputeLoop) {
  recordDstAccessImpl(storeOp, copyInfos, dstSlice, outermostInnerComputeLoop,
                      /*emitGuard=*/false);
}

void collectDstStoreAccess(affine::AffineStoreOp storeOp,
                           CopyInfoMap &copyInfos, int dstSlice,
                           Operation *outermostInnerComputeLoop) {
  collectDstStoreAccessImpl(storeOp, copyInfos, dstSlice,
                            outermostInnerComputeLoop);
}

void collectDstStoreAccess(memref::StoreOp storeOp, CopyInfoMap &copyInfos,
                           int dstSlice, Operation *outermostInnerComputeLoop) {
  collectDstStoreAccessImpl(storeOp, copyInfos, dstSlice,
                            outermostInnerComputeLoop);
}

// Collect a single load access and determine whether it needs an accumulation
// guard.
template <typename LoadTy>
static void collectDstLoadWithAccumAnalysisImpl(
    LoadTy loadOp, int64_t operandIdx, ValueRange carriedOutputRegions,
    ArrayRef<int64_t> accumOperandIndices, CopyInfoMap &copyInfos, int dstSlice,
    Operation *outermostInnerComputeLoop, bool noAccumGuard) {
  const bool emitGuard = shouldGuardDstLoadForAccumulation(
      loadOp, operandIdx, carriedOutputRegions, accumOperandIndices,
      noAccumGuard);
  recordDstAccessImpl(loadOp, copyInfos, dstSlice, outermostInnerComputeLoop,
                      emitGuard);
}

void collectDstLoadWithAccumAnalysis(affine::AffineLoadOp loadOp,
                                     int64_t operandIdx,
                                     ValueRange carriedOutputRegions,
                                     ArrayRef<int64_t> accumOperandIndices,
                                     CopyInfoMap &copyInfos, int dstSlice,
                                     Operation *outermostInnerComputeLoop,
                                     bool noAccumGuard) {
  collectDstLoadWithAccumAnalysisImpl(loadOp, operandIdx, carriedOutputRegions,
                                      accumOperandIndices, copyInfos, dstSlice,
                                      outermostInnerComputeLoop, noAccumGuard);
}

void collectDstLoadWithAccumAnalysis(memref::LoadOp loadOp, int64_t operandIdx,
                                     ValueRange carriedOutputRegions,
                                     ArrayRef<int64_t> accumOperandIndices,
                                     CopyInfoMap &copyInfos, int dstSlice,
                                     Operation *outermostInnerComputeLoop,
                                     bool noAccumGuard) {
  collectDstLoadWithAccumAnalysisImpl(loadOp, operandIdx, carriedOutputRegions,
                                      accumOperandIndices, copyInfos, dstSlice,
                                      outermostInnerComputeLoop, noAccumGuard);
}

std::pair<Operation *, mlir::IRMapping>
cloneAffineLoopSkeleton(PatternRewriter &rewriter, Operation *loopNestOrOp) {
  Operation *skeleton = nullptr;
  mlir::IRMapping mapper;
  if (mlir::isa<affine::AffineForOp>(loopNestOrOp)) {
    skeleton = rewriter.clone(*loopNestOrOp, mapper);
    skeleton->walk([&](Operation *op) {
      if (!mlir::isa<affine::AffineForOp, affine::AffineYieldOp,
                     affine::AffineApplyOp>(op)) {
        op->dropAllUses();
        rewriter.eraseOp(op);
      }
    });
  }
  return {skeleton, mapper};
}

template <typename LoadOrStoreTy>
void emitDstCopyNest(
    PatternRewriter &rewriter, Operation *loopNestOrOp,
    ArrayRef<LoadStoreRecord<LoadOrStoreTy>> loadStoreRecords,
    llvm::function_ref<void(PatternRewriter &, LoadStoreRecord<LoadOrStoreTy>,
                            AffineMap, ValueRange, AffineMap, ValueRange)>
        copyGenerator,
    llvm::function_ref<void(PatternRewriter &, LoadStoreRecord<LoadOrStoreTy>,
                            AffineMap, ValueRange)>
        accessReplacer,
    bool disableL1Acc) {
  if (loadStoreRecords.empty()) {
    return;
  }

  // Pre-clone the unguarded copy nest (shared by all records that don't
  // need a per-IV guard).
  Operation *copyLoop = nullptr;
  mlir::IRMapping copyLoopMapper;
  if (disableL1Acc) {
    std::tie(copyLoop, copyLoopMapper) =
        cloneAffineLoopSkeleton(rewriter, loopNestOrOp);
  }

  for (auto record : loadStoreRecords) {
    auto loadStoreLoc = record.loadStore.getLoc();
    auto loadStoreIndices = record.loadStore.getIndices();
    auto loadStoreMap = record.loadStore.getMap();
    auto loadStoreMemRefType = record.loadStore.getMemRefType();

    if (disableL1Acc) {
      mlir::IRMapping irMapper = copyLoopMapper;
      if (!record.guardIVs.empty()) {
        const bool isBcastGuard = record.bcast.has_value();
        // TODO(wenbinlyuTT): #6516 WA to put all bcast inits to the top of
        // the compute tiling loops.
        if (isBcastGuard && copyLoop) {
          rewriter.setInsertionPoint(copyLoop);
        }
        if (!isBcastGuard) {
          auto guard = createLoadLoopGuard(rewriter, record.loadStore.getLoc(),
                                           record.guardIVs, isBcastGuard);
          rewriter.setInsertionPointToStart(&guard.getThenRegion().front());
          auto [_, guardedMapper] =
              cloneAffineLoopSkeleton(rewriter, loopNestOrOp);
          irMapper = guardedMapper;
          rewriter.setInsertionPointAfter(guard);
        }
      }

      Block *fromScope = record.loadStore->getBlock();
      Block *toScope = irMapper.lookupOrNull(fromScope);
      if (toScope) {
        Operation *terminator = toScope->getTerminator();
        if (terminator) {
          rewriter.setInsertionPoint(terminator);
        } else {
          rewriter.setInsertionPointToEnd(toScope);
        }
      }

      auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
          buildIndices(rewriter, loadStoreLoc, irMapper, loadStoreIndices,
                       record.dstSlice, loadStoreMap, loadStoreMemRefType,
                       loopNestOrOp);
      copyGenerator(rewriter, record, l1AccessMap, l1AccessIndices,
                    dstAccessMap, dstAccessIndices);
    }

    {
      mlir::IRMapping dummyIRMapper;
      rewriter.setInsertionPoint(record.loadStore);
      auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
          buildIndices(rewriter, loadStoreLoc, dummyIRMapper, loadStoreIndices,
                       record.dstSlice, loadStoreMap, loadStoreMemRefType,
                       loopNestOrOp);
      accessReplacer(rewriter, record, dstAccessMap, dstAccessIndices);
    }
  }
}

// Explicit instantiations for the four LoadStoreOpTy variants used.
template void emitDstCopyNest<affine::AffineLoadOp>(
    PatternRewriter &, Operation *,
    ArrayRef<LoadStoreRecord<affine::AffineLoadOp>>,
    llvm::function_ref<void(PatternRewriter &,
                            LoadStoreRecord<affine::AffineLoadOp>, AffineMap,
                            ValueRange, AffineMap, ValueRange)>,
    llvm::function_ref<void(PatternRewriter &,
                            LoadStoreRecord<affine::AffineLoadOp>, AffineMap,
                            ValueRange)>,
    bool);
template void emitDstCopyNest<affine::AffineStoreOp>(
    PatternRewriter &, Operation *,
    ArrayRef<LoadStoreRecord<affine::AffineStoreOp>>,
    llvm::function_ref<void(PatternRewriter &,
                            LoadStoreRecord<affine::AffineStoreOp>, AffineMap,
                            ValueRange, AffineMap, ValueRange)>,
    llvm::function_ref<void(PatternRewriter &,
                            LoadStoreRecord<affine::AffineStoreOp>, AffineMap,
                            ValueRange)>,
    bool);

scf::IfOp createLoadLoopGuard(PatternRewriter &rewriter, Location loc,
                              ValueRange guardIVs, bool isBcastGuard) {
  if (guardIVs.empty()) {
    return nullptr;
  }

  // Build `guard = INIT op (iv0 cmp 0) op (iv1 cmp 0) op ...` where:
  //   - bcast init guard:  init=true,  cmp=eq, op=AND  -> "all IVs at 0"
  //   - accumulation guard: init=false, cmp=ne, op=OR  -> "any IV not 0"
  //
  // Each iteration folds the next per-IV comparison into the running `guard`
  // value; we are NOT overwriting -- the previous `guard` is the LHS of the
  // AND/OR.
  Value guard =
      rewriter
          .create<arith::ConstantOp>(loc, rewriter.getI1Type(),
                                     rewriter.getBoolAttr(isBcastGuard))
          .getResult();

  const auto cmpPredicate =
      isBcastGuard ? arith::CmpIPredicate::eq : arith::CmpIPredicate::ne;

  for (Value guardIV : guardIVs) {
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, guardIV.getType(), rewriter.getIntegerAttr(guardIV.getType(), 0));
    Value cmp =
        rewriter.create<arith::CmpIOp>(loc, cmpPredicate, guardIV, zero);
    if (isBcastGuard) {
      guard = rewriter.create<arith::AndIOp>(loc, guard, cmp).getResult();
    } else {
      guard = rewriter.create<arith::OrIOp>(loc, guard, cmp).getResult();
    }
  }

  return rewriter.create<scf::IfOp>(loc, guard);
}

std::pair<AffineMap, SmallVector<Value>>
buildLinearizedDstAccess(PatternRewriter &rewriter, Operation *op, int dstSlice,
                         Operation *linalgRoot) {
  SmallVector<affine::AffineForOp> enclosingLoops;
  Operation *current = op->getParentOp();
  while (current) {
    if (auto affineFor = dyn_cast<affine::AffineForOp>(current)) {
      enclosingLoops.push_back(affineFor);
      if (linalgRoot && current == linalgRoot) {
        break;
      }
    }
    current = current->getParentOp();
  }

  if (enclosingLoops.empty()) {
    return {AffineMap::getConstantMap(dstSlice, rewriter.getContext()), {}};
  }

  std::reverse(enclosingLoops.begin(), enclosingLoops.end());

  unsigned numDims = enclosingLoops.size();
  SmallVector<int64_t> strides(numDims, 1);
  int64_t stride = 1;
  for (int i = numDims - 1; i >= 0; --i) {
    strides[i] = stride;
    if (enclosingLoops[i].hasConstantUpperBound()) {
      stride *= enclosingLoops[i].getConstantUpperBound();
    }
  }

  AffineExpr linearExpr = getAffineConstantExpr(
      static_cast<int64_t>(dstSlice) * stride, rewriter.getContext());
  for (unsigned i = 0; i < numDims; ++i) {
    AffineExpr dimExpr = getAffineDimExpr(i, rewriter.getContext());
    linearExpr = linearExpr + dimExpr * strides[i];
  }

  AffineMap accessMap =
      AffineMap::get(numDims, 0, linearExpr, rewriter.getContext());

  SmallVector<Value> accessIndices;
  for (auto loop : enclosingLoops) {
    accessIndices.push_back(loop.getInductionVar());
  }

  return {accessMap, accessIndices};
}

void fixDstIntermediateResults(PatternRewriter &rewriter, Location loc,
                               Value dst,
                               const DstIntermediatesMap &dstIntermediates) {
  auto dstType = dyn_cast<MemRefType>(dst.getType());
  if (!dstType) {
    return;
  }

  for (const auto &[op, dstInfo] : dstIntermediates) {
    int dstSlice = dstInfo.dstSlice;

    rewriter.setInsertionPoint(op);

    auto [storeMap, storeIndices] =
        buildLinearizedDstAccess(rewriter, op, dstSlice, dstInfo.outermostLoop);

    rewriter.setInsertionPointAfter(op);

    Value originalResult = op->getResult(0);
    Type originalType = originalResult.getType();
    Value valueToStore = originalResult;
    Operation *castOp = nullptr;
    bool needsTypeCast = (originalType != dstType.getElementType());

    if (needsTypeCast) {
      auto cast = rewriter.create<d2m::DstReinterpretCastOp>(
          loc, dstType.getElementType(), valueToStore);
      valueToStore = cast.getResult();
      castOp = cast.getOperation();
    }

    auto storeOp = rewriter.create<affine::AffineStoreOp>(
        loc, valueToStore, dst, storeMap, storeIndices);

    auto loadedResult =
        rewriter.create<affine::AffineLoadOp>(loc, dst, storeMap, storeIndices);

    Value replacementValue = loadedResult.getResult();
    Operation *castBackOp = nullptr;
    if (needsTypeCast) {
      auto castBack = rewriter.create<d2m::DstReinterpretCastOp>(
          loc, originalType, replacementValue);
      replacementValue = castBack.getResult();
      castBackOp = castBack.getOperation();
    }

    rewriter.replaceUsesWithIf(
        originalResult, replacementValue, [&](mlir::OpOperand &operand) {
          Operation *owner = operand.getOwner();
          return owner != storeOp && owner != castOp && owner != castBackOp;
        });
  }
}

bool isDstScopeIV(Value iv, Operation *linalgRoot) {
  if (!linalgRoot) {
    return true;
  }
  if (auto blockArg = dyn_cast<BlockArgument>(iv)) {
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    return linalgRoot == parentOp || linalgRoot->isProperAncestor(parentOp);
  }
  if (auto *defOp = iv.getDefiningOp()) {
    return linalgRoot == defOp || linalgRoot->isProperAncestor(defOp);
  }
  return false;
}

std::tuple<AffineMap, SmallVector<Value>, AffineMap, SmallVector<Value>>
buildIndices(PatternRewriter &rewriter, Location loc,
             const mlir::IRMapping &irMapper, ValueRange currentIndices,
             int dstSlice, AffineMap map, MemRefType cbType,
             Operation *linalgRoot) {
  AffineMap l1AccessMap = map;
  SmallVector<Value> l1AccessIndices =
      llvm::to_vector(llvm::map_range(currentIndices, [&](Value index) {
        return irMapper.lookupOrDefault(index);
      }));

  ArrayRef<int64_t> cbShape = cbType.getShape();

  SmallVector<Value> dstOperands;
  SmallVector<int64_t> dstDims;

  unsigned numResults = map.getNumResults();
  for (unsigned resultDim = 0;
       resultDim < numResults && resultDim < cbShape.size(); ++resultDim) {
    AffineExpr expr = map.getResult(resultDim);

    SmallVector<unsigned, 2> dimPositions;
    expr.walk([&](AffineExpr e) {
      if (auto dimExpr = mlir::dyn_cast<AffineDimExpr>(e)) {
        if (!llvm::is_contained(dimPositions, dimExpr.getPosition())) {
          dimPositions.push_back(dimExpr.getPosition());
        }
      }
    });

    if (dimPositions.size() != 1 || dimPositions[0] >= currentIndices.size()) {
      continue;
    }

    unsigned operandIdx = dimPositions[0];
    if (isDstScopeIV(currentIndices[operandIdx], linalgRoot)) {
      dstOperands.push_back(
          irMapper.lookupOrDefault(currentIndices[operandIdx]));
      dstDims.push_back(cbShape[resultDim]);
    }
  }

  unsigned numDstDims = dstOperands.size();
  if (numDstDims == 0) {
    AffineMap dstAccessMap =
        AffineMap::getConstantMap(dstSlice, rewriter.getContext());
    return {l1AccessMap, l1AccessIndices, dstAccessMap, {}};
  }

  int64_t stride = 1;
  SmallVector<int64_t> strides(numDstDims, 1);
  for (int i = numDstDims - 1; i >= 0; --i) {
    strides[i] = stride;
    if (i < static_cast<int>(dstDims.size())) {
      stride *= dstDims[i];
    }
  }

  AffineExpr linearExpr = getAffineConstantExpr(
      static_cast<int64_t>(dstSlice) * stride, rewriter.getContext());

  for (unsigned i = 0; i < numDstDims; ++i) {
    AffineExpr dimExpr = getAffineDimExpr(i, rewriter.getContext());
    linearExpr = linearExpr + dimExpr * strides[i];
  }

  AffineMap dstAccessMap =
      AffineMap::get(numDstDims, 0, linearExpr, rewriter.getContext());
  return {l1AccessMap, l1AccessIndices, dstAccessMap, dstOperands};
}

void insertPackerL1AccGuard(PatternRewriter &rewriter, Location loc,
                            AcquireDstOp acquireDst, Value loopIV) {
  rewriter.setInsertionPointAfter(acquireDst);
  Value secondIterationValue = getSecondIterationValue(rewriter, loc, loopIV);
  Value cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                              loopIV, secondIterationValue);
  auto ifOp = rewriter.create<scf::IfOp>(loc, cond);
  rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
  Value enableFlag = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
  rewriter.create<SetL1AccumulateOp>(loc, enableFlag);
}

// ---------------------------------------------------------------------------
// Shared orchestration: insertDstRegisterAccessFinalize
// ---------------------------------------------------------------------------

bool insertDstRegisterAccessFinalize(
    PatternRewriter &rewriter, GenericOp gOp, Region &region,
    unsigned dstCapacity, Operation *outermostInnerComputeLoop,
    bool disableL1Acc, CopyInfoMap &copyInfos,
    DstIntermediatesMap &dstIntermediates,
    llvm::function_ref<void(PatternRewriter &, Location, Value,
                            const CopyInfoMap &, bool)>
        emitDataCopies) {
  assert(region.getBlocks().size() == 1);
  if (hasAcquireDstOp(region)) {
    return false;
  }

  if (copyInfos.empty()) {
    return false;
  }

  Location loc = gOp.getLoc();

  // Insert acquire_dst.
  bool isScfForLoop = isa_and_nonnull<scf::ForOp>(outermostInnerComputeLoop);
  AcquireDstOp acquireDst = insertAcquireDst(
      rewriter, loc, region, copyInfos, outermostInnerComputeLoop, dstCapacity,
      /*insertInsideLoop=*/isScfForLoop);
  Value dst = acquireDst.getResult();

  Value l1AccLoopIV = nullptr;
  if (!disableL1Acc) {
    // L1-acc must be triggered by the outermost ancestor *reduction* loop --
    // i.e. an outer loop that does NOT index the output store. Using the
    // outermost ancestor unconditionally is incorrect for batched matmuls,
    // where the outermost loop is parallel (e.g. the batch dim) and turning
    // on L1-acc on its second iteration would cause the packer to accumulate
    // a fresh batch's tile into uninitialized L1 contents at a different
    // output address.
    l1AccLoopIV = findOutermostReductionLoopIVForL1Acc(
        acquireDst.getOperation(), copyInfos);
    if (!l1AccLoopIV) {
      LDBG() << "Skipping L1 accumulation insertion: no outer reduction loop "
                "with trip count > 1";
      disableL1Acc = true;
    }
  }

  // Emit path-specific data copy loops.
  emitDataCopies(rewriter, loc, dst, copyInfos, disableL1Acc);

  // Insert optional L1 accumulation guard.
  if (!disableL1Acc) {
    insertPackerL1AccGuard(rewriter, loc, acquireDst, l1AccLoopIV);
  }

  // Fix intermediate DST results.
  fixDstIntermediateResults(rewriter, loc, dst, dstIntermediates);

  // When there's no outermost compute loop (loop was canonicalized away),
  // the acquire_dst may have been placed at the start of the region before
  // remote_load ops. Move it to just before its first use to ensure compute
  // ops are contiguous (important for SplitUnifiedThread which separates out
  // contiguous regions of compute ops into synchronized regions). When there
  // IS an outermost compute loop, the insertion point was already set
  // correctly (inside or before the loop body), so no move is needed.
  if (!outermostInnerComputeLoop) {
    if (Operation *firstUser =
            ttmlir::utils::findFirstUserInBlock(acquireDst)) {
      acquireDst->moveBefore(firstUser);
    }
  }

  return true;
}

} // namespace detail
} // namespace mlir::tt::d2m
