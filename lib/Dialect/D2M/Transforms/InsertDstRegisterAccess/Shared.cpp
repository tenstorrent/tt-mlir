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

#include <algorithm>
#include <type_traits>

#define DEBUG_TYPE "D2MInsertDstRegisterAccess"

namespace mlir::tt::d2m {

namespace {

static std::optional<int64_t>
tryGetConstantAffineTripCount(affine::AffineForOp affineFor) {
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

static int64_t getRequiredConstantAffineTripCount(affine::AffineForOp loop) {
  std::optional<int64_t> tripCount = tryGetConstantAffineTripCount(loop);
  TT_assertv(tripCount.has_value(),
             "DST register access linearization requires constant-bounded "
             "affine.for loops");
  return tripCount.value_or(1);
}

static Operation *findEnclosingLinalgRoot(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent->hasAttr("d2m.linalg_root")) {
      return parent;
    }
  }
  return nullptr;
}

static SmallVector<affine::AffineForOp>
collectEnclosingAffineLoopsForDstAccess(Operation *op, Operation *linalgRoot) {
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
  std::reverse(enclosingLoops.begin(), enclosingLoops.end());
  return enclosingLoops;
}

} // namespace

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

  walkResult = moduleOp->walk(
      [&](OperandLoadStoreRegisterOpInterface computeOp) -> WalkResult {
        Operation *op = computeOp.getOperation();
        Operation *linalgRoot = findEnclosingLinalgRoot(op);
        for (affine::AffineForOp loop :
             collectEnclosingAffineLoopsForDstAccess(op, linalgRoot)) {
          if (!tryGetConstantAffineTripCount(loop)) {
            op->emitOpError()
                << "requires constant-bounded affine.for loops for DST "
                   "register access linearization";
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });

  if (walkResult.wasInterrupted()) {
    return failure();
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

static bool accessDependsOnIV(Operation *op, DstAccessKind kind, Value iv) {
  switch (kind) {
  case DstAccessKind::AffineLoad:
    return accessDependsOnIV(cast<affine::AffineLoadOp>(op), iv);
  case DstAccessKind::AffineStore:
    return accessDependsOnIV(cast<affine::AffineStoreOp>(op), iv);
  case DstAccessKind::MemrefLoad:
    return accessDependsOnIV(cast<memref::LoadOp>(op), iv);
  case DstAccessKind::MemrefStore:
    return accessDependsOnIV(cast<memref::StoreOp>(op), iv);
  }
  llvm_unreachable("unknown DstAccessKind");
}

static bool accessDependsOnIV(const DstAccess &access, Value iv) {
  return accessDependsOnIV(access.op, access.kind, iv);
}

static Operation *getLoopOpForIV(Value iv) {
  auto blockArg = mlir::dyn_cast<BlockArgument>(iv);
  if (!blockArg) {
    return nullptr;
  }
  return blockArg.getOwner()->getParentOp();
}

static std::optional<bool> isReductionBlockingLoop(Operation *loopOp) {
  auto blockingLoopAttr =
      loopOp->getAttrOfType<IntegerAttr>("d2m.blocking_loop");
  if (!blockingLoopAttr) {
    return std::nullopt;
  }

  if (loopOp->hasAttr("d2m.reduction_loop")) {
    return true;
  }

  auto genericOp = loopOp->getParentOfType<GenericOp>();
  if (!genericOp) {
    return false;
  }

  int64_t dim = blockingLoopAttr.getInt();
  auto iteratorTypes = genericOp.getIteratorTypes();
  if (iteratorTypes.empty() || dim < 0 ||
      static_cast<size_t>(dim) >= iteratorTypes.size()) {
    return false;
  }

  auto iteratorType =
      mlir::cast<ttcore::IteratorTypeAttr>(iteratorTypes[dim]).getValue();
  return iteratorType == ttcore::IteratorType::Reduction;
}

static SmallVector<Value> getGuardLoopIVs(Operation *loadOrStore,
                                          DstAccessKind kind,
                                          Operation *contextOp) {
  SmallVector<Value> guardIVs;
  for (Value loopIV : collectAncestorLoopIVs(contextOp)) {
    Operation *loopOp = getLoopOpForIV(loopIV);

    if (loopOp) {
      std::optional<bool> isReduction = isReductionBlockingLoop(loopOp);
      if (isReduction.has_value() && !*isReduction) {
        continue;
      }
    }

    if (!accessDependsOnIV(loadOrStore, kind, loopIV)) {
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

// Returns true iff any store recorded for this region depends on `iv`.
// "Depends on" includes transitive dependence through subview indices.
static bool anyOutputStoreDependsOnIV(const CopyInfoMap &copyInfos, Value iv) {
  for (const auto &[loopOrOp, copyInfo] : copyInfos) {
    for (const auto &access : copyInfo.stores) {
      if (access.op && accessDependsOnIV(access, iv)) {
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
    return tryGetConstantAffineTripCount(affineFor);
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

Value findClosestReductionLoopIVForL1Acc(Operation *acquireDstOp,
                                         const CopyInfoMap &copyInfos) {
  // `collectAncestorLoopIVs` returns IVs in outermost-to-innermost order
  // (it reverses the upward walk).
  SmallVector<Value> ancestorIVs = collectAncestorLoopIVs(acquireDstOp);
  for (Value iv : llvm::reverse(ancestorIVs)) {
    Operation *loopOp = getLoopOpForIV(iv);
    if (!loopOp) {
      continue;
    }

    std::optional<bool> isReduction = isReductionBlockingLoop(loopOp);
    if (isReduction.has_value() && !*isReduction) {
      continue;
    }
    if (!isReduction.has_value() && anyOutputStoreDependsOnIV(copyInfos, iv)) {
      // Non-blocking fallback: if this loop indexes the output, it is parallel,
      // not a reduction.
      continue;
    }

    // Found the closest reduction loop. Only enable L1-acc if the loop
    // actually iterates more than once (otherwise the per-iteration
    // accumulation guard would never fire and we'd just emit dead code).
    std::optional<int64_t> tripCount = tryGetConstantTripCount(loopOp);
    if (tripCount.has_value() && *tripCount <= 1) {
      continue;
    }
    return iv;
  }
  return nullptr;
}

static int64_t
computeDstLinearizationFootprint(ArrayRef<affine::AffineForOp> enclosingLoops) {
  int64_t footprint = 1;
  for (affine::AffineForOp loop : enclosingLoops) {
    footprint *= getRequiredConstantAffineTripCount(loop);
  }
  return footprint;
}

static int64_t computeLinearizedDstSliceBaseIndex(Operation *op, int dstSlice,
                                                  Operation *linalgRoot) {
  return static_cast<int64_t>(dstSlice) *
         computeDstLinearizationFootprint(
             collectEnclosingAffineLoopsForDstAccess(op, linalgRoot));
}

void setDstScratchIndex(OperandLoadStoreRegisterOpInterface computeOp,
                        int scratchSlice, Operation *linalgRoot) {
  TT_assertv(computeOp.getNumDstScratchSlices() == 1,
             "setDstScratchIndex supports exactly one scratch slice");
  Operation *op = computeOp.getOperation();
  int64_t dstIndex =
      computeLinearizedDstSliceBaseIndex(op, scratchSlice, linalgRoot);
  op->setAttr("dst_scratch_index",
              mlir::IntegerAttr::get(
                  mlir::IntegerType::get(op->getContext(), 64), dstIndex));
}

static Value getFirstIterationValue(PatternRewriter &rewriter, Location loc,
                                    Value loopIV) {
  auto zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 0));

  auto ivBlockArg = mlir::dyn_cast<BlockArgument>(loopIV);
  if (!ivBlockArg) {
    return zero;
  }

  auto *ownerBlock = ivBlockArg.getOwner();
  if (!ownerBlock) {
    return zero;
  }

  auto *ownerOp = ownerBlock->getParentOp();
  if (!ownerOp) {
    return zero;
  }

  if (auto scfFor = mlir::dyn_cast<scf::ForOp>(ownerOp)) {
    return scfFor.getLowerBound();
  }

  if (auto affineFor = mlir::dyn_cast<affine::AffineForOp>(ownerOp)) {
    if (affineFor.hasConstantLowerBound()) {
      return rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(),
          rewriter.getIntegerAttr(rewriter.getIndexType(),
                                  affineFor.getConstantLowerBound()));
    }

    AffineMap lowerBoundMap = affineFor.getLowerBoundMap();
    if (lowerBoundMap.getNumResults() == 1) {
      return rewriter.create<affine::AffineApplyOp>(
          loc, lowerBoundMap, affineFor.getLowerBoundOperands());
    }
  }

  return zero;
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
    for (const auto &access : copyInfo.loads) {
      updateInfo(access.getMemRefType(), access.dstSlice);
    }
    for (const auto &access : copyInfo.stores) {
      updateInfo(access.getMemRefType(), access.dstSlice);
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

// ---------------------------------------------------------------------------
// DstAccess / CopyInfo
// ---------------------------------------------------------------------------

bool DstAccess::isLoad() const {
  return kind == DstAccessKind::AffineLoad || kind == DstAccessKind::MemrefLoad;
}

bool DstAccess::isStore() const {
  return kind == DstAccessKind::AffineStore ||
         kind == DstAccessKind::MemrefStore;
}

bool DstAccess::isAffine() const {
  return kind == DstAccessKind::AffineLoad ||
         kind == DstAccessKind::AffineStore;
}

bool DstAccess::isMemref() const {
  return kind == DstAccessKind::MemrefLoad ||
         kind == DstAccessKind::MemrefStore;
}

Location DstAccess::getLoc() const { return op->getLoc(); }

Value DstAccess::getMemRef() const {
  switch (kind) {
  case DstAccessKind::AffineLoad:
    return cast<affine::AffineLoadOp>(op).getMemref();
  case DstAccessKind::AffineStore:
    return cast<affine::AffineStoreOp>(op).getMemref();
  case DstAccessKind::MemrefLoad:
    return cast<memref::LoadOp>(op).getMemRef();
  case DstAccessKind::MemrefStore:
    return cast<memref::StoreOp>(op).getMemRef();
  }
  llvm_unreachable("unknown DstAccessKind");
}

MemRefType DstAccess::getMemRefType() const {
  return cast<MemRefType>(getMemRef().getType());
}

AffineMap DstAccess::getAffineMap() const {
  TT_assert(isAffine());
  if (kind == DstAccessKind::AffineLoad) {
    return cast<affine::AffineLoadOp>(op).getAffineMap();
  }
  return cast<affine::AffineStoreOp>(op).getAffineMap();
}

ValueRange DstAccess::getAffineIndices() const {
  TT_assert(isAffine());
  if (kind == DstAccessKind::AffineLoad) {
    return cast<affine::AffineLoadOp>(op).getIndices();
  }
  return cast<affine::AffineStoreOp>(op).getIndices();
}

ValueRange DstAccess::getMemrefIndices() const {
  TT_assert(isMemref());
  if (kind == DstAccessKind::MemrefLoad) {
    return cast<memref::LoadOp>(op).getIndices();
  }
  return cast<memref::StoreOp>(op).getIndices();
}

void CopyInfo::record(DstAccess access) {
  if (access.isLoad()) {
    loads.push_back(std::move(access));
  } else {
    stores.push_back(std::move(access));
  }
}

void CopyInfo::record(affine::AffineLoadOp load, int dstSlice,
                      ArrayRef<Value> guardIVs) {
  record(DstAccess(load.getOperation(), DstAccessKind::AffineLoad, dstSlice,
                   std::nullopt, guardIVs));
}

void CopyInfo::record(affine::AffineLoadOp load, d2m::TileBcastOp bcast,
                      int dstSlice, ArrayRef<Value> guardIVs) {
  record(DstAccess(load.getOperation(), DstAccessKind::AffineLoad, dstSlice,
                   bcast, guardIVs));
}

void CopyInfo::record(affine::AffineStoreOp store, int dstSlice,
                      ArrayRef<Value>) {
  record(DstAccess(store.getOperation(), DstAccessKind::AffineStore, dstSlice,
                   std::nullopt, {}));
}

void CopyInfo::record(memref::LoadOp load, int dstSlice,
                      ArrayRef<Value> guardIVs) {
  record(DstAccess(load.getOperation(), DstAccessKind::MemrefLoad, dstSlice,
                   std::nullopt, guardIVs));
}

void CopyInfo::record(memref::StoreOp store, int dstSlice, ArrayRef<Value>) {
  record(DstAccess(store.getOperation(), DstAccessKind::MemrefStore, dstSlice,
                   std::nullopt, {}));
}

// Core recording logic: record a load/store with an optional guard.
static void recordDstAccessImpl(Operation *loadOrStore, DstAccessKind kind,
                                std::optional<d2m::TileBcastOp> bcast,
                                CopyInfoMap &copyInfos, int dstSlice,
                                Operation *outermostInnerComputeLoop,
                                bool emitGuard) {
  if (!outermostInnerComputeLoop) {
    outermostInnerComputeLoop = loadOrStore;
  }

  auto [iter, _] = copyInfos.try_emplace(outermostInnerComputeLoop);

  Value memref;
  switch (kind) {
  case DstAccessKind::AffineLoad:
    memref = cast<affine::AffineLoadOp>(loadOrStore).getMemref();
    break;
  case DstAccessKind::AffineStore:
    memref = cast<affine::AffineStoreOp>(loadOrStore).getMemref();
    break;
  case DstAccessKind::MemrefLoad:
    memref = cast<memref::LoadOp>(loadOrStore).getMemRef();
    break;
  case DstAccessKind::MemrefStore:
    memref = cast<memref::StoreOp>(loadOrStore).getMemRef();
    break;
  }

  SmallVector<Value> guardIVs;
  if (lookThroughSubView(memref) && emitGuard) {
    guardIVs = getGuardLoopIVs(loadOrStore, kind, outermostInnerComputeLoop);
  }

  iter->second.record(DstAccess(loadOrStore, kind, dstSlice, bcast, guardIVs));
}

void recordDstAccess(affine::AffineLoadOp op, CopyInfoMap &copyInfos,
                     int dstSlice, Operation *outermostInnerComputeLoop,
                     bool emitGuard) {
  recordDstAccessImpl(op.getOperation(), DstAccessKind::AffineLoad,
                      std::nullopt, copyInfos, dstSlice,
                      outermostInnerComputeLoop, emitGuard);
}

void recordDstAccess(affine::AffineStoreOp op, CopyInfoMap &copyInfos,
                     int dstSlice, Operation *outermostInnerComputeLoop,
                     bool emitGuard) {
  recordDstAccessImpl(op.getOperation(), DstAccessKind::AffineStore,
                      std::nullopt, copyInfos, dstSlice,
                      outermostInnerComputeLoop, emitGuard);
}

void recordDstAccess(memref::LoadOp op, CopyInfoMap &copyInfos, int dstSlice,
                     Operation *outermostInnerComputeLoop, bool emitGuard) {
  recordDstAccessImpl(op.getOperation(), DstAccessKind::MemrefLoad,
                      std::nullopt, copyInfos, dstSlice,
                      outermostInnerComputeLoop, emitGuard);
}

void recordDstAccess(memref::StoreOp op, CopyInfoMap &copyInfos, int dstSlice,
                     Operation *outermostInnerComputeLoop, bool emitGuard) {
  recordDstAccessImpl(op.getOperation(), DstAccessKind::MemrefStore,
                      std::nullopt, copyInfos, dstSlice,
                      outermostInnerComputeLoop, emitGuard);
}

void recordDstAccess(affine::AffineLoadOp loadOp, d2m::TileBcastOp bcastOp,
                     CopyInfoMap &copyInfos, int dstSlice,
                     Operation *outermostInnerComputeLoop, bool emitGuard) {
  recordDstAccessImpl(loadOp.getOperation(), DstAccessKind::AffineLoad, bcastOp,
                      copyInfos, dstSlice, outermostInnerComputeLoop,
                      emitGuard);
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
static void collectDstStoreAccessImpl(Operation *storeOp, DstAccessKind kind,
                                      CopyInfoMap &copyInfos, int dstSlice,
                                      Operation *outermostInnerComputeLoop) {
  recordDstAccessImpl(storeOp, kind, std::nullopt, copyInfos, dstSlice,
                      outermostInnerComputeLoop, /*emitGuard=*/false);
}

void collectDstStoreAccess(affine::AffineStoreOp storeOp,
                           CopyInfoMap &copyInfos, int dstSlice,
                           Operation *outermostInnerComputeLoop) {
  collectDstStoreAccessImpl(storeOp.getOperation(), DstAccessKind::AffineStore,
                            copyInfos, dstSlice, outermostInnerComputeLoop);
}

void collectDstStoreAccess(memref::StoreOp storeOp, CopyInfoMap &copyInfos,
                           int dstSlice, Operation *outermostInnerComputeLoop) {
  collectDstStoreAccessImpl(storeOp.getOperation(), DstAccessKind::MemrefStore,
                            copyInfos, dstSlice, outermostInnerComputeLoop);
}

// Collect a single load access and determine whether it needs an accumulation
// guard.
template <typename LoadTy>
static void collectDstLoadWithAccumAnalysisImpl(
    LoadTy loadOp, DstAccessKind kind, int64_t operandIdx,
    ValueRange carriedOutputRegions, ArrayRef<int64_t> accumOperandIndices,
    CopyInfoMap &copyInfos, int dstSlice, Operation *outermostInnerComputeLoop,
    bool noAccumGuard) {
  const bool emitGuard = shouldGuardDstLoadForAccumulation(
      loadOp, operandIdx, carriedOutputRegions, accumOperandIndices,
      noAccumGuard);
  recordDstAccessImpl(loadOp.getOperation(), kind, std::nullopt, copyInfos,
                      dstSlice, outermostInnerComputeLoop, emitGuard);
}

void collectDstLoadWithAccumAnalysis(affine::AffineLoadOp loadOp,
                                     int64_t operandIdx,
                                     ValueRange carriedOutputRegions,
                                     ArrayRef<int64_t> accumOperandIndices,
                                     CopyInfoMap &copyInfos, int dstSlice,
                                     Operation *outermostInnerComputeLoop,
                                     bool noAccumGuard) {
  collectDstLoadWithAccumAnalysisImpl(loadOp, DstAccessKind::AffineLoad,
                                      operandIdx, carriedOutputRegions,
                                      accumOperandIndices, copyInfos, dstSlice,
                                      outermostInnerComputeLoop, noAccumGuard);
}

void collectDstLoadWithAccumAnalysis(memref::LoadOp loadOp, int64_t operandIdx,
                                     ValueRange carriedOutputRegions,
                                     ArrayRef<int64_t> accumOperandIndices,
                                     CopyInfoMap &copyInfos, int dstSlice,
                                     Operation *outermostInnerComputeLoop,
                                     bool noAccumGuard) {
  collectDstLoadWithAccumAnalysisImpl(loadOp, DstAccessKind::MemrefLoad,
                                      operandIdx, carriedOutputRegions,
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

void replaceLoadWithDst(PatternRewriter &rewriter, const DstAccess &access,
                        Value dst, AffineMap dstAccessMap,
                        ValueRange dstAccessIndices) {
  switch (access.kind) {
  case DstAccessKind::AffineLoad: {
    auto dstLoad = rewriter.create<affine::AffineLoadOp>(
        access.getLoc(), dst, dstAccessMap, dstAccessIndices);
    if (access.bcast.has_value()) {
      // Rewrites IR only; `access` is not mutated (bcast op is erased).
      Operation *bcastOp = *access.bcast;
      cast<d2m::TileBcastOp>(bcastOp).getResult().replaceAllUsesWith(
          dstLoad.getResult());
      rewriter.eraseOp(bcastOp);
    } else {
      rewriter.replaceOp(access.op, dstLoad.getResult());
    }
    break;
  }
  case DstAccessKind::MemrefLoad: {
    auto dstLoad = rewriter.create<affine::AffineLoadOp>(
        access.getLoc(), dst, dstAccessMap, dstAccessIndices);
    rewriter.replaceOp(access.op, dstLoad.getResult());
    break;
  }
  default:
    llvm_unreachable("replaceLoadWithDst expects a load access");
  }
}

void replaceStoreWithDst(PatternRewriter &rewriter, const DstAccess &access,
                         Value dst, AffineMap dstAccessMap,
                         ValueRange dstAccessIndices) {
  switch (access.kind) {
  case DstAccessKind::AffineStore: {
    auto storeOp = cast<affine::AffineStoreOp>(access.op);
    Value valueToStore = storeOp.getValue();
    auto dstType = cast<MemRefType>(dst.getType());
    if (valueToStore.getType() != dstType.getElementType()) {
      valueToStore =
          rewriter
              .create<d2m::DstReinterpretCastOp>(
                  storeOp.getLoc(), dstType.getElementType(), valueToStore)
              .getResult();
    }
    rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
        storeOp, valueToStore, dst, dstAccessMap, dstAccessIndices);
    break;
  }
  case DstAccessKind::MemrefStore: {
    auto storeOp = cast<memref::StoreOp>(access.op);
    Value valueToStore = storeOp.getValue();
    auto dstType = cast<MemRefType>(dst.getType());
    if (valueToStore.getType() != dstType.getElementType()) {
      valueToStore =
          rewriter
              .create<d2m::DstReinterpretCastOp>(
                  storeOp.getLoc(), dstType.getElementType(), valueToStore)
              .getResult();
    }
    rewriter.create<affine::AffineStoreOp>(storeOp.getLoc(), valueToStore, dst,
                                           dstAccessMap, dstAccessIndices);

    auto dstLoad = rewriter.create<affine::AffineLoadOp>(
        storeOp.getLoc(), dst, dstAccessMap, dstAccessIndices);
    Value packValue = dstLoad.getResult();
    auto cbType = cast<MemRefType>(storeOp.getMemRef().getType());
    if (packValue.getType() != cbType.getElementType()) {
      packValue = rewriter
                      .create<d2m::DstReinterpretCastOp>(
                          storeOp.getLoc(), cbType.getElementType(), packValue)
                      .getResult();
    }
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        storeOp, packValue, storeOp.getMemRef(), storeOp.getIndices());
    break;
  }
  default:
    llvm_unreachable("replaceStoreWithDst expects a store access");
  }
}

void generateLoadSideCopy(PatternRewriter &rewriter, const DstAccess &access,
                          Value dst, AffineMap l1AccessMap,
                          ValueRange l1AccessIndices, AffineMap dstAccessMap,
                          ValueRange dstAccessIndices) {
  auto loc = access.getLoc();
  Value cb = access.getMemRef();

  switch (access.kind) {
  case DstAccessKind::AffineLoad: {
    auto cbLoad = rewriter.create<affine::AffineLoadOp>(loc, cb, l1AccessMap,
                                                        l1AccessIndices);
    Value valueToStore = cbLoad.getResult();
    if (access.bcast.has_value()) {
      rewriter.setInsertionPointAfter(cbLoad);
      Operation *bcastOp = *access.bcast;
      auto *clonedBcast = rewriter.clone(*bcastOp);
      clonedBcast->setOperand(0, valueToStore);
      valueToStore = clonedBcast->getResult(0);
    }
    rewriter.create<affine::AffineStoreOp>(loc, valueToStore, dst, dstAccessMap,
                                           dstAccessIndices);
    break;
  }
  case DstAccessKind::MemrefLoad: {
    auto cbLoad =
        rewriter.create<memref::LoadOp>(loc, cb, access.getMemrefIndices());
    rewriter.create<affine::AffineStoreOp>(loc, cbLoad.getResult(), dst,
                                           dstAccessMap, dstAccessIndices);
    break;
  }
  default:
    llvm_unreachable("generateLoadSideCopy expects a load access");
  }
}

void generateStoreSideCopy(PatternRewriter &rewriter, const DstAccess &access,
                           Value dst, AffineMap l1AccessMap,
                           ValueRange l1AccessIndices, AffineMap dstAccessMap,
                           ValueRange dstAccessIndices) {
  auto loc = access.getLoc();
  Value cb = access.getMemRef();

  switch (access.kind) {
  case DstAccessKind::AffineStore: {
    auto dstLoad = rewriter.create<affine::AffineLoadOp>(loc, dst, dstAccessMap,
                                                         dstAccessIndices);
    Value valueToStore = dstLoad.getResult();
    auto cbType = cast<MemRefType>(cb.getType());
    if (valueToStore.getType() != cbType.getElementType()) {
      valueToStore = rewriter
                         .create<d2m::DstReinterpretCastOp>(
                             loc, cbType.getElementType(), valueToStore)
                         .getResult();
    }
    rewriter.create<affine::AffineStoreOp>(loc, valueToStore, cb, l1AccessMap,
                                           l1AccessIndices);
    break;
  }
  case DstAccessKind::MemrefStore:
    // Memref stores use in-place replaceStoreWithDst only (no upfront CB copy).
    break;
  default:
    llvm_unreachable("generateStoreSideCopy expects a store access");
  }
}

void emitDstCopyNest(PatternRewriter &rewriter, Operation *loopNestOrOp,
                     Value dst, ArrayRef<DstAccess> accesses, bool isLoadSide,
                     bool cloneLoopNest, bool disableL1Acc) {
  if (accesses.empty()) {
    return;
  }

  Operation *copyLoop = nullptr;
  mlir::IRMapping copyLoopMapper;
  if (disableL1Acc && cloneLoopNest) {
    std::tie(copyLoop, copyLoopMapper) =
        cloneAffineLoopSkeleton(rewriter, loopNestOrOp);
  }

  for (const DstAccess &access : accesses) {
    const bool useInPlace = !cloneLoopNest || access.isMemref();

    if (useInPlace) {
      mlir::IRMapping emptyIRMapper;
      rewriter.setInsertionPoint(access.op);

      AffineMap l1AccessMap;
      SmallVector<Value> l1AccessIndices;
      AffineMap dstAccessMap;
      SmallVector<Value> dstAccessIndices;

      if (access.isMemref()) {
        dstAccessMap =
            AffineMap::getConstantMap(access.dstSlice, rewriter.getContext());
        l1AccessIndices.reserve(access.getMemrefIndices().size());
        for (Value index : access.getMemrefIndices()) {
          l1AccessIndices.push_back(emptyIRMapper.lookupOrDefault(index));
        }
      } else {
        std::tie(l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices) =
            buildIndices(rewriter, access.getLoc(), emptyIRMapper, access,
                         loopNestOrOp);
      }

      if (disableL1Acc) {
        if (isLoadSide) {
          generateLoadSideCopy(rewriter, access, dst, l1AccessMap,
                               l1AccessIndices, dstAccessMap, dstAccessIndices);
        } else if (access.isAffine()) {
          generateStoreSideCopy(rewriter, access, dst, l1AccessMap,
                                l1AccessIndices, dstAccessMap,
                                dstAccessIndices);
        }
      }

      if (isLoadSide) {
        replaceLoadWithDst(rewriter, access, dst, dstAccessMap,
                           dstAccessIndices);
      } else {
        replaceStoreWithDst(rewriter, access, dst, dstAccessMap,
                            dstAccessIndices);
      }
      continue;
    }

    // Cloned affine loop nest path (unscheduled loads/stores).
    TT_assert(access.isAffine());
    auto loadStoreLoc = access.getLoc();

    if (disableL1Acc) {
      mlir::IRMapping irMapper = copyLoopMapper;
      if (!access.guardIVs.empty()) {
        const bool isBcastGuard = access.bcast.has_value();
        // TODO(wenbinlyuTT): #6516 WA to put all bcast inits to the top of
        // the compute tiling loops.
        if (isBcastGuard && copyLoop) {
          rewriter.setInsertionPoint(copyLoop);
        }
        if (!isBcastGuard) {
          auto guard = createLoadLoopGuard(rewriter, access.getLoc(),
                                           access.guardIVs, isBcastGuard);
          rewriter.setInsertionPointToStart(&guard.getThenRegion().front());
          auto [_, guardedMapper] =
              cloneAffineLoopSkeleton(rewriter, loopNestOrOp);
          irMapper = guardedMapper;
          rewriter.setInsertionPointAfter(guard);
        }
      }

      Block *fromScope = access.op->getBlock();
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
          buildIndices(rewriter, loadStoreLoc, irMapper, access, loopNestOrOp);

      if (isLoadSide) {
        generateLoadSideCopy(rewriter, access, dst, l1AccessMap,
                             l1AccessIndices, dstAccessMap, dstAccessIndices);
      } else {
        generateStoreSideCopy(rewriter, access, dst, l1AccessMap,
                              l1AccessIndices, dstAccessMap, dstAccessIndices);
      }
    }

    {
      mlir::IRMapping dummyIRMapper;
      rewriter.setInsertionPoint(access.op);
      auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
          buildIndices(rewriter, loadStoreLoc, dummyIRMapper, access,
                       loopNestOrOp);
      if (isLoadSide) {
        replaceLoadWithDst(rewriter, access, dst, dstAccessMap,
                           dstAccessIndices);
      } else {
        replaceStoreWithDst(rewriter, access, dst, dstAccessMap,
                            dstAccessIndices);
      }
    }
  }
}

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
  SmallVector<affine::AffineForOp> enclosingLoops =
      collectEnclosingAffineLoopsForDstAccess(op, linalgRoot);

  if (enclosingLoops.empty()) {
    return {AffineMap::getConstantMap(dstSlice, rewriter.getContext()), {}};
  }

  unsigned numDims = enclosingLoops.size();
  SmallVector<int64_t> strides(numDims, 1);
  SmallVector<int64_t> lowerBounds(numDims, 0);
  SmallVector<int64_t> steps(numDims, 1);
  int64_t stride = 1;
  for (int i = numDims - 1; i >= 0; --i) {
    affine::AffineForOp loop = enclosingLoops[i];
    strides[i] = stride;
    lowerBounds[i] = loop.getConstantLowerBound();
    steps[i] = loop.getStepAsInt();
    stride *= getRequiredConstantAffineTripCount(loop);
  }

  AffineExpr linearExpr = getAffineConstantExpr(
      static_cast<int64_t>(dstSlice) * stride, rewriter.getContext());
  for (unsigned i = 0; i < numDims; ++i) {
    AffineExpr dimExpr = getAffineDimExpr(i, rewriter.getContext());
    if (lowerBounds[i] != 0) {
      dimExpr = dimExpr - lowerBounds[i];
    }
    if (steps[i] != 1) {
      dimExpr = dimExpr.floorDiv(steps[i]);
    }
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
             const mlir::IRMapping &irMapper, const DstAccess &access,
             Operation *linalgRoot) {
  TT_assert(access.isAffine());
  AffineMap map = access.getAffineMap();
  ValueRange currentIndices = access.getAffineIndices();
  int dstSlice = access.dstSlice;
  MemRefType cbType = access.getMemRefType();

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
  Operation *loopOp = getLoopOpForIV(loopIV);
  if (!loopOp) {
    return;
  }

  rewriter.setInsertionPointAfter(acquireDst);
  Value firstIterationValue = getFirstIterationValue(rewriter, loc, loopIV);
  Value isFirstIteration = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, loopIV, firstIterationValue);
  Value disableFlag = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
  Value enableFlag = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
  Value flag = rewriter.create<arith::SelectOp>(loc, isFirstIteration,
                                                disableFlag, enableFlag);
  rewriter.create<SetL1AccumulateOp>(loc, flag);

  // Packer L1-acc is sticky. Scope it to the reduction loop so enclosing
  // parallel M/N reblock iterations start from a clean packer state.
  rewriter.setInsertionPointAfter(loopOp);
  Value resetFlag = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
  rewriter.create<SetL1AccumulateOp>(loc, resetFlag);
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
    // L1-acc must be triggered by the closest ancestor reduction loop: the
    // innermost enclosing loop that does NOT index the output store. Outer
    // parallel loops, and outer loops that only look reduction-like because
    // the immediate store is to a scratch buffer, are not valid triggers.
    l1AccLoopIV = findClosestReductionLoopIVForL1Acc(acquireDst.getOperation(),
                                                     copyInfos);
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
