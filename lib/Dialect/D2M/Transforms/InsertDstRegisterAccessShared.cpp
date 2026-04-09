// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/InsertDstRegisterAccessShared.h"

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
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"

#include <type_traits>

#define DEBUG_TYPE "D2MInsertDstRegisterAccess"

namespace mlir::tt::d2m {

// ---------------------------------------------------------------------------
// Preconditions
// ---------------------------------------------------------------------------

LogicalResult
verifyInsertDstRegisterAccessPreconditions(ModuleOp moduleOp) {
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
// DstStackAllocator implementation
// ---------------------------------------------------------------------------

DstStackAllocator::DstStackAllocator(unsigned dstSliceCapacityIn) {
  dstSliceCapacity = dstSliceCapacityIn;
  initSliceStack();
}

unsigned DstStackAllocator::allocate(bool isStore) {
  assert(!sliceStack.empty() && "Out of dst slices");

  currSliceIndex = sliceStack.pop_back_val();

  if (isStore) {
    outputQueue.push_back(currSliceIndex);
  } else {
    inputStack.push_back(currSliceIndex);
  }

  LDBG() << "========== ALLOCATE ==========";

  std::string sliceStackStr = "SliceStack = ";
  for (auto it : sliceStack) {
    sliceStackStr += std::to_string(it) + ",";
  }
  LDBG() << sliceStackStr << " --> " << currSliceIndex;

  std::string inputStackStr = "InputStack = ";
  for (auto it : inputStack) {
    inputStackStr += std::to_string(it) + ",";
  }
  LDBG() << inputStackStr;

  std::string outputStackStr = "OutputStack = ";
  for (auto it : outputQueue) {
    outputStackStr += std::to_string(it) + ",";
  }
  LDBG() << outputStackStr;

  return currSliceIndex;
}

unsigned DstStackAllocator::deallocate() {
  assert(!(inputStack.empty() && outputQueue.empty()) &&
         "Deallocating non-existent dst slice");

  unsigned id;

  if (!inputStack.empty()) {
    id = inputStack.pop_back_val();
  } else {
    if (outputQueue.size() > 1) {
      id = outputQueue.at(outputQueue.size() - 2);
      outputQueue.erase(outputQueue.end() - 2);
    } else {
      id = outputQueue.back();
      outputQueue.pop_back();
    }
  }

  sliceStack.push_back(id);

  LDBG() << "======== DEALLOCATE =========";

  std::string sliceStackStr = "SliceStack = ";
  for (auto it : sliceStack) {
    sliceStackStr += std::to_string(it) + ",";
  }
  LDBG() << sliceStackStr;

  std::string inputStackStr = "InputStack = ";
  for (auto it : inputStack) {
    inputStackStr += std::to_string(it) + ",";
  }
  LDBG() << inputStackStr;

  std::string outputStackStr = "OutputStack = ";
  for (auto it : outputQueue) {
    outputStackStr += std::to_string(it) + ",";
  }
  LDBG() << outputStackStr << " --> " << id;

  return id;
}

unsigned DstStackAllocator::getFirstInputSliceIndex() {
  assert(!inputStack.empty() && "No input slots allocated");
  return inputStack.front();
}

void DstStackAllocator::deallocateAllButFirstInput() {
  assert(inputStack.size() >= 1 && "Need at least one input to keep");

  unsigned firstInput = inputStack.front();
  inputStack.erase(inputStack.begin());

  while (!inputStack.empty()) {
    unsigned id = inputStack.pop_back_val();
    sliceStack.push_back(id);

    LDBG() << "======== DEALLOCATE (keeping first) =========";
    std::string sliceStackStr = "SliceStack = ";
    for (auto it : sliceStack) {
      sliceStackStr += std::to_string(it) + ",";
    }
    LDBG() << sliceStackStr;
  }

  currSliceIndex = firstInput;
}

void DstStackAllocator::initSliceStack() {
  assert(dstSliceCapacity > 0 && dstSliceCapacity <= 16);

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
  bool found = false;
  op->walk([&](d2m::TileMatmulOp) {
    found = true;
    return WalkResult::interrupt();
  });
  return found;
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

OperationTypes getOperationTypes(GenericOp gOp, unsigned regionIndex) {
  OperationTypes types;
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

std::pair<Type, int>
inferDstInfoFromAllAccesses(const CopyInfoMap &copyInfos) {
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
      } else if (auto affineFor = dyn_cast<affine::AffineForOp>(
                     outermostInnerComputeLoop)) {
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
    } else if (auto allocOp = mlir::dyn_cast<memref::AllocOp>(definingOp)) {
      Value assocOperand = GenericOp::findAssocOperand(allocOp);
      if (!assocOperand) {
        return nullptr;
      }
      Value cb = GenericOp::findAssocCBByOperand(allocOp.getOperation(),
                                                 assocOperand);
      if (cb) {
        return cb;
      }
      return nullptr;
    }
  }
  if (mlir::isa<BlockArgument>(memref)) {
    return memref;
  }
  return nullptr;
}

// Template implementation for collectDstLoadOrStore.
template <typename LoadOrStoreTy>
static void collectDstLoadOrStoreImpl(GenericOp gOp,
                                      LoadOrStoreTy loadOrStore,
                                      CopyInfoMap &copyInfos, int dstSlice,
                                      Operation *outermostInnerComputeLoop,
                                      bool noAccumGuard) {
  if (!outermostInnerComputeLoop) {
    outermostInnerComputeLoop = loadOrStore;
  }

  auto [iter, _] = copyInfos.try_emplace(outermostInnerComputeLoop);
  Value assocCB = lookThroughSubView(loadOrStore.getMemRef());

  SmallVector<Value> guardIVs;
  if (assocCB && !noAccumGuard) {
    guardIVs = getGuardLoopIVs(loadOrStore, outermostInnerComputeLoop);
  }

  iter->second.record(loadOrStore, dstSlice, guardIVs);
}

// Explicit instantiations for the four load/store types.
void collectDstLoadOrStore(GenericOp gOp, affine::AffineLoadOp loadOrStore,
                           CopyInfoMap &copyInfos, int dstSlice,
                           Operation *outermostInnerComputeLoop,
                           bool noAccumGuard) {
  collectDstLoadOrStoreImpl(gOp, loadOrStore, copyInfos, dstSlice,
                            outermostInnerComputeLoop, noAccumGuard);
}

void collectDstLoadOrStore(GenericOp gOp, affine::AffineStoreOp loadOrStore,
                           CopyInfoMap &copyInfos, int dstSlice,
                           Operation *outermostInnerComputeLoop,
                           bool noAccumGuard) {
  collectDstLoadOrStoreImpl(gOp, loadOrStore, copyInfos, dstSlice,
                            outermostInnerComputeLoop, noAccumGuard);
}

void collectDstLoadOrStore(GenericOp gOp, memref::LoadOp loadOrStore,
                           CopyInfoMap &copyInfos, int dstSlice,
                           Operation *outermostInnerComputeLoop,
                           bool noAccumGuard) {
  collectDstLoadOrStoreImpl(gOp, loadOrStore, copyInfos, dstSlice,
                            outermostInnerComputeLoop, noAccumGuard);
}

void collectDstLoadOrStore(GenericOp gOp, memref::StoreOp loadOrStore,
                           CopyInfoMap &copyInfos, int dstSlice,
                           Operation *outermostInnerComputeLoop,
                           bool noAccumGuard) {
  collectDstLoadOrStoreImpl(gOp, loadOrStore, copyInfos, dstSlice,
                            outermostInnerComputeLoop, noAccumGuard);
}

void collectDstLoadThenBcast(GenericOp gOp, affine::AffineLoadOp loadOp,
                             d2m::TileBcastOp bcastOp, CopyInfoMap &copyInfos,
                             int dstSlice,
                             Operation *outermostInnerComputeLoop) {
  if (!outermostInnerComputeLoop) {
    outermostInnerComputeLoop = loadOp;
  }

  auto [iter, _] = copyInfos.try_emplace(outermostInnerComputeLoop);
  Value assocCB = lookThroughSubView(loadOp.getMemRef());

  SmallVector<Value> guardIVs;
  if (assocCB) {
    guardIVs = getGuardLoopIVs(loadOp, outermostInnerComputeLoop);
  }

  iter->second.record(loadOp, bcastOp, dstSlice, guardIVs);
}

scf::IfOp createLoadLoopGuard(PatternRewriter &rewriter, Location loc,
                               ValueRange guardIVs, bool isBcastGuard) {
  if (guardIVs.empty()) {
    return nullptr;
  }

  Value guard =
      rewriter
          .create<arith::ConstantOp>(loc, rewriter.getI1Type(),
                                     rewriter.getBoolAttr(isBcastGuard))
          .getResult();

  const auto cmpPredicate =
      isBcastGuard ? arith::CmpIPredicate::eq : arith::CmpIPredicate::ne;

  auto zero = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIndexType(),
      rewriter.getIntegerAttr(rewriter.getIndexType(), 0));

  for (Value guardIV : guardIVs) {
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
buildLinearizedDstAccess(PatternRewriter &rewriter, Operation *op,
                         int dstSlice, Operation *linalgRoot) {
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

    auto [storeMap, storeIndices] = buildLinearizedDstAccess(
        rewriter, op, dstSlice, dstInfo.outermostLoop);

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

    auto loadedResult = rewriter.create<affine::AffineLoadOp>(
        loc, dst, storeMap, storeIndices);

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

    if (dimPositions.size() != 1 ||
        dimPositions[0] >= currentIndices.size()) {
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
    bool enableL1Acc, CopyInfoMap &copyInfos,
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
  AcquireDstOp acquireDst =
      insertAcquireDst(rewriter, loc, region, copyInfos,
                       outermostInnerComputeLoop, dstCapacity,
                       /*insertInsideLoop=*/isScfForLoop);
  Value dst = acquireDst.getResult();

  Value l1AccLoopIV = nullptr;
  if (enableL1Acc) {
    SmallVector<Value> loopIVsInScope =
        collectAncestorLoopIVs(acquireDst.getOperation());
    if (!loopIVsInScope.empty()) {
      l1AccLoopIV = loopIVsInScope.front();
    }
    if (!l1AccLoopIV) {
      LDBG() << "Skipping L1 accumulation insertion: no in-scope loop IV";
      enableL1Acc = false;
    }
  }

  // Emit path-specific data copy loops.
  emitDataCopies(rewriter, loc, dst, copyInfos, enableL1Acc);

  // Insert optional L1 accumulation guard.
  if (enableL1Acc) {
    insertPackerL1AccGuard(rewriter, loc, acquireDst, l1AccLoopIV);
  }

  // Fix intermediate DST results.
  fixDstIntermediateResults(rewriter, loc, dst, dstIntermediates);

  return true;
}

} // namespace detail
} // namespace mlir::tt::d2m
