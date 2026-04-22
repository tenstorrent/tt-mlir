// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_TRANSFORMS_INSERTDSTREGISTERACCESSSHARED_H
#define TTMLIR_DIALECT_D2M_TRANSFORMS_INSERTDSTREGISTERACCESSSHARED_H

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <deque>
#include <optional>

namespace mlir::tt::d2m {

class GenericOp;

// ---------------------------------------------------------------------------
// Preconditions
// ---------------------------------------------------------------------------

LogicalResult verifyInsertDstRegisterAccessPreconditions(ModuleOp moduleOp);

// ---------------------------------------------------------------------------
// Shared data structures
// ---------------------------------------------------------------------------

namespace detail {

struct OperationTypes {
  bool hasComputeOps = false;
  bool hasLinalgGeneric = false;
  bool hasMarkedAffineLoops = false;
};

template <typename LoadOrStoreTy>
struct LoadStoreRecord {
  LoadOrStoreTy loadStore = nullptr;
  std::optional<d2m::TileBcastOp> bcast = std::nullopt;
  int dstSlice = -1;
  SmallVector<Value> guardIVs = {};

  LoadStoreRecord(LoadOrStoreTy loadStore,
                  std::optional<d2m::TileBcastOp> bcast, int dstSlice,
                  ArrayRef<Value> guardIVs)
      : loadStore(loadStore), bcast(bcast), dstSlice(dstSlice),
        guardIVs(guardIVs.begin(), guardIVs.end()) {}
};

struct CopyInfo {
  void record(affine::AffineLoadOp load, int dstSlice,
              ArrayRef<Value> guardIVs) {
    loads.emplace_back(load, std::nullopt, dstSlice, guardIVs);
  }

  void record(affine::AffineLoadOp load, d2m::TileBcastOp bcast, int dstSlice,
              ArrayRef<Value> guardIVs) {
    loads.emplace_back(load, bcast, dstSlice, guardIVs);
  }

  void record(affine::AffineStoreOp store, int dstSlice, ArrayRef<Value>) {
    stores.emplace_back(store, std::nullopt, dstSlice, ArrayRef<Value>{});
  }

  void record(memref::LoadOp load, int dstSlice, ArrayRef<Value> guardIVs) {
    memrefLoads.emplace_back(load, std::nullopt, dstSlice, guardIVs);
  }

  void record(memref::StoreOp store, int dstSlice, ArrayRef<Value>) {
    memrefStores.emplace_back(store, std::nullopt, dstSlice, ArrayRef<Value>{});
  }

  SmallVector<LoadStoreRecord<affine::AffineLoadOp>> loads;
  SmallVector<LoadStoreRecord<affine::AffineStoreOp>> stores;
  SmallVector<LoadStoreRecord<memref::LoadOp>> memrefLoads;
  SmallVector<LoadStoreRecord<memref::StoreOp>> memrefStores;
};

using CopyInfoMap = DenseMap<Operation *, CopyInfo>;

struct DstIntermediateResult {
  int dstSlice;
  Operation *outermostLoop;
};
using DstIntermediatesMap = DenseMap<Operation *, DstIntermediateResult>;

struct DstAccessCollection {
  CopyInfoMap copyInfos;
  DstIntermediatesMap dstIntermediates;
};

// ---------------------------------------------------------------------------
// DST slice allocators
// ---------------------------------------------------------------------------

// Simple bump allocator -- used by the unscheduled path.
class DstSliceAllocationState {
public:
  int allocate() { return nextSliceIndex++; }
  void setStoreToDst() { storedToDst = true; }
  bool didStoreToDst() { return storedToDst; }
  int getCurrSliceIndex() { return nextSliceIndex - 1; }

private:
  int64_t nextSliceIndex = 0;
  bool storedToDst = false;
};

// Stack-based allocator -- used by the scheduled path.
class DstStackAllocator {
public:
  DstStackAllocator() = delete;
  explicit DstStackAllocator(unsigned dstSliceCapacityIn);

  unsigned allocate(bool isStore = false);
  unsigned deallocate();
  void setStoreToDst() { storedToDst = true; }
  bool didStoreToDst() { return storedToDst; }
  unsigned getCurrSliceIndex() { return currSliceIndex; }
  unsigned getFirstInputSliceIndex();
  void deallocateAllButFirstInput();

private:
  unsigned dstSliceCapacity = 0;
  unsigned currSliceIndex = 0;
  SmallVector<unsigned, 16> inputStack;
  std::deque<unsigned> outputQueue;
  SmallVector<unsigned, 16> sliceStack;
  bool storedToDst = false;

  void initSliceStack();
};

// ---------------------------------------------------------------------------
// Shared utility free functions
// ---------------------------------------------------------------------------

SmallVector<Value> collectAncestorLoopIVs(Operation *op);

bool hasTileMatmul(Operation *op);

bool hasAcquireDstOp(Region &region);

OperationTypes getOperationTypes(GenericOp gOp, unsigned regionIndex);

std::pair<Type, int> inferDstInfoFromAllAccesses(const CopyInfoMap &copyInfos);

AcquireDstOp insertAcquireDst(PatternRewriter &rewriter, Location loc,
                              Region &region, const CopyInfoMap &copyInfos,
                              Operation *outermostInnerComputeLoop,
                              unsigned dstCapacity, bool insertInsideLoop);

Value lookThroughSubView(Value memref);

Value stripDstRegionWrappers(Value memref);

bool isSameLogicalMemRefRegion(Value lhs, Value rhs);

SmallVector<Value>
getObviousCarriedOutputRegions(OperandLoadStoreRegisterOpInterface computeOp);

SmallVector<int64_t>
getAccumClassificationOperandIndices(OperandLoadStoreRegisterOpInterface op);

void recordDstAccess(affine::AffineLoadOp loadOrStore, CopyInfoMap &copyInfos,
                     int dstSlice, Operation *outermostInnerComputeLoop,
                     bool emitGuard);
void recordDstAccess(affine::AffineStoreOp loadOrStore, CopyInfoMap &copyInfos,
                     int dstSlice, Operation *outermostInnerComputeLoop,
                     bool emitGuard);
void recordDstAccess(memref::LoadOp loadOrStore, CopyInfoMap &copyInfos,
                     int dstSlice, Operation *outermostInnerComputeLoop,
                     bool emitGuard);
void recordDstAccess(memref::StoreOp loadOrStore, CopyInfoMap &copyInfos,
                     int dstSlice, Operation *outermostInnerComputeLoop,
                     bool emitGuard);
void recordDstAccess(affine::AffineLoadOp loadOp, d2m::TileBcastOp bcastOp,
                     CopyInfoMap &copyInfos, int dstSlice,
                     Operation *outermostInnerComputeLoop, bool emitGuard);

void collectDstStoreAccess(affine::AffineStoreOp storeOp,
                           CopyInfoMap &copyInfos, int dstSlice,
                           Operation *outermostInnerComputeLoop);
void collectDstStoreAccess(memref::StoreOp storeOp, CopyInfoMap &copyInfos,
                           int dstSlice, Operation *outermostInnerComputeLoop);

void collectDstLoadWithAccumAnalysis(affine::AffineLoadOp loadOp,
                                     int64_t operandIdx,
                                     ArrayRef<Value> carriedOutputRegions,
                                     ArrayRef<int64_t> accumOperandIndices,
                                     CopyInfoMap &copyInfos, int dstSlice,
                                     Operation *outermostInnerComputeLoop,
                                     bool noAccumGuard = false);
void collectDstLoadWithAccumAnalysis(memref::LoadOp loadOp, int64_t operandIdx,
                                     ArrayRef<Value> carriedOutputRegions,
                                     ArrayRef<int64_t> accumOperandIndices,
                                     CopyInfoMap &copyInfos, int dstSlice,
                                     Operation *outermostInnerComputeLoop,
                                     bool noAccumGuard = false);

scf::IfOp createLoadLoopGuard(PatternRewriter &rewriter, Location loc,
                              ValueRange guardIVs, bool isBcastGuard);

std::pair<AffineMap, SmallVector<Value>>
buildLinearizedDstAccess(PatternRewriter &rewriter, Operation *op, int dstSlice,
                         Operation *linalgRoot = nullptr);

void fixDstIntermediateResults(PatternRewriter &rewriter, Location loc,
                               Value dst,
                               const DstIntermediatesMap &dstIntermediates);

bool isDstScopeIV(Value iv, Operation *linalgRoot);

std::tuple<AffineMap, SmallVector<Value>, AffineMap, SmallVector<Value>>
buildIndices(PatternRewriter &rewriter, Location loc,
             const mlir::IRMapping &irMapper, ValueRange currentIndices,
             int dstSlice, AffineMap map, MemRefType cbType,
             Operation *linalgRoot = nullptr);

void insertPackerL1AccGuard(PatternRewriter &rewriter, Location loc,
                            AcquireDstOp acquireDst, Value loopIV);

// ---------------------------------------------------------------------------
// Shared orchestration helper
//
// Performs steps common to both scheduled and unscheduled paths after
// DST accesses have been collected:
//   - insert acquire_dst
//   - call the path-specific data copy emitter (provided as a callback)
//   - optional L1-acc guard insertion
//   - fix DST intermediate results
// ---------------------------------------------------------------------------

bool insertDstRegisterAccessFinalize(
    PatternRewriter &rewriter, GenericOp gOp, Region &region,
    unsigned dstCapacity, Operation *outermostInnerComputeLoop,
    bool enableL1Acc, CopyInfoMap &copyInfos,
    DstIntermediatesMap &dstIntermediates,
    llvm::function_ref<void(PatternRewriter &, Location, Value,
                            const CopyInfoMap &, bool)>
        emitDataCopies);

} // namespace detail

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_TRANSFORMS_INSERTDSTREGISTERACCESSSHARED_H
