// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::tt::d2m {

// ---------------------------------------------------------------------------
// Preconditions
// ---------------------------------------------------------------------------

LogicalResult verifyInsertDstRegisterAccessPreconditions(ModuleOp moduleOp);

// ---------------------------------------------------------------------------
// Shared data structures
// ---------------------------------------------------------------------------

namespace detail {

// Coarse classification of what the compute region of a d2m.generic looks
// like at the time the DST-insertion pass runs.  Used by both passes to
// decide whether the region is a candidate for rewrite.
struct DstRegionOpClassification {
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

// `MapVector` is used (instead of `DenseMap`) so that iteration order is
// the deterministic insertion order; this matters for IR emission stability
// across runs and for reproducible debug output.
using CopyInfoMap = llvm::MapVector<Operation *, CopyInfo>;

struct DstIntermediateResult {
  int dstSlice;
  Operation *outermostLoop;
};
using DstIntermediatesMap = llvm::MapVector<Operation *, DstIntermediateResult>;

struct DstAccessCollection {
  CopyInfoMap copyInfos;
  DstIntermediatesMap dstIntermediates;
};

// DST slice allocator types live in their respective pass .cpp files
// (`DstSliceAllocationState` in InsertDstRegisterAccessUnscheduled.cpp,
// `DstStackAllocator` in InsertDstRegisterAccessScheduled.cpp), since each is
// only used by one pass.

// ---------------------------------------------------------------------------
// Shared utility free functions
// ---------------------------------------------------------------------------

SmallVector<Value> collectAncestorLoopIVs(Operation *op);

bool hasTileMatmul(Operation *op);

// True iff `op` is any tile-level reduction op (FPU or SFPU variant).
bool isTileReductionOp(Operation *op);

// Stamp a pass-allocated scratch slice onto the op's `dst_scratch_index`
// attribute for the TTKernel lowering to consume.  Today only supports ops
// that need exactly one scratch slice.
void setDstScratchIndex(OperandLoadStoreRegisterOpInterface computeOp,
                        int scratchSlice);

bool hasAcquireDstOp(Region &region);

DstRegionOpClassification classifyDstRegionOps(GenericOp gOp,
                                               unsigned regionIndex);

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

void recordDstAccess(affine::AffineLoadOp op, CopyInfoMap &copyInfos,
                     int dstSlice, Operation *outermostInnerComputeLoop,
                     bool emitGuard);
void recordDstAccess(affine::AffineStoreOp op, CopyInfoMap &copyInfos,
                     int dstSlice, Operation *outermostInnerComputeLoop,
                     bool emitGuard);
void recordDstAccess(memref::LoadOp op, CopyInfoMap &copyInfos, int dstSlice,
                     Operation *outermostInnerComputeLoop, bool emitGuard);
void recordDstAccess(memref::StoreOp op, CopyInfoMap &copyInfos, int dstSlice,
                     Operation *outermostInnerComputeLoop, bool emitGuard);
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
                                     ValueRange carriedOutputRegions,
                                     ArrayRef<int64_t> accumOperandIndices,
                                     CopyInfoMap &copyInfos, int dstSlice,
                                     Operation *outermostInnerComputeLoop,
                                     bool noAccumGuard = false);
void collectDstLoadWithAccumAnalysis(memref::LoadOp loadOp, int64_t operandIdx,
                                     ValueRange carriedOutputRegions,
                                     ArrayRef<int64_t> accumOperandIndices,
                                     CopyInfoMap &copyInfos, int dstSlice,
                                     Operation *outermostInnerComputeLoop,
                                     bool noAccumGuard = false);

scf::IfOp createLoadLoopGuard(PatternRewriter &rewriter, Location loc,
                              ValueRange guardIVs, bool isBcastGuard);

// Clone the loop scaffolding of `loopNestOrOp` (affine.for / yield / apply
// only) and return the cloned root + the IRMapping that maps original IVs
// to the clone. If `loopNestOrOp` is not an `affine::AffineForOp`, returns
// {nullptr, empty mapping}.
std::pair<Operation *, mlir::IRMapping>
cloneAffineLoopSkeleton(PatternRewriter &rewriter, Operation *loopNestOrOp);

// Shared "wrap a compute loop with a cloned CB<->DST copy nest" emitter
// used by both the scheduled and unscheduled paths.
//
// For each record in `loadStoreRecords`, this:
//   1. Emits a copy op via `copyGenerator` inside a clone of `loopNestOrOp`
//      (the *shared* clone built once up front); records that carry a
//      non-empty `guardIVs` get a fresh, per-record clone wrapped in an
//      `scf.if` (accumulation / bcast init guard).
//   2. Rewrites the original load/store via `accessReplacer` so it now
//      goes through the DST register.
//
// `enableL1Acc=true` skips the upfront copy nest entirely (the L1 acc
// guard preserves the running tile).
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
    bool enableL1Acc = false);

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
