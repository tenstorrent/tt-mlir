// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_TRANSFORMS_INSERTDSTREGISTERACCESS_SHARED_H
#define TTMLIR_DIALECT_D2M_TRANSFORMS_INSERTDSTREGISTERACCESS_SHARED_H

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

// ---------------------------------------------------------------------------
// Shared DST slice allocator
//
// Slots come from a free list (`sliceStack`) and are bound to one of four
// explicit roles:
//
//   - `inputStack`     : operand reads of the in-progress compute op, popped
//                        LIFO (most recent input is deallocated first).
//   - `currentOutput`  : the result tile of the most recently emitted
//                        compute op.  Its value may still be consumed
//                        in-place by the next compute op.
//   - `retiredOutputs` : prior outputs whose consumer has already been
//                        emitted; they are dead but not yet recycled.
//   - `scratchSlots`   : per-op private scratch (e.g. SFPU int reductions
//                        via `getNumDstScratchSlices()`).  Owned by the op
//                        for the lifetime of the region; never recycled,
//                        and deliberately not tracked by `inputStack`,
//                        `currentOutput`, or `currSliceIndex`.
//
// `currSliceIndex` is the most recently allocated *operand or output* slot
// (never a scratch slot) so callers wanting to overwrite-in-place can grab
// it directly.
//
// The scheduled pass uses the full API (LIFO reuse via `deallocate()` and
// `deallocateAllButFirstInput()`).  The unscheduled pass uses only
// `allocate()` / `allocateScratch()` / `getCurrSliceIndex()`, which gives
// it the same bump-allocator behavior its old `DstSliceAllocationState`
// had: every call returns a fresh slot in 0,1,2,... order.
// ---------------------------------------------------------------------------

class DstStackAllocator {
public:
  DstStackAllocator() = delete;
  explicit DstStackAllocator(unsigned dstSliceCapacityIn)
      : dstSliceCapacity(dstSliceCapacityIn) {
    initSliceStack();
  }

  unsigned allocate(bool isStore = false);
  unsigned allocateScratch();
  unsigned deallocate();
  void setStoreToDst() { storedToDst = true; }
  bool didStoreToDst() const { return storedToDst; }
  unsigned getCurrSliceIndex() const { return currSliceIndex; }
  unsigned getFirstInputSliceIndex() const;
  void deallocateAllButFirstInput();

private:
  unsigned dstSliceCapacity = 0;
  unsigned currSliceIndex = 0;
  SmallVector<unsigned, 16> inputStack;
  std::optional<unsigned> currentOutput;
  SmallVector<unsigned, 4> retiredOutputs;
  SmallVector<unsigned, 4> scratchSlots;
  SmallVector<unsigned, 16> sliceStack;
  bool storedToDst = false;

  void initSliceStack();
};

// ---------------------------------------------------------------------------
// Shared utility free functions
// ---------------------------------------------------------------------------

SmallVector<Value> collectAncestorLoopIVs(Operation *op);

bool hasTileMatmul(Operation *op);

// True iff `op` is any tile-level reduction op (FPU or SFPU variant).
bool isTileReductionOp(Operation *op);

// Returns true iff the packer hardware supports L1 accumulation when packing
// into an output buffer of element type `dt`. The packer L1-acc path goes
// through a fixed set of native formats; block-float (Bfp*) outputs are NOT
// supported and would silently produce garbage results.
//
// Supported set follows tt-llk `PACK_L1_ACC_FORMATS`:
//   Float32, Float16, BFloat16 (Float16_b), Int32, UInt8.
bool isPackerL1AccumulationSupportedDataType(ttcore::DataType dt);

// Returns true iff every `d2m.tile_matmul` in `loopOp` has a result tile
// element type that is supported by the packer L1-accumulation path. Returns
// true if there are no `d2m.tile_matmul` ops (caller is responsible for
// gating on `hasTileMatmul`).
bool allTileMatmulOutputsSupportPackerL1Acc(Operation *loopOp);

// Find the outermost ancestor reduction loop IV of `acquireDst`, where
// "reduction" means: no output store recorded in `copyInfos` depends on the
// loop's induction variable. This is the loop whose iterations accumulate
// into the same output L1 slot, and is therefore the correct trigger for
// switching the packer to L1-acc mode.
//
// Returns nullptr if there is no such reduction loop, or if the candidate
// reduction loop has a constant trip count <= 1 (in which case L1-acc is
// not needed and the trigger comparison would never fire correctly).
Value findOutermostReductionLoopIVForL1Acc(Operation *acquireDstOp,
                                           const CopyInfoMap &copyInfos);

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
// `disableL1Acc=false` (i.e. L1 acc on) skips the upfront copy nest
// entirely (the L1 acc guard preserves the running tile).
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
    bool disableL1Acc = true);

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
    bool disableL1Acc, CopyInfoMap &copyInfos,
    DstIntermediatesMap &dstIntermediates,
    llvm::function_ref<void(PatternRewriter &, Location, Value,
                            const CopyInfoMap &, bool)>
        emitDataCopies);

} // namespace detail

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_TRANSFORMS_INSERTDSTREGISTERACCESS_SHARED_H
