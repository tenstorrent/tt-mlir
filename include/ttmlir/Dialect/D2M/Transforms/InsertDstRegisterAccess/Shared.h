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

enum class DstAccessKind : uint8_t {
  AffineLoad,
  AffineStore,
  MemrefLoad,
  MemrefStore,
};

struct DstAccess {
  Operation *op = nullptr;
  DstAccessKind kind = DstAccessKind::AffineLoad;
  int dstSlice = -1;
  std::optional<d2m::TileBcastOp> bcast = std::nullopt;
  SmallVector<Value> guardIVs = {};

  DstAccess() = default;
  DstAccess(Operation *opIn, DstAccessKind kindIn, int dstSliceIn,
            std::optional<d2m::TileBcastOp> bcastIn, ArrayRef<Value> guardIVsIn)
      : op(opIn), kind(kindIn), dstSlice(dstSliceIn), bcast(bcastIn),
        guardIVs(guardIVsIn.begin(), guardIVsIn.end()) {}

  bool isLoad() const;
  bool isStore() const;
  bool isAffine() const;
  bool isMemref() const;

  Location getLoc() const;
  Value getMemRef() const;
  MemRefType getMemRefType() const;

  AffineMap getAffineMap() const;
  ValueRange getAffineIndices() const;
  ValueRange getMemrefIndices() const;
};

struct CopyInfo {
  void record(DstAccess access);

  void record(affine::AffineLoadOp load, int dstSlice,
              ArrayRef<Value> guardIVs);
  void record(affine::AffineLoadOp load, d2m::TileBcastOp bcast, int dstSlice,
              ArrayRef<Value> guardIVs);
  void record(affine::AffineStoreOp store, int dstSlice, ArrayRef<Value>);
  void record(memref::LoadOp load, int dstSlice, ArrayRef<Value> guardIVs);
  void record(memref::StoreOp store, int dstSlice, ArrayRef<Value>);

  SmallVector<DstAccess> loads;
  SmallVector<DstAccess> stores;
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
// DstSliceAllocator
//
// Free-slot pool for the DST register.  Slots are bump-allocated from
// `sliceStack` and reclaimed only via `deallocateAllButFirstInput()` (the
// multi-input fold case); `inputStack` exists so that path has the
// current op's operand slots to free.  `scratchSlots` is a separate pool
// so that a later compute op can never pick a scratch slot as a
// candidate for in-place reuse.
// ---------------------------------------------------------------------------

class DstSliceAllocator {
public:
  DstSliceAllocator() = delete;
  explicit DstSliceAllocator(unsigned dstSliceCapacityIn)
      : dstSliceCapacity(dstSliceCapacityIn) {
    initSliceStack();
  }

  unsigned allocateInput();
  unsigned allocateOutput();
  unsigned allocateScratch();

  void setStoreToDst() { storedToDst = true; }
  bool didStoreToDst() const { return storedToDst; }

  unsigned getCurrSliceIndex() const;
  unsigned getFirstInputSliceIndex() const;
  void deallocateAllButFirstInput();

private:
  unsigned dstSliceCapacity = 0;
  std::optional<unsigned> currSliceIndex;
  SmallVector<unsigned, 16> inputStack;
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

// Find the closest ancestor reduction loop IV of `acquireDst`, where
// "reduction" means: no output store recorded in `copyInfos` depends on the
// loop's induction variable. The closest such loop is the loop whose adjacent
// iterations accumulate into the same output L1 slot, and is therefore the
// correct trigger for switching the packer to L1-acc mode.
//
// Returns nullptr if there is no qualifying reduction loop with trip count > 1.
// Reduction loops with constant trip count <= 1 are skipped because L1-acc is
// not needed for them.
Value findClosestReductionLoopIVForL1Acc(Operation *acquireDstOp,
                                         const CopyInfoMap &copyInfos);

// Stamp a pass-allocated scratch slice onto the op's `dst_scratch_index`
// attribute for the TTKernel lowering to consume.  Today only supports ops
// that need exactly one scratch slice.
void setDstScratchIndex(OperandLoadStoreRegisterOpInterface computeOp,
                        int scratchSlice, Operation *linalgRoot);

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

void replaceLoadWithDst(PatternRewriter &rewriter, const DstAccess &access,
                        Value dst, AffineMap dstAccessMap,
                        ValueRange dstAccessIndices);

void replaceStoreWithDst(PatternRewriter &rewriter, const DstAccess &access,
                         Value dst, AffineMap dstAccessMap,
                         ValueRange dstAccessIndices);

void generateLoadSideCopy(PatternRewriter &rewriter, const DstAccess &access,
                          Value dst, AffineMap l1AccessMap,
                          ValueRange l1AccessIndices, AffineMap dstAccessMap,
                          ValueRange dstAccessIndices);

void generateStoreSideCopy(PatternRewriter &rewriter, const DstAccess &access,
                           Value dst, AffineMap l1AccessMap,
                           ValueRange l1AccessIndices, AffineMap dstAccessMap,
                           ValueRange dstAccessIndices);

scf::IfOp createLoadLoopGuard(PatternRewriter &rewriter, Location loc,
                              ValueRange guardIVs, bool isBcastGuard);

// Clone the loop scaffolding of `loopNestOrOp` (affine.for / yield / apply
// only) and return the cloned root + the IRMapping that maps original IVs
// to the clone. If `loopNestOrOp` is not an `affine::AffineForOp`, returns
// {nullptr, empty mapping}.
std::pair<Operation *, mlir::IRMapping>
cloneAffineLoopSkeleton(PatternRewriter &rewriter, Operation *loopNestOrOp);

// Shared CB<->DST copy nest emitter used by both scheduled and unscheduled
// paths.  When `cloneLoopNest` is true (unscheduled), affine accesses are
// wrapped in a cloned affine.for skeleton; when false (scheduled in-place),
// copy ops are emitted at each access site.  Memref accesses always use the
// in-place path with a constant DST slice map.
void emitDstCopyNest(PatternRewriter &rewriter, Operation *loopNestOrOp,
                     Value dst, ArrayRef<DstAccess> accesses, bool isLoadSide,
                     bool cloneLoopNest, bool disableL1Acc = true);

std::pair<AffineMap, SmallVector<Value>>
buildLinearizedDstAccess(PatternRewriter &rewriter, Operation *op, int dstSlice,
                         Operation *linalgRoot = nullptr);

void fixDstIntermediateResults(PatternRewriter &rewriter, Location loc,
                               Value dst,
                               const DstIntermediatesMap &dstIntermediates);

bool isDstScopeIV(Value iv, Operation *linalgRoot);

std::tuple<AffineMap, SmallVector<Value>, AffineMap, SmallVector<Value>>
buildIndices(PatternRewriter &rewriter, Location loc,
             const mlir::IRMapping &irMapper, const DstAccess &access,
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
