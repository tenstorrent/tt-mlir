// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"

#include <variant>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTDSTREGISTERACCESS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

#define DEBUG_TYPE "D2MInsertDstRegisterAccess"

namespace {

//===----------------------------------------------------------------------===//
// Wrapper types to abstract over affine vs memref/scf ops.
//===----------------------------------------------------------------------===//

// Wrapper for load operations (affine::AffineLoadOp or memref::LoadOp).
class LoadOpVariant {
public:
  LoadOpVariant() = default;
  LoadOpVariant(affine::AffineLoadOp op) : op(op) {}
  LoadOpVariant(memref::LoadOp op) : op(op) {}

  bool isValid() const { return !std::holds_alternative<std::monostate>(op); }
  bool isAffine() const {
    return std::holds_alternative<affine::AffineLoadOp>(op);
  }
  bool isMemref() const { return std::holds_alternative<memref::LoadOp>(op); }

  affine::AffineLoadOp getAffine() const {
    return std::get<affine::AffineLoadOp>(op);
  }
  memref::LoadOp getMemref() const { return std::get<memref::LoadOp>(op); }

  Operation *getOperation() const {
    if (isAffine()) {
      return getAffine().getOperation();
    }
    if (isMemref()) {
      return getMemref().getOperation();
    }
    return nullptr;
  }

  Value getMemRef() const {
    if (isAffine()) {
      return getAffine().getMemRef();
    }
    if (isMemref()) {
      return getMemref().getMemRef();
    }
    return nullptr;
  }

  MemRefType getMemRefType() const {
    return mlir::cast<MemRefType>(getMemRef().getType());
  }

  Location getLoc() const { return getOperation()->getLoc(); }

  Value getResult() const {
    if (isAffine()) {
      return getAffine().getResult();
    }
    if (isMemref()) {
      return getMemref().getResult();
    }
    return nullptr;
  }

  // Get indices for memref ops, empty for affine (uses map).
  ValueRange getIndices() const {
    if (isMemref()) {
      return getMemref().getIndices();
    }
    return ValueRange{};
  }

  // Create from a Value's defining op.
  static LoadOpVariant fromValue(Value v) {
    if (auto affineLoad = v.getDefiningOp<affine::AffineLoadOp>()) {
      return LoadOpVariant(affineLoad);
    }
    if (auto memrefLoad = v.getDefiningOp<memref::LoadOp>()) {
      return LoadOpVariant(memrefLoad);
    }
    return LoadOpVariant();
  }

private:
  std::variant<std::monostate, affine::AffineLoadOp, memref::LoadOp> op;
};

// Wrapper for store operations (affine::AffineStoreOp or memref::StoreOp).
class StoreOpVariant {
public:
  StoreOpVariant() = default;
  StoreOpVariant(affine::AffineStoreOp op) : op(op) {}
  StoreOpVariant(memref::StoreOp op) : op(op) {}

  bool isValid() const { return !std::holds_alternative<std::monostate>(op); }
  bool isAffine() const {
    return std::holds_alternative<affine::AffineStoreOp>(op);
  }
  bool isMemref() const { return std::holds_alternative<memref::StoreOp>(op); }

  affine::AffineStoreOp getAffine() const {
    return std::get<affine::AffineStoreOp>(op);
  }
  memref::StoreOp getMemref() const { return std::get<memref::StoreOp>(op); }

  Operation *getOperation() const {
    if (isAffine()) {
      return getAffine().getOperation();
    }
    if (isMemref()) {
      return getMemref().getOperation();
    }
    return nullptr;
  }

  Value getMemRef() const {
    if (isAffine()) {
      return getAffine().getMemRef();
    }
    if (isMemref()) {
      return getMemref().getMemRef();
    }
    return nullptr;
  }

  MemRefType getMemRefType() const {
    return mlir::cast<MemRefType>(getMemRef().getType());
  }

  Location getLoc() const { return getOperation()->getLoc(); }

  Value getValue() const {
    if (isAffine()) {
      return getAffine().getValue();
    }
    if (isMemref()) {
      return getMemref().getValue();
    }
    return nullptr;
  }

  // Get indices for memref ops.
  ValueRange getIndices() const {
    if (isMemref()) {
      return getMemref().getIndices();
    }
    return ValueRange{};
  }

  // Try to cast from Operation*, store empty monostate otherwise.
  static StoreOpVariant fromOperation(Operation *op) {
    if (auto affineStore = dyn_cast<affine::AffineStoreOp>(op)) {
      return StoreOpVariant(affineStore);
    }
    if (auto memrefStore = dyn_cast<memref::StoreOp>(op)) {
      return StoreOpVariant(memrefStore);
    }
    return StoreOpVariant();
  }

private:
  std::variant<std::monostate, affine::AffineStoreOp, memref::StoreOp> op;
};

// Wrapper for loop operations (affine::AffineForOp or scf::ForOp).
class ForOpVariant {
public:
  ForOpVariant() = default;
  ForOpVariant(affine::AffineForOp op) : op(op) {}
  ForOpVariant(scf::ForOp op) : op(op) {}

  bool isValid() const { return !std::holds_alternative<std::monostate>(op); }
  bool isAffine() const {
    return std::holds_alternative<affine::AffineForOp>(op);
  }
  bool isScf() const { return std::holds_alternative<scf::ForOp>(op); }

  affine::AffineForOp getAffine() const {
    return std::get<affine::AffineForOp>(op);
  }
  scf::ForOp getScf() const { return std::get<scf::ForOp>(op); }

  Operation *getOperation() const {
    if (isAffine()) {
      return getAffine().getOperation();
    }
    if (isScf()) {
      return getScf().getOperation();
    }
    return nullptr;
  }

  Region &getRegion() const {
    if (isAffine()) {
      return getAffine().getRegion();
    }
    return getScf().getRegion();
  }

  bool hasAttr(StringRef name) const { return getOperation()->hasAttr(name); }

  void removeAttr(StringRef name) const { getOperation()->removeAttr(name); }

  void setAttr(StringRef name, Attribute value) const {
    getOperation()->setAttr(name, value);
  }

  // Try to cast from Operation*, store empty monostate otherwise.
  static ForOpVariant fromOperation(Operation *op) {
    if (auto affineFor = dyn_cast<affine::AffineForOp>(op)) {
      return ForOpVariant(affineFor);
    }
    if (auto scfFor = dyn_cast<scf::ForOp>(op)) {
      return ForOpVariant(scfFor);
    }
    return ForOpVariant();
  }

private:
  std::variant<std::monostate, affine::AffineForOp, scf::ForOp> op;
};

struct OperationTypes {
  bool hasComputeOps = false;
  bool hasLinalgGeneric = false;
  bool hasMarkedLoops = false; // For loop op with d2m.linalg_root.
};

static bool hasTileMatmul(linalg::GenericOp linalgGenericOp) {
  bool hasTileMatmul = false;
  linalgGenericOp->walk([&](d2m::TileMatmulOp) {
    hasTileMatmul = true;
    return WalkResult::interrupt();
  });
  return hasTileMatmul;
}

struct D2MInsertDstRegisterAccessRewriter final
    : public OpRewritePattern<GenericOp> {
public:
  D2MInsertDstRegisterAccessRewriter(mlir::MLIRContext *ctx, bool useTileMatmul,
                                     unsigned maxDstPhysicalSizeTiles)
      : OpRewritePattern<GenericOp>(ctx), useTileMatmul(useTileMatmul),
        maxDstPhysicalSizeTiles(maxDstPhysicalSizeTiles) {};

  // Records a CB<->DST load/store op, which DST slice it accesses, and
  // some special considerations for looping over the tensor shard while doing
  // DST accumulation/broadcast.
  struct LoadStoreRecord {
    LoadOpVariant load;
    StoreOpVariant store;
    std::optional<d2m::TileBcastOp> bcast = std::nullopt;
    int dstSlice = -1;
    std::set<int64_t> guardDims = {};

    // Constructor for loads.
    LoadStoreRecord(LoadOpVariant load, std::optional<d2m::TileBcastOp> bcast,
                    int dstSlice, const std::set<int64_t> &guardDims)
        : load(load), bcast(bcast), dstSlice(dstSlice), guardDims(guardDims) {}

    // Constructor for stores.
    LoadStoreRecord(StoreOpVariant store, int dstSlice,
                    const std::set<int64_t> &guardDims)
        : store(store), dstSlice(dstSlice), guardDims(guardDims) {}

    bool isLoad() const { return load.isValid(); }
    bool isStore() const { return store.isValid(); }

    Operation *getOperation() const {
      if (isLoad()) {
        return load.getOperation();
      }
      return store.getOperation();
    }

    Location getLoc() const { return getOperation()->getLoc(); }

    Value getMemRef() const {
      if (isLoad()) {
        return load.getMemRef();
      }
      return store.getMemRef();
    }

    MemRefType getMemRefType() const {
      return mlir::cast<MemRefType>(getMemRef().getType());
    }
  };

  // Stores all DST<->CB loads/stores that are under the same loop nest.
  struct CopyInfo {
    void recordLoad(LoadOpVariant load, int dstSlice,
                    const std::set<int64_t> &guardDims) {
      loads.emplace_back(load, std::nullopt, dstSlice, guardDims);
    }

    void recordLoad(LoadOpVariant load, d2m::TileBcastOp bcast, int dstSlice,
                    const std::set<int64_t> &guardDims) {
      loads.emplace_back(load, bcast, dstSlice, guardDims);
    }

    void recordStore(StoreOpVariant store, int dstSlice,
                     const std::set<int64_t> &guardDims = {}) {
      stores.emplace_back(store, dstSlice, guardDims);
    }

    bool empty() const { return loads.empty() && stores.empty(); }

    SmallVector<LoadStoreRecord> loads;
    SmallVector<LoadStoreRecord> stores;
  };

  using CopyInfoMap = DenseMap<Operation *, CopyInfo>;

  // Maps a compute op whose result will be consumed by another compute op, to
  // its assigned DST slice and its ancestor loop nest.
  struct DstIntermediateResult {
    int dstSlice;
    Operation *outermostLoop;
  };
  using DstIntermediatesMap = DenseMap<Operation *, DstIntermediateResult>;

  struct DstAccessCollection {
    CopyInfoMap copyInfos;
    DstIntermediatesMap dstIntermediates;
  };

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

  class DstStackAllocator {
  public:
    DstStackAllocator() = delete;

    DstStackAllocator(unsigned dstSliceCapacityIn) {
      dstSliceCapacity = dstSliceCapacityIn;
      initSliceStack();
    }

    unsigned allocate(bool isStore = false) {
      assert(!sliceStack.empty() && "Out of dst slices");

      currSliceIndex = sliceStack.pop_back_val();

      if (isStore) {
        outputQueue.push_back(currSliceIndex);
      } else {
        inputStack.push_back(currSliceIndex);
      }

      return currSliceIndex;
    }

    unsigned deallocate() {
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
      LDBG() << "Deallocated slot " << id;

      return id;
    }

    void setStoreToDst() { storedToDst = true; }
    bool didStoreToDst() { return storedToDst; }
    unsigned getCurrSliceIndex() { return currSliceIndex; }

  private:
    unsigned dstSliceCapacity = 0;
    unsigned currSliceIndex = 0;
    SmallVector<unsigned, 16> inputStack;
    std::deque<unsigned> outputQueue;
    SmallVector<unsigned, 16> sliceStack;
    bool storedToDst = false;

    void initSliceStack() {
      assert(dstSliceCapacity > 0 && dstSliceCapacity <= 16);
      for (int i = dstSliceCapacity - 1; i >= 0; --i) {
        sliceStack.push_back(static_cast<unsigned>(i));
      }
    }
  };

  LogicalResult matchAndRewrite(GenericOp gOp,
                                PatternRewriter &rewriter) const final {

    bool modified = false;
    for (unsigned regionIndex = 0; regionIndex < gOp.getNumRegions();
         regionIndex++) {
      ThreadType threadType = gOp.getRegionThreadType(regionIndex);
      if (threadType != ThreadType::Unified &&
          threadType != ThreadType::Compute) {
        continue;
      }

      Region *genericRegion = &gOp.getRegion(regionIndex);
      Block &block = genericRegion->getBlocks().front();

      // Check if this region has any operations that this pass can handle.
      OperationTypes opTypes = getOperationTypes(gOp, regionIndex);
      if (!opTypes.hasComputeOps && !opTypes.hasLinalgGeneric &&
          !opTypes.hasMarkedLoops) {
        return failure();
      }

      // Check if there are any DST-using ops. If not (e.g., passthrough loops
      // with only load/store), skip this region.
      Type largestDstType =
          utils::getRegionLargestDstElemTypeOrNull(*genericRegion);
      if (!largestDstType) {
        continue;
      }
      const unsigned dstCapacity =
          ttcore::getOpChipDescAttr(gOp).getDstLogicalSizeTiles(
              largestDstType, false, maxDstPhysicalSizeTiles);

      // Process linalg.generic ops that were not converted by LinalgToAffine
      WalkResult walkResult =
          block.walk([&](linalg::GenericOp linalgGenericOp) {
            if (!useTileMatmul && hasTileMatmul(linalgGenericOp)) {
              if (!gOp.isExplicitDatamovementForm()) {
                if (rewriteTileMatmulAsTileMatmulBlock(
                        rewriter, gOp, *genericRegion, linalgGenericOp,
                        dstCapacity, modified)) {
                  return WalkResult::interrupt();
                }
                return WalkResult::advance();
              }
            }
            return WalkResult::interrupt();
          });

      if (walkResult.wasInterrupted()) {
        return rewriter.notifyMatchFailure(
            gOp, "linalg.generic operations were not converted to loops");
      }

      // Process loops marked by LinalgToAffine pass.
      block.walk([&](Operation *op) {
        ForOpVariant forOp = ForOpVariant::fromOperation(op);
        if (!forOp.isValid()) {
          return WalkResult::advance();
        }

        if (!forOp.hasAttr("d2m.linalg_root")) {
          return WalkResult::advance();
        }

        forOp.removeAttr("d2m.linalg_root");

        Region &dstRegisterAccessRegion = forOp.getRegion();
        modified |=
            insertDstRegisterAccess(rewriter, gOp, dstRegisterAccessRegion,
                                    dstCapacity, forOp.getOperation());

        return WalkResult::advance();
      });
    }
    return success(modified);
  }

  static bool
  insertDstRegisterAccess(PatternRewriter &rewriter, GenericOp gOp,
                          Region &region, unsigned dstCapacity,
                          Operation *outermostInnerComputeLoop = nullptr) {
    assert(region.getBlocks().size() == 1);
    if (hasAcquireDstOp(region)) {
      return false;
    }

    bool isScheduled = outermostInnerComputeLoop->hasAttr("d2m.scheduled");
    outermostInnerComputeLoop->removeAttr("d2m.scheduled");

    Location loc = gOp.getLoc();

    // 1. Collect relevant DST accesses.
    auto [copyInfos, dstIntermediates] =
        isScheduled
            ? collectDstAccessesScheduled(
                  gOp, region, outermostInnerComputeLoop, dstCapacity)
            : collectDstAccesses(gOp, region, outermostInnerComputeLoop);

    if (copyInfos.empty()) {
      return false;
    }

    // 2. Insert acquire dst.
    AcquireDstOp acquireDst =
        insertAcquireDst(rewriter, loc, region, copyInfos,
                         outermostInnerComputeLoop, dstCapacity);
    Value dst = acquireDst.getResult();

    // 3. Generate data copy loops to/from dst.
    if (isScheduled) {
      dataCopyGenerateScheduled(rewriter, loc, dst, copyInfos);
    } else {
      dataCopyGenerate(rewriter, loc, dst, copyInfos);
    }

    // 4. Fix the passing of intermediate results through the DST.
    fixDstIntermediateResults(rewriter, loc, dst, dstIntermediates);

    return true;
  }

  static bool hasAcquireDstOp(Region &region) {
    return !region.getOps<AcquireDstOp>().empty();
  }

  static OperationTypes getOperationTypes(GenericOp gOp, unsigned regionIndex) {
    OperationTypes types;
    types.hasComputeOps = gOp.hasComputeOpsInRegion(regionIndex);

    Region *genericRegion = &gOp.getRegion(regionIndex);
    Block &block = genericRegion->getBlocks().front();

    block.walk([&](Operation *op) {
      if (isa<linalg::GenericOp>(op)) {
        types.hasLinalgGeneric = true;
      } else if (auto forOp = ForOpVariant::fromOperation(op);
                 forOp.isValid()) {
        if (forOp.hasAttr("d2m.linalg_root")) {
          types.hasMarkedLoops = true;
        }
      }
    });

    return types;
  }

  // Returns the element type and max DST slot index needed.
  static std::pair<Type, int>
  inferDstInfoFromAllAccesses(const CopyInfoMap &copyInfos) {
    Type elemType = nullptr;
    int maxDstSlice = -1;

    auto updateInfo = [&](MemRefType memref, int idx) {
      if (elemType == nullptr) {
        elemType = memref.getElementType();
      }
      maxDstSlice = std::max(maxDstSlice, idx);
    };

    for (auto [loopNest, copyInfo] : copyInfos) {
      for (auto &record : copyInfo.loads) {
        updateInfo(record.getMemRefType(), record.dstSlice);
      }
      for (auto &record : copyInfo.stores) {
        updateInfo(record.getMemRefType(), record.dstSlice);
      }
    }
    TT_assert(elemType != nullptr);
    TT_assert(maxDstSlice >= 0);
    return {elemType, maxDstSlice};
  }

  static AcquireDstOp insertAcquireDst(PatternRewriter &rewriter, Location loc,
                                       Region &region,
                                       const CopyInfoMap &copyInfos,
                                       Operation *outermostInnerComputeLoop,
                                       unsigned dstCapacity) {
    assert(!copyInfos.empty());
    if (outermostInnerComputeLoop) {
      rewriter.setInsertionPoint(outermostInnerComputeLoop);
    } else {
      rewriter.setInsertionPointToStart(&region.front());
    }

    auto [elemType, maxDstSlice] = inferDstInfoFromAllAccesses(copyInfos);
    TT_assertv(maxDstSlice < static_cast<int>(dstCapacity),
               "Insufficient DST capacity for all operands.");

    // DST is 1D array of tile slots (flat indexing for scratch tile support).
    MemRefType dstType = MemRefType::get(
        {static_cast<int64_t>(dstCapacity)}, elemType,
        mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext()),
        rewriter.getAttr<ttcore::MemorySpaceAttr>(
            ttcore::MemorySpace::RegisterDst));

    return rewriter.create<AcquireDstOp>(loc, dstType);
  }

  static BlockArgument lookThroughSubView(Value memref) {
    while (auto subView = mlir::dyn_cast_or_null<memref::SubViewOp>(
               memref.getDefiningOp())) {
      memref = subView.getSource();
    }
    if (auto *definingOp = memref.getDefiningOp()) {
      if (mlir::isa<d2m::WaitOp, d2m::ReserveOp>(definingOp)) {
        memref = definingOp->getOperand(0);
      } else if (auto allocOp = mlir::dyn_cast<memref::AllocOp>(definingOp)) {
        // memref.alloc: find the associated operand by tracing uses, then
        // find the corresponding CB block argument
        Value assocOperand = GenericOp::findAssocOperand(allocOp);
        if (!assocOperand) {
          return nullptr;
        }
        Value cb = GenericOp::findAssocCBByOperand(allocOp.getOperation(),
                                                   assocOperand);
        return mlir::dyn_cast<BlockArgument>(cb);
      }
    }
    return mlir::dyn_cast<BlockArgument>(memref);
  }

  // Collect a single load and determine its loop guard.
  static void
  collectDstLoad(GenericOp gOp, LoadOpVariant load, CopyInfoMap &copyInfos,
                 int dstSlice, Operation *outermostInnerComputeLoop,
                 std::optional<d2m::TileBcastOp> bcast = std::nullopt) {
    if (!outermostInnerComputeLoop) {
      outermostInnerComputeLoop = load.getOperation();
    }

    auto [iter, _] = copyInfos.try_emplace(outermostInnerComputeLoop);
    BlockArgument blockArg = lookThroughSubView(load.getMemRef());

    std::set<int64_t> guardDims = {};
    if (blockArg && !gOp.isExplicitDatamovementForm()) {
      auto nonParticipatingLoopDims =
          gOp.getNonParticipatingLoopDims(blockArg.getArgNumber());
      auto iteratorTypes = gOp.getIteratorTypesValue();

      bool isConstantIndexed =
          nonParticipatingLoopDims.size() == iteratorTypes.size();

      if (!isConstantIndexed) {
        for (int64_t dim : nonParticipatingLoopDims) {
          // For bcast, check parallel; otherwise check reduction.
          if (bcast.has_value()) {
            TT_assert(iteratorTypes[dim] == ttcore::IteratorType::Parallel);
          } else {
            TT_assert(iteratorTypes[dim] == ttcore::IteratorType::Reduction);
          }
          guardDims.insert(dim);
        }
      }
    }

    if (bcast.has_value()) {
      iter->second.recordLoad(load, *bcast, dstSlice, guardDims);
    } else {
      iter->second.recordLoad(load, dstSlice, guardDims);
    }
  }

  // Collect a single store.
  static void collectDstStore(GenericOp gOp, StoreOpVariant store,
                              CopyInfoMap &copyInfos, int dstSlice,
                              Operation *outermostInnerComputeLoop) {
    if (!outermostInnerComputeLoop) {
      outermostInnerComputeLoop = store.getOperation();
    }

    auto [iter, _] = copyInfos.try_emplace(outermostInnerComputeLoop);
    iter->second.recordStore(store, dstSlice);
  }

  // Walk all compute ops in the region and collect loads/stores.
  static DstAccessCollection
  collectDstAccesses(GenericOp gOp, Region &region,
                     Operation *outermostInnerComputeLoop) {
    CopyInfoMap copyInfos;
    DstSliceAllocationState dstSliceAllocationState;
    DstIntermediatesMap dstIntermediates;
    DenseSet<Operation *> collectedLoads;

    region.walk([&](OperandLoadStoreRegisterOpInterface computeOp) {
      auto notDstMemspace = [](Value memref) {
        return memref && ttcore::getMemorySpace(memref) !=
                             ttcore::MemorySpace::RegisterDst;
      };

      // Collect CB->DST loads for this op's operands.
      for (int64_t operandIdx : computeOp.getOperandsLoadFromDstRegister()) {
        if (computeOp.isScalarOperand(operandIdx)) {
          continue;
        }

        Value operand = computeOp->getOperand(operandIdx);
        LoadOpVariant load = LoadOpVariant::fromValue(operand);

        if (load.isValid() && notDstMemspace(load.getMemRef())) {
          if (collectedLoads.contains(load.getOperation())) {
            continue;
          }
          collectedLoads.insert(load.getOperation());
          collectDstLoad(gOp, load, copyInfos,
                         dstSliceAllocationState.allocate(),
                         outermostInnerComputeLoop);
        }
      }

      const bool dstRegInPlace = computeOp.getDstRegInPlace();

      for (auto *user : computeOp->getUsers()) {
        StoreOpVariant store = StoreOpVariant::fromOperation(user);

        if (store.isValid() && notDstMemspace(store.getMemRef())) {
          assert(!dstSliceAllocationState.didStoreToDst() &&
                 "Multiple stores from last op to dst not supported");

          const bool rhsIsScalar =
              computeOp->getNumOperands() > 1 && computeOp.isScalarOperand(1);

          int dstSlice = -1;
          if (dstRegInPlace || rhsIsScalar) {
            bool isUnaryOp = computeOp->getNumOperands() == 1;
            bool isTileMatmul = mlir::isa<d2m::TileMatmulOp>(computeOp);
            bool isReduction = mlir::isa<d2m::TileReduceMaxOp>(computeOp) ||
                               mlir::isa<d2m::TileReduceSumOp>(computeOp);
            assert((isUnaryOp || isTileMatmul || isReduction || rhsIsScalar) &&
                   "Only unary ops, tile matmul, reductions, and tile+scalar "
                   "ops supported for destination register in place");
            dstSlice = dstSliceAllocationState.getCurrSliceIndex();
          } else {
            dstSlice = dstSliceAllocationState.allocate();
            dstSliceAllocationState.setStoreToDst();
          }
          collectDstStore(gOp, store, copyInfos, dstSlice,
                          outermostInnerComputeLoop);
        } else if (user->hasTrait<D2MGenericRegionComputeOpTrait>()) {
          assert(computeOp->hasOneUse() &&
                 "Currently we do not support multiple "
                 "users in the same compute dst region.");
          assert(computeOp->getNumResults() == 1);
          assert(!dstIntermediates.contains(computeOp));

          bool reuseSlot =
              computeOp.getDstRegInPlace() ||
              (computeOp->getNumOperands() > 1 && computeOp.isScalarOperand(1));
          int dstSlice = reuseSlot ? dstSliceAllocationState.getCurrSliceIndex()
                                   : dstSliceAllocationState.allocate();

          if (mlir::isa<d2m::TileBcastOp>(computeOp)) {
            LoadOpVariant loadOp =
                LoadOpVariant::fromValue(computeOp->getOperand(0));
            TT_assert(loadOp.isValid());
            auto bcastOp = mlir::cast<d2m::TileBcastOp>(computeOp);
            collectDstLoad(gOp, loadOp, copyInfos, dstSlice,
                           outermostInnerComputeLoop, bcastOp);
          } else {
            dstIntermediates[computeOp] = {dstSlice, outermostInnerComputeLoop};
          }
        }
      }
    });
    return {copyInfos, dstIntermediates};
  }

  // Collect scheduled path.
  static DstAccessCollection
  collectDstAccessesScheduled(GenericOp gOp, Region &region,
                              Operation *outermostInnerComputeLoop,
                              unsigned dstCapacity) {
    CopyInfoMap copyInfos;
    DstStackAllocator dstStackAllocator(dstCapacity);
    DstIntermediatesMap dstIntermediates;

    LDBG() << "=== collectDstAccessesScheduled START (capacity=" << dstCapacity
           << ") ===";

    region.walk<WalkOrder::PreOrder>([&](OperandLoadStoreRegisterOpInterface
                                             computeOp) {
      auto notDstMemspace = [](Value memref) {
        return memref && ttcore::getMemorySpace(memref) !=
                             ttcore::MemorySpace::RegisterDst;
      };

      int numLoads = 0;
      SmallVector<int32_t> intermediateSlots;

      LDBG() << "Processing: " << computeOp->getName().getStringRef().str();

      auto isIntermediateLive = [&](Operation *intermediateOp) -> bool {
        for (Operation *user : intermediateOp->getUsers()) {
          if (user == computeOp.getOperation()) {
            return true;
          }
          if (user->hasTrait<D2MGenericRegionComputeOpTrait>() &&
              !dstIntermediates.contains(user)) {
            return true;
          }
        }
        return false;
      };

      for (const auto &[intermediateOp, info] : dstIntermediates) {
        if (isIntermediateLive(intermediateOp)) {
          intermediateSlots.push_back(info.dstSlice);
          LDBG() << "  Reserved LIVE intermediate slot DST[" << info.dstSlice
                 << "]";
        }
      }

      for (int64_t operandIdx : computeOp.getOperandsLoadFromDstRegister()) {
        if (computeOp.isScalarOperand(operandIdx)) {
          continue;
        }

        ++numLoads;

        Value operand = computeOp->getOperand(operandIdx);
        LoadOpVariant load = LoadOpVariant::fromValue(operand);

        if (load.isValid() && notDstMemspace(load.getMemRef())) {
          int32_t slot = dstStackAllocator.allocate();
          while (llvm::is_contained(intermediateSlots, slot)) {
            slot = dstStackAllocator.allocate();
          }
          collectDstLoad(gOp, load, copyInfos, slot, outermostInnerComputeLoop);
        }
      }

      for (auto *user : computeOp->getUsers()) {
        StoreOpVariant store = StoreOpVariant::fromOperation(user);

        if (store.isValid() && notDstMemspace(store.getMemRef())) {
          assert(!dstStackAllocator.didStoreToDst() &&
                 "Multiple stores from last op to dst not supported");

          bool dstRegInPlace = computeOp.getDstRegInPlace();
          bool rhsIsScalar =
              computeOp->getNumOperands() > 1 && computeOp.isScalarOperand(1);

          int64_t dstSliceIndex = -1;
          if (dstRegInPlace || rhsIsScalar) {
            bool isUnaryOp = computeOp->getNumOperands() == 1;
            bool isTileMatmul = mlir::isa<d2m::TileMatmulOp>(computeOp);
            bool isReduction = mlir::isa<d2m::TileReduceMaxOp>(computeOp) ||
                               mlir::isa<d2m::TileReduceSumOp>(computeOp);
            assert((isUnaryOp || isTileMatmul || isReduction || rhsIsScalar) &&
                   "Only unary ops, tile matmul, reductions, and tile+scalar "
                   "ops supported for destination register in place");
            dstSliceIndex = dstStackAllocator.getCurrSliceIndex();
          } else {
            dstSliceIndex = dstStackAllocator.allocate(true);
            dstStackAllocator.setStoreToDst();
          }
          collectDstStore(gOp, store, copyInfos, dstSliceIndex,
                          outermostInnerComputeLoop);
        } else if (user->hasTrait<D2MGenericRegionComputeOpTrait>()) {
          assert(computeOp->hasOneUse() &&
                 "Currently we do not support multiple "
                 "users in the same compute dst region.");
          assert(computeOp->getNumResults() == 1);
          assert(!dstIntermediates.contains(computeOp));

          bool overwriteInput =
              computeOp.getDstRegInPlace() ||
              (computeOp->getNumOperands() > 1 && computeOp.isScalarOperand(1));

          int32_t allocatedIndex = (overwriteInput)
                                       ? dstStackAllocator.getCurrSliceIndex()
                                       : dstStackAllocator.allocate(true);

          LDBG() << "INTERMEDIATE: "
                 << computeOp->getName().getStringRef().str() << " -> DST["
                 << allocatedIndex << "]";

          dstIntermediates[computeOp] = {allocatedIndex,
                                         outermostInnerComputeLoop};

          if (!overwriteInput) {
            for (int i = 0; i < numLoads; ++i) {
              dstStackAllocator.deallocate();
            }
          }
        }
      }
    });

    // Handle passthrough case (load directly stored without compute).
    region.walk([&](Operation *storeOpGeneric) {
      StoreOpVariant store = StoreOpVariant::fromOperation(storeOpGeneric);
      if (!store.isValid()) {
        return WalkResult::advance();
      }

      if (ttcore::getMemorySpace(store.getMemRef()) ==
          ttcore::MemorySpace::RegisterDst) {
        return WalkResult::advance();
      }

      LoadOpVariant load = LoadOpVariant::fromValue(store.getValue());
      if (!load.isValid()) {
        return WalkResult::advance();
      }

      if (ttcore::getMemorySpace(load.getMemRef()) ==
          ttcore::MemorySpace::RegisterDst) {
        return WalkResult::advance();
      }

      auto memrefType = load.getMemRefType();
      if (!mlir::isa<ttcore::TileType>(memrefType.getElementType())) {
        return WalkResult::advance();
      }

      int dstSlice = dstStackAllocator.allocate();
      collectDstLoad(gOp, load, copyInfos, dstSlice, outermostInnerComputeLoop);
      collectDstStore(gOp, store, copyInfos, dstSlice,
                      outermostInnerComputeLoop);

      return WalkResult::advance();
    });

    return {copyInfos, dstIntermediates};
  }

  // Generate data copy for non-scheduled path.
  static void dataCopyGenerate(PatternRewriter &rewriter, Location loc,
                               Value dst, const CopyInfoMap &copyInfos) {
    for (const auto &[loopNestOrOp, copyInfo] : copyInfos) {
      rewriter.setInsertionPointAfter(loopNestOrOp);
      auto insertionPointAfterLoopNest = rewriter.saveInsertionPoint();

      // Process loads and load-bcasts.
      rewriter.setInsertionPoint(loopNestOrOp);
      for (auto &record : copyInfo.loads) {
        AffineMap dstAccessMap =
            AffineMap::getConstantMap(record.dstSlice, rewriter.getContext());

        // Generate CB->DST copy op.
        rewriter.setInsertionPoint(loopNestOrOp);
        if (record.load.isAffine()) {
          auto affineLoad = record.load.getAffine();
          auto cbLoad = rewriter.create<affine::AffineLoadOp>(
              record.getLoc(), affineLoad.getMemRef(), affineLoad.getMap(),
              affineLoad.getIndices());
          Value valueToStore = cbLoad.getResult();

          if (record.bcast.has_value()) {
            d2m::TileBcastOp bcastOp = *record.bcast;
            auto *clonedBcast = rewriter.clone(*bcastOp.getOperation());
            clonedBcast->setOperand(0, valueToStore);
            valueToStore = clonedBcast->getResult(0);
          }

          rewriter.create<affine::AffineStoreOp>(
              record.getLoc(), valueToStore, dst, dstAccessMap, ValueRange{});

          // Replace original load op.
          rewriter.setInsertionPoint(affineLoad);
          auto dstLoad = rewriter.create<affine::AffineLoadOp>(
              record.getLoc(), dst, dstAccessMap, ValueRange{});
          if (record.bcast.has_value()) {
            d2m::TileBcastOp bcastOp = *record.bcast;
            bcastOp.getResult().replaceAllUsesWith(dstLoad.getResult());
            rewriter.eraseOp(bcastOp);
          } else {
            rewriter.replaceOp(affineLoad, dstLoad.getResult());
          }
        } else {
          auto memrefLoad = record.load.getMemref();
          auto cbLoad = rewriter.create<memref::LoadOp>(
              record.getLoc(), memrefLoad.getMemRef(), memrefLoad.getIndices());
          Value valueToStore = cbLoad.getResult();

          if (record.bcast.has_value()) {
            d2m::TileBcastOp bcastOp = *record.bcast;
            auto *clonedBcast = rewriter.clone(*bcastOp.getOperation());
            clonedBcast->setOperand(0, valueToStore);
            valueToStore = clonedBcast->getResult(0);
          }

          rewriter.create<affine::AffineStoreOp>(
              record.getLoc(), valueToStore, dst, dstAccessMap, ValueRange{});

          // Replace original load op.
          rewriter.setInsertionPoint(memrefLoad);
          auto dstLoad = rewriter.create<affine::AffineLoadOp>(
              record.getLoc(), dst, dstAccessMap, ValueRange{});
          if (record.bcast.has_value()) {
            d2m::TileBcastOp bcastOp = *record.bcast;
            bcastOp.getResult().replaceAllUsesWith(dstLoad.getResult());
            rewriter.eraseOp(bcastOp);
          } else {
            rewriter.replaceOp(memrefLoad, dstLoad.getResult());
          }
        }
      }

      // Process stores.
      rewriter.restoreInsertionPoint(insertionPointAfterLoopNest);
      for (auto &record : copyInfo.stores) {
        AffineMap dstAccessMap =
            AffineMap::getConstantMap(record.dstSlice, rewriter.getContext());

        // Generate DST->CB copy.
        auto dstLoad = rewriter.create<affine::AffineLoadOp>(
            record.getLoc(), dst, dstAccessMap, ValueRange{});
        Value valueToStore = dstLoad.getResult();

        auto cbType = record.getMemRefType();
        if (valueToStore.getType() != cbType.getElementType()) {
          valueToStore =
              rewriter
                  .create<d2m::DstReinterpretCastOp>(
                      record.getLoc(), cbType.getElementType(), valueToStore)
                  .getResult();
        }

        if (record.store.isAffine()) {
          auto affineStore = record.store.getAffine();
          rewriter.create<affine::AffineStoreOp>(
              record.getLoc(), valueToStore, affineStore.getMemRef(),
              affineStore.getMap(), affineStore.getIndices());

          // Replace original store op.
          rewriter.setInsertionPoint(affineStore);
          auto dstType = mlir::cast<MemRefType>(dst.getType());
          Value storeValue = affineStore.getValue();
          if (storeValue.getType() != dstType.getElementType()) {
            storeValue =
                rewriter
                    .create<d2m::DstReinterpretCastOp>(
                        record.getLoc(), dstType.getElementType(), storeValue)
                    .getResult();
          }
          rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
              affineStore, storeValue, dst, dstAccessMap, ValueRange{});
        } else {
          auto memrefStore = record.store.getMemref();
          rewriter.create<memref::StoreOp>(record.getLoc(), valueToStore,
                                           memrefStore.getMemRef(),
                                           memrefStore.getIndices());

          // Replace original store op.
          rewriter.setInsertionPoint(memrefStore);
          auto dstType = mlir::cast<MemRefType>(dst.getType());
          Value storeValue = memrefStore.getValue();
          if (storeValue.getType() != dstType.getElementType()) {
            storeValue =
                rewriter
                    .create<d2m::DstReinterpretCastOp>(
                        record.getLoc(), dstType.getElementType(), storeValue)
                    .getResult();
          }
          rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
              memrefStore, storeValue, dst, dstAccessMap, ValueRange{});
        }
      }
    }
  }

  // Generate data copy for scheduled path (in-place stores).
  static void dataCopyGenerateScheduled(PatternRewriter &rewriter, Location loc,
                                        Value dst,
                                        const CopyInfoMap &copyInfos) {
    for (const auto &[loopNestOrOp, copyInfo] : copyInfos) {
      rewriter.setInsertionPoint(loopNestOrOp);

      // Process load ops.
      for (auto &record : copyInfo.loads) {
        AffineMap dstAccessMap =
            AffineMap::getConstantMap(record.dstSlice, rewriter.getContext());

        rewriter.setInsertionPoint(record.load.getOperation());

        // Generate CB->DST copy.
        Value cbLoadResult;
        if (record.load.isAffine()) {
          auto affineLoad = record.load.getAffine();
          auto cbLoad = rewriter.create<affine::AffineLoadOp>(
              record.getLoc(), affineLoad.getMemRef(), affineLoad.getMap(),
              affineLoad.getIndices());
          cbLoadResult = cbLoad.getResult();
        } else {
          auto memrefLoad = record.load.getMemref();
          auto cbLoad = rewriter.create<memref::LoadOp>(
              record.getLoc(), memrefLoad.getMemRef(), memrefLoad.getIndices());
          cbLoadResult = cbLoad.getResult();
        }

        rewriter.create<affine::AffineStoreOp>(record.getLoc(), cbLoadResult,
                                               dst, dstAccessMap, ValueRange{});

        // Replace original load with DST load.
        auto dstLoad = rewriter.create<affine::AffineLoadOp>(
            record.getLoc(), dst, dstAccessMap, ValueRange{});

        if (record.load.isAffine()) {
          rewriter.replaceOp(record.load.getAffine(), dstLoad.getResult());
        } else {
          rewriter.replaceOp(record.load.getMemref(), dstLoad.getResult());
        }
      }

      // Process stores in-place.
      for (auto &record : copyInfo.stores) {
        AffineMap dstAccessMap =
            AffineMap::getConstantMap(record.dstSlice, rewriter.getContext());

        Location storeLoc = record.getLoc();
        Value cb = record.store.getMemRef();
        Value valueToStore = record.store.getValue();

        auto dstType = mlir::cast<MemRefType>(dst.getType());
        if (valueToStore.getType() != dstType.getElementType()) {
          rewriter.setInsertionPoint(record.store.getOperation());
          valueToStore =
              rewriter
                  .create<d2m::DstReinterpretCastOp>(
                      storeLoc, dstType.getElementType(), valueToStore)
                  .getResult();
        }

        // Step 1: Store to DST.
        rewriter.setInsertionPoint(record.store.getOperation());
        auto dstStore = rewriter.create<affine::AffineStoreOp>(
            storeLoc, valueToStore, dst, dstAccessMap, ValueRange{});

        // Step 2: Load from DST and store to CB.
        rewriter.setInsertionPointAfter(dstStore);
        auto dstLoad = rewriter.create<affine::AffineLoadOp>(
            storeLoc, dst, dstAccessMap, ValueRange{});

        Value packValue = dstLoad.getResult();
        auto cbType = mlir::cast<MemRefType>(cb.getType());
        if (packValue.getType() != cbType.getElementType()) {
          packValue = rewriter
                          .create<d2m::DstReinterpretCastOp>(
                              storeLoc, cbType.getElementType(), packValue)
                          .getResult();
        }

        if (record.store.isAffine()) {
          auto affineStore = record.store.getAffine();
          rewriter.create<affine::AffineStoreOp>(storeLoc, packValue, cb,
                                                 affineStore.getMap(),
                                                 affineStore.getIndices());
        } else {
          auto memrefStore = record.store.getMemref();
          rewriter.create<memref::StoreOp>(storeLoc, packValue, cb,
                                           memrefStore.getIndices());
        }

        rewriter.eraseOp(record.store.getOperation());
      }
    }
  }

  // Fix intermediate results through DST.
  static void
  fixDstIntermediateResults(PatternRewriter &rewriter, Location loc, Value dst,
                            const DstIntermediatesMap &dstIntermediates) {
    auto dstType = dyn_cast<MemRefType>(dst.getType());
    if (!dstType) {
      return;
    }

    for (const auto &[op, dstInfo] : dstIntermediates) {
      int dstSlice = dstInfo.dstSlice;

      rewriter.setInsertionPointAfter(op);

      AffineMap storeMap =
          AffineMap::getConstantMap(dstSlice, rewriter.getContext());

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
          loc, valueToStore, dst, storeMap, ValueRange{});

      auto loadedResult = rewriter.create<affine::AffineLoadOp>(
          loc, dst, storeMap, ValueRange{});

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

  static bool rewriteTileMatmulAsTileMatmulBlock(
      PatternRewriter &rewriter, GenericOp gOp, Region &region,
      linalg::GenericOp linalgGenericOp, unsigned dstCapacity, bool &modified) {
    assert(linalgGenericOp.getInputs().size() == 2 &&
           "Expected exactly 2 input for tile matmul");
    assert(linalgGenericOp.getOutputs().size() == 1 &&
           "Expected exactly 1 output for tile matmul");

    Value inputAMemref = linalgGenericOp.getInputs()[0];
    Value inputBMemref = linalgGenericOp.getInputs()[1];
    Value outputCMemref = linalgGenericOp.getOutputs()[0];

    rewriter.setInsertionPoint(linalgGenericOp);

    auto linalgLoops = linalg::linalgOpToAffineLoops(rewriter, linalgGenericOp);
    if (failed(linalgLoops)) {
      return false;
    }
    rewriter.eraseOp(linalgGenericOp);
    modified |= insertDstRegisterAccess(
        rewriter, gOp, region, dstCapacity,
        !linalgLoops.value().empty() ? linalgLoops.value().front() : nullptr);

    Operation *outerLoop = linalgLoops.value()[0];
    Block *parentBlk = outerLoop->getBlock();
    auto insertPos = std::next(Block::iterator(outerLoop));

    rewriter.setInsertionPoint(parentBlk, insertPos);
    for (Operation *loopOp : llvm::reverse(linalgLoops.value())) {
      rewriter.eraseOp(loopOp);
    }
    rewriter.create<d2m::TileMatmulBlockOp>(gOp.getLoc(), inputAMemref,
                                            inputBMemref, outputCMemref);
    return true;
  }

  // Generates load loop guards
  static scf::IfOp createLoadLoopGuard(PatternRewriter &rewriter, Location loc,
                                       const std::set<int64_t> &guardDims,
                                       const bool isBcastGuard) {
    if (guardDims.empty()) {
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

    for (int64_t idx : guardDims) {
      Value iterIdx = rewriter.create<d2m::IterIndexOp>(loc, idx);
      Value cmp =
          rewriter.create<arith::CmpIOp>(loc, cmpPredicate, iterIdx, zero);
      if (isBcastGuard) {
        guard = rewriter.create<arith::AndIOp>(loc, guard, cmp).getResult();
      } else {
        guard = rewriter.create<arith::OrIOp>(loc, guard, cmp).getResult();
      }
    }

    return rewriter.create<scf::IfOp>(loc, guard);
  }

  bool useTileMatmul = false;
  unsigned maxDstPhysicalSizeTiles = 0;
};
} // namespace

namespace {
template <typename TileReduceOp>
class D2MPackerMaskResetRewriter : public OpRewritePattern<TileReduceOp> {
public:
  using OpRewritePattern<TileReduceOp>::OpRewritePattern;

  Value index(OpBuilder &rewriter, Location loc, int64_t val) const {
    return rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                              rewriter.getIndexAttr(val));
  }

  LogicalResult matchAndRewrite(TileReduceOp op,
                                PatternRewriter &rewriter) const final {

    bool packerResetFound = false;
    op->getBlock()->walk([&](Operation *op) {
      if (auto packerReset =
              mlir::dyn_cast_or_null<d2m::PackerMaskResetOp>(op)) {
        packerResetFound = true;
      }
    });
    if (packerResetFound) {
      return failure();
    }

    rewriter.setInsertionPointAfter(op);
    ReduceDim reduceDim = op.getReduceDim();
    SmallVector<int64_t> loopBounds =
        op->template getParentOfType<GenericOp>().getLoopBounds();

    scf::IfOp ifOp;
    if (reduceDim == ReduceDim::R) {
      auto iterIndex = rewriter.create<d2m::IterIndexOp>(
          op.getLoc(), static_cast<int64_t>(1));
      auto condOp = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, iterIndex,
          index(rewriter, op.getLoc(), loopBounds[1] - 1));
      ifOp = rewriter.create<scf::IfOp>(op.getLoc(), condOp);
    } else if (reduceDim == ReduceDim::C) {
      auto iterIndex = rewriter.create<d2m::IterIndexOp>(
          op.getLoc(), static_cast<int64_t>(0));
      auto condOp = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, iterIndex,
          index(rewriter, op.getLoc(), loopBounds[0] - 1));
      ifOp = rewriter.create<scf::IfOp>(op.getLoc(), condOp);
    } else if (reduceDim == ReduceDim::RC) {
      auto iterIndexR = rewriter.create<d2m::IterIndexOp>(
          op.getLoc(), static_cast<int64_t>(1));
      auto iterIndexC = rewriter.create<d2m::IterIndexOp>(
          op.getLoc(), static_cast<int64_t>(0));
      auto condOp = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, iterIndexR,
          index(rewriter, op.getLoc(), loopBounds[1] - 1));
      auto condOp2 = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, iterIndexC,
          index(rewriter, op.getLoc(), loopBounds[0] - 1));
      auto finalCondOp =
          rewriter.create<arith::OrIOp>(op.getLoc(), condOp, condOp2);
      ifOp = rewriter.create<scf::IfOp>(op.getLoc(), finalCondOp);
    }
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    rewriter.create<d2m::PackerMaskResetOp>(op.getLoc());

    return success();
  }
};

} // namespace

namespace {
class D2MInsertDstRegisterAccess
    : public impl::D2MInsertDstRegisterAccessBase<D2MInsertDstRegisterAccess> {
public:
  using impl::D2MInsertDstRegisterAccessBase<
      D2MInsertDstRegisterAccess>::D2MInsertDstRegisterAccessBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // Check precondition: linalg.generic ops should be converted to affine,
    // EXCEPT those with tile_matmul when useTileMatmul=false (they'll be
    // handled by the tile_matmul_block rewrite in the pattern).
    WalkResult walkResult = moduleOp->walk([&](linalg::GenericOp op) {
      // Allow linalg ops with tile_matmul when useTileMatmul=false.
      if (!useTileMatmul && hasTileMatmul(op)) {
        return WalkResult::advance();
      }
      // All other linalg ops should have been converted.
      return WalkResult::interrupt();
    });

    if (walkResult.wasInterrupted()) {
      moduleOp.emitOpError()
          << "found linalg.generic operations that were not converted to "
             "affine loops. Please run --d2m-linalg-to-affine before the "
             "--d2m-insert-dst-register-access pass.";
      return signalPassFailure();
    }

    MLIRContext *ctx = moduleOp.getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<D2MInsertDstRegisterAccessRewriter>(
        ctx, useTileMatmul, maxDstPhysicalSizeTiles.getValue());

    patterns.add<D2MPackerMaskResetRewriter<TileReduceSumOp>,
                 D2MPackerMaskResetRewriter<TileReduceMaxOp>>(ctx);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
