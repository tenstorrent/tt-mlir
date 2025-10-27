// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Analysis/Allocation/Planner.h"
#include "ttmlir/Dialect/D2M/Analysis/Allocation/Utils.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Utils.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <numeric>
#include <optional>

//===---------------------------------------------------------------------===//
namespace llvm {
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const SmallBitVector &obj) {
  constexpr char digits[2] = {'0', '1'};

  os << '<';
  for (SmallBitVector::size_type i = 0; i < obj.size(); ++i) {
    os << digits[obj.test(i)];
  }
  return os << '>';
}
} // namespace llvm
//===---------------------------------------------------------------------===//
namespace mlir::tt::d2m {

#define GEN_PASS_DEF_D2MALLOCATE
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Helper definitions.
//===----------------------------------------------------------------------===//

inline ttcore::MemorySpace getMemorySpace(MemRefType memref,
                                          ttcore::MemorySpace dflt) {
  auto memSpace = memref.getMemorySpace();
  return memSpace ? mlir::cast<ttcore::MemorySpaceAttr>(memSpace).getValue()
                  : dflt;
}

inline bool isDeviceMemorySpace(MemRefType memref, ttcore::MemorySpace dflt) {
  return ttcore::isDeviceMemorySpace(getMemorySpace(memref, dflt));
}

//===----------------------------------------------------------------------===//
// Helper classes.
//===----------------------------------------------------------------------===//
namespace {

using Planner = allocation::Planner;

using ttcore::MemorySpace;
using PlannerSpace = Planner::Space;

using AllocSizeT = Planner::AllocSizeT;
using SequenceT = Planner::SequenceT;
using IndexT = Planner::IndexT;
using LiveRange = Planner::LiveRange;

using allocation::AsOperandPrinter;
using allocation::asSeq;
using allocation::asShape;
using allocation::is_operation_v;
using allocation::ordinal;

struct MemorySpaceInfo {

  MemorySpaceInfo() = default;
  MemorySpaceInfo(AllocSizeT baseAddress, AllocSizeT maxAddress,
                  AllocSizeT alignment)
      : baseAddress(baseAddress), maxAddress(maxAddress), alignment(alignment) {
    TT_assert(baseAddress < maxAddress);
    TT_assert(baseAddress % alignment == 0);
    TT_assert(maxAddress % alignment == 0);
  }

  // Valid address range is [baseAddress, maxAddress).

  AllocSizeT baseAddress = 0;
  AllocSizeT maxAddress = 0;
  AllocSizeT alignment = 0;

  static constexpr std::size_t kMaxEnumValForMemorySpace =
      (ttcore::getMaxEnumValForMemorySpace() + 1);
};

using MemorySpaces =
    std::array<MemorySpaceInfo, MemorySpaceInfo::kMaxEnumValForMemorySpace>;

inline PlannerSpace asPlannerSpace(MemorySpace memspace) {
  switch (memspace) {
  case MemorySpace::DeviceDRAM: {
    return PlannerSpace::Spill;
  }
  case MemorySpace::DeviceL1: {
    return PlannerSpace::Scratch;
  }
  default: {
    llvm_unreachable("expected device memory space input");
  }
  }
}

inline MemorySpace asMemorySpace(PlannerSpace space) {
  switch (space) {
  case PlannerSpace::Scratch: {
    return MemorySpace::DeviceL1;
  }
  case PlannerSpace::Spill: {
    return MemorySpace::DeviceDRAM;
  }
  default: {
    llvm_unreachable("expected planner space input");
  }
  }
}

struct LivenessClosure {
  Operation *lastOp;
  LiveRange live;
};

template <typename T>
using space_specific = std::array<T, ordinal(PlannerSpace::end)>;

struct MemrefValueContext { // TODO rename AllocValueContext?
  MemRefType type;
  // All generic op users of this alloc (immediate or through a chain of
  // view/steam layout ops).
  llvm::DenseSet<d2m::GenericOp> genericUsers;
  // // "Raw" allocation request size in bytes (i.e. not aligned up for
  // // any particular memspace).
  // AllocSizeT size = -1;
  // TODO aligned for the associated memspace
  space_specific<AllocSizeT> size = {-1, -1};
  // Live range of this value, starting with the defining op itself
  // and extending to its last user.
  LiveRange live = {-1, -1};
  // `true` iff this value is ineligible for spilling because
  // it has users that are not generic ops.
  bool hasNonGenericUsers = false;
  // `true` iff this value acts as the output of at least one
  // generic op.
  bool usedForOutput = false;
  // `Planner`s spill outcome for this decision variable.
  // TODO replace with PlannerSpace var (memref type could be obtained from
  // `allocCtx`)?
  std::optional<MemorySpace> remappedMemSpace;

  // Fields used to link this Value to a `Planner` decision variable.

  int32_t varIndex = -1; // Needed to retrieve `Planner::Variable::placement`.
  int32_t reqIndex = -1; // Needed to retrieve `Planner::Request::offset`.
};

using DefUseChain = llvm::SmallVector<Operation *, 4>;

struct OperandContext {
  // Index of this operand for the associated generic op.
  size_t operandIndex = 0;
  // This collects the set of ops defining an operand all the way to its
  // root `memref::AllocOp` or block arg.
  // TODO(vroubtsov) consider making operand Value the 0th slot in here, get rid
  // of `operandIndex` etc.
  DefUseChain defChain;
  // TODO(vroubtsov)
  MemRefType bufferType; // TODO doc this is always in L1
  MemRefType streamType; // TODO with reblocked map
  // `true` is if this corresponds to a generic op output.
  bool isOutput = false;
  // `true` if this operand requires streaming regardless of
  // possible spilling (e.g. because of an intra-core data movement
  // pattern.)
  bool requiresStream = false;

  // Fields used to link this Value to a `Planner` decision variable.

  // Needed to retrieve `Planner::Request::offset` for this operand stream's
  // storage buffer, both `Scratch` and `Spill` alternatives.
  space_specific<int32_t> reqIndex = {-1, -1};
};

// A map linking `OperandContext`s with their originating `Value`s (defined
// by `memref.alloc`s or passed as block args).
using OperandContextMap = llvm::SmallMapVector<mlir::Value, OperandContext, 4>;

struct GenericOpContext {
  // Context info for each of this generic ops list of operands, in declaration
  // order. (Note that the latter relies on `SmallMapVector` structure).
  OperandContextMap operands;
  // TODO(vroubtsov)
  SmallVector<int64_t> scaleFactors;
  // Generic ops in "DMA-only" form currently operate in alias mode
  // and do not use operand streams.
  bool isDMAOnly = false;
};

struct SequenceMapping {
  // Within a func body scope, maps logical time (preorder) positions
  // to their `Operation`s.
  llvm::SmallVector<Operation *> positionMap;
  // Inverse of `positionMap`.
  DenseMap<Operation *, SequenceT> operationMap;

  SequenceT size() const { return positionMap.size(); }

  SequenceT operator[](Operation *op) const {
    auto i = operationMap.find(op);
    TT_debugv(i != operationMap.end(), "failed to map {}", *op);
    return i->second;
  }

  template <typename ConcreteOp>
  auto operator[](ConcreteOp op) const
      -> std::enable_if_t<is_operation_v<ConcreteOp>, SequenceT> {
    return this->operator[](op.getOperation());
  }
};

using PlannerProblems =
    std::array<Planner::Problem, MemorySpaceInfo::kMaxEnumValForMemorySpace>;

struct FuncAnalysisData {
  SequenceMapping sequencing;
  llvm::DenseMap<mlir::Value, MemrefValueContext> memrefs;
  llvm::DenseMap<d2m::GenericOp, GenericOpContext> generics;
  PlannerProblems problems; // Only using L1 and DRAM slots.

  const Planner::Problem &problem(MemorySpace memspace) const {
    return problems[ordinal(memspace)];
  }

  Planner::Problem &problem(MemorySpace memspace) {
    return const_cast<Planner::Problem &>(
        (const_cast<const FuncAnalysisData *>(this))->problem(memspace));
  }
};

} // namespace
//===----------------------------------------------------------------------===//
// Pass implementation.
//===----------------------------------------------------------------------===//
namespace {
class D2MAllocate final : public impl::D2MAllocateBase<D2MAllocate> {
  using Base = impl::D2MAllocateBase<D2MAllocate>;

  using Base::Base;

  MemorySpaces memSpaces;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    TT_ALLOC_DEBUG("configured with options: {}", to_string(*this));

    memSpaces = [this, moduleOp]() {
      ttcore::SystemDescAttr systemDesc =
          ttcore::getCurrentScopeSystemDesc(moduleOp);
      ttcore::ChipDescAttr chipDesc = systemDesc.getChipDescs().front();
      return getMemorySpaces(chipDesc, testAssumeL1Capacity);
    }();

    TT_ALLOC_DEBUG("using memspaces:\n"
                   "  L1:\t[{}, {}),\n"
                   "  DRAM:\t[{}, {})",
                   memSpaces[ordinal(MemorySpace::DeviceL1)].baseAddress,
                   memSpaces[ordinal(MemorySpace::DeviceL1)].maxAddress,
                   memSpaces[ordinal(MemorySpace::DeviceDRAM)].baseAddress,
                   memSpaces[ordinal(MemorySpace::DeviceDRAM)].maxAddress);

    if (moduleOp
            ->walk([&](func::FuncOp funcOp) -> WalkResult {
              return runOnFunc(funcOp);
            })
            .wasInterrupted()) {
      signalPassFailure();
    }
  }

  LogicalResult runOnFunc(func::FuncOp funcOp) {
    if (funcOp.isDeclaration()) {
      return success();
    }

    FuncAnalysisData analysis;

    if (failed(analyzeAllocOps(funcOp, analysis))) {
      return failure();
    }

    if (failed(analyzeGenericOps(funcOp, analysis))) {
      return failure();
    }

    if (failed(prepareMemoryPlanner(funcOp, analysis))) {
      return failure();
    }

    if (failed(runMemoryPlanner(funcOp, analysis))) {
      return failure();
    }

    if (failed(assignAllocAddresses(funcOp, analysis))) {
      return failure();
    }

    if (failed(insertOperandStreams(funcOp, analysis))) {
      return failure();
    }

    return success();
  }

  LogicalResult analyzeAllocOps(func::FuncOp funcOp,
                                FuncAnalysisData &analysis) {
    [[maybe_unused]] AsOperandPrinter asOperand{funcOp};

    ttcore::DeviceAttr device = ttcore::lookupDevice(funcOp);
    Block &funcBody = funcOp.getBody().front();

    IRRewriter rewriter(funcOp->getContext());

    // All `memref.alloc`s will need to be placed into memspaces, therefore
    // collect all of them (regardless of whether they define operands of
    // generic ops or something else).

    // Start with SSA liveness for `func`.

    mlir::Liveness liveness(funcOp.getOperation());
    const mlir::LivenessBlockInfo *li = liveness.getLiveness(&funcBody);

    // (a) Build `Operation` <-> preorder position mappings for the
    //  (unmodified) `funcOp` IR.
    // (b) Collect a separate set of "ops of interest", which are
    // `memref.alloc`s as well as certain ops that we imbue with semantics
    //  of extending liveness of their memref operands.

    llvm::DenseMap<Operation *, LivenessClosure> livenessJoinGraph;

    funcBody.walk<WalkOrder::PreOrder>([&](Operation *op) {
      const SequenceT position = analysis.sequencing.size();

      analysis.sequencing.operationMap[op] = position;
      analysis.sequencing.positionMap.emplace_back(op);

      if (llvm::isa<memref::AllocOp, d2m::ViewLayoutOp, d2m::StreamLayoutOp>(
              op)) {
        TT_assert(op->getNumResults() == 1u);
        Value result = op->getResult(0);

        Operation *firstOp = li->getStartOperation(result);
        Operation *lastOp = li->getEndOperation(result, firstOp);

        LivenessClosure &closure = livenessJoinGraph[op];
        closure.lastOp = lastOp;
        closure.live = {position, -1};
      }
    });
    TT_debug(analysis.sequencing.operationMap.size() ==
             analysis.sequencing.positionMap.size());

    // Ops in `livenessJoinGraph` form a graph of Values and their users where
    // some Values have their original SSA liveness "extended" by stream op
    // users (d2m.view_layout, d2m.stream_layout).
    //
    // We calculate the "last use position" by computing for each value
    // the max over its users over a traversal through this graph.

    for (auto &[op, closure] : livenessJoinGraph) {
      // Initial maxLast values are from the SSA liveness calculation.
      auto i = analysis.sequencing.operationMap.find(closure.lastOp);
      TT_debug(i != analysis.sequencing.operationMap.end());
      closure.live.last = i->second;
    }

    // TODO(vroubtsov) this is retained from v2, but now there is an opportunity
    // to merge live range and def/use chain calculations into a single pass
    // step
    // TODO(vroubtsov) non-recursive impl?
    for (auto &[op, closure] : livenessJoinGraph) {
      closure.live.last = resolve(op, livenessJoinGraph);

      // Copy liveness results into our alloc set.
      if (memref::AllocOp allocOp = llvm::dyn_cast<memref::AllocOp>(op)) {
        TT_assertv(!allocOp->use_empty(),
                   "didn't expect an alloc op without uses: {}",
                   asOperand(allocOp));

        MemrefValueContext &memrefCtx = analysis.memrefs[allocOp];

        memrefCtx.type = mlir::cast<MemRefType>(op->getResult(0).getType());
        if (isDeviceMemorySpace(memrefCtx.type, MemorySpace::System)) {
          memrefCtx.size =
              getAlignedAllocSizes(rewriter, memrefCtx.type, memSpaces, device);
        }

        memrefCtx.live = closure.live;
      }
    }

    TT_ALLOC_DEBUG("collected {} root memref context(s)",
                   analysis.memrefs.size());
    return success();
  }

  static llvm::SmallVector<OperandContext>
  getOperandContexts(d2m::GenericOp genericOp) {
    const std::size_t outputsStart =
        genericOp.getOutputs().getBeginOperandIndex();
    ArrayAttr iteratorTypes = genericOp.getIteratorTypes();

    llvm::SmallVector<OperandContext> result;

    for (std::size_t operandIndex = 0;
         operandIndex < genericOp.getNumOperands(); ++operandIndex) {
      OperandContext &operandCtx = result.emplace_back();

      operandCtx.operandIndex = operandIndex;
      operandCtx.isOutput = (operandIndex >= outputsStart);

      if (operandCtx.isOutput) {
        // L1 outputs are currently allocated in L1 so won't use streams unless
        // allowed to do so in `allowL1OutputSpilling` mode.
        // DRAM outputs always need to be spilled.
        continue;
      }

      // A core participating in a reduction dim necessarily requires
      // non-local data movement unless it is the only core involved
      // in that dim.
      // Similar logic applies to a broadcast dim.
      const AffineMap indexingMap = genericOp.getIndexingMap(operandIndex);
      const auto bcastDims = indexingMap.getBroadcastDims();
      const llvm::SmallSet<unsigned, 4> bcastDimIndex(bcastDims.begin(),
                                                      bcastDims.end());
      operandCtx.requiresStream = llvm::any_of(
          llvm::seq(indexingMap.getNumResults()), [&](unsigned resultIndex) {
            if (bcastDimIndex.contains(resultIndex)) {
              return true;
            }
            const auto dimPosition = indexingMap.getDimPosition(resultIndex);
            ttcore::IteratorType iteratorType =
                mlir::cast<ttcore::IteratorTypeAttr>(iteratorTypes[dimPosition])
                    .getValue();
            return (iteratorType == ttcore::IteratorType::Reduction);
          });

      // Note: even if `operandCtx.requiresStream` is left false here, a stream
      // may still be inserted in the final IR, e.g. to read the operand from
      // DRAM.
    }

    // TODO(vroubtsov) possible edge case issue with this design if the same
    // `Value` appears in more than one operand slot?
    TT_debug(result.size() == genericOp.getNumOperands());

    return result;
  }

  static SmallVector<int64_t> rescaleLeft(SmallVector<int64_t> shape,
                                          SmallVector<int64_t> factors) {
    const std::size_t rank = factors.size();
    TT_assertv(shape.size() == 2 * rank, "expected full grid/shard shape");

    SmallVector<int64_t> r(2 * rank);
    for (std::size_t d = 0; d < rank; ++d) {
      const auto factor = factors[d];
      r[d] = shape[d] * factor;
      r[d + rank] = shape[rank + d] / factor;
      TT_debug(r[d] * r[d + rank] == shape[d] * shape[d + rank]);
    }

    return r;
  }

  SmallVector<int64_t> getGridExtents(d2m::GenericOp genericOp) {
    auto maps = genericOp.getIndexingMapsValue();
    auto flatInverseMap =
        utils::concatInversePermutationMap(maps, /*reverse=*/false);

    auto shapes = genericOp.getOperandGridShapes();
    SmallVector<int64_t> flattenedShapes;
    for (auto v : shapes) {
      flattenedShapes.append(v.begin(), v.end());
    }

    return flatInverseMap.compose(flattenedShapes);
  }
  SmallVector<int64_t> getShardExtents(d2m::GenericOp genericOp) {
    auto maps = genericOp.getIndexingMapsValue();
    auto flatInverseMap =
        utils::concatInversePermutationMap(maps, /*reverse=*/false);

    auto shapes = genericOp.getOperandShardShapes(false);
    SmallVector<int64_t> flattenedShapes;
    for (auto v : shapes) {
      flattenedShapes.append(v.begin(), v.end());
    }

    return flatInverseMap.compose(flattenedShapes);
  }

  /// Return a bitmask that indicates which of the dims are "participating"
  /// (defined as dims that are used by any of the `genericOp`'s output
  /// expressions.
  llvm::SmallBitVector getParticipatingDimMask(d2m::GenericOp genericOp) {
    TT_debugv(genericOp.getDpsInits().size() == 1u, "expected 1 op output");
    AffineMap outputMap = genericOp.getIndexingMapsValue().back();

    const std::size_t rank = outputMap.getNumDims();

    llvm::SmallBitVector mask(rank, false);
    for (std::size_t d = 0; d < rank; ++d) {
      mask[d] = outputMap.isFunctionOfDim(d);
    }
    return mask;
  }

  /// Calculate full "shard-only" blocking factors (defined as full blocking
  /// factors divided by blocking factors)
  SmallVector<int64_t> getShardBlockFactors(d2m::GenericOp genericOp) {
    SmallVector<int64_t> r = genericOp.getFullBlockFactors();
    SmallVector<int64_t> blockFactors = genericOp.getBlockFactorsValue();
    TT_debug(r.size() == blockFactors.size());
    for (std::size_t d = 0; d < r.size(); ++d) {
      TT_debugv(blockFactors[d] > 0, "unxpected block factor {} for dim {}",
                blockFactors[d], d);
      r[d] /= blockFactors[d];
    }

    return r;
  }

  static std::array<AffineMap, 2>
  getRescalingMaps(ArrayRef<int64_t> factors,
                   ArrayRef<int64_t> /* old */ extents, MLIRContext *ctx) {
    const auto rank = factors.size();

    SmallVector<AffineExpr> dilation;
    SmallVector<AffineExpr> contraction;
    for (std::size_t d = 0; d < rank; ++d) {
      const auto factor = mlir::getAffineConstantExpr(factors[d], ctx);
      const auto extent = mlir::getAffineConstantExpr(extents[d], ctx);
      dilation.push_back(mlir::getAffineDimExpr(d, ctx).floorDiv(factor));
      contraction.push_back((mlir::getAffineDimExpr(d, ctx) % factor) * extent);
    }

    return {AffineMap::get(rank, 0, dilation, ctx),
            AffineMap::get(rank, 0, contraction, ctx)};
  }

  [[maybe_unused]] static std::array<AffineExpr, 2>
  getRescaling(std::size_t dimIndex, int64_t factor, int64_t /* old */ extent,
               MLIRContext *ctx) {
    return {mlir::getAffineDimExpr(dimIndex, ctx).floorDiv(factor),
            (mlir::getAffineDimExpr(dimIndex, ctx) % factor) * extent +
                mlir::getAffineDimExpr(dimIndex + /*TODO*/ 2, ctx)};
  }

  static SmallVector<int64_t>
  calculateBlockFactors(ArrayRef<int64_t> gridShape,
                        const SmallVector<AffineMap> &indexingMaps,
                        /* TODO ValueRange? */ ArrayRef<MemRefType> operands) {
    const auto output = operands.back();
    const ttcore::DeviceLayoutInterface layout =
        ttcore::getDeviceLayout(output);
    TT_debug(layout != nullptr);
    TT_assert(layout.getGridShape(output) == gridShape);

    auto flatInverseMap =
        utils::concatInversePermutationMap(indexingMaps, /*reverse=*/true);

    SmallVector<int64_t> flattenedOperandGridShapes;
    for (MemRefType operand : llvm::reverse(operands)) {
      const ttcore::DeviceLayoutInterface layout =
          ttcore::getDeviceLayout(operand);
      TT_debug(layout != nullptr);
      auto gridShape = layout.getGridShape(operand);
      flattenedOperandGridShapes.append(gridShape.begin(), gridShape.end());
    }

    for (std::size_t i = 0; i < gridShape.size(); ++i) {
      flattenedOperandGridShapes[i] /= gridShape[i];
    }

    return flatInverseMap.compose(flattenedOperandGridShapes);
  }

  LogicalResult analyzeGenericOps(func::FuncOp funcOp,
                                  FuncAnalysisData &analysis) {

    [[maybe_unused]] AsOperandPrinter asOperand{funcOp};

    ttcore::DeviceAttr device = ttcore::lookupDevice(funcOp);
    Block &funcBody = funcOp.getBody().front();
    MLIRContext *const ctx = funcOp->getContext();

    IRRewriter rewriter(funcOp->getContext());

    using OperationSet = llvm::SmallPtrSet<Operation *, 4>;

    // Temp state to help set `MemrefValueContext::hasNonGenericUsers`.
    llvm::DenseMap<memref::AllocOp, OperationSet> genericUseClosure;

    [[maybe_unused]] int32_t genericsInDMAOnlyForm = 0;

    funcBody.walk([&](d2m::GenericOp genericOp) {
      GenericOpContext &genericCtx = analysis.generics[genericOp];

      // Detect generic ops in "DMA-only" form, they must not
      // insert operand streams and therefore have no memory allocation
      // needs possibly associated with those.
      genericCtx.isDMAOnly = genericOp.isDMAOnlyForm();
      genericsInDMAOnlyForm += genericCtx.isDMAOnly;

      // TODO(vroubtsov) at some point it makes sense to merge
      // getOperandContexts() into this step and/or break out
      // stream insertion into its own pass

      const std::size_t rank = genericOp->getNumOperands();

      // Decide which operands might/must have streams. Note that
      // the actual stream creation decision is only final after
      // we have a feasible planner solution.
      llvm::SmallVector<OperandContext> operandCtxs =
          getOperandContexts(genericOp);

      TT_ALLOC_DEBUG(
          "*** {}: grid {}, factors {}, full factors {}, grid "
          "extents {}, shard extents {}, SHARD-ONLY factors {}, MASK {}",
          asOperand(genericOp), asShape(genericOp.getGrid().getShape()),
          asSeq(genericOp.getBlockFactorsValue()),
          asSeq(genericOp.getFullBlockFactors()),
          asSeq(getGridExtents(genericOp)), asSeq(getShardExtents(genericOp)),
          asSeq(getShardBlockFactors(genericOp)),
          getParticipatingDimMask(genericOp));

      [[maybe_unused]] const bool useMinPolicy =
          (testBufferSizePolicy == "min");

      // Allow rescaling in "min" mode only and only for non-participating
      // dims:
      genericCtx.scaleFactors.resize(rank, 1);
      if (useMinPolicy) {
        const SmallVector<int64_t> shardFactors =
            getShardBlockFactors(genericOp);
        const llvm::SmallBitVector participationMask =
            getParticipatingDimMask(genericOp);
        for (auto d = participationMask.find_first_unset(); d >= 0;
             d = participationMask.find_next_unset(d)) {
          genericCtx.scaleFactors[d] = shardFactors[d];
        }
      }
      TT_ALLOC_DEBUG("genericCtx.scaleFactors = {}",
                     asSeq(genericCtx.scaleFactors));
      TT_debug(genericCtx.scaleFactors.size() == rank);

      SmallVector<int64_t> iota(4);
      std::iota(iota.begin(), iota.end(), 0);

      SmallVector<int64_t> gridExtents = getGridExtents(genericOp);
      SmallVector<int64_t> shardExtents = getShardExtents(genericOp);
      TT_debug((gridExtents.size() == rank and shardExtents.size() == rank));
      for (std::size_t d = 0; d < rank; ++d) {
        gridExtents[d] *= genericCtx.scaleFactors[d];
        shardExtents[d] /= genericCtx.scaleFactors[d];
      }

      const ttcore::MemorySpaceAttr l1Attr =
          ttcore::MemorySpaceAttr::get(ctx, ttcore::MemorySpace::DeviceL1);

      for (auto [operandIndex, operand] :
           llvm::enumerate(genericOp.getOperands())) {
        OperandContext &operandCtx = operandCtxs[operandIndex];

        {
          const AffineMap indexingMap = genericOp.getIndexingMap(operandIndex);
          TT_ALLOC_TRACE("\t[{}] operand map (projected permutation: {}): {}",
                         operandIndex, indexingMap.isProjectedPermutation(),
                         indexingMap);

          // const AffineMap dilatedMap = indexingMap.compose(rescaling[0]);
          // const AffineMap contractedMap =
          // indexingMap.compose(rescaling[1]);

          // TT_ALLOC_DEBUG("\t[{}] operand map (projected permutation: {}):
          // {},"
          //                " scale factors: {}, rescaling maps: dilated {}
          //                and " "contracted {}", operandIndex,
          //                indexingMap.isProjectedPermutation(), indexingMap,
          //                asShape(genericCtx.scaleFactors), dilatedMap,
          //                contractedMap);

          const auto operandType = mlir::cast<MemRefType>(operand.getType());
          // const auto operandLayout =
          // mlir::cast<ttcore::DeviceLayoutInterface>(
          //     operandType.getLayout());

          // const ArrayRef<int64_t> gridShape =
          //     operandLayout.getGridShape(operandType);
          // const ArrayRef<int64_t> shardShape =
          //     operandLayout.getShardShape(operandType);

          SmallVector<int64_t> gridShapeRescaled =
              indexingMap.compose(gridExtents);
          SmallVector<int64_t> shardShapeRescaled =
              indexingMap.compose(shardExtents);

          AffineMap reblockingMap = operandType.getLayout().getAffineMap();

          SmallVector<int64_t> indexing = indexingMap.compose(iota);
          TT_ALLOC_TRACE("\tindexing {}", asShape(indexing));

          for (std::size_t pos = 0; pos < indexing.size(); ++pos) {
            const int64_t d = indexing[pos];
            TT_debug_limit(d, iota.size());

            auto rescaling = getRescaling(pos, genericCtx.scaleFactors[d],
                                          shardExtents[d], ctx);

            reblockingMap = reblockingMap.replace(
                mlir::getAffineDimExpr(pos, ctx), rescaling[0], 4, 0);
            reblockingMap = reblockingMap.replace(
                mlir::getAffineDimExpr(pos + 2, ctx), rescaling[1], 4, 0);
          }
          TT_ALLOC_TRACE("\treblocking map: {}", reblockingMap);

          std::tie(operandCtx.bufferType, operandCtx.streamType) =
              getStreamBuffer(gridShapeRescaled, shardShapeRescaled,
                              reblockingMap, operandType.getElementType(),
                              numStreamBuffers, l1Attr);

          const int64_t bufferSizeBytes =
              getStreamBufferSizeBytes(operandCtx.bufferType, device);

          TT_ALLOC_DEBUG(
              "\n\t[{}] storage buffer ({} byte(s)): {};\n\tstream type: {}",
              operandIndex, bufferSizeBytes, operandCtx.bufferType,
              operandCtx.streamType);
        }

        // For later IR mutation, it is convenient at this point to gather
        // all chains of ops defining operand inputs.
        Value memref =
            getOperandDefChain(genericOp, operand, operandCtx.defChain);

        const auto &[i, inserted] = analysis.memrefs.try_emplace(memref);
        MemrefValueContext &memrefCtx = i->second;

        memrefCtx.genericUsers.insert(genericOp);
        memrefCtx.usedForOutput |= operandCtx.isOutput;

        if (inserted) {
          // These were not discovered by the earlier `analyzeAllocOps()`, it
          // could only happen if the value is a block arg.
          TT_debugv(mlir::isa<BlockArgument>(memref),
                    "expected a block arg: {}", memref);
          memrefCtx.type = mlir::cast<MemRefType>(memref.getType());
          memrefCtx.size =
              getAlignedAllocSizes(rewriter, memrefCtx.type, memSpaces, device);
        } else {
          // An existing `analysis.memrefs` entry means `operand` is
          // ultimately rooted in a `memref::AllocOp`.
          memref::AllocOp allocOp =
              mlir::cast<memref::AllocOp>(memref.getDefiningOp());

          OperationSet &allocOpGenericUsers = genericUseClosure[allocOp];
          allocOpGenericUsers.insert(genericOp.getOperation());
          allocOpGenericUsers.insert(operandCtx.defChain.begin(),
                                     operandCtx.defChain.end());
        }

        genericCtx.operands.try_emplace(memref, std::move(operandCtx));
      }
      TT_debug(genericCtx.operands.size() == genericOp.getNumOperands());
    });

    if (TT_DEBUG_ENABLED()) {
      for ([[maybe_unused]] auto &[value, valueCtx] : analysis.memrefs) {
        TT_ALLOC_TRACE("\t{}:\t{} generic user(s), [{}, {}], {} byte(s)",
                       asOperand(value), valueCtx.genericUsers.size(),
                       valueCtx.live.first, valueCtx.live.last,
                       asSeq(valueCtx.size));
      };
    }

    // Alloc ops that have users other than a `func.return` or `d2m.generic`
    // will be marked as ineligible for memspace remapping.

    [[maybe_unused]] int32_t allocsWithNonGenericUsers = 0;

    for (auto &[allocOp, users] : genericUseClosure) {
      for (Operation *user : allocOp->getUsers()) {
        if (!llvm::isa<func::ReturnOp>(user) && !users.contains(user)) {
          analysis.memrefs[allocOp].hasNonGenericUsers = true;
          ++allocsWithNonGenericUsers;
        }
      }
    }

    TT_ALLOC_DEBUG("collected {} generic op context(s) ({} DMA-only)",
                   analysis.generics.size(), genericsInDMAOnlyForm);
    TT_ALLOC_DEBUG("found {} alloc(s) with non-generic use",
                   allocsWithNonGenericUsers);
    return success();
  }

  LogicalResult prepareMemoryPlanner(func::FuncOp funcOp,
                                     FuncAnalysisData &analysis) {
    ttcore::DeviceAttr device = ttcore::lookupDevice(funcOp);
    IRRewriter rewriter(funcOp->getContext());

    Planner::Problem &problem = analysis.problem(MemorySpace::DeviceL1);

    // Each `analysis.memrefs` entry defines an allocation planner decision
    // variable. These can be of different origins:
    // (1) A memref defined by a `memref.alloc` backing a generic op operand
    // and
    //     potentially associated with a stream and its buffer.
    // (2) A memref that backs a generic op operand but is not defined by an
    // op
    //     inside `funcOp` (i.e. passed in as a block argument). We may insert
    //     a stream for this operand and will therefore need to allocate this
    //     stream's buffer.
    // (3) A memref defined by a "standalone" `memref.alloc` that needs no
    //     generic op streaming but will still need a valid L1/DRAM memory
    //     address assigned.

    for (auto &[memref, memrefCtx] : analysis.memrefs) {
      const MemorySpace memspace =
          getMemorySpace(memrefCtx.type, MemorySpace::System);
      if (!ttcore::isDeviceMemorySpace(memspace)) {
        continue;
      }
      // Invariant established earlier: all `analysis.memrefs` in DRAM/L1
      // have 'type' and 'size' set.
      TT_debug(
          (memrefCtx.type != nullptr &&
           llvm::all_of(memrefCtx.size, [](auto size) { return size >= 0; })));

      TT_debug(memrefCtx.varIndex < 0);
      memrefCtx.varIndex =
          problem.def([&, &memref = memref,
                       &memrefCtx = memrefCtx](Planner::VariableBuilder &b) {
            // If `memref` is being defined inside `funcOp` and is initially
            // placed in L1, it will require scratch memory to hold its tensor
            // data.
            if (memref.getDefiningOp<memref::AllocOp>() &&
                memspace == MemorySpace::DeviceL1) {
              memrefCtx.reqIndex =
                  b.request(PlannerSpace::Scratch,
                            memrefCtx.size[ordinal(asPlannerSpace(memspace))],
                            memrefCtx.live.first, memrefCtx.live.last);
            }

            // This decision variable must be bound to its incoming memspace
            // in any of these cases:
            //  - if it is placed in DRAM *explicitly*;
            //  - if it has non-generic op users or has zero generic op users;
            //  - if it the output of a generic op and the enabled pass
            //  options do not allow output spilling.
            const bool bound =
                (memspace == MemorySpace::DeviceDRAM) ||
                memrefCtx.genericUsers.empty() ||
                memrefCtx.hasNonGenericUsers ||
                (memrefCtx.usedForOutput && !allowL1OutputSpilling);
            if (bound) {
              b.bind(asPlannerSpace(memspace));
            }

            // For each possible variable placement, add mem requests for L1
            // stream buffers if the variable must be streamed when it backs a
            // generic op operand.
            for (PlannerSpace placement = PlannerSpace::begin;
                 placement < PlannerSpace::end; ++placement) {

              const MemorySpace placementMemspace = asMemorySpace(placement);
              if (bound && placementMemspace != memspace) {
                // A bound variable only needs its domain populated for its
                // fixed (incoming) memspace.
                continue;
              }

              const auto &memInfo = memSpaces[ordinal(placementMemspace)];

              for (d2m::GenericOp user : memrefCtx.genericUsers) {
                GenericOpContext &genericCtx = analysis.generics[user];
                if (genericCtx.isDMAOnly) {
                  continue;
                }
                OperandContext &operandCtx =
                    genericCtx.operands.find(memref)->second;

                // An operand stream is required under any of these
                // conditions:
                // - streaming was earlier determined as required due to
                // inter-core data movement needs;
                // - the final memref placement is `Spill` (i.e. DRAM
                // memspace).
                if (operandCtx.requiresStream ||
                    (placement == PlannerSpace::Spill)) {
                  TT_debug(operandCtx.bufferType != nullptr);
                  const AllocSizeT bufferSize = ttmlir::utils::alignUp(
                      getStreamBufferSizeBytes(operandCtx.bufferType, device),
                      memInfo.alignment);

                  // Because we will insert stream buffer allocs just before
                  // generic ops themselves, without any other interposing
                  // allocs, it is mathematically correct to see all such
                  // buffers' live ranges as a single position coinciding with
                  // the generic op's logical time.
                  const SequenceT firstAndLast = analysis.sequencing[user];

                  TT_debug(operandCtx.reqIndex[ordinal(placement)] < 0);
                  operandCtx.reqIndex[ordinal(placement)] = b.request(
                      placement, bufferSize, firstAndLast, firstAndLast);
                }
              }
            }
          });
    }

    TT_ALLOC_TRACE("L1 planner problem:\n{}", problem);
    return success();
  }

  LogicalResult runMemoryPlanner(func::FuncOp funcOp,
                                 FuncAnalysisData &analysis) {
    // Solve the "main" problem, spilling and allocating.
    {
      auto &problem = analysis.problem(MemorySpace::DeviceL1);

      const auto &memInfo = memSpaces[ordinal(MemorySpace::DeviceL1)];
      const auto memUsageLimit = memInfo.maxAddress - memInfo.baseAddress;

      [[maybe_unused]] const auto stats =
          Planner::spillAllocate(problem, memUsageLimit);
      TT_ALLOC_DEBUG("L1 solution stats: {}", stats);

      if (stats.memUsage > memUsageLimit) {
        return funcOp.emitOpError()
               << "required L1 memory usage " << stats.memUsage
               << " exceeds memory capacity "
               << (memInfo.maxAddress - memInfo.baseAddress)
               << " (usable space is [" << memInfo.baseAddress << ", "
               << memInfo.maxAddress << "))";
      }
      TT_ALLOC_DEBUG("L1 solution verified: {}", Planner::verify(problem));
      TT_ALLOC_TRACE("L1 solution:{}", problem);
    }
    // What's left is an easier problem, which is just an allocation problem
    // formed by all variables with `Spill` placements (either decided so
    // above by the planner or because so bound).
    {
      const auto &L1solution = analysis.problem(MemorySpace::DeviceL1);
      auto &problem = analysis.problem(MemorySpace::DeviceDRAM);

      const auto &memInfo = memSpaces[ordinal(MemorySpace::DeviceDRAM)];

      for (auto &[memref, memrefCtx] : analysis.memrefs) {
        if (!isDeviceMemorySpace(memrefCtx.type, MemorySpace::System)) {
          continue;
        }

        const auto placement =
            L1solution.variable(memrefCtx.varIndex).placement;
        if (placement == PlannerSpace::Spill) {
          memrefCtx.varIndex = problem.def(
              [&, &allocCtx = memrefCtx](Planner::VariableBuilder &b) {
                allocCtx.reqIndex = b.request(
                    PlannerSpace::Scratch, allocCtx.size[ordinal(placement)],
                    allocCtx.live.first, allocCtx.live.last);
              });
          memrefCtx.remappedMemSpace = MemorySpace::DeviceDRAM;
        } else {
          // This `memref` remains in scratch memory and we have its solution
          // parameters in `analysis.problem(PlannerSpace::Scratch)`.
          memrefCtx.remappedMemSpace = MemorySpace::DeviceL1;
        }
      }

      if (!problem.empty()) {
        problem.reset(PlannerSpace::Scratch);
        TT_ALLOC_TRACE("DRAM planner problem:\n{}", problem);

        // Now we just allocate(), not spillAllocate().

        [[maybe_unused]] const auto stats = Planner::allocate(problem);

        const auto memUsageLimit = memInfo.maxAddress - memInfo.baseAddress;
        if (stats.memUsage > memUsageLimit) {
          return funcOp.emitOpError()
                 << "required DRAM memory usage " << stats.memUsage
                 << " exceeds memory capacity "
                 << (memInfo.maxAddress - memInfo.baseAddress)
                 << " (usable space is [" << memInfo.baseAddress << ", "
                 << memInfo.maxAddress << "))";

          TT_ALLOC_DEBUG("DRAM solution verified: {}",
                         Planner::verify(problem));
          TT_ALLOC_TRACE("DRAM solution:{}", problem);
        }
      }
    }

    return success();
  }

  // Sweep through all collected allocs (not just those associated with
  // operand streams) and set their address/alignment attribute, *without*
  // changing their memspace. The latter will be fixed up in a subsequent step
  // (which will also restore the IR to a valid state).
  LogicalResult assignAllocAddresses(func::FuncOp funcOp,
                                     FuncAnalysisData &analysis) {
    [[maybe_unused]] AsOperandPrinter asOperand{funcOp};

    IRRewriter rewriter(funcOp->getContext());

    for (auto &[memref, memrefCtx] : analysis.memrefs) {
      if (!isDeviceMemorySpace(memrefCtx.type, MemorySpace::System)) {
        continue;
      }
      memref::AllocOp allocOp = memref.getDefiningOp<memref::AllocOp>();
      if (!allocOp) {
        continue;
      }
      TT_debugv(memrefCtx.remappedMemSpace.has_value(),
                "should have been placed: {}", asOperand(memref));

      const MemorySpace remappedMemorySpace = *memrefCtx.remappedMemSpace;
      const auto &solution = analysis.problem(remappedMemorySpace);
      const auto &memInfo = memSpaces[ordinal(remappedMemorySpace)];

      assignAddressAndAlignment(rewriter, allocOp,
                                solution.request(memrefCtx.reqIndex).offset,
                                memInfo);
    }

    return success();
  }

  // Sweep through all collected generic ops and make two simultaneous
  // modifications to their operand def chains:
  //  - modify root alloc ops and any view layout ops to be in the final
  //  memspace decided by the planner;
  //  - insert stream layout ops together with their stream buffer allocs.
  //
  // Additionally, it also seems easier to insert corresponding dealloc ops
  // here instead of a separate pass step.
  LogicalResult insertOperandStreams(func::FuncOp funcOp,
                                     const FuncAnalysisData &analysis) {
    IRRewriter rewriter(funcOp->getContext());

    const auto &L1solution = analysis.problem(MemorySpace::DeviceL1);
    const auto &L1memInfo = memSpaces[ordinal(MemorySpace::DeviceL1)];

    llvm::DenseSet<Operation *> visited;
    for (auto [genericOp, genericCtx] : analysis.generics) { // TODO const ref
      if (genericCtx.isDMAOnly) {
        continue;
      }

      // TT_ALLOC_DEBUG("-------------BEFORE:");
      // genericOp->dump();

      int32_t operandIndex = 0;
      for (const auto &[memref, operandCtx] : genericCtx.operands) {
        TT_debug(analysis.memrefs.contains(memref));
        const MemrefValueContext &memrefCtx = analysis.memrefs.at(memref);

        const MemorySpace remappedMemorySpace = *memrefCtx.remappedMemSpace;

        for (Operation *opOnChain : operandCtx.defChain) {
          if (!visited.insert(opOnChain).second) {
            // Assigning final memspace is idempotent, but no need to do this
            // repeatedly.
            continue;
          }
          llvm::TypeSwitch<Operation *, void>(opOnChain)
              .Case([&](memref::AllocOp op) {
                remap(rewriter, op, remappedMemorySpace);
                insertDealloc(rewriter, op, memrefCtx.live.last,
                              analysis.sequencing);
              })
              .Case([&](d2m::ViewLayoutOp op) {
                remap(rewriter, op, remappedMemorySpace);
              });
        }

        auto &operand = genericOp->getOpOperand(operandIndex);

        if (operandCtx.isOutput) {
          // // An output view may need to be added.

          // if (failed(
          //         insertOutputView(rewriter, operand, genericOp,
          //         operandCtx))) {
          //   return failure();
          // }
        } else {
          // The above may have changed memspace attributes of ops in the
          // operand's def chain; inserting a matching `stream_layout` next
          // will restore IR to a valid form.

          if (operandCtx.requiresStream ||
              (remappedMemorySpace == MemorySpace::DeviceDRAM)) {

            const PlannerSpace finalPlacement =
                asPlannerSpace(remappedMemorySpace);
            TT_debug(operandCtx.reqIndex[ordinal(finalPlacement)] >= 0);
            const Planner::Request &req = L1solution.request(
                operandCtx.reqIndex[ordinal(finalPlacement)]);

            // Note that this will take care of inserting the dealloc for the
            // stream buffer.
            if (failed(insertStream(rewriter, operand, genericOp, req,
                                    operandCtx, remappedMemorySpace, L1memInfo,
                                    analysis.sequencing))) {
              return failure();
            }
          }
        }

        ++operandIndex;
      }

      // TODO do this elsewhere/better (use mapped/cloned generic op)

      SmallVector<MemRefType> mroperands;
      for (auto [v, ctx] : genericCtx.operands) {
        mroperands.push_back(llvm::cast<MemRefType>(
            (genericOp->getOpOperand(ctx.operandIndex)).get().getType()));
      }
      const SmallVector<int64_t> blockFactors =
          calculateBlockFactors(genericOp.getGrid().getShape(),
                                genericOp.getIndexingMapsValue(), mroperands);
      TT_ALLOC_DEBUG("SETTING BF TO {}", asShape(blockFactors));
      genericOp->setAttr("block_factors",
                         rewriter.getI64ArrayAttr(blockFactors));

      // TT_ALLOC_DEBUG("-------------AFTER:");
      // genericOp->dump();
    }

    return success();
  }

  static void assignAddressAndAlignment(RewriterBase &rewriter,
                                        memref::AllocOp op,
                                        Planner::AllocSizeT offset,
                                        const MemorySpaceInfo &info) {

    const AllocSizeT address = info.baseAddress + offset;

    rewriter.startOpModification(op);
    {
      op.setAlignment(info.alignment);
      op->setAttr("address", rewriter.getI64IntegerAttr(address));
    };
    rewriter.finalizeOpModification(op);
  }

  static LogicalResult
  insertStream(RewriterBase &rewriter, OpOperand &operand, d2m::GenericOp op,
               const Planner::Request &req, const OperandContext &operandCtx,
               MemorySpace remappedMemspace, const MemorySpaceInfo &info,
               const SequenceMapping &sequencing) {
    const MemRefType bufferType = operandCtx.bufferType;
    TT_debug(bufferType != nullptr);

    OpBuilder::InsertionGuard guard(rewriter);
    {
      // Allocate a new stream buffer for `operand`, by design the insertion
      // must be just before the `op` to have the same live range.

      rewriter.setInsertionPoint(op);

      auto bufferAllocOp =
          rewriter.create<memref::AllocOp>(op.getLoc(), bufferType);

      assignAddressAndAlignment(rewriter, bufferAllocOp, req.offset, info);
      insertDealloc(rewriter, bufferAllocOp, req.last, sequencing);

      // A new stream_layout will read the original `operand`'s defchain
      // via storage `buffer` and its return value becomes the `op`'s
      // operand.

      MemRefType streamType = operandCtx.streamType;
      TT_debug(streamType != nullptr);
      // TODO better design for this
      streamType = remap(rewriter, streamType, remappedMemspace);

      auto streamOp = rewriter.create<d2m::StreamLayoutOp>(
          op.getLoc(), /* result */ streamType, /* input */ operand.get(),
          /* storage */ bufferAllocOp);

      rewriter.modifyOpInPlace(op,
                               [&]() { operand.assign(streamOp.getResult()); });

      // TODO(vroubtsov) use IRMapper to clone or applySignatureConversion()?

      MemRefType newArgType = MemRefType::get(
          mlir::cast<ttcore::DeviceLayoutInterface>(streamType.getLayout())
              .getShardShape(streamType),
          streamType.getElementType(), nullptr, streamType.getMemorySpace());

      for (Region &region : op->getRegions()) {
        TT_assert(region.hasOneBlock());
        Block &block = region.getBlocks().front();

        BlockArgument arg = block.getArgument(operandCtx.operandIndex);
        BlockArgument newArg = block.insertArgument(operandCtx.operandIndex,
                                                    newArgType, arg.getLoc());
        TT_ALLOC_DEBUG("  arg: {} - > {}", arg, newArg);
        rewriter.replaceAllUsesWith(arg, newArg);
        block.eraseArgument(operandCtx.operandIndex + 1);
      }

      // TT_ALLOC_DEBUG("INSERTED STREAM {}", streamOp);
    }
    return success();
  }

  static LogicalResult insertOutputView(RewriterBase &rewriter,
                                        OpOperand &operand, d2m::GenericOp op,
                                        const OperandContext &operandCtx) {
    const MemRefType bufferType = operandCtx.bufferType;
    TT_debug(bufferType != nullptr);

    OpBuilder::InsertionGuard guard(rewriter);
    {
      rewriter.setInsertionPoint(op);

      const MemRefType viewType = operandCtx.streamType;
      TT_debug(viewType != nullptr);

      auto viewOp = rewriter.create<d2m::ViewLayoutOp>(
          op.getLoc(), /* result */ viewType, /* input */ operand.get());
      // /* reinterpretLayout */ true);

      rewriter.modifyOpInPlace(op,
                               [&]() { operand.assign(viewOp.getResult()); });

      // TODO(vroubtsov) use IRMapper to clone or applySignatureConversion()?

      MemRefType newArgType = MemRefType::get(
          mlir::cast<ttcore::DeviceLayoutInterface>(viewType.getLayout())
              .getShardShape(viewType),
          viewType.getElementType(), nullptr, viewType.getMemorySpace());

      for (Region &region : op->getRegions()) {
        TT_assert(region.hasOneBlock());
        Block &block = region.getBlocks().front();

        BlockArgument arg = block.getArgument(operandCtx.operandIndex);
        BlockArgument newArg = block.insertArgument(operandCtx.operandIndex,
                                                    newArgType, arg.getLoc());
        TT_ALLOC_DEBUG("  arg: {} - > {}", arg, newArg);
        rewriter.replaceAllUsesWith(arg, newArg);
        block.eraseArgument(operandCtx.operandIndex + 1);
      }

      TT_ALLOC_DEBUG("INSERTED VIEW {}", viewOp);
    }

    return success();
  }

  // Populates `chain` with the sequence of operations that
  // start with the `operand`'s defining op and end with a `memref::AllocOp`
  // if one is found through a sequence, possibly empty, of
  // `view/stream_layout` inputs.
  //
  // Returns the last `Value` thus discovered, which is either a result of a
  // `memref::AllocOp` or a block argument.
  static Value getOperandDefChain(d2m::GenericOp genericOp, Value operand,
                                  DefUseChain &chain) {
    Operation *definingOp = operand.getDefiningOp();
    if (!definingOp) {
      TT_debug(mlir::isa<BlockArgument>(operand));
      return operand;
    }
    chain.emplace_back(definingOp);

    // Note: a canonicalizer pass would have collapse all `d2m.view_layout`
    // chains but we don't rely on that here.
    return llvm::TypeSwitch<Operation *, Value>(definingOp)
        .Case([&](memref::AllocOp op) { return operand; })
        .Case([&](d2m::ViewLayoutOp op) {
          return getOperandDefChain(genericOp, op.getInput(), chain);
        })
        .Case([&](d2m::StreamLayoutOp op) {
          return getOperandDefChain(genericOp, op.getInput(), chain);
        });
  }

  // Factor out defaults passed into DeviceAttr::getMemrefSizeBytes()
  // for operand memrefs.
  static int64_t getMemrefSizeBytes(MemRefType bufferType,
                                    ttcore::DeviceAttr device) {
    // A tighter size calculation is possible for memrefs that don't map to
    // CBs which we don't attempt here except to ignore `buffers` multipler.
    return device.getMemrefSizeBytes(bufferType, 0, false);
  }

  // Factor out defaults passed into DeviceAttr::getMemrefSizeBytes()
  // for stream buffer memrefs.
  static int64_t getStreamBufferSizeBytes(MemRefType bufferType,
                                          ttcore::DeviceAttr device) {
    TT_assertv(ttcore::getMemorySpace(bufferType) ==
                   ttcore::MemorySpace::DeviceL1,
               "stream buffers must be allocated in L1");
    // Stream buffers map to CBs and therefore are subject to CB size
    // alignment requirements (invoke with `pageSize` default of 0).
    return device.getMemrefSizeBytes(bufferType, 0, true);
  }

  static space_specific<int64_t>
  getAlignedAllocSizes(RewriterBase &rewriter, MemRefType type,
                       const MemorySpaces &memSpaces,
                       ttcore::DeviceAttr device) {
    space_specific<int64_t> sizes = {-1, -1};
    for (PlannerSpace placement = PlannerSpace::begin;
         placement < PlannerSpace::end; ++placement) {
      const MemorySpace memspace = asMemorySpace(placement);

      sizes[ordinal(placement)] = ttmlir::utils::alignUp(
          getMemrefSizeBytes(remap(rewriter, type, memspace), device),
          memSpaces[ordinal(memspace)].alignment);
    }

    return sizes;
  }

  static void insertDealloc(RewriterBase &rewriter, memref::AllocOp allocOp,
                            Planner::SequenceT position,
                            const SequenceMapping &sequencing) {
    Operation *lastOp = sequencing.positionMap[position];
    if (!llvm::isa<func::ReturnOp>(lastOp)) {
      OpBuilder::InsertionGuard guard(rewriter);
      {
        rewriter.setInsertionPointAfter(lastOp);
        rewriter.create<memref::DeallocOp>(lastOp->getLoc(),
                                           allocOp.getResult());
      }
    }
  }

  static MemRefType remap(RewriterBase &rewriter, MemRefType memrefType,
                          MemorySpace memspace) {
    return MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                           memrefType.getLayout(),
                           rewriter.getAttr<ttcore::MemorySpaceAttr>(memspace));
  }

  static void remap(RewriterBase &rewriter, memref::AllocOp op,
                    MemorySpace memspace) {
    auto memref = op.getMemref();
    MemRefType memrefType = memref.getType();
    MemRefType newType = remap(rewriter, memrefType, memspace);

    rewriter.modifyOpInPlace(op, [&]() { memref.setType(newType); });
  }

  static void remap(RewriterBase &rewriter, d2m::ViewLayoutOp op,
                    MemorySpace memspace) {
    auto memref = op->getResult(0);
    MemRefType memrefType = llvm::cast<MemRefType>(memref.getType());
    MemRefType newType = remap(rewriter, memrefType, memspace);

    rewriter.modifyOpInPlace(op, [&]() { memref.setType(newType); });
  }

  // Recursive helper for `analyzeAllocOps(func::FuncOp funcOp...)`.
  // Note: the overall traversal cost can be reduced by memoizing
  // final maxLast values and/or visiting Values in a reverse topological
  // sort order. This is not done at the moment.
  static SequenceT
  resolve(Operation *op,
          const llvm::DenseMap<Operation *, LivenessClosure> &graph) {

    auto opClosure = graph.find(op);
    TT_assert(opClosure != graph.end());
    SequenceT last = opClosure->second.live.last;

    for (Operation *user : op->getResult(0).getUsers()) {
      if (graph.contains(user)) {
        if (llvm::isa<d2m::ViewLayoutOp, d2m::StreamLayoutOp>(user)) {
          last = std::max(last, resolve(user, graph));
        }
      }
    }

    return last;
  }

  static std::tuple<MemRefType, MemRefType>
  getStreamBuffer(ArrayRef<int64_t> gridShape, ArrayRef<int64_t> shardShape,
                  AffineMap map, Type elementType, uint32_t buffers,
                  ttcore::MemorySpaceAttr memSpaceAttr) {
    TT_debug(gridShape.size() == shardShape.size());

    SmallVector<int64_t> bufferShape(gridShape.begin(), gridShape.end());
    bufferShape.append(shardShape.begin(), shardShape.end());

    const auto bufferLayout =
        ttcore::ShardLayoutAttr::get(shardShape, elementType, buffers);
    const auto streamLayout =
        ttcore::ViewLayoutAttr::get(map.getContext(), map);

    // TODO pass different memspaces

    return {
        MemRefType::get(bufferShape, elementType, bufferLayout, memSpaceAttr),
        MemRefType::get(bufferShape, elementType, streamLayout, memSpaceAttr)};
  }

  static MemorySpaces getMemorySpaces(ttcore::ChipDescAttr chipDesc,
                                      AllocSizeT l1CapacityOverride) {
    MemorySpaces info;
    {
      // Currently, we only need some slots in 'info'.

      const AllocSizeT l1Size =
          l1CapacityOverride > 0
              ? (chipDesc.getL1UnreservedBase() + l1CapacityOverride)
              : chipDesc.getL1Size();

      info[ordinal(MemorySpace::DeviceL1)] =
          MemorySpaceInfo(chipDesc.getL1UnreservedBase(), l1Size,
                          chipDesc.getNocL1AddressAlignBytes());

      info[ordinal(MemorySpace::DeviceDRAM)] = MemorySpaceInfo(
          chipDesc.getDramUnreservedBase(), chipDesc.getDramChannelSize(),
          chipDesc.getNocDRAMAddressAlignBytes());
    }
    return info;
  }

  [[maybe_unused]] friend std::string to_string(D2MAllocate obj) {
    std::stringstream s;
    s << "{\n";
    s << "  num-stream-buffers: " << obj.numStreamBuffers << "\n";
    s << "  allow-l1-output-spilling: " << obj.allowL1OutputSpilling << "\n";
    s << "  test-assume-l1-capacity: " << obj.testAssumeL1Capacity << "\n";
    s << "  test-buffer-size-policy: " << obj.testBufferSizePolicy << "\n";
    s << "}";

    return s.str();
  }
};
} // namespace

} // namespace mlir::tt::d2m
//===---------------------------------------------------------------------===//
