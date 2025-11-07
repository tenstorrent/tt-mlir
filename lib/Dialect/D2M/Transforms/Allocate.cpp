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
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <numeric>
#include <optional>

//===---------------------------------------------------------------------===//
namespace mlir::tt::d2m {

#define GEN_PASS_DEF_D2MALLOCATE
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Helper definitions.
//===----------------------------------------------------------------------===//

inline bool isDeviceMemorySpace(MemRefType memref, ttcore::MemorySpace dflt) {
  return ttcore::isDeviceMemorySpace(ttcore::getMemorySpace(memref, dflt));
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
using allocation::concatToVector;
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

  [[maybe_unused]] friend std::string to_string(const MemorySpaceInfo &obj) {
    std::stringstream s;
    s << '[' << obj.baseAddress << ", " << obj.maxAddress << ')'
      << ", alignment " << obj.alignment;
    return s.str();
  }

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

using LivenessClosureGraph = llvm::DenseMap<Operation *, LivenessClosure>;

template <typename T>
using SpaceSpecific = std::array<T, ordinal(PlannerSpace::end)>;

struct MemrefValueContext {
  MemRefType type;
  // All generic op users of this alloc (immediate or through a chain of
  // view/steam layout ops).
  llvm::DenseSet<d2m::GenericOp> genericUsers;
  // Allocation request size in bytes (i.e. aligned for scratch/spill spaces,
  // respectively).
  SpaceSpecific<AllocSizeT> allocSize = {-1, -1};
  // Live range of this value, starting with the defining op itself
  // and extending to its last user.
  LiveRange live = {-1, -1};
  // `true` iff this value is ineligible for remapping (has non-generic
  // users, has a generic explicit datamovement user, etc).
  bool isMemspaceBound = false;
  // `true` iff this value acts as the output of at least one
  // generic op.
  bool usedForOutput = false;
  // `Planner`s spill outcome for this decision variable.
  // TODO(vroubtsov) replace with PlannerSpace var?
  std::optional<MemorySpace> remappedMemSpace;

  // Fields used to link this Value to a `Planner` decision variable.

  int32_t varIndex = -1; // Needed to retrieve `Planner::Variable::placement`.
  int32_t reqIndex = -1; // Needed to retrieve `Planner::Request::offset`.
};

using OperandDefChain = llvm::SmallVector<Operation *, 4>;

struct OperandContext {
  // Link to the operand in the incoming IR.
  OpOperand *operand = nullptr;

  int32_t operandIndex() const {
    TT_assert(operand != nullptr);
    return operand->getOperandNumber();
  }

  // This collects the set of ops defining an operand all the way to its
  // root `memref::AllocOp` or block arg.
  OperandDefChain defChain;
  // `true` is if this corresponds to a generic op output.
  bool isOutput = false;
  // `true` if this operand requires streaming regardless of
  // possible spilling (e.g. because of an intra-core data movement
  // pattern.)
  bool requiresStream = false;
  // To be able to plan possible pressure on L1, this precomputes
  // the type of the stream buffer this operand would have.
  MemRefType bufferType;

  // Fields used to link this Value to a `Planner` decision variable.

  // Needed to retrieve `Planner::Request::offset` for this operand stream's
  // storage buffer, both `Scratch` and `Spill` alternatives.
  SpaceSpecific<int32_t> reqIndex = {-1, -1};
};

// A map linking `OperandContext`s with their originating `Value`s (defined
// by `memref.alloc`s or passed as block args).
using OperandContextMap = llvm::SmallMapVector<mlir::Value, OperandContext, 4>;

struct GenericOpContext {
  // Context info for each of this generic ops list of operands, in declaration
  // order. (Note that the latter relies on `SmallMapVector` structure).
  OperandContextMap operands;
  // Pre-computed block factors for the modified op.
  SmallVector<int64_t> reblockedFactors;
  // Generic ops in "DMA-only" form currently operate in alias mode
  // and do not use operand streams.
  bool isDMAOnly = false;
  // Generic ops in "explicit datamovement" form have no static
  // iteration space (indexing maps, etc) information.
  bool isExplicitDatamovement = false;
};

struct SequenceMapping {
  // Within a func body scope, maps logical time (preorder) positions
  // to their `Operation`s.
  llvm::SmallVector<Operation *> positionMap;
  // Inverse of `positionMap`.
  DenseMap<Operation *, SequenceT> operationMap;

  SequenceT size() const { return positionMap.size(); }
  bool valid() const { return positionMap.size() == operationMap.size(); }

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
  ttcore::MemorySpaceAttr L1Attr = nullptr;
  ttcore::MemorySpaceAttr DRAMAttr = nullptr;

  [[maybe_unused]] friend std::string to_string(const D2MAllocate &obj) {
    std::stringstream s;
    s << "{\n";
    s << "\tnum-stream-buffers: " << obj.numStreamBuffers << "\n";
    s << "\tallow-l1-output-spilling: " << obj.allowL1OutputSpilling << "\n";
    s << "\ttest-assume-l1-capacity: " << obj.testAssumeL1Capacity << "\n";
    s << "\ttest-buffer-size-policy: " << obj.testBufferSizePolicy << "\n";
    s << "}";
    return s.str();
  }

  void runOnOperation() override {
    TT_ALLOC_DEBUG("configured with options: {}", to_string(*this));

    // Set some instance state:

    ModuleOp moduleOp = getOperation();

    memSpaces = [this, moduleOp]() {
      ttcore::SystemDescAttr systemDesc =
          ttcore::getCurrentScopeSystemDesc(moduleOp);
      ttcore::ChipDescAttr chipDesc = systemDesc.getChipDescs().front();
      return getMemorySpaces(chipDesc, testAssumeL1Capacity);
    }();
    TT_ALLOC_DEBUG("using memspaces:\n\tDRAM\t{}\n\tL1\t{}",
                   to_string(memSpaces[ordinal(MemorySpace::DeviceDRAM)]),
                   to_string(memSpaces[ordinal(MemorySpace::DeviceL1)]));

    using namespace ttcore;

    auto *ctx = &getContext();
    L1Attr = MemorySpaceAttr::get(ctx, MemorySpace::DeviceL1);
    DRAMAttr = MemorySpaceAttr::get(ctx, MemorySpace::DeviceDRAM);

    // Run a sequence of FuncOp-scoped steps:

    if (moduleOp
            ->walk([&](func::FuncOp funcOp) -> WalkResult {
              return runOnFunc(funcOp);
            })
            .wasInterrupted()) {
      return signalPassFailure();
    }
  }

  LogicalResult runOnFunc(func::FuncOp funcOp) {
    if (funcOp.isDeclaration()) {
      return success();
    }

    // This pass works with two sets of IR objects: memrefs (defined by
    // memref.alloc's) and generic ops (with operands that are either raw
    // memrefs or views/streams of those).
    //
    // The IR is allowed to contain "standalone" allocs that don't feed into
    // generic ops (TODO(vroubtsov) these won't become CBs, so can this
    // assumption be removed?). Conversely, generic ops are allowed to have
    // their operands rooted at memrefs that are not allocated withing `funcOp`,
    // e.g. passed in as func arguments. Therefore, the two
    // sets of memref values, (a) those allocated within `funcOp1 and (b) those
    // defining generic op operands are incomparable (neither is a subset of the
    // other). We try to track this carefully.

    FuncAnalysisData analysis;

    if (failed(analyzeLiveness(funcOp, analysis))) {
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

  /// Populate `analysis.sequencing` and `analysis.memrefs`:
  ///
  /// - Discover `memref.alloc`s already present in `funcOp` and collect them
  ///   into `analysis.memrefs`.
  /// - For each memref value:
  ///   - Calculate effective liveness by extending `mlir::Liveness`
  ///     ranges with uses by view/stream ops.
  ///   - Pre-compute memory footprint if placed within L1 or DRAM.
  ///
  LogicalResult analyzeLiveness(func::FuncOp funcOp,
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

    LivenessClosureGraph livenessJoinGraph;

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
    TT_debug(analysis.sequencing.valid());

    // Ops in `livenessJoinGraph` form a graph of Values and their users where
    // some Values have their original SSA liveness "extended" by stream op
    // users (d2m.view_layout, d2m.stream_layout).
    //
    // We calculate the "last use position" by computing for each value
    // the max over its users over a traversal through this graph.

    for (auto &[op, closure] : livenessJoinGraph) {
      // Initial last values are from the SSA liveness calculation.
      auto i = analysis.sequencing.operationMap.find(closure.lastOp);
      TT_debug(i != analysis.sequencing.operationMap.end());
      closure.live.last = i->second;
    }

    // TODO(vroubtsov) this is retained from v2, but now there is an opportunity
    // to merge live range and def/use chain calculations into a single step.
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
        memrefCtx.live = closure.live;
      }
    }

    // Convenient at this point to add func args to `analysis.memrefs` as well.
    for (auto arg : funcOp.getArguments()) {
      if (auto memrefType = mlir::dyn_cast<MemRefType>(arg.getType())) {
        MemrefValueContext &memrefCtx = analysis.memrefs[arg];
        TT_debug(memrefCtx.live.first < 0);

        memrefCtx.type = memrefType;
        // `memrefCtx.live` will not be used, leave it in "not set" state.
      }
    }

    for (auto &[_, memrefCtx] : analysis.memrefs) {
      if (isDeviceMemorySpace(memrefCtx.type, MemorySpace::System)) {
        memrefCtx.allocSize =
            getAlignedAllocSizes(rewriter, memrefCtx.type, memSpaces, device);
      }
    }

    TT_ALLOC_DEBUG("collected {} memref context(s)", analysis.memrefs.size());
    TT_debug(analysis.sequencing.valid());

    return success();
  }

  // Internal helper used by `analyzeGenericOps()` to create a new analysis
  // entry for `genericOp`.
  GenericOpContext &createGenericContext(FuncAnalysisData &analysis,
                                         d2m::GenericOp genericOp) {
    GenericOpContext &genericCtx = analysis.generics[genericOp];

    // Detect generic ops in "DMA-only" form: they must not
    // insert operand streams and therefore have no associated memory
    // allocation needs.
    genericCtx.isDMAOnly = genericOp.isDMAOnlyForm();

    // Detect generic ops in "explicit datamovement" form: they do not have
    // iteration space (indexing maps, etc) information and can't be analyzed
    // by this pass. However, a `GenericOpContext` entry must still be created
    // at this point.
    genericCtx.isExplicitDatamovement = genericOp.isExplicitDatamovementForm();

    return genericCtx;
  }

  // Internal helper used by `analyzeGenericOps()` to create analysis entries
  // for each operand of `genericOp`.
  void createOperandContexts(FuncAnalysisData &analysis,
                             d2m::GenericOp genericOp,
                             GenericOpContext &genericCtx) {
    [[maybe_unused]] AsOperandPrinter asOperand{genericOp->getParentOp()};
    [[maybe_unused]] ttcore::DeviceAttr device =
        ttcore::lookupDevice(genericOp);

    const bool haveIterationSpaceInfo = !genericCtx.isExplicitDatamovement;
    const bool useMinPolicy = (testBufferSizePolicy == "min");

    using OperationSet = llvm::SmallPtrSet<Operation *, 4>;

    // This is temp state to help set `MemrefValueContext::isMemspaceBound`.
    // This maps every `memref::AllocOp` to a union set of `Operation`s
    // that are seen on the use/def paths leading to their downstream
    // `d2m::GenericOp`s. Later, these sets will be intersected
    // with `memref::AllocOp->getUsers()` to detect if there are
    // any user not contained within the union sets.
    llvm::DenseMap<memref::AllocOp, OperationSet> genericUseClosure;

    const std::size_t outputsStart =
        genericOp.getOutputs().getBeginOperandIndex();

    SmallVector<AffineMap> indexingMaps;
    SmallVector<ttcore::IteratorType> iteratorTypes;

    SmallVector<int64_t> gridExtents;
    SmallVector<int64_t> shardExtents;
    SmallVector<int64_t> inputTileFactors;
    SmallVector<int64_t> outputTileFactors;

    if (haveIterationSpaceInfo) {
      // Do some analysis common to all `genericOp` operands.

      indexingMaps = genericOp.getIndexingMapsValue();
      iteratorTypes = genericOp.getIteratorTypesValue();

      const std::size_t rank = genericOp.getNumDims();

      const SmallVector<SmallVector<int64_t>> gridShapes =
          genericOp.getOperandGridShapes();
      const SmallVector<SmallVector<int64_t>> shardShapes =
          genericOp.getOperandShardShapes();

      std::tie(gridExtents, shardExtents) = getGridAndShardExtents(genericOp);
      std::tie(inputTileFactors, outputTileFactors) =
          getOperandTileShapes(genericOp);

      SmallVector<int64_t> blockFactors = genericOp.getBlockFactorsValue();

      TT_ALLOC_DEBUG(
          "analyzing {}: grid {}, block factors {}, full factors {}, grid "
          "extents {}, shard extents {}, shard block factors {}, "
          "participating "
          "dim mask {}",
          asOperand(genericOp), asShape(genericOp.getGrid().getShape()),
          asSeq(blockFactors), asSeq(genericOp.getFullBlockFactors()),
          asSeq(gridExtents), asSeq(shardExtents),
          asSeq(getShardBlockFactors(genericOp)),
          getParticipatingDimMask(genericOp));

      // WIP: for now, commit to just one buffer size "scaling policy"
      // (requested by `testBufferSizePolicy`) before examining individual
      // operands. This choice will be captured in new block factors
      // `genericCtx.reblockedFactors` and grid/shard dim extents in
      // `grid/shardExtents`.
      {
        SmallVector<int64_t> rescaling(rank, 1);
        if (useMinPolicy) {
          const llvm::BitVector participationMask =
              getParticipatingDimMask(genericOp);
          const SmallVector<int64_t> shardFactors =
              getShardBlockFactors(genericOp);
          for (auto d = participationMask.find_first_unset(); d >= 0;
               d = participationMask.find_next_unset(d)) {
            rescaling[d] = shardFactors[d];
          }
        }

        for (std::size_t d = 0; d < rank; ++d) {
          gridExtents[d] *= rescaling[d];
          shardExtents[d] /= rescaling[d];

          blockFactors[d] *= rescaling[d];
        }
        TT_ALLOC_DEBUG("rescaling {}, new block factors {}", asSeq(rescaling),
                       asSeq(blockFactors));
      }
      genericCtx.reblockedFactors = blockFactors;
      TT_debug((gridExtents.size() == rank and shardExtents.size() == rank));
    }

    // Do some operand-specific analysis.

    for (auto [operandIndex, operand] :
         llvm::enumerate(genericOp->getOpOperands())) {
      OperandContext operandCtx;

      operandCtx.operand = &operand;
      operandCtx.isOutput = (operandIndex >= outputsStart);

      // Find `operand`s "root" memref and the op chain that links to it
      // (this sets `operandCtx.defChain`). Populate `operandCtx.defChain`
      // and update this memref's slot in `analysis.memrefs`.

      Value root =
          getOperandDefChain(genericOp, operand.get(), operandCtx.defChain);

      TT_debug(analysis.memrefs.contains(root));
      MemrefValueContext &memrefCtx = analysis.memrefs.find(root)->second;

      memrefCtx.genericUsers.insert(genericOp);
      memrefCtx.isMemspaceBound |= genericCtx.isExplicitDatamovement;
      memrefCtx.usedForOutput |= operandCtx.isOutput;

      if (memref::AllocOp allocOp = root.getDefiningOp<memref::AllocOp>()) {
        // Update the union set of all `allocOp` generic users.
        OperationSet &allocOpGenericUsers = genericUseClosure[allocOp];
        allocOpGenericUsers.insert(genericOp.getOperation());
        allocOpGenericUsers.insert(operandCtx.defChain.begin(),
                                   operandCtx.defChain.end());
      }

      if (haveIterationSpaceInfo) {
        // Decide which operands might/must have streams. Note that
        // the actual stream creation decision is only final after
        // we have a feasible planner solution.

        const AffineMap &indexingMap = indexingMaps[operandIndex];

        if (operandCtx.isOutput) {
          // L1 outputs are currently allocated in L1 so won't use streams
          // unless allowed to do so in `allowL1OutputSpilling` mode. DRAM
          // outputs always need to be streamed.
        } else {
          // A core participating in a reduction dim necessarily requires
          // non-local data movement unless it is the only core involved
          // in that dim. Similar logic applies to a broadcast dim.
          const auto bcastDims = indexingMap.getBroadcastDims();
          const llvm::SmallSet<unsigned, 4> bcastDimIndex(bcastDims.begin(),
                                                          bcastDims.end());
          operandCtx.requiresStream =
              llvm::any_of(llvm::seq(indexingMap.getNumResults()),
                           [&](unsigned resultIndex) {
                             if (bcastDimIndex.contains(resultIndex)) {
                               return true;
                             }
                             const auto dimPosition =
                                 indexingMap.getDimPosition(resultIndex);
                             return (iteratorTypes[dimPosition] ==
                                     ttcore::IteratorType::Reduction);
                           });
        }

        // To know the exact L1 memory pressure, we need to know the type/size
        // of this operand's stream if one were to be inserted.

        const AffineMap canonicalMap = canonicalizeBroadcasts(indexingMap);

        const SmallVector<int64_t> gridShapeRescaled =
            canonicalMap.compose(gridExtents);

        SmallVector<int64_t> shardShapeRescaled =
            canonicalMap.compose(shardExtents);
        // TODO(vroubtsov) not sure if there's a better option right now for
        // adjusting to input/output tile shape changes:
        if (operandCtx.isOutput) {
          for (std::size_t t = 0; t < 2; ++t) {
            const std::size_t d = shardShapeRescaled.size() - 2 + t;
            shardShapeRescaled[d] =
                (shardShapeRescaled[d] * inputTileFactors[t]) /
                outputTileFactors[t];
          }
        }

        const auto operandType =
            mlir::cast<MemRefType>(operand.get().getType());

        operandCtx.bufferType = getStreamBufferType(
            gridShapeRescaled, shardShapeRescaled, operandType.getElementType(),
            L1Attr, numStreamBuffers);
        TT_ALLOC_TRACE("[operand #{}] selected stream buffer ({} byte(s)): {}",
                       operandIndex,
                       getStreamBufferSizeBytes(operandCtx.bufferType, device),
                       operandCtx.bufferType);
        TT_debug(getStreamBufferSizeBytes(operandCtx.bufferType, device) > 0);
      }

      // Finally, insert `operandCtx` into `genericCtx`.

      // TODO(vroubtsov) is it possible for incoming IR to link to the same
      // memref Value in different block arg slots? Guard against that
      // explicitly for now.

      const auto [_, inserted] =
          genericCtx.operands.try_emplace(root, std::move(operandCtx));
      TT_assertv(inserted,
                 "memref used by more than one generic operand slot?");
    }
    TT_debug(genericCtx.operands.size() == genericOp.getNumOperands());

    // `genericUseClosure` is complete, use it to update
    // `MemrefValueContext::isMemspaceBound`:

    for (auto &[allocOp, users] : genericUseClosure) {
      for (Operation *user : allocOp->getUsers()) {
        if (!llvm::isa<func::ReturnOp>(user) && !users.contains(user)) {
          analysis.memrefs[allocOp].isMemspaceBound |= true;
        }
      }
    }
  }

  /// Populate `analysis.generics`:
  ///
  /// - Collect `d2m.generic`s present in `funcOp` into `analysis.generics`.
  /// - For each `d2m.generic`, build a block of operand context structs
  ///   parallel to the op's IR operands:
  ///   - Each operand's context links to the def/use chain rooted at its
  ///     defining memref value (used for overwriting the memspaces of all
  ///     contained ops should this memref be chosen for spilling).
  ///   - Each operand's context memoizes several decisions should a
  ///     `d2m.stream_layout` be inserted for this operand at a later step:
  ///     - Types of the stream and its buffer, which in turn determine the
  ///       allocation size for the buffer alloc and ensuing generic op
  ///       reblocking.
  ///
  /// Note that each decision to spill a memref alloc is binary while the stream
  /// buffer sizing decision is in theory k-ary. As work-in-progress, the latter
  /// decision is currently clamped to binary based on `testBufferSizePolicy`.
  ///
  LogicalResult analyzeGenericOps(func::FuncOp funcOp,
                                  FuncAnalysisData &analysis) {

    [[maybe_unused]] AsOperandPrinter asOperand{funcOp};
    [[maybe_unused]] ttcore::DeviceAttr device = ttcore::lookupDevice(funcOp);

    Block &funcBody = funcOp.getBody().front();
    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);

    [[maybe_unused]] int32_t genericsInDMAOnlyForm = 0;
    [[maybe_unused]] int32_t genericsInExplicitDatamovementForm = 0;

    funcBody.walk([&](d2m::GenericOp genericOp) {
      GenericOpContext &genericCtx = createGenericContext(analysis, genericOp);

      genericsInDMAOnlyForm += genericCtx.isDMAOnly;
      genericsInExplicitDatamovementForm += genericCtx.isExplicitDatamovement;

      createOperandContexts(analysis, genericOp, genericCtx);
    });

    if (TT_DEBUG_ENABLED()) {
      for ([[maybe_unused]] auto &[value, valueCtx] : analysis.memrefs) {
        TT_ALLOC_TRACE("\t{}:\t[{}, "
                       "{}], {} byte(s), {} generic user(s), is memspace "
                       "bound: {}, used for output: {}",
                       asOperand(value), valueCtx.live.first,
                       valueCtx.live.last, asSeq(valueCtx.allocSize),
                       valueCtx.genericUsers.size(), valueCtx.isMemspaceBound,
                       valueCtx.usedForOutput);
      };
    }

    TT_ALLOC_DEBUG("collected {} generic op context(s) ({} DMA-only, {} "
                   "explicit datamovement)",
                   analysis.generics.size(), genericsInDMAOnlyForm,
                   genericsInExplicitDatamovementForm);

    return success();
  }

  /// Each `analysis.memrefs` entry defines an allocation planner decision
  /// variable. These can be of different origins:
  ///
  /// (1) A memref defined by a `memref.alloc` backing a generic op operand
  ///     and potentially associated with a stream and its buffer.
  /// (2) A memref that backs a generic op operand but is not defined by an
  ///     op inside `funcOp` (i.e. passed in as a block argument). We may
  ///     insert a stream for this operand and will therefore need to
  ///     allocate this stream's buffer.
  /// (3) A memref defined by a "standalone" `memref.alloc` that needs no
  ///     generic op streaming but will still need a valid L1/DRAM memory
  ///     address assigned.
  ///
  LogicalResult prepareMemoryPlanner(func::FuncOp funcOp,
                                     FuncAnalysisData &analysis) {
    [[maybe_unused]] AsOperandPrinter asOperand{funcOp};

    ttcore::DeviceAttr device = ttcore::lookupDevice(funcOp);
    IRRewriter rewriter(funcOp->getContext());

    Planner::Problem &problem = analysis.problem(MemorySpace::DeviceL1);

    for (auto &[memref, memrefCtx] : analysis.memrefs) {
      const MemorySpace memspace =
          ttcore::getMemorySpace(memrefCtx.type, MemorySpace::System);
      if (!ttcore::isDeviceMemorySpace(memspace)) {
        continue;
      }
      // Invariant established earlier: all `analysis.memrefs` in DRAM/L1
      // have 'type' and 'size' set.
      TT_debug((memrefCtx.type != nullptr &&
                llvm::all_of(memrefCtx.allocSize,
                             [](auto size) { return size >= 0; })));

      TT_debug(memrefCtx.varIndex < 0);
      memrefCtx.varIndex = problem.def([&, &memref = memref,
                                        &memrefCtx = memrefCtx](
                                           Planner::VariableBuilder &b) {
        // If `memref` is being defined inside `funcOp` and is initially
        // placed in L1, it will require scratch memory to hold its tensor
        // data.
        if (memref.getDefiningOp<memref::AllocOp>() &&
            memspace == MemorySpace::DeviceL1) {
          memrefCtx.reqIndex =
              b.request(PlannerSpace::Scratch,
                        memrefCtx.allocSize[ordinal(asPlannerSpace(memspace))],
                        memrefCtx.live.first, memrefCtx.live.last);
        }

        // This decision variable must be bound to its incoming memspace
        // in any of these cases:
        //  - if it is placed in DRAM *explicitly*;
        //  - if the incoming IR indicates that this alloc should be pinned to
        //  it current memspace;
        //  - if it is the output of a generic op and the enabled pass options
        //  do not allow output spilling;
        //  - (edge case) if it has zero generic op users;
        //
        const bool bound =
            (memspace == MemorySpace::DeviceDRAM) ||
            memrefCtx.isMemspaceBound ||
            (memrefCtx.usedForOutput && !allowL1OutputSpilling) ||
            memrefCtx.genericUsers.empty();
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
              // Generics in "DMA only" form do not use streams.
              continue;
            }
            if (genericCtx.isExplicitDatamovement) {
              // Generics in "explicit datamovement" form manage their own
              // streams which should already be present in the incoming IR.
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
              operandCtx.reqIndex[ordinal(placement)] =
                  b.request(placement, bufferSize, firstAndLast, firstAndLast);
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
                allocCtx.reqIndex =
                    b.request(PlannerSpace::Scratch,
                              allocCtx.allocSize[ordinal(placement)],
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

  /// Sweep through all collected generic ops and make several in-place
  /// modifications:
  ///  - modify root alloc ops and any view layout ops to be in the final
  ///    memspace decided by the planner;
  ///  - insert stream layout ops together with their stream buffer allocs.
  ///  - fix block factors
  ///
  /// Additionally, it also seems easier to insert corresponding dealloc ops
  /// here instead of a separate pass step.
  LogicalResult insertOperandStreams(func::FuncOp funcOp,
                                     const FuncAnalysisData &analysis) {
    IRRewriter rewriter(funcOp->getContext());

    const auto &L1solution = analysis.problem(MemorySpace::DeviceL1);
    const auto &L1memInfo = memSpaces[ordinal(MemorySpace::DeviceL1)];

    llvm::DenseSet<Operation *> visited;
    for (const auto &[genericOp, genericCtx] : analysis.generics) {
      if (genericCtx.isDMAOnly) {
        // Generics in "DMA only" form do not use streams.
        continue;
      }
      if (genericCtx.isExplicitDatamovement) {
        // Generics in "explicit datamovement" form manage their own
        // streams which should already be present in the incoming IR.
        continue;
      }

      for (const auto &[memref, operandCtx] : genericCtx.operands) {
        TT_debug(analysis.memrefs.contains(memref));
        const MemrefValueContext &memrefCtx = analysis.memrefs.at(memref);

        const MemorySpace remappedMemSpace = *memrefCtx.remappedMemSpace;

        for (Operation *opOnChain : operandCtx.defChain) {
          if (!visited.insert(opOnChain).second) {
            // Assigning final memspace is idempotent, but no need to do this
            // repeatedly.
            continue;
          }
          llvm::TypeSwitch<Operation *, void>(opOnChain)
              .Case([&](memref::AllocOp op) {
                remap(rewriter, op, remappedMemSpace);
                insertDealloc(rewriter, op, memrefCtx.live.last,
                              analysis.sequencing);
              })
              .Case([&](d2m::ViewLayoutOp op) {
                remap(rewriter, op, remappedMemSpace);
              });
        }

        auto &operand = *operandCtx.operand;

        if (operandCtx.requiresStream ||
            (remappedMemSpace == MemorySpace::DeviceDRAM)) {
          if (operandCtx.isOutput) {
            // We get here if output streaming was enabled and the output
            // ended up spilled to DRAM. But there is no full support for
            // output streams/views as of yet, so for now just bail.
            TT_ALLOC_ERROR("inserting output streams not implemented yet");
            return failure();
          } /* else */

          // The above modifications may have changed memspace attributes
          // of ops in the operand's def chain; inserting a matching
          // `stream_layout` next will restore IR to a valid form.

          const PlannerSpace finalPlacement = asPlannerSpace(remappedMemSpace);
          TT_debug(operandCtx.reqIndex[ordinal(finalPlacement)] >= 0);
          const Planner::Request &req =
              L1solution.request(operandCtx.reqIndex[ordinal(finalPlacement)]);

          if (failed(insertStream(rewriter, operand, genericOp, req, operandCtx,
                                  (remappedMemSpace == MemorySpace::DeviceDRAM
                                       ? DRAMAttr
                                       : L1Attr),
                                  L1memInfo, analysis.sequencing))) {
            return failure();
          }
        }
      }

      // Fix up CB ops in the body:

      for (Region &region : genericOp->getRegions()) {
        TT_assert(region.hasOneBlock());
        Block &block = region.getBlocks().front();

        block.walk([&](Operation *blockOp) {
          llvm::TypeSwitch<Operation *, void>(blockOp)
              .Case([&](d2m::ReserveOp op) {
                op->getResult(0).setType(
                    op.getCbType().getUnderlyingAs<MemRefType>());
              })
              .Case([&](d2m::WaitOp op) {
                op->getResult(0).setType(
                    op.getCbType().getUnderlyingAs<MemRefType>());
              });
        });
      }

      // Fix up block factors:

      const auto blockFactorsAttrName =
          const_cast<GenericOp &>(genericOp).getBlockFactorsAttrName();
      genericOp->setAttr(blockFactorsAttrName,
                         rewriter.getI64ArrayAttr(genericCtx.reblockedFactors));
    }

    return success();
  }

  LogicalResult insertStream(RewriterBase &rewriter, OpOperand &operand,
                             d2m::GenericOp op, const Planner::Request &req,
                             const OperandContext &operandCtx,
                             ttcore::MemorySpaceAttr remappedMemspace,
                             const MemorySpaceInfo &info,
                             const SequenceMapping &sequencing) {
    const MemRefType bufferType = operandCtx.bufferType;
    TT_debug(bufferType != nullptr);

    OpBuilder::InsertionGuard guard(rewriter);

    // Allocate a new stream buffer for `operand`, by design the insertion
    // must be just before the `op` to have the same live range.

    rewriter.setInsertionPoint(op);

    auto bufferAllocOp =
        rewriter.create<memref::AllocOp>(op.getLoc(), bufferType);

    assignAddressAndAlignment(rewriter, bufferAllocOp, req.offset, info);
    insertDealloc(rewriter, bufferAllocOp, req.last, sequencing);

    const auto oldOperandType = mlir::cast<MemRefType>(operand.get().getType());
    const AffineMap reblockingMap = utils::calculateReblockMap(
        oldOperandType.getShape(), bufferType.getShape(),
        rewriter.getContext());

    const MemRefType streamType =
        getStreamType(bufferType.getShape(), reblockingMap,
                      oldOperandType.getElementType(), remappedMemspace);

    auto streamOp = rewriter.create<d2m::StreamLayoutOp>(
        op.getLoc(), /* result */ streamType, /* input */ operand.get(),
        /* storage */ bufferAllocOp);

    rewriter.startOpModification(op);
    {
      operand.assign(streamOp.getResult());

      const MemRefType newArgMemRefType = MemRefType::get(
          mlir::cast<ttcore::DeviceLayoutInterface>(streamType.getLayout())
              .getShardShape(streamType),
          streamType.getElementType(), nullptr, L1Attr);
      const CBType newCBArgType = d2m::CBType::get(newArgMemRefType);

      for (Region &region : op->getRegions()) {
        TT_assert(region.hasOneBlock());
        Block &block = region.getBlocks().front();

        const auto operandIndex = operandCtx.operand->getOperandNumber();
        BlockArgument arg = block.getArgument(operandIndex);
        BlockArgument newArg =
            block.insertArgument(operandIndex, newCBArgType, arg.getLoc());
        rewriter.replaceAllUsesWith(arg, newArg);
        block.eraseArgument(operandIndex + 1);
      }
    }
    rewriter.finalizeOpModification(op);

    return success();
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

  /// @return `map` with all broadcast result expressions replaced with const-1
  /// expression
  static AffineMap canonicalizeBroadcasts(AffineMap map) {
    auto *ctx = map.getContext();

    // This could almost be a simple AffineMap::replace() but need to make sure
    // only complete `0`-result expressions are replaced, not other possible
    // zero const terms within result expression trees, however unlikely that
    // seems.

    const auto replacement = mlir::getAffineConstantExpr(1, ctx);
    SmallVector<AffineExpr> exprs;

    for (auto expr : map.getResults()) {
      if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
        if (constExpr.getValue() == 0) {
          exprs.push_back(replacement);
          continue;
        }
      }
      exprs.push_back(expr);
    }

    return AffineMap::get(map.getNumDims(), map.getNumSymbols(), exprs, ctx);
  }

  static MemRefType getStreamBufferType(ArrayRef<int64_t> gridShape,
                                        ArrayRef<int64_t> shardShape,
                                        Type elementType,
                                        ttcore::MemorySpaceAttr memSpaceAttr,
                                        uint32_t buffers) {
    TT_debug(gridShape.size() == shardShape.size());

    const SmallVector<int64_t> fullShape =
        concatToVector<int64_t>(gridShape, shardShape);
    const auto bufferLayout =
        ttcore::ShardLayoutAttr::get(shardShape, elementType, buffers);

    return MemRefType::get(fullShape, elementType, bufferLayout, memSpaceAttr);
  }

  static MemRefType getStreamType(ArrayRef<int64_t> fullShape, AffineMap map,
                                  Type elementType,
                                  ttcore::MemorySpaceAttr memSpaceAttr) {
    const auto streamLayout =
        ttcore::ViewLayoutAttr::get(map.getContext(), map);

    return MemRefType::get(fullShape, elementType, streamLayout, memSpaceAttr);
  }

  static std::tuple</* input */ SmallVector<int64_t>,
                    /* output */ SmallVector<int64_t>>
  getOperandTileShapes(d2m::GenericOp genericOp) {
    const Type inputElementType =
        mlir::cast<MemRefType>(genericOp.getOperands().front().getType())
            .getElementType();
    for (std::size_t operandIndex = 1;
         operandIndex < genericOp.getOutputs().getBeginOperandIndex();
         ++operandIndex) {
      TT_assertv(inputElementType ==
                     mlir::cast<MemRefType>(
                         genericOp->getOperand(operandIndex).getType())
                         .getElementType(),
                 "expected no change in tile shapes across generic op inputs");
    }

    const Type outputElementType =
        mlir::cast<MemRefType>(genericOp.getOperands().back().getType())
            .getElementType();

    return {getEffectiveTileShape(inputElementType),
            getEffectiveTileShape(outputElementType)};
  }

  /// @return tile shape of `elementType` or `{1, 1}` if it isn't a TileType
  static SmallVector<int64_t> getEffectiveTileShape(Type elementType) {
    if (auto tileType = mlir::dyn_cast<ttcore::TileType>(elementType)) {
      TT_debug(tileType.getRank() == 2);
      return SmallVector<int64_t>(tileType.getShape());
    }
    return {1, 1};
  }

  static std::tuple</* grid */ SmallVector<int64_t>,
                    /* shard */ SmallVector<int64_t>>
  getGridAndShardExtents(d2m::GenericOp genericOp) {
    auto flatInverseMap = utils::concatInversePermutationMap(
        genericOp.getIndexingMapsValue(), /*reverse=*/false);

    return {flatInverseMap.compose(
                concatToVector(genericOp.getOperandGridShapes())),
            flatInverseMap.compose(
                concatToVector(genericOp.getOperandShardShapes()))};
  }

  /// Return a bitmask that indicates which of the dims are "participating"
  /// (defined as dims that are used by any of the `genericOp`'s output
  /// expressions.
  static llvm::BitVector getParticipatingDimMask(d2m::GenericOp genericOp) {
    TT_debug(genericOp.getOutputs().size() == 1u);
    AffineMap outputMap = genericOp.getIndexingMapsValue().back();

    const std::size_t rank = outputMap.getNumDims();

    llvm::BitVector mask(rank, false);
    for (std::size_t d = 0; d < rank; ++d) {
      mask[d] = outputMap.isFunctionOfDim(d);
    }
    return mask;
  }

  /// Calculate full "shard-only" blocking factors (defined as full blocking
  /// factors divided by blocking factors)
  static SmallVector<int64_t> getShardBlockFactors(d2m::GenericOp genericOp) {
    SmallVector<int64_t> r = genericOp.getFullBlockFactors();
    SmallVector<int64_t> blockFactors = genericOp.getBlockFactorsValue();
    TT_debug(r.size() == blockFactors.size());
    for (std::size_t d = 0; d < r.size(); ++d) {
      TT_debugv(blockFactors[d] > 0, "unexpected block factor {} for dim {}",
                blockFactors[d], d);
      r[d] /= blockFactors[d];
    }

    return r;
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

  // Populates `chain` with the sequence of operations that
  // start with the `operand`'s defining op and end with a `memref::AllocOp`
  // if one is found through a sequence, possibly empty, of
  // `view/stream_layout` inputs.
  //
  // Returns the last `Value` thus discovered, which is either a result of a
  // `memref::AllocOp` or a block argument.
  static Value getOperandDefChain(d2m::GenericOp genericOp, Value start,
                                  llvm::SmallVector<Operation *, 4> &chain) {
    Operation *definingOp = start.getDefiningOp();
    if (!definingOp) {
      TT_debug(mlir::isa<BlockArgument>(start));
      return start;
    }
    chain.emplace_back(definingOp);

    // Note: a canonicalizer pass would have collapsed all `d2m.view_layout`
    // chains but we don't rely on that here.
    return llvm::TypeSwitch<Operation *, Value>(definingOp)
        .Case([&](memref::AllocOp op) { return start; })
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

  /// @return aligned allocation sizes for a `type` buffer for different
  /// placements
  static SpaceSpecific<int64_t>
  getAlignedAllocSizes(RewriterBase &rewriter, MemRefType type,
                       const MemorySpaces &memSpaces,
                       ttcore::DeviceAttr device) {
    SpaceSpecific<int64_t> sizes = {-1, -1};
    for (PlannerSpace placement = PlannerSpace::begin;
         placement < PlannerSpace::end; ++placement) {
      const MemorySpace memspace = asMemorySpace(placement);

      sizes[ordinal(placement)] = ttmlir::utils::alignUp(
          getMemrefSizeBytes(remap(rewriter, type, memspace), device),
          memSpaces[ordinal(memspace)].alignment);
    }

    return sizes;
  }

  /// @return 'memrefType' with given memory space override
  static MemRefType remap(RewriterBase &rewriter, MemRefType memrefType,
                          MemorySpace memspace) {
    return MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                           memrefType.getLayout(),
                           rewriter.getAttr<ttcore::MemorySpaceAttr>(memspace));
  }

  /// @return 'op' with given memory space override
  static void remap(RewriterBase &rewriter, memref::AllocOp op,
                    MemorySpace memspace) {
    auto memref = op.getMemref();
    MemRefType memrefType = memref.getType();
    MemRefType newType = remap(rewriter, memrefType, memspace);

    rewriter.modifyOpInPlace(op, [&]() { memref.setType(newType); });
  }

  /// @return 'op' with given memory space override
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
  static SequenceT resolve(Operation *op, const LivenessClosureGraph &graph) {

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
};
} // namespace

} // namespace mlir::tt::d2m
//===---------------------------------------------------------------------===//
