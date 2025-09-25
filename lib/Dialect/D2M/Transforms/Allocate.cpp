// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Analysis/Allocation/Planner.h"
#include "ttmlir/Dialect/D2M/Analysis/Allocation/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Utils.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <optional>

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
using allocation::is_operation_v;
using allocation::ordinal;

struct MemorySpaceInfo {

  MemorySpaceInfo() = default;
  MemorySpaceInfo(AllocSizeT baseAddress, AllocSizeT maxAddress,
                  AllocSizeT alignment)
      : baseAddress(baseAddress), maxAddress(maxAddress), alignment(alignment) {
    TT_assert(baseAddress % alignment == 0);
    TT_assert(baseAddress < maxAddress);
    TT_assert(baseAddress % alignment == 0);
    TT_assert(baseAddress < maxAddress);
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

struct MemrefValueContext {
  MemRefType type;
  // All generic op users of this alloc (immediate or through a chain of
  // view/steam layout ops).
  llvm::DenseSet<d2m::GenericOp> genericUsers;
  // "Raw" allocation request size in bytes (i.e. not aligned up for
  // any particular memspace).
  AllocSizeT size = -1;
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
  std::optional<MemorySpace> remappedMemSpace;

  // Fields used to link this Value to a `Planner` decision variable.

  int32_t varIndex = -1; // Needed to retrieve `Planner::Variable::placement`.
  int32_t reqIndex = -1; // Needed to retrieve `Planner::Request::offset`.
};

using DefUseChain = llvm::SmallVector<Operation *, 4>;

struct OperandContext {
  // This collects the set of ops defining an operand all the way to its
  // root `memref::AllocOp` or block arg.
  DefUseChain defChain;
  // Cached memref type for the defining Value, set lazily.
  MemRefType bufferType;
  // `true` is if this corresponds to a generic op output.
  bool isOutput = false;
  // `true` if this operand requires streaming regardless of
  // possible spilling (e.g. because of an intra-core data movement
  // pattern.)
  bool requiresStream = false;

  // Fields used to link this Value to a `Planner` decision variable.

  // Needed to retrieve `Planner::Request::offset` for this operand stream's
  // storage buffer, both `Scratch` and `Spill` alternatives.
  std::array<int32_t, ordinal(PlannerSpace::end)> reqIndex = {-1, -1};
};

// A map linking `OperandContext`s with their originating `Value`s (defined
// by `memref.alloc`s or passed as block args).
using OperandContextMap = llvm::SmallMapVector<mlir::Value, OperandContext, 4>;

struct GenericOpContext {
  // Context info for each of this generic ops list of operands, in declaration
  // order. (Note that the latter relies on `SmallMapVector` structure).
  OperandContextMap operands;
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

    memSpaces = [moduleOp]() {
      ttcore::SystemDescAttr systemDesc =
          ttcore::getCurrentScopeSystemDesc(moduleOp);
      ttcore::ChipDescAttr chipDesc = systemDesc.getChipDescs().front();
      return getMemorySpaces(chipDesc);
    }();

    TT_ALLOC_DEBUG("configured with {{num-stream-buffers: {}, "
                   "allow-output-spilling: {}}",
                   numStreamBuffers, allowOutputSpilling);

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

    if (failed(analyzeOperandStreams(funcOp, analysis))) {
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

    // All `memref.alloc`s will need to be placed into memspaces, therefore
    // collect all of them (regardless of whether they define operands of
    // generic ops or something else).

    // Start with SSA liveness for `func`.

    mlir::Liveness liveness(funcOp.getOperation());
    const mlir::LivenessBlockInfo *li = liveness.getLiveness(&funcBody);

    // (a) Build `Operation` <-> preorder position mappings for the
    // (unmodified) `funcOp` IR.
    //  (b) Collect a separate set of "ops of interest", which are
    //  `memref.alloc`s as well as certain ops that we imbue with semantics
    //   of extending liveness of their memref operands.

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
          memrefCtx.size = device.getMemrefSizeBytes(memrefCtx.type);
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

      operandCtx.isOutput = (operandIndex >= outputsStart);

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

  LogicalResult analyzeOperandStreams(func::FuncOp funcOp,
                                      FuncAnalysisData &analysis) {

    [[maybe_unused]] AsOperandPrinter asOperand{funcOp};

    ttcore::DeviceAttr device = ttcore::lookupDevice(funcOp);
    Block &funcBody = funcOp.getBody().front();

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

      // Decide which operands might/must have streams. Note that
      // the actual stream creation decision is only final after
      // we have a feasible planner solution.
      llvm::SmallVector<OperandContext> streams = getOperandContexts(genericOp);
      TT_debug(streams.size() == genericOp.getNumOperands());

      for (std::size_t operandIndex = 0;
           operandIndex < genericOp.getNumOperands(); ++operandIndex) {
        auto operand = genericOp->getOperand(operandIndex);
        OperandContext &streamCtx = streams[operandIndex];
        // For later IR mutation, it is convenient at this point to gather
        // all chains of ops defining operand inputs.
        Value memref =
            getOperandDefChain(genericOp, operand, streamCtx.defChain);

        const auto &[i, inserted] = analysis.memrefs.try_emplace(memref);
        MemrefValueContext &memrefCtx = i->second;

        memrefCtx.genericUsers.insert(genericOp);
        memrefCtx.usedForOutput |= streamCtx.isOutput;

        if (inserted) {
          // These were not discovered by the earlier `analyzeAllocOps()`, it
          // could only happen if the value is a block arg.
          TT_debugv(mlir::isa<BlockArgument>(memref),
                    "expected a block arg: {}", memref);
          memrefCtx.type = mlir::cast<MemRefType>(memref.getType());
          memrefCtx.size = device.getMemrefSizeBytes(memrefCtx.type);
        } else {
          // An existing `analysis.memrefs` entry means `operand` is ultimately
          // rooted in a `memref::AllocOp`.
          memref::AllocOp allocOp =
              mlir::cast<memref::AllocOp>(memref.getDefiningOp());

          OperationSet &allocOpGenericUsers = genericUseClosure[allocOp];
          allocOpGenericUsers.insert(genericOp.getOperation());
          allocOpGenericUsers.insert(streamCtx.defChain.begin(),
                                     streamCtx.defChain.end());
        }

        genericCtx.operands.try_emplace(memref, std::move(streamCtx));
      }
      TT_debug(genericCtx.operands.size() == genericOp.getNumOperands());
    });

    if (TT_DEBUG_ENABLED()) {
      for ([[maybe_unused]] auto &[value, valueCtx] : analysis.memrefs) {
        TT_ALLOC_TRACE("\t{}:\t{} generic user(s), [{}, {}], {} byte(s)",
                       asOperand(value), valueCtx.genericUsers.size(),
                       valueCtx.live.first, valueCtx.live.last, valueCtx.size);
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
    // (1) A memref defined by a `memref.alloc` backing a generic op operand and
    //     potentially associated with a stream and its buffer.
    // (2) A memref that backs a generic op operand but is not defined by an op
    //     inside `funcOp` (i.e. passed in as a block argument). We may insert a
    //     stream for this operand and will therefore need to allocate this
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
      TT_debugv((memrefCtx.type != nullptr && memrefCtx.size >= 0),
                "memref: {}", memref);

      TT_debug(memrefCtx.varIndex < 0);
      memrefCtx.varIndex =
          problem.def([&, &memref = memref,
                       &memrefCtx = memrefCtx](Planner::VariableBuilder &b) {
            // If `memref` is being defined inside `funcOp` and is initially
            // placed in L1, it will require scratch memory to hold its tensor
            // data.
            if (memref.getDefiningOp<memref::AllocOp>() &&
                memspace == MemorySpace::DeviceL1) {
              const auto &memInfo = memSpaces[ordinal(MemorySpace::DeviceL1)];

              memrefCtx.reqIndex = b.request(
                  PlannerSpace::Scratch,
                  ttmlir::utils::alignUp(memrefCtx.size, memInfo.alignment),
                  memrefCtx.live.first, memrefCtx.live.last);
            }

            // This decision variable must be bound to its incoming memspace in
            // any of these cases:
            //  - if it is placed in DRAM *explicitly*;
            //  - if it has non-generic op users or has zero generic op users;
            //  - if it the output of a generic op and the enabled pass options
            //  do not allow output spilling.
            const bool bound =
                (memspace == MemorySpace::DeviceDRAM) ||
                memrefCtx.genericUsers.empty() ||
                memrefCtx.hasNonGenericUsers ||
                (memrefCtx.usedForOutput && !allowOutputSpilling);
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

                // An operand stream is required under any of these conditions:
                // - streaming was earlier determined as required due to
                // inter-core data movement needs;
                // - the final memref placement is `Spill` (i.e. DRAM memspace).
                if (operandCtx.requiresStream ||
                    (placement == PlannerSpace::Spill)) {
                  if (!operandCtx.bufferType) {
                    // In principle, buffer shape/size could depend on whether
                    // the stream is out of L1 or DRAM... but not right now.
                    operandCtx.bufferType = selectStreamBuffer(
                        rewriter, memrefCtx.type, numStreamBuffers);
                  }
                  const AllocSizeT bufferSize =
                      device.getMemrefSizeBytes(operandCtx.bufferType);

                  // Because we will insert stream buffer allocs just before
                  // generic ops themselves, without any other interposing
                  // allocs, it is mathematically correct to see all such
                  // buffers' live ranges as single position coinciding with the
                  // generic op's logical time.
                  const SequenceT firstAndLast = analysis.sequencing[user];

                  TT_debug(operandCtx.reqIndex[ordinal(placement)] < 0);
                  operandCtx.reqIndex[ordinal(placement)] = b.request(
                      placement,
                      ttmlir::utils::alignUp(bufferSize, memInfo.alignment),
                      firstAndLast, firstAndLast);
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
    // formed by all variables with `Spill` placements (either decided so above
    // by the planner or because so bound).
    {
      const auto &L1solution = analysis.problem(MemorySpace::DeviceL1);
      auto &problem = analysis.problem(MemorySpace::DeviceDRAM);

      const auto &memInfo = memSpaces[ordinal(MemorySpace::DeviceDRAM)];

      // Note that we already have valid live ranges but must re-calculate
      // request sizes using DRAM alignment.

      for (auto &[memref, memrefCtx] : analysis.memrefs) {
        if (!isDeviceMemorySpace(memrefCtx.type, MemorySpace::System)) {
          continue;
        }

        if (L1solution.variable(memrefCtx.varIndex).placement ==
            PlannerSpace::Spill) {
          memrefCtx.varIndex = problem.def(
              [&, &allocCtx = memrefCtx](Planner::VariableBuilder &b) {
                allocCtx.reqIndex = b.request(
                    PlannerSpace::Scratch,
                    ttmlir::utils::alignUp(allocCtx.size, memInfo.alignment),
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

    for (const auto &[genericOp, genericCtx] : analysis.generics) {
      if (genericCtx.isDMAOnly) {
        continue;
      }
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

        // The above may have changed memspace attributes of ops in the
        // operand's def chain; inserting a matching `stream_layout` next
        // will restore IR to a valid form.

        if (operandCtx.requiresStream ||
            (remappedMemorySpace == MemorySpace::DeviceDRAM)) {

          const PlannerSpace finalPlacement =
              asPlannerSpace(remappedMemorySpace);
          TT_debug(operandCtx.reqIndex[ordinal(finalPlacement)] >= 0);
          const Planner::Request &req =
              L1solution.request(operandCtx.reqIndex[ordinal(finalPlacement)]);

          // Note that this will take care of inserting the dealloc for the
          // stream buffer.
          auto &operand = genericOp->getOpOperand(operandIndex);
          if (failed(insertStream(rewriter, operand, genericOp, req, operandCtx,
                                  L1memInfo, analysis.sequencing))) {
            return failure();
          }
        }

        ++operandIndex;
      }
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
               const MemorySpaceInfo &info, const SequenceMapping &sequencing) {
    auto operandMemrefType = mlir::cast<MemRefType>(operand.get().getType());

    OpBuilder::InsertionGuard guard(rewriter);
    {
      // By design, must insert just before the generic op.
      rewriter.setInsertionPoint(op);

      auto streamAttr = rewriter.getAttr<ttcore::ViewLayoutAttr>(
          rewriter.getMultiDimIdentityMap(operandMemrefType.getRank()));
      auto streamMemref = MemRefType::get(
          operandMemrefType.getShape(), operandMemrefType.getElementType(),
          streamAttr, operandMemrefType.getMemorySpace());

      TT_debug(operandCtx.bufferType != nullptr);
      auto bufferMemref = operandCtx.bufferType;
      auto buffer = rewriter.create<memref::AllocOp>(op.getLoc(), bufferMemref);

      assignAddressAndAlignment(rewriter, buffer, req.offset, info);
      insertDealloc(rewriter, buffer, req.last, sequencing);

      auto stream = rewriter.create<d2m::StreamLayoutOp>(
          op.getLoc(), streamMemref, operand.get(), buffer);

      rewriter.modifyOpInPlace(op,
                               [&]() { operand.assign(stream.getResult()); });
    }
    return success();
  }

  // Populates `chain` with the sequence of operations that
  // start with the `operand`'s defining op and end with a `memref::AllocOp` if
  // one is found through a sequence, possibly empty, of `view/stream_layout`
  // inputs.
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

  // TODO(vroubtsov) this is currently mocked up to use single tile buffers
  static llvm::SmallVector<int64_t, 4>
  selectStreamBufferShape(MemRefType operandType) {
    ttcore::DeviceLayoutInterface layout =
        mlir::cast<ttcore::DeviceLayoutInterface>(operandType.getLayout());
    ArrayRef<int64_t> gridShape = layout.getGridShape(operandType);

    llvm::SmallVector<int64_t, 4> shape(gridShape);
    shape.resize(operandType.getRank(), 1);
    return shape;
  }

  static MemRefType selectStreamBuffer(RewriterBase &rewriter,
                                       MemRefType operandType,
                                       uint32_t buffers) {
    llvm::SmallVector<int64_t> bufferShape =
        selectStreamBufferShape(operandType);
    auto bufferLayout = ttcore::ShardLayoutAttr::get(
        ArrayRef(bufferShape).take_back(bufferShape.size() / 2),
        operandType.getElementType(), /*buffers=*/buffers);
    return MemRefType::get(
        ArrayRef(bufferShape), operandType.getElementType(), bufferLayout,
        rewriter.getAttr<ttcore::MemorySpaceAttr>(MemorySpace::DeviceL1));
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

  static MemorySpaces getMemorySpaces(ttcore::ChipDescAttr chipDesc) {
    std::array<MemorySpaceInfo, MemorySpaceInfo::kMaxEnumValForMemorySpace>
        info;
    // Currently, we only need some slots in 'info'.
    {
      info[ordinal(MemorySpace::DeviceL1)] =
          MemorySpaceInfo(chipDesc.getL1UnreservedBase(), chipDesc.getL1Size(),
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
