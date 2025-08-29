// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/Analysis/Allocation/Planner.h"
#include "ttmlir/Dialect/TTIR/Analysis/Allocation/Utils.h"
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

// ----------------------------------------------------------------------------
namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRALLOCATE
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

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

using namespace allocation;

using AllocSizeT = Planner::AllocSizeT;
using SequenceT = Planner::SequenceT;
using IndexT = Planner::IndexT;

using LiveRange = Planner::LiveRange;

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

inline Planner::Space asPlannerSpace(ttcore::MemorySpace memspace) {
  switch (memspace) {
  case ttcore::MemorySpace::DeviceDRAM: {
    return Planner::Space::Spill;
  }
  case ttcore::MemorySpace::DeviceL1: {
    return Planner::Space::Scratch;
  }
  default: {
    llvm_unreachable("expected device memory space input");
  }
  }
}

inline ttcore::MemorySpace asMemorySpace(Planner::Space space) {
  switch (space) {
  case Planner::Space::Scratch: {
    return ttcore::MemorySpace::DeviceL1;
  }
  case Planner::Space::Spill: {
    return ttcore::MemorySpace::DeviceDRAM;
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
  // All generic op users of this alloc (immediate or through view/steam layout
  // ops).
  llvm::DenseSet<ttir::GenericOp> genericUsers;
  // "Raw" allocation request size in bytes (unaligned for any particular
  // memspace).
  AllocSizeT size = -1; // TODO this is available from the op itself
  // Live range of this alloc, starting with the op itself and
  // extending to its latest user.
  LiveRange live = {-1, -1};

  bool hasNonGenericUsers = false;
  bool isSomeonesOutput = false; // TODO rename

  int32_t varIndex = -1; // needed to retrieve Variable::placement
  int32_t reqIndex = -1; // needed to retrieve offset of Variable's alloc
  std::optional<ttcore::MemorySpace> remappedMemSpace;
};

using DefUseChain = llvm::SmallVector<Operation *, 4>;

struct OperandStream {
  DefUseChain defUseChain;
  MemRefType bufferType; // set lazily
  bool isOutput = false;
  bool requiresStream =
      false; // be it L1 or DRAM TODO rename smth like requiresInterCoreStream ?

  std::array<int32_t, ordinal(Planner::Space::end)> reqIndex = {-1, -1};
};

// root memref -> OperandStream
using OperandStreamMap = llvm::SmallMapVector<mlir::Value, OperandStream, 4>;

struct GenericOpContext {
  // Root definitions of the use-def chain for each of this generic ops'
  // list of operands.
  OperandStreamMap operands;
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

struct FuncAnalysisData {
  SequenceMapping sequencing;
  llvm::DenseMap<mlir::Value, MemrefValueContext> memrefs;
  llvm::DenseMap<ttir::GenericOp, GenericOpContext> generics;
  std::array<Planner::Problem, ordinal(Planner::Space::end)> problems;

  const Planner::Problem &problem(Planner::Space space) const {
    return problems[ordinal(space)];
  }

  Planner::Problem &problem(Planner::Space space) {
    return const_cast<Planner::Problem &>(
        (const_cast<const FuncAnalysisData *>(this))->problem(space));
  }
};

} // namespace
//===----------------------------------------------------------------------===//
// Pass implementation.
//===----------------------------------------------------------------------===//
namespace {
class TTIRAllocate final : public impl::TTIRAllocateBase<TTIRAllocate> {
  using Base = impl::TTIRAllocateBase<TTIRAllocate>;

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

    TT_ALLOC_DEBUG("configured with 'allow-output-spilling' = {}",
                   allowOutputSpilling);

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

  // TODO rename?
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

      if (llvm::isa<memref::AllocOp, ttir::ViewLayoutOp, ttir::StreamLayoutOp>(
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
    // users (ttir.view_layout, ttir.stream_layout).
    //
    // We calculate the "last use position" by computing for each value
    // the max over its users over a traversal through this graph.

    for (auto &[op, closure] : livenessJoinGraph) {
      // Initial maxLast values are from the SSA liveness calculation.
      auto i = analysis.sequencing.operationMap.find(closure.lastOp);
      TT_debug(i != analysis.sequencing.operationMap.end());
      closure.live.last = i->second;
    }

    // TODO non-recursive impl?
    for (auto &[op, closure] : livenessJoinGraph) {
      closure.live.last = resolve(op, livenessJoinGraph);

      // Copy liveness results into our alloc set.
      if (memref::AllocOp allocOp = llvm::dyn_cast<memref::AllocOp>(op)) {
        TT_assertv(!allocOp->use_empty(),
                   "didn't expect an alloc op without uses: {}",
                   asOperand(allocOp));

        MemrefValueContext &memrefCtx = analysis.memrefs[allocOp];

        memrefCtx.type = mlir::cast<MemRefType>(op->getResult(0).getType());
        if (isDeviceMemorySpace(memrefCtx.type, ttcore::MemorySpace::System)) {
          memrefCtx.size = device.getMemrefSizeBytes(memrefCtx.type);
        }

        memrefCtx.live = closure.live;
      }
    }

    TT_ALLOC_DEBUG("collected {} root memref context(s)",
                   analysis.memrefs.size());
    return success();
  }

  LogicalResult analyzeOperandStreams(func::FuncOp funcOp,
                                      FuncAnalysisData &analysis) {

    [[maybe_unused]] AsOperandPrinter asOperand{funcOp};

    ttcore::DeviceAttr device = ttcore::lookupDevice(funcOp);
    Block &funcBody = funcOp.getBody().front();

    // Temp state to help set `MemrefValueContext::hasNonGenericUsers`.
    using OperationSet = llvm::SmallPtrSet<Operation *, 4>;
    llvm::DenseMap<memref::AllocOp, OperationSet>
        genericUsersMap; // TODO rename

    funcBody.walk([&](ttir::GenericOp genericOp) {
      // Decide which operands might/must have streams. Note that
      // the actual stream creation decision is only final after
      // we have the memory planner's placement solution.
      llvm::SmallVector<OperandStream> streams = getOperandStreams(genericOp);
      TT_debug(streams.size() == genericOp.getNumOperands());

      GenericOpContext &genericCtx = analysis.generics[genericOp];
      for (std::size_t operandIndex = 0;
           operandIndex < genericOp.getNumOperands(); ++operandIndex) {
        auto operand = genericOp->getOperand(operandIndex);
        OperandStream &stream = streams[operandIndex];
        // For later IR mutation, it is convenient at this point to gather
        // all chains of ops defining operand inputs.
        Value memref =
            getOperandDefChain(genericOp, operand, stream.defUseChain);

        const auto &[i, inserted] = analysis.memrefs.try_emplace(memref);
        MemrefValueContext &memrefCtx = i->second;

        memrefCtx.genericUsers.insert(genericOp);
        memrefCtx.isSomeonesOutput |= stream.isOutput;

        if (inserted) {
          // These were not discovered by the earlier walk.
          TT_debugv(mlir::isa<BlockArgument>(memref),
                    "expected a block arg: {}", memref);
          memrefCtx.type = mlir::cast<MemRefType>(memref.getType());
          memrefCtx.size = device.getMemrefSizeBytes(memrefCtx.type);
        } else {
          // An existing `analysis.memrefs` entry means `operand` is ultimately
          // rooted in a `memref::AllocOp`.
          memref::AllocOp allocOp =
              mlir::cast<memref::AllocOp>(memref.getDefiningOp());

          // Track a closure of `allocOp`'s users along the generic/alloc
          // use-def chains.
          OperationSet &allocOpGenericUsers = genericUsersMap[allocOp];

          allocOpGenericUsers.insert(genericOp.getOperation());
          allocOpGenericUsers.insert(stream.defUseChain.begin(),
                                     stream.defUseChain.end());
        }

        genericCtx.operands.try_emplace(memref, std::move(stream));
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

    // Alloc ops that have users other than a `func.return` or `ttir.generic`
    // will be marked as ineligible for memspace remapping.

    [[maybe_unused]] int32_t allocsWithNonGenericUsers = 0;

    for (auto &[allocOp, users] : genericUsersMap) {
      for (Operation *user : allocOp->getUsers()) {
        if (!llvm::isa<func::ReturnOp>(user) && !users.contains(user)) {
          analysis.memrefs[allocOp].hasNonGenericUsers = true;
          ++allocsWithNonGenericUsers;
        }
      }
    }

    TT_ALLOC_DEBUG("collected {} generic op context(s)",
                   analysis.generics.size());
    TT_ALLOC_DEBUG("found {} alloc(s) with non-generic use",
                   allocsWithNonGenericUsers);
    return success();
  }

  // Form a placement problem.
  LogicalResult prepareMemoryPlanner(func::FuncOp funcOp,
                                     FuncAnalysisData &analysis) {
    ttcore::DeviceAttr device = ttcore::lookupDevice(funcOp);
    IRRewriter rewriter(funcOp->getContext());

    Planner::Problem &problem = analysis.problem(Planner::Space::Scratch);

    // Each `analysis.memrefs` entry defines an allocation planner decision
    // variable. These can be of different origins:
    // (1) A memref defined by a `memref.alloc` backing a generic op operand and
    //     potentially associated with a stream and its buffer.
    // (2) A memref that backs a generic op operand but is not defined by an op
    //     inside `funcOp` (i.e. passed as a block argument). We may insert a
    //     stream for this operand and will therefore need to allocate this
    //     stream's buffer.
    // (3) A memref defined by a "standalone" `memref.alloc` that needs no
    //     generic op streaming but will still need a valid L1/DRAM memory
    //     address assigned.

    for (auto &[memref, memrefCtx] : analysis.memrefs) {
      const ttcore::MemorySpace memspace =
          getMemorySpace(memrefCtx.type, ttcore::MemorySpace::System);
      if (!ttcore::isDeviceMemorySpace(memspace)) {
        continue;
      }
      // Invariant established earlier: all `analysis.memrefs` in DRAM/L1 have
      // 'type' and 'size' set.
      TT_debugv((memrefCtx.type != nullptr && memrefCtx.size >= 0),
                "memref: {}", memref);

      TT_debug(memrefCtx.varIndex < 0);
      memrefCtx.varIndex = problem.def([&, &memref = memref,
                                        &memrefCtx = memrefCtx](
                                           Planner::VariableBuilder &b) {
        // If `memref` is being defined inside `funcOp` and is initially placed
        // in L1, it will require scratch memory to hold its tensor data.
        if (memref.getDefiningOp<memref::AllocOp>() &&
            memspace == ttcore::MemorySpace::DeviceL1) {
          const auto &memInfo =
              memSpaces[ordinal(ttcore::MemorySpace::DeviceL1)];

          memrefCtx.reqIndex = b.request(
              Planner::Space::Scratch,
              ttmlir::utils::alignUp(memrefCtx.size, memInfo.alignment),
              memrefCtx.live.first, memrefCtx.live.last);
        }

        // This decision variable must be bound to its incoming memspace in
        // any of these cases:
        //  - if it is placed in DRAM explicitly;
        //  - if it has non-generic op users or has zero generic op users;
        //  - if it the output of of a generic op and the pass options do not
        //  allow output spilling.
        const bool bound = (memspace == ttcore::MemorySpace::DeviceDRAM) ||
                           memrefCtx.genericUsers.empty() ||
                           memrefCtx.hasNonGenericUsers ||
                           (memrefCtx.isSomeonesOutput && !allowOutputSpilling);
        if (bound) {
          b.bind(asPlannerSpace(memspace));
        }

        // For each possible variable placement, add mem requests for L1 stream
        // buffers if the variable must be streamed when it backs a generic op
        // operand.
        for (Planner::Space placement = Planner::Space::begin;
             placement < Planner::Space::end; ++placement) {

          const ttcore::MemorySpace placementMemspace =
              asMemorySpace(placement);
          if (bound && placementMemspace != memspace) {
            // A bound variable only needs its domain populated for its fixed
            // (incoming) memspace.
            continue;
          }

          const auto &memInfo = memSpaces[ordinal(placementMemspace)];

          for (ttir::GenericOp user : memrefCtx.genericUsers) {
            GenericOpContext &genericCtx = analysis.generics[user];
            OperandStream &stream = genericCtx.operands.find(memref)->second;

            // An operand stream is required under any of these conditions:
            // - streaming was earlier determined as required due to inter-core
            // data movement needs;
            // - the memref placement is in scratch memory, i.e. DRAM.
            if (stream.requiresStream || (placement == Planner::Space::Spill)) {
              if (!stream.bufferType) {
                // In principle, buffer shape/size could depend on whether the
                // stream is out of L1 or DRAM... but not right now.
                stream.bufferType =
                    selectStreamBuffer(rewriter, memrefCtx.type);
              }
              const AllocSizeT bufferSize =
                  device.getMemrefSizeBytes(stream.bufferType);

              // Because we will insert stream buffer allocs just before generic
              // ops themselves, without any other interposing allocs, it is
              // mathematically correct to see all such buffers' live ranges as
              // coinciding with the generic op's logical time.
              const SequenceT firstAndLast = analysis.sequencing[user];

              TT_debug(stream.reqIndex[ordinal(placement)] < 0);
              stream.reqIndex[ordinal(placement)] = b.request(
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
    {
      auto &problem = analysis.problem(Planner::Space::Scratch);

      const auto &memInfo = memSpaces[ordinal(ttcore::MemorySpace::DeviceL1)];
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
    {
      const auto &L1solution = analysis.problem(Planner::Space::Scratch);
      auto &problem = analysis.problem(Planner::Space::Spill);

      const auto &memInfo = memSpaces[ordinal(ttcore::MemorySpace::DeviceDRAM)];

      // Form an allocation problem out of all L1 variables with `spill`
      // placements (either mapped or bound).
      //
      // Note that we already have valid live ranges but must re-calculate
      // request sizes using DRAM alignment.

      for (auto &[memref, memrefCtx] : analysis.memrefs) {
        if (!isDeviceMemorySpace(memrefCtx.type, ttcore::MemorySpace::System)) {
          continue;
        }

        if (Planner::Space::Spill ==
            L1solution.variable(memrefCtx.varIndex).placement) {
          memrefCtx.varIndex = problem.def(
              [&, &allocCtx = memrefCtx](Planner::VariableBuilder &b) {
                allocCtx.reqIndex = b.request(
                    Planner::Space::Scratch,
                    ttmlir::utils::alignUp(allocCtx.size, memInfo.alignment),
                    allocCtx.live.first, allocCtx.live.last);
              });
          memrefCtx.remappedMemSpace = ttcore::MemorySpace::DeviceDRAM;
        } else {
          // This `memref` remains in scratch memory and we have its solution
          // parameters in `analysis.problem(Planner::Space::Scratch)`.
          memrefCtx.remappedMemSpace = ttcore::MemorySpace::DeviceL1;
        }
      }

      if (!problem.empty()) {
        problem.reset(Planner::Space::Scratch);
        TT_ALLOC_TRACE("DRAM planner problem:\n{}", problem);

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

  // Sweep through all allocs in the incoming IR (not just those associated with
  // operand streams) and set their address/alignment attribute, *without*
  // changing their memspace. The latter will be fixed in a subsequent step
  // (which will also restore the IR to a valid state).
  LogicalResult
  assignAllocAddresses(func::FuncOp funcOp,
                       /* TODO const */ FuncAnalysisData &analysis) {
    [[maybe_unused]] AsOperandPrinter asOperand{funcOp};

    IRRewriter rewriter(funcOp->getContext());

    for (auto &[memref, memrefCtx] : analysis.memrefs) {
      if (!isDeviceMemorySpace(memrefCtx.type, ttcore::MemorySpace::System)) {
        continue;
      }
      memref::AllocOp allocOp = memref.getDefiningOp<memref::AllocOp>();
      if (!allocOp) {
        continue;
      }
      TT_debugv(memrefCtx.remappedMemSpace.has_value(),
                "should have been placed: {}", asOperand(memref));

      const auto &solution =
          analysis.problem(asPlannerSpace(*memrefCtx.remappedMemSpace));
      const auto &memInfo = memSpaces[ordinal(*memrefCtx.remappedMemSpace)];

      assign(rewriter, allocOp, solution.request(memrefCtx.reqIndex).offset,
             memInfo);
    }

    return success();
  }

  LogicalResult insertOperandStreams(func::FuncOp funcOp,
                                     const FuncAnalysisData &analysis) {
    IRRewriter rewriter(funcOp->getContext());

    const auto &L1solution = analysis.problem(Planner::Space::Scratch);
    const auto &L1memInfo = memSpaces[ordinal(ttcore::MemorySpace::DeviceL1)];

    llvm::DenseSet<Operation *> visited;

    for (const auto &[genericOp, genericCtx] : analysis.generics) {
      int32_t operandIndex = 0;
      for (const auto &[memref, stream] : genericCtx.operands) {

        // Walk the use-def chain in `stream` and:
        // - modify root alloc ops to be in the memspace as decided by the
        // planner
        // - modify view layout ops to have correct memspace typing
        // - insert stream layout ops together with their stream buffer allocs

        TT_debug(analysis.memrefs.contains(memref));
        const MemrefValueContext &memrefCtx = analysis.memrefs.at(memref);

        const ttcore::MemorySpace remappedMemorySpace =
            memrefCtx.remappedMemSpace.value();

        for (Operation *opOnChain : stream.defUseChain) {
          // Even though assigning final memspace is idempotent,
          // don't do this repeatedly.
          if (!visited.insert(opOnChain).second) {
            continue;
          }
          llvm::TypeSwitch<Operation *, void>(opOnChain)
              .Case([&](memref::AllocOp op) {
                remap(rewriter, op, remappedMemorySpace);
                insertDealloc(rewriter, op, memrefCtx.live.last,
                              analysis.sequencing);
              })
              .Case([&](ttir::ViewLayoutOp op) {
                remap(rewriter, op, remappedMemorySpace);
              });
        }

        // The above may have changed memspace attributes of ops in the
        // operand's use-def chain; inserting a matching `stream_layout` next
        // will restore IR to a valid form.

        if (stream.requiresStream ||
            (remappedMemorySpace == ttcore::MemorySpace::DeviceDRAM)) {

          const Planner::Space finalPlacement =
              asPlannerSpace(remappedMemorySpace);
          TT_debug(stream.reqIndex[ordinal(finalPlacement)] >= 0);
          const Planner::Request &req =
              L1solution.request(stream.reqIndex[ordinal(finalPlacement)]);

          // Note that this will take care of inserting the dealloc for the
          // stream buffer.
          auto &operand = genericOp->getOpOperand(operandIndex);
          if (failed(insertStream(rewriter, operand, genericOp, req, L1memInfo,
                                  analysis.sequencing))) {
            return failure();
          }
        }

        ++operandIndex;
      }
    }

    return success();
  }

  // TODO rename (only analyzes part of `OperandStream`)?..
  static llvm::SmallVector<OperandStream>
  getOperandStreams(ttir::GenericOp genericOp) {
    const std::size_t outputsStart =
        genericOp.getOutputs().getBeginOperandIndex();
    ArrayAttr iteratorTypes = genericOp.getIteratorTypes();

    llvm::SmallVector<OperandStream> result;

    for (std::size_t operandIndex = 0;
         operandIndex < genericOp.getNumOperands(); ++operandIndex) {
      OperandStream &stream = result.emplace_back();

      stream.isOutput = (operandIndex >= outputsStart);

      // A core participating in a reduction dim necessarily requires
      // non-local data movement unless it is the only core involved
      // in that dim.
      //
      // Similar logic applies to a broadcast dim.
      const AffineMap indexingMap = genericOp.getIndexingMap(operandIndex);
      const auto bcastDims = indexingMap.getBroadcastDims();
      const llvm::SmallSet<unsigned, 4> bcastDimIndex(bcastDims.begin(),
                                                      bcastDims.end());
      stream.requiresStream = llvm::any_of(
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

      // Note: even if `params.requiresStream` is left false here, a stream
      // may still be inserted, e.g. to read the operand from DRAM
    }

    // TODO possible issue/edge case when the same Value appears
    // in more than one operand slot?
    TT_debug(result.size() == genericOp.getNumOperands());
    return result;
  }

  static void assign(RewriterBase &rewriter, memref::AllocOp op,
                     Planner::AllocSizeT offset, const MemorySpaceInfo &info) {

    const AllocSizeT address = info.baseAddress + offset;

    rewriter.startOpModification(op);
    {
      op.setAlignment(info.alignment);
      op->setAttr("address", rewriter.getI64IntegerAttr(address));
    };
    rewriter.finalizeOpModification(op);
  }

  static MemRefType remap(RewriterBase &rewriter, MemRefType memrefType,
                          ttcore::MemorySpace memspace) {
    return MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                           memrefType.getLayout(),
                           rewriter.getAttr<ttcore::MemorySpaceAttr>(memspace));
  }

  static void remap(RewriterBase &rewriter, memref::AllocOp op,
                    ttcore::MemorySpace memspace) {
    auto memref = op.getMemref();
    MemRefType memrefType = memref.getType();
    MemRefType newType = remap(rewriter, memrefType, memspace);

    rewriter.modifyOpInPlace(op, [&]() { memref.setType(newType); });
  }

  static void remap(RewriterBase &rewriter, ttir::ViewLayoutOp op,
                    ttcore::MemorySpace memspace) {
    auto memref = op->getResult(0); // TODO name
    MemRefType memrefType = llvm::cast<MemRefType>(memref.getType());
    MemRefType newType = remap(rewriter, memrefType, memspace);

    rewriter.modifyOpInPlace(op, [&]() { memref.setType(newType); });
  }

  // TODO this is mocked up, will eventually be a more complex decision
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
                                       MemRefType operandType) {
    llvm::SmallVector<int64_t> bufferShape =
        selectStreamBufferShape(operandType);
    auto bufferLayout = ttcore::ShardLayoutAttr::get(
        ArrayRef(bufferShape).take_back(bufferShape.size() / 2),
        operandType.getElementType(), /*buffers=*/1);
    return MemRefType::get(ArrayRef(bufferShape), operandType.getElementType(),
                           bufferLayout,
                           rewriter.getAttr<ttcore::MemorySpaceAttr>(
                               ttcore::MemorySpace::DeviceL1));
  }

  static LogicalResult insertStream(RewriterBase &rewriter, OpOperand &operand,
                                    ttir::GenericOp op,
                                    const Planner::Request &req,
                                    const MemorySpaceInfo &info,
                                    const SequenceMapping &sequencing) {
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

      // TODO this buffer type selection decision is repeated here, it was
      // already made by an analysis step
      auto bufferMemref = selectStreamBuffer(rewriter, operandMemrefType);
      auto buffer = rewriter.create<memref::AllocOp>(op.getLoc(), bufferMemref);
      assign(rewriter, buffer, req.offset, info);
      insertDealloc(rewriter, buffer, req.last, sequencing);

      auto stream = rewriter.create<ttir::StreamLayoutOp>(
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
  static Value getOperandDefChain(ttir::GenericOp genericOp, Value operand,
                                  DefUseChain &chain) {
    Operation *definingOp = operand.getDefiningOp();
    if (!definingOp) {
      TT_debug(mlir::isa<BlockArgument>(operand));
      return operand;
    }
    chain.emplace_back(definingOp);

    // Note: a canonicalizer pass would have collapse all `ttir.view_layout`
    // chains but we don't rely on that here.
    return llvm::TypeSwitch<Operation *, Value>(definingOp)
        .Case([&](memref::AllocOp op) { return operand; })
        .Case([&](ttir::ViewLayoutOp op) {
          return getOperandDefChain(genericOp, op.getInput(), chain);
        })
        .Case([&](ttir::StreamLayoutOp op) {
          return getOperandDefChain(genericOp, op.getInput(), chain);
        });
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
        if (llvm::isa<ttir::ViewLayoutOp, ttir::StreamLayoutOp>(user)) {
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
      info[ordinal(ttcore::MemorySpace::DeviceL1)] =
          MemorySpaceInfo(chipDesc.getL1UnreservedBase(), chipDesc.getL1Size(),
                          chipDesc.getNocL1AddressAlignBytes());

      info[ordinal(ttcore::MemorySpace::DeviceDRAM)] = MemorySpaceInfo(
          chipDesc.getDramUnreservedBase(), chipDesc.getDramChannelSize(),
          chipDesc.getNocDRAMAddressAlignBytes());
    }
    return info;
  }
};
} // namespace

} // namespace mlir::tt::ttir
// ----------------------------------------------------------------------------
