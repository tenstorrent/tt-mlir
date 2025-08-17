// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/Analysis/AllocationDefs.h"
#include "ttmlir/Dialect/TTIR/Analysis/AllocationPlanner.h"
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
  return memSpace ? mlir::cast<ttcore::MemorySpaceAttr>(memref.getMemorySpace())
                        .getValue()
                  : dflt;
}
//===----------------------------------------------------------------------===//
// Helper classes.
//===----------------------------------------------------------------------===//
namespace {

using Planner = AllocationPlanner;

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

struct AllocOpContext {
  // All generic op users of this alloc (immediate or through view/steam layout
  // ops).
  llvm::DenseSet<ttir::GenericOp> genericUsers;
  // "Raw" allocation request size in bytes (unaligned for any particular
  // memspace).
  AllocSizeT size = -1; // TODO this is available from the op itself
  // Live range of this alloc, starting with the op itself and
  // extending to its latest user.
  LiveRange live = {-1, -1};
  ttcore::MemorySpace memspace;    // TODO is available from the op itself
  bool hasNonGenericUsers = false;
  bool isSomeonesOutput = false;

  int32_t varIndex = -1; // needed to retrieve Variable::placement
  int32_t reqIndex = -1; // needed to retrieve offset of Variable's alloc
  std::optional<ttcore::MemorySpace> finalMemSpace;
};

struct OperandStreamParams {
  OperandStreamParams(int32_t operandIndex) : operandIndex(operandIndex) {}

  MemRefType bufferType;
  int32_t operandIndex;
  bool isOutput = false;
  bool requiresStream = false; // be it L1 or DRAM

  std::array<int32_t, Planner::Space::limit> reqIndex = {-1, -1};
};

struct GenericOpContext {
  // Root definitions of the use-def chain for each of this generic ops'
  // list of operands. This map reflects only those `GenericOp->getOperands()`
  // that originate in allocs.
  llvm::SmallMapVector<memref::AllocOp, OperandStreamParams, 4> operands;
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

  template <typename ConcreteOp> // TODO restrict to ops
  SequenceT operator[](ConcreteOp op) const {
    return this->operator[](op.getOperation());
  }
};

struct FuncAnalysisData {
  SequenceMapping mapping;
  llvm::DenseMap<memref::AllocOp, AllocOpContext> allocOps;     // owns values
  llvm::DenseMap<ttir::GenericOp, GenericOpContext> genericOps; // owns values
  std::array<Planner::Problem, Planner::Space::limit> problems;

  const Planner::Problem &problem(Planner::Space space) const {
    return problems[space];
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

    if (failed(runAnalyzeAllocOps(funcOp, analysis))) {
      return failure();
    }

    if (failed(runAnalyzeGenericOps(funcOp, analysis))) {
      return failure();
    }

    if (failed(runAnalyzeStreams(funcOp, analysis))) {
      return failure();
    }

    if (failed(runMemoryPlanner(funcOp, analysis))) {
      return failure();
    }

    if (failed(runAssignAddresses(funcOp, analysis))) {
      return failure();
    }

    if (failed(runInsertStreams(funcOp, analysis))) {
      return failure();
    }

    return success();
  }

  LogicalResult runAnalyzeAllocOps(func::FuncOp funcOp,
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
      const SequenceT position = analysis.mapping.size();

      analysis.mapping.operationMap[op] = position;
      analysis.mapping.positionMap.emplace_back(op);

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
    TT_debug(analysis.mapping.operationMap.size() ==
             analysis.mapping.positionMap.size());

    // Ops in `livenessJoinGraph` form a graph of Values and their users where
    // some Values have their original SSA liveness "extended" by stream op
    // users (ttir.view_layout, ttir.stream_layout).
    //
    // We calculate the "last use position" by computing for each value
    // the max over its users over a traversal through this graph.

    for (auto &[op, closure] : livenessJoinGraph) {
      // Initial maxLast values are from the SSA liveness calculation.
      auto i = analysis.mapping.operationMap.find(closure.lastOp);
      TT_debug(i != analysis.mapping.operationMap.end());
      closure.live.last = i->second;
    }

    // TODO non-recursive impl
    for (auto &[op, closure] : livenessJoinGraph) {
      closure.live.last = resolve(op, livenessJoinGraph);

      // Copy liveness results into our alloc set.
      if (memref::AllocOp allocOp = llvm::dyn_cast<memref::AllocOp>(op)) {
        TT_assertv(!allocOp->use_empty(),
                   "didn't expect an alloc op without uses: {}",
                   asOperand(allocOp));

        AllocOpContext &allocCtx = analysis.allocOps[allocOp];
        allocCtx.live = closure.live;
        const auto memrefType = allocOp.getType();

        allocCtx.memspace =
            getMemorySpace(memrefType, ttcore::MemorySpace::System);

        if (isDeviceMemorySpace(allocCtx.memspace)) {
          allocCtx.size = device.getMemrefSizeBytes(memrefType);
        }
      }
    }

    TT_ALLOC_DEBUG("collected {} root alloc(s)", analysis.allocOps.size());
    return success();
  }

  LogicalResult runAnalyzeGenericOps(func::FuncOp funcOp,
                                     FuncAnalysisData &analysis) {
    Block &funcBody = funcOp.getBody().front();

    // All `ttir.generic`s will need to make decisions about their operand
    // streams, so collect all of them.
    //
    // Note that the set of `memref.alloc`s that can be reached as root defs
    // of all generic operands could be a proper subset of `analysis.allocOps`.
    // Such `memref.alloc`s with non-generic users are ineligible for memspace
    // remapping because this pass doesn't (currently) deal with non-generic
    // ops.

    llvm::DenseMap<memref::AllocOp, llvm::SmallPtrSet<Operation *, 4>>
        allocPathVisitSets; // TODO rename

    funcBody.walk([&](ttir::GenericOp genericOp) {
      GenericOpContext &genericCtx = analysis.genericOps[genericOp];

      llvm::SmallVector<OperandStreamParams> streams =
          analyzeOperandStreams(genericOp);

      const int32_t outputsStart =
          genericOp.getOutputs().getBeginOperandIndex();

      int32_t operandIndex = 0;
      for (Value operand : genericOp->getOperands()) {
        llvm::SmallVector<Operation *> path;

        memref::AllocOp allocOp = findRootAlloc(operand, path);
        if (allocOp) {
          auto i = analysis.allocOps.find(allocOp);
          TT_debug(i != analysis.allocOps.end());
          AllocOpContext &allocCtx = i->second;

          allocCtx.genericUsers.insert(genericOp);

          if (operandIndex >= outputsStart) {
            allocCtx.isSomeonesOutput = true;
          }

          // Track the full set of ops along the generic/alloc use-def chains.
          auto &allocOpPathVisits = allocPathVisitSets[allocOp];

          allocOpPathVisits.insert(genericOp.getOperation());
          allocOpPathVisits.insert(path.begin(), path.end());
        }
        genericCtx.operands.try_emplace(allocOp,
                                        std::move(streams[operandIndex]));
        ++operandIndex;
      }
    });

    for (auto &[allocOp, pathVisits] : allocPathVisitSets) {
      for (Operation *user : allocOp->getUsers()) {
        if (!pathVisits.contains(user)) {
          analysis.allocOps[allocOp].hasNonGenericUsers = true;
        }
      }
    }

    TT_ALLOC_DEBUG("collected {} generic(s)", analysis.genericOps.size());
    return success();
  }

  LogicalResult runAnalyzeStreams(func::FuncOp funcOp,
                                  FuncAnalysisData &analysis) {
    [[maybe_unused]] AsOperandPrinter asOperand{funcOp};

    // Convert 'analysis' into an allocation plan problem. There are two
    // levels of decision variables:
    //
    // 1. for all allocs, their memspace placements sized for L1 and DRAM;
    // 2. for all generics, stream buffer sizes for those operands that are
    // being placed in DRAM or require L1 streams due to non-local data
    // movement.
    //
    // Note:
    // - TODO currently decision #2 is fixed
    // - TODO (this is inaccurate, mem pressure also reduces because of
    // shorter liferanges) we require that an operand spill from L1 to DRAM
    // result in strict memory pressure improvement, i.e. stream buffer
    // sizes must be less than their original alloc sizes;
    // - not all alloc operands are eligible for a memspace change (e.g.
    // those with non-generic op users aren't).

    // (1) if an alloc is explicitly set to DRAM, we leave it in DRAM;
    //  otherwise the alloc's placement is a decision variable
    // (2) an operand may require a stream (out of L1 or DRAM) due to
    //  a data movement non-local to some cores; otherwise a stream will
    //  be inserted if the final placement is in DRAM;
    //  in any case, a stream will require a stream buffer reservation in L1
    // (3) if an alloc defines an operand to a non-generic op, we leave its
    //  placement as-is, the alloc will not be a decision variable

    ttcore::DeviceAttr device = ttcore::lookupDevice(funcOp);
    IRRewriter rewriter(funcOp->getContext());

    for (auto &[allocOp, ctx] : analysis.allocOps) {
      TT_ALLOC_TRACE("{}:\t[{}, {}] {} byte(s), {} user(s)", asOperand(allocOp),
                     ctx.live.first, ctx.live.last, ctx.size,
                     ctx.genericUsers.size());
    }

    // Form a placement problem.

    Planner::Problem &problem = analysis.problem(Planner::Space::Scratch);

    for (auto &[allocOp, allocCtx] : analysis.allocOps) {
      if (!ttcore::isDeviceMemorySpace(allocCtx.memspace)) {
        continue; // TODO somebody needs to deal with system allocs...
      }
      TT_debug(allocCtx.varIndex < 0);
      allocCtx.varIndex = problem.def([&, &allocOp = allocOp,
                                       &allocCtx = allocCtx](
                                          Planner::VariableBuilder &b) {
        // Scratch mem request for the root alloc itself.
        if (allocCtx.memspace == ttcore::MemorySpace::DeviceL1) {
          const auto &memInfo =
              memSpaces[ordinal(ttcore::MemorySpace::DeviceL1)];

          allocCtx.reqIndex = b.request(
              Planner::Space::Scratch,
              ttmlir::utils::alignUp(allocCtx.size, memInfo.alignment),
              allocCtx.live.first, allocCtx.live.last);
        }

        // Mem requests for stream buffers associated with streams
        // out of `allocOp`.

        for (Planner::Space placement = Planner::Space::first;
             placement < Planner::Space::limit; ++placement) {

          const auto &memInfo = memSpaces[ordinal(asMemorySpace(placement))];

          // if any of generic users of `allocOp` must read it via a
          // stream, there will be a scratch buffer request for the
          // corresponding operand:

          for (ttir::GenericOp user : allocCtx.genericUsers) {
            GenericOpContext &genericCtx = analysis.genericOps[user];
            OperandStreamParams &params =
                genericCtx.operands.find(allocOp)->second;

            if ((placement == Planner::Space::Spill) || params.requiresStream) {
              // In principle, buffer shape/size could depend on whether the
              // stream is out of L1 or DRAM. But not right now.
              if (!params.bufferType) {
                params.bufferType =
                    selectStreamBuffer(rewriter, allocOp.getType());
              }
              const AllocSizeT bufferSize =
                  device.getMemrefSizeBytes(params.bufferType);

              const SequenceT time = analysis.mapping[user];
              TT_debug(params.reqIndex[placement] < 0);

              params.reqIndex[placement] = b.request(
                  placement,
                  ttmlir::utils::alignUp(bufferSize, memInfo.alignment), time,
                  time);
            }
          }
        }

        // An alloc is not a free decision variables if:
        //  - it is the output of some `ttir.generic` op but
        // `allow-output-spilling` pass option is disabled, or
        //  - it was placed into DRAM explicitly, or
        //  - it has a user that is not a `ttir.generic` op

        if ((allocCtx.isSomeonesOutput && !allowOutputSpilling) ||
            (allocCtx.memspace == ttcore::MemorySpace::DeviceDRAM) ||
            (allocCtx.genericUsers.empty() || allocCtx.hasNonGenericUsers)) {
          b.bind(asPlannerSpace(allocCtx.memspace));
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

      // Form an allocation problem out of all L1 variables with spill
      // placements (either mapped or bound).
      //
      // Note that we can already have valid live ranges but must re-calculate
      // request sizes using DRAM alignment.

      for (auto &[allocOp, allocCtx] : analysis.allocOps) {
        if (!ttcore::isDeviceMemorySpace(allocCtx.memspace)) {
          continue; // TODO somebody needs to deal with system allocs...
        }

        if (Planner::Space::Spill ==
            L1solution.variable(allocCtx.varIndex).placement) {
          allocCtx.varIndex = problem.def(
              [&, &allocCtx = allocCtx](Planner::VariableBuilder &b) {
                allocCtx.reqIndex = b.request(
                    Planner::Space::Scratch,
                    ttmlir::utils::alignUp(allocCtx.size, memInfo.alignment),
                    allocCtx.live.first, allocCtx.live.last);
              });
          allocCtx.finalMemSpace = ttcore::MemorySpace::DeviceDRAM;
        } else {
          // This `allocOp` remains in scratch memory and we have it solution
          // parameters in `analysis.problem(Planner::Space::SCRATCH)`.
          allocCtx.finalMemSpace = ttcore::MemorySpace::DeviceL1;
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

  // Sweep through all allocs in the incoming IR (not associated with operand
  // streams) and set their address/alignment attribute, *without* changing
  // their memspace. The latter is not done here to avoid breaking IR validity
  // and will be fixed in a subsequent step.
  LogicalResult runAssignAddresses(func::FuncOp funcOp,
                                   const FuncAnalysisData &analysis) {
    [[maybe_unused]] AsOperandPrinter asOperand{funcOp};

    IRRewriter rewriter(funcOp->getContext());

    for (auto &[allocOp, allocCtx] : analysis.allocOps) {
      if (!ttcore::isDeviceMemorySpace(allocCtx.memspace)) {
        continue; // TODO somebody needs to deal with system allocs...
      }
      TT_debugv(allocCtx.finalMemSpace.has_value(),
                "should have been placed: {}", asOperand(allocOp));

      const auto &solution =
          analysis.problem(asPlannerSpace(*allocCtx.finalMemSpace));
      const auto &memInfo = memSpaces[ordinal(*allocCtx.finalMemSpace)];

      assign(rewriter, allocOp, solution.request(allocCtx.reqIndex).offset,
             memInfo);
    }

    return success();
  }

  LogicalResult runInsertStreams(func::FuncOp funcOp,
                                 const FuncAnalysisData &analysis) {
    IRRewriter rewriter(funcOp->getContext());

    const auto &L1solution = analysis.problem(Planner::Space::Scratch);
    const auto &L1memInfo = memSpaces[ordinal(ttcore::MemorySpace::DeviceL1)];

    llvm::DenseSet<Operation *> visited;

    for (const auto &[genericOp, genericCtx] : analysis.genericOps) {
      for (const auto &[allocOp, params] : genericCtx.operands) {

        // Walk the use-def chain starting with `operand` and:
        // - modify root alloc ops to be in the memspace as decided by the
        // planner
        // - modify view layout ops to have correct memspace typing
        // - insert stream layout ops together with their stream buffer allocs

        OpOperand &operand = genericOp->getOpOperand(params.operandIndex);

        llvm::SmallVector<Operation *, 4> chain;
        if (failed(getUseDefChain(genericOp, operand.get(), chain))) {
          return failure();
        }

        TT_debug(analysis.allocOps.contains(allocOp));
        const AllocOpContext &allocCtx = analysis.allocOps.at(allocOp);

        const ttcore::MemorySpace remappedMemorySpace =
            allocCtx.finalMemSpace.value();

        for (Operation *opOnChain : chain) {
          // Even though assigning final memspace is idempotent,
          // don't do this repeatedly.
          if (!visited.insert(opOnChain).second) {
            continue;
          }
          llvm::TypeSwitch<Operation *, void>(opOnChain)
              .Case([&](memref::AllocOp op) {
                remap(rewriter, op, remappedMemorySpace);
                insertDealloc(rewriter, op, allocCtx.live.last,
                              analysis.mapping);
              })
              .Case([&](ttir::ViewLayoutOp op) {
                remap(rewriter, op, remappedMemorySpace);
              });
        }

        // The above may have changed memspace attributes of ops in the
        // operand's use-def chain; inserting a matching `stream_layout` next
        // will restore IR to a valid form.

        if (params.requiresStream ||
            (remappedMemorySpace == ttcore::MemorySpace::DeviceDRAM)) {

          const Planner::Space finalPlacement =
              asPlannerSpace(remappedMemorySpace);
          TT_debug(params.reqIndex[finalPlacement] >= 0);
          const Planner::Request &req =
              L1solution.request(params.reqIndex[finalPlacement]);

          // Note that this will take care of inserting the dealloc for the
          // stream buffer.
          if (failed(insertStream(rewriter, operand, genericOp, req, L1memInfo,
                                  analysis.mapping))) {
            return failure();
          }
        }
      }
    }

    return success();
  }

  static llvm::SmallVector<OperandStreamParams> analyzeOperandStreams(ttir::GenericOp genericOp) {
    const int32_t outputsStart = genericOp.getOutputs().getBeginOperandIndex();
    ArrayAttr iteratorTypes = genericOp.getIteratorTypes();

    llvm::SmallVector<OperandStreamParams> result;

    for (int32_t operandIndex = 0;
        operandIndex < static_cast<int32_t>(genericOp.getNumOperands());
        ++operandIndex) {
      OperandStreamParams &params = result.emplace_back(operandIndex);

      // TODO check logic here for outputs
      if (operandIndex >= outputsStart) {
        params.isOutput = true;
        // No stream (NOC ops) will be needed if 'operand' is already
        // allocated in L1 ("alias" mode), which is currently guaranteed
        // to be the case for outputs.
        params.requiresStream = false;
      } else {
        // A core participating in a reduction dim necessarily requires
        // non-local data movement unless it is the only core involved
        // in that dim.
        //
        // Similar logic applies to a broadcast dim.
        const AffineMap indexingMap = genericOp.getIndexingMap(operandIndex);
        const auto bcastDims = indexingMap.getBroadcastDims();
        const llvm::SmallSet<unsigned, 4> bcastDimIndex(bcastDims.begin(),
                                                        bcastDims.end());
        params.requiresStream = llvm::any_of(
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

        // Note: even if `params.requiresStream` is left false here, a stream may
        // still be inserted, e.g. to read the operand from DRAM
      };
    }

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

  static LogicalResult
  getUseDefChain(ttir::GenericOp genericOp, Value operand,
                 llvm::SmallVector<Operation *, 4> &chain) {
    llvm::TypeSwitch<Operation *, void>(operand.getDefiningOp())
        .Case(
            [&](memref::AllocOp op) { chain.emplace_back(op.getOperation()); })
        .Case([&](ttir::ViewLayoutOp op) {
          chain.emplace_back(op.getOperation());
          return getUseDefChain(genericOp, op.getInput(), chain);
        })
        .Case([&](ttir::StreamLayoutOp op) {
          return genericOp->emitOpError(
              "didn't expect to walk through a stream at this point");
        });
    return success();
  }

  // TODO this will eventually be a more complex decision
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
                                    const SequenceMapping &mapping) {
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

      // TODO this buffer type selection decision is repeated here, it was already made by
      // an analysis step
      auto bufferMemref = selectStreamBuffer(rewriter, operandMemrefType);
      auto buffer = rewriter.create<memref::AllocOp>(op.getLoc(), bufferMemref);
      assign(rewriter, buffer, req.offset, info);
      insertDealloc(rewriter, buffer, req.last, mapping);

      auto stream = rewriter.create<ttir::StreamLayoutOp>(
          op.getLoc(), streamMemref, operand.get(), buffer);

      rewriter.modifyOpInPlace(op,
                               [&]() { operand.assign(stream.getResult()); });
    }
    return success();
  }

  static memref::AllocOp findRootAlloc(Value v,
                                       llvm::SmallVector<Operation *> &path) {
    // Note: a canonicalizer pass would collapse all `ttir.view_layout` chains
    // but we don't rely on that here.
    return llvm::TypeSwitch<Operation *, memref::AllocOp>(v.getDefiningOp())
        .Case([&](memref::AllocOp op) {
          path.emplace_back(op.getOperation());
          return op;
        })
        .Case([&](ttir::ViewLayoutOp op) {
          path.emplace_back(op.getOperation());
          return findRootAlloc(op.getInput(), path);
        })
        .Case([&](ttir::StreamLayoutOp op) {
          path.emplace_back(op.getOperation());
          return findRootAlloc(op.getInput(), path);
        })
        .Default([&](Operation *op) { return nullptr; });
  }

  static void insertDealloc(RewriterBase &rewriter, memref::AllocOp allocOp,
                            Planner::SequenceT position,
                            const SequenceMapping &mapping) {
    Operation *lastOp = mapping.positionMap[position];
    if (!llvm::isa<func::ReturnOp>(lastOp)) {
      OpBuilder::InsertionGuard guard(rewriter);
      {
        rewriter.setInsertionPointAfter(lastOp);
        rewriter.create<memref::DeallocOp>(lastOp->getLoc(),
                                           allocOp.getResult());
      }
    }
  }

  // Recursive helper for `runAnalyzeBuffers(func::FuncOp funcOp...)`.
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
