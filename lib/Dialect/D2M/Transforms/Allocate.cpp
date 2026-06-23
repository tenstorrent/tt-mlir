// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Analysis/Allocation/Planner.h"
#include "ttmlir/Dialect/D2M/Analysis/Allocation/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
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

using namespace d2m::allocation;

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
  // The type of how this value is seen within the scope, either
  // the actual type of a `memref.alloc` or the result of a ttnn bridge cast.
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

  // `true` iff this alloc is defined inside a d2m::GenericOp region.
  // Such allocs are L1-only (no spilling) and do not participate in
  // stream insertion or dealloc insertion at the func-body level.
  bool isInsideGeneric = false;
  // `true` iff this in-generic alloc is a speculative stream buffer that was
  // skipped during aliasing because its root intermediate might be spilled.
  // Once the planner decides, the post-planner step will either alias it away
  // (root stayed in L1) or keep it for streaming (root was spilled).
  bool speculativeStreamBuffer = false;
};

using OperandDefChain = llvm::SmallVector<Operation *, 4>;
using OperationSet = llvm::SmallPtrSet<Operation *, 4>;

// The single root discovered by `analyzeOperandDefChain`.
// Normal operands produce one; composite views produce one per input tensor.
struct ChainRoot {
  Value root = nullptr;
  MemRefType type = nullptr;
  OperandDefChain defChain;
};

struct OperandContext {
  // Link to the operand in the incoming IR.
  OpOperand *operand = nullptr;

  int32_t operandIndex() const {
    TT_assert(operand != nullptr);
    return operand->getOperandNumber();
  }

  // The Value (either a memref.alloc or a block arg) that is the source of
  // this operand's data, possibly through a view/cast chain. For composite
  // operands this is the first root.
  Value primaryRoot;
  // This collects the set of ops defining an operand all the way to its root
  // `memref::AllocOp` or block arg. For composite operands all chains and roots
  // are collected separately.
  SmallVector<ChainRoot> chainRoots;
  // `true` is if this corresponds to a generic op output.
  bool isOutput = false;
};

// A map linking `OperandContext`s with their originating `Value`s (defined
// by `memref.alloc`s or passed as block args).
using OperandContextList = llvm::SmallVector<OperandContext, 4>;

struct GenericOpContext {
  // Context info for each of this generic op's operands, in declaration
  // order (parallel to the generic op's operand list).
  OperandContextList operands;
  // Pre-computed block factors for the modified op.
  SmallVector<int64_t> reblockedFactors;
  // Generic ops in "explicit datamovement" form have no static
  // iteration space (indexing maps, etc) information.
  bool isExplicitDatamovement = false;
};

struct SequenceMapping {
  // Within a func body scope, maps logical time (postorder) positions
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
  llvm::MapVector<mlir::Value, MemrefValueContext> memrefs;
  llvm::MapVector<d2m::GenericOp, GenericOpContext> generics;
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
    // std::stringstream s;
    std::string str;
    llvm::raw_string_ostream s{str};
    s << "{\n";
    s << "\tnum-stream-buffers: " << obj.numStreamBuffers << "\n";
    s << "\tallow-l1-output-spilling: " << obj.allowL1OutputSpilling << "\n";
    s << "\tstream-insert-policy: " << obj.streamInsertPolicy << "\n";
    s << "\tforce-spill-to-dram-if-legal: " << obj.forceSpillToDramIfLegal
      << "\n";
    s << "\tavailable-l1-addr-range: "
      << asSeq(llvm::to_vector(obj.availableL1AddrRange)) << "\n";
    s << "\ttest-assume-l1-capacity: " << obj.testAssumeL1Capacity << "\n";
    s << "}";
    return s.str();
  }

  bool useAlwaysStreamPolicy() const {
    return (streamInsertPolicy == "always");
  }

  void runOnOperation() override {
    TT_ALLOC_DEBUG("configured with options: {}", to_string(*this));

    // Set some instance state:

    ModuleOp moduleOp = getOperation();

    memSpaces = [this, moduleOp]() {
      ttcore::SystemDescAttr systemDesc =
          ttcore::getCurrentScopeSystemDesc(moduleOp);
      ttcore::ChipDescAttr chipDesc = systemDesc.getChipDescs().front();
      return getMemorySpaces(chipDesc, llvm::to_vector(availableL1AddrRange),
                             testAssumeL1Capacity);
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

    TT_ALLOC_DEBUG("processing '{}()' ...", funcOp.getSymName());

    // This pass works with two sets of IR objects: memrefs (defined by
    // memref.alloc's) and generic ops (with operands that are either raw
    // memrefs or views of those).
    //
    // The IR is allowed to contain "standalone" allocs that don't feed into
    // generic ops. Conversely, generic ops are allowed to have
    // their operands rooted at memrefs that are not allocated within `funcOp`,
    // e.g. passed in as func arguments. Therefore, the two
    // sets of memref values, (a) those allocated within `funcOp` and (b) those
    // defining generic op operands are incomparable (neither is a subset of the
    // other). We try to track this carefully. We also allow memref allocs
    // inside generic ops that are not tied to any operand loads/stores (e.g.
    // scratch buffers).

    FuncAnalysisData analysis;

    if (failed(validateGenericOpForms(funcOp))) {
      return failure();
    }

    if (failed(remapGenericRegionAllocs(funcOp))) {
      return failure();
    }

    if (failed(analyzeLiveness(funcOp, analysis))) {
      return failure();
    }

    if (failed(analyzeGenericOps(funcOp, analysis))) {
      return failure();
    }

    if (failed(materializeAliasedLoadStore(funcOp, analysis))) {
      return failure();
    }

    if (failed(analyzeGenericRegionAllocs(funcOp, analysis))) {
      return failure();
    }

    if (failed(prepareMemoryPlanner(funcOp, analysis))) {
      return failure();
    }

    if (failed(runMemoryPlanner(funcOp, analysis))) {
      return failure();
    }

    if (failed(materializeUnspilledIntermediateAliasedLoadStore(funcOp,
                                                                analysis))) {
      return failure();
    }

    if (failed(assignAllocAddresses(funcOp, analysis))) {
      return failure();
    }

    if (failed(updateMemorySpaces(funcOp, analysis))) {
      return failure();
    }

    if (failed(insertDeallocs(funcOp, analysis))) {
      return failure();
    }

    return success();
  }

  /// Validate that all generic ops are in affine blocked form. Generic ops
  /// in explicit datamovement form are exempt from this check.
  ///
  LogicalResult validateGenericOpForms(func::FuncOp funcOp) {
    LogicalResult result = success();

    funcOp.walk([&](d2m::GenericOp genericOp) {
      if (genericOp.isExplicitDatamovementForm() ||
          genericOp.isExternalSymbolForm() || !genericOp.isUnifiedForm()) {
        return;
      }
      if (!genericOp.isAffineBlockedForm()) {
        genericOp.emitOpError("expected generic op to be in affine blocked "
                              "form before allocation");
        result = failure();
      }
    });

    return result;
  }

  /// Walk all GenericOp operations and remap any memref.alloc operations
  /// inside their regions to L1 memory space. Also trace the uses of each
  /// alloc to find compute operations and update their result types to match
  /// the alloc's memref type.
  ///
  LogicalResult remapGenericRegionAllocs(func::FuncOp funcOp) {
    IRRewriter rewriter(funcOp->getContext());
    Block &funcBody = funcOp.getBody().front();

    funcBody.walk([&](d2m::GenericOp genericOp) {
      for (Region &region : genericOp->getRegions()) {
        region.walk([&](memref::AllocOp allocOp) {
          // First remap the alloc to L1
          remap(rewriter, allocOp, MemorySpace::DeviceL1);

          // Get the result type of the alloc
          Value allocResult = allocOp.getResult();
          MemRefType allocType = mlir::cast<MemRefType>(allocResult.getType());

          // Trace uses of the alloc result to find compute ops
          llvm::SmallVector<Operation *> worklist;
          llvm::DenseSet<Operation *> visited;

          // Initialize worklist with direct users
          for (Operation *user : allocResult.getUsers()) {
            worklist.push_back(user);
          }

          // Process the worklist to trace through all uses
          while (!worklist.empty()) {
            Operation *op = worklist.pop_back_val();

            if (!visited.insert(op).second) {
              continue; // Already visited
            }

            // Check if this is a compute operation with results
            if (op->getNumResults() > 0) {
              // Re-type only the DPS result that aliases this alloc.
              auto dpsOp = mlir::dyn_cast<DestinationStyleOpInterface>(op);
              llvm::SmallVector<unsigned, 2> resultsToRetype;
              if (dpsOp && op->getNumResults() == dpsOp.getNumDpsInits()) {
                for (OpOperand &init : dpsOp.getDpsInitsMutable()) {
                  if (init.get() == allocResult) {
                    resultsToRetype.push_back(
                        dpsOp.getTiedOpResult(&init).getResultNumber());
                  }
                }
              } else {
                // Preserve legacy behavior for non-DPS/asymmetric ops.
                for (unsigned i = 0; i < op->getNumResults(); ++i) {
                  resultsToRetype.push_back(i);
                }
              }

              rewriter.modifyOpInPlace(op, [&]() {
                for (unsigned idx : resultsToRetype) {
                  OpResult result = op->getResult(idx);
                  if (mlir::isa<MemRefType>(result.getType())) {
                    result.setType(allocType);
                  }
                }
              });

              // Continue tracing through this operation's results
              for (OpResult result : op->getResults()) {
                for (Operation *userOp : result.getUsers()) {
                  worklist.push_back(userOp);
                }
              }
            }
          }
        });
      }
    });

    return success();
  }

  /// Populate `analysis.sequencing` and `analysis.memrefs`:
  ///
  /// - Discover `memref.alloc`s already present in `funcOp` and collect them
  ///   into `analysis.memrefs`.
  /// - For each memref value:
  ///   - Calculate effective liveness by extending `mlir::Liveness`
  ///     ranges with uses by view ops.
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

    // (a) Build `Operation` <-> postorder position mappings for the
    //  (unmodified) `funcOp` IR.
    // (b) Collect a separate set of "ops of interest", which are
    // `memref.alloc`s as well as certain ops that we imbue with semantics
    //  of extending liveness of their memref operands.

    LivenessClosureGraph livenessJoinGraph;

    funcBody.walk<WalkOrder::PostOrder>([&](Operation *op) {
      const SequenceT position = analysis.sequencing.size();

      analysis.sequencing.operationMap[op] = position;
      analysis.sequencing.positionMap.emplace_back(op);

      if (op->getBlock() != &funcBody) {
        return;
      }

      if (llvm::isa<memref::AllocOp, d2m::ViewLayoutOp, d2m::CompositeViewOp,
                    d2m::CreateGlobalSemaphoreOp>(op)) {
        // Skip memref.alloc operations that have a genericOp as parent
        if (llvm::isa<memref::AllocOp>(op) &&
            op->getParentOfType<d2m::GenericOp>()) {
          return;
        }

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
    // some Values have their original SSA liveness "extended" by view op
    // users (d2m.view_layout).
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

        const MemRefType memrefType =
            llvm::cast<MemRefType>(allocOp->getResultTypes().front());
        MemrefValueContext &memrefCtx = addMemrefValueContext(
            rewriter, analysis, allocOp, memrefType, device);
        memrefCtx.live = closure.live;
      }
    }

    TT_ALLOC_DEBUG("collected {} memref context(s)", analysis.memrefs.size());
    TT_debug(analysis.sequencing.valid());

    return success();
  }

  /// Walk all GenericOp regions and register every memref.alloc found inside
  /// them in `analysis.memrefs`.  These allocs are pinned to L1 (no spilling)
  /// and use the parent GenericOp's sequence position as their live range.
  ///
  /// Must run after `analyzeGenericOps` so that `operandCtx.bufferType` is
  /// available for computing the double-buffered allocation size of operand
  /// allocs that will be streamed.
  LogicalResult analyzeGenericRegionAllocs(func::FuncOp funcOp,
                                           FuncAnalysisData &analysis) {
    IRRewriter rewriter(funcOp->getContext());
    ttcore::DeviceAttr device = ttcore::lookupDevice(funcOp);
    Block &funcBody = funcOp.getBody().front();
    const auto &L1memInfo = memSpaces[ordinal(MemorySpace::DeviceL1)];

    LogicalResult result = success();
    funcBody.walk([&](d2m::GenericOp genericOp) {
      SequenceT genericSeqPos = analysis.sequencing[genericOp];

      // Register all in-generic allocs.
      // The internal alloc needs a planner-assigned L1 address and will be
      // stamped with CBLayoutAttr, even for explicit generics, so that it can
      // be hoisted correctly as a CB later in the HoistCBAllocs pass.
      genericOp->walk([&](memref::AllocOp allocOp) {
        std::optional<int32_t> numBuffers = std::nullopt;
        if (allocOp->getAttr("d2m.scratch_buffer")) {
          numBuffers = 1;
        } else if (allocOp->getAttr("d2m.synchronized_buffer")) {
          // Skip allocating, this is a contract from fusion and compute
          // lowering saying we will not actually use this buffer and it's just
          // a placeholder, so we can safely skip it.
          if (!allocOp->getAttr("d2m.compute_intermediate")) {
            numBuffers =
                allocOp->getAttrOfType<IntegerAttr>("d2m.synchronized_buffer")
                    .getInt();
          }
        } else {
          // We can allow this in the future but asserting for now to check it's
          // not used.
          allocOp->emitError(
              "Alloc op not tagged with any recognized attributes");
          result = failure();
          return WalkResult::interrupt();
        }

        if (numBuffers.has_value()) {
          MemrefValueContext &ctx =
              addMemrefValueContext(rewriter, analysis, allocOp.getResult(),
                                    allocOp.getType(), device);
          ctx.live = {genericSeqPos, genericSeqPos};
          ctx.isInsideGeneric = true;
          ctx.isMemspaceBound = true;
          ctx.speculativeStreamBuffer =
              allocOp->hasAttr("d2m.speculative_stream_buffer");
          if (ctx.speculativeStreamBuffer) {
            allocOp->removeAttr("d2m.speculative_stream_buffer");
          }
          TT_assertv(ttcore::getMemorySpace(allocOp.getType()) ==
                         ttcore::MemorySpace::DeviceL1,
                     "generic allocs must be allocated in L1");
          ctx.allocSize[ordinal(asPlannerSpace(MemorySpace::DeviceL1))] =
              ttmlir::utils::alignUp(
                  numBuffers.value() *
                      getMemrefSizeBytes(allocOp.getType(), device),
                  L1memInfo.alignment);
        }

        return WalkResult::advance();
      });
    });

    if (failed(result)) {
      return result;
    }

    TT_ALLOC_DEBUG("collected {} in-generic memref alloc(s)",
                   llvm::count_if(analysis.memrefs, [](const auto &entry) {
                     return entry.second.isInsideGeneric;
                   }));

    return success();
  }

  // Internal helper for populating `analysis.memrefs`, either
  // during the initial sweep for `memref.alloc`s or during analysis
  // of generic operands.
  MemrefValueContext &addMemrefValueContext(RewriterBase &rewriter,
                                            FuncAnalysisData &analysis,
                                            Value root, MemRefType memrefType,
                                            ttcore::DeviceAttr device) {
    TT_assert(root != nullptr);
    TT_assert(memrefType != nullptr);

    auto [entry, inserted] = analysis.memrefs.try_emplace(root);
    MemrefValueContext &memrefCtx = entry->second;
    if (inserted) {
      memrefCtx.type = memrefType;
      memrefCtx.allocSize =
          getAlignedAllocSizes(rewriter, memrefType, memSpaces, device);
    } else {
      TT_debug(memrefCtx.type == memrefType);
      TT_debug(llvm::all_of(memrefCtx.allocSize,
                            [](auto size) { return size > 0; }));
    }

    return memrefCtx;
  }

  // Internal helper used by `analyzeGenericOps()` to create a new analysis
  // entry for `genericOp`.
  GenericOpContext &createGenericContext(FuncAnalysisData &analysis,
                                         d2m::GenericOp genericOp) {
    GenericOpContext &genericCtx = analysis.generics[genericOp];

    // Detect generic ops in "explicit datamovement" form: they do not have
    // iteration space (indexing maps, etc) information and can't be analyzed
    // by this pass. However, a `GenericOpContext` entry must still be created
    // at this point.
    genericCtx.isExplicitDatamovement = genericOp.isExplicitDatamovementForm();

    return genericCtx;
  }

  // Internal helper used by `analyzeGenericOps()` to create analysis entries
  // and allocation usage closure for each operand of `genericOp`.
  void createOperandContexts(
      FuncAnalysisData &analysis, d2m::GenericOp genericOp,
      GenericOpContext &genericCtx,
      llvm::DenseMap<memref::AllocOp, OperationSet> &genericUseClosure) {
    [[maybe_unused]] AsOperandPrinter asOperand{genericOp->getParentOp()};
    [[maybe_unused]] ttcore::DeviceAttr device =
        ttcore::lookupDevice(genericOp);

    IRRewriter rewriter(genericOp->getContext());

    const bool haveIterationSpaceInfo = !genericCtx.isExplicitDatamovement;

    SmallVector<AffineMap> indexingMaps;
    SmallVector<ttcore::IteratorType> iteratorTypes;

    SmallVector<int64_t> gridExtents;
    SmallVector<int64_t> shardExtents;

    if (haveIterationSpaceInfo) {
      // Do some analysis common to all `genericOp` operands.

      indexingMaps = genericOp.getIndexingMapsValue();
      iteratorTypes = genericOp.getIteratorTypesValue();

      const std::size_t rank = genericOp.getNumDims();

      std::tie(gridExtents, shardExtents) = getGridAndShardExtents(genericOp);

      SmallVector<int64_t> blockFactors = genericOp.getBlockFactorsValue();
      const SmallVector<int64_t> originalBlockFactors = blockFactors;

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

      for (std::size_t d = 0; d < rank; ++d) {
        TT_assert(blockFactors[d] % originalBlockFactors[d] == 0);
        const int64_t rescaling = blockFactors[d] / originalBlockFactors[d];
        gridExtents[d] *= rescaling;
        TT_assert(shardExtents[d] % rescaling == 0);
        shardExtents[d] /= rescaling;
      }
      genericCtx.reblockedFactors = blockFactors;
      TT_debug((gridExtents.size() == rank and shardExtents.size() == rank));
    }

    // Do some operand-specific analysis.

    for (auto [operandIndex, operand] :
         llvm::enumerate(genericOp.getInputsAndOutputsMutable())) {

      OperandContext operandCtx;

      operandCtx.operand = &operand;
      operandCtx.isOutput = genericOp.isOutputOperandIdx(operandIndex);

      // Find the operand's root memref(s) and the op chain(s) that links to
      // them.
      SmallVector<ChainRoot> chainRoots =
          analyzeOperandDefChain(genericOp, operand.get());
      TT_assert(!chainRoots.empty());
      operandCtx.chainRoots = chainRoots;

      // Use the first root as the primary root for the OperandContext.
      operandCtx.primaryRoot = chainRoots.front().root;

      if (TT_DEBUG_ENABLED()) {
        for ([[maybe_unused]] const auto &[rootIdx, chainRoot] :
             llvm::enumerate(chainRoots)) {
          TT_ALLOC_DEBUG(
              "\tadding memref value ctx (root {}/{}): root {}, memref type {}",
              rootIdx, chainRoots.size(), asOperand(chainRoot.root),
              chainRoot.type);
        }
      }

      // Register ALL roots in analysis.memrefs and genericUseClosure.
      for (const ChainRoot &chainRoot : chainRoots) {
        MemrefValueContext &rootMemrefCtx = addMemrefValueContext(
            rewriter, analysis, chainRoot.root, chainRoot.type, device);

        rootMemrefCtx.genericUsers.insert(genericOp);
        rootMemrefCtx.isMemspaceBound |= genericCtx.isExplicitDatamovement;
        rootMemrefCtx.usedForOutput |= operandCtx.isOutput;

        if (memref::AllocOp allocOp =
                chainRoot.root.getDefiningOp<memref::AllocOp>()) {
          OperationSet &allocOpGenericUsers = genericUseClosure[allocOp];
          allocOpGenericUsers.insert(genericOp.getOperation());
          allocOpGenericUsers.insert(chainRoot.defChain.begin(),
                                     chainRoot.defChain.end());
        }
      }

      // Finally, insert `operandCtx` into `genericCtx`.
      // Even for composite view it's one OperandContext per GenericOp operand.
      genericCtx.operands.push_back(std::move(operandCtx));
    }
    TT_assert(genericCtx.operands.size() ==
              genericOp.getInputsAndOutputs().size());
  }

  /// Populate `analysis.generics`:
  ///
  /// - Collect `d2m.generic`s present in `funcOp` into `analysis.generics`.
  /// - For each `d2m.generic`, build a block of operand context structs
  ///   parallel to the op's IR operands:
  ///   - Each operand's context links to the def/use chain rooted at its
  ///     defining memref value (used for overwriting the memspaces of all
  ///     contained ops should this memref be chosen for spilling).
  ///   - Each operand's context memoizes the buffer type which determines
  ///     the allocation size and shard shape for the in-generic CB alloc.
  ///
  LogicalResult analyzeGenericOps(func::FuncOp funcOp,
                                  FuncAnalysisData &analysis) {

    [[maybe_unused]] AsOperandPrinter asOperand{funcOp};
    [[maybe_unused]] ttcore::DeviceAttr device = ttcore::lookupDevice(funcOp);

    Block &funcBody = funcOp.getBody().front();
    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);

    // Collect the full union set of generic users plus def/use-chain ops for
    // each root alloc across the function. After all generics have contributed,
    // these sets will be intersected with `memref::AllocOp->getUsers()` to tell
    // whether an alloc has true non-generic external users and is thus bound.
    llvm::DenseMap<memref::AllocOp, OperationSet> genericUseClosure;
    [[maybe_unused]] int32_t genericsInExplicitDatamovementForm = 0;

    funcBody.walk([&](d2m::GenericOp genericOp) {
      GenericOpContext &genericCtx = createGenericContext(analysis, genericOp);

      genericsInExplicitDatamovementForm += genericCtx.isExplicitDatamovement;

      createOperandContexts(analysis, genericOp, genericCtx, genericUseClosure);
    });

    // `genericUseClosure` is complete, use it to update
    // `MemrefValueContext::isMemspaceBound`.
    for (auto &[allocOp, users] : genericUseClosure) {
      for (Operation *user : allocOp->getUsers()) {
        // Nested remote ops are part of the surrounding generic op's operand
        // path even though they also directly use the root alloc, skip them.
        if (llvm::isa<d2m::RemoteLoadOp, d2m::RemoteStoreOp>(user)) {
          if (auto parentGeneric = user->getParentOfType<d2m::GenericOp>();
              parentGeneric && users.contains(parentGeneric.getOperation())) {
            continue;
          }
        }
        if (!llvm::isa<func::ReturnOp>(user) && !users.contains(user)) {
          analysis.memrefs[allocOp.getResult()].isMemspaceBound = true;
        }
      }
    }

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

    TT_ALLOC_DEBUG("collected {} generic op context(s) ({} explicit "
                   "datamovement)",
                   analysis.generics.size(),
                   genericsInExplicitDatamovementForm);

    return success();
  }

  /// Each `analysis.memrefs` entry defines an allocation planner decision
  /// variable. These can be of different origins:
  ///
  /// (1) A memref defined by a `memref.alloc` backing a generic op operand.
  /// (2) A memref that backs a generic op operand but is not defined by an
  ///     op inside `funcOp` (i.e. passed in as a block argument).
  /// (3) A memref defined by a "standalone" `memref.alloc` that needs no
  ///     generic op streaming but will still need a valid L1/DRAM memory
  ///     address assigned.
  ///
  LogicalResult prepareMemoryPlanner(func::FuncOp funcOp,
                                     FuncAnalysisData &analysis) {
    [[maybe_unused]] AsOperandPrinter asOperand{funcOp};

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
      memrefCtx.varIndex =
          problem.def([&, &memref = memref,
                       &memrefCtx = memrefCtx](Planner::VariableBuilder &b) {
            // If `memref` is being defined inside `funcOp` and is initially
            // placed in L1, it will require scratch memory to hold its tensor
            // data.
            if (memref.getDefiningOp<memref::AllocOp>() &&
                memspace == MemorySpace::DeviceL1) {
              memrefCtx.reqIndex = b.request(
                  PlannerSpace::Scratch,
                  memrefCtx.allocSize[ordinal(asPlannerSpace(memspace))],
                  memrefCtx.live.first, memrefCtx.live.last);
            }

            // This decision variable must be bound to its incoming memspace
            // in any of these cases:
            //  - if it is placed in DRAM *explicitly*;
            //  - if the incoming IR indicates that this alloc should be pinned
            //    to its current memspace in any other explicit way (aggregated
            //    into `isMemspaceBound`);
            //  - if it is a generic output and output spilling is disabled, or
            //    if it is a terminal generic output. When output spilling is
            //    enabled, only outputs with downstream generic users remain
            //    spillable because the allocator can remap their def/use chain
            //    and insert producer/consumer streams as needed;
            //  - (edge case) if it has zero generic op users;
            const bool isIntermediateMemref =
                memrefCtx.usedForOutput && memrefCtx.genericUsers.size() > 1;
            const bool bindOutputToL1 =
                memrefCtx.usedForOutput &&
                (!allowL1OutputSpilling || !isIntermediateMemref);
            const bool bound = (memspace == MemorySpace::DeviceDRAM) ||
                               memrefCtx.isMemspaceBound || bindOutputToL1 ||
                               memrefCtx.genericUsers.empty();
            const bool forceSpillToDram = forceSpillToDramIfLegal && !bound;
            if (bound) {
              b.bind(asPlannerSpace(memspace));
            } else if (forceSpillToDram) {
              b.bind(PlannerSpace::Spill);
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

  /// Sweep through all collected generic ops and modify root
  /// alloc ops and any view layout ops to be in the final
  /// memspace decided by the planner.
  LogicalResult updateMemorySpaces(func::FuncOp funcOp,
                                   FuncAnalysisData &analysis) {
    IRRewriter rewriter(funcOp->getContext());

    llvm::DenseSet<Operation *> visited;
    for (auto &[genericOp, genericCtx] : analysis.generics) {
      for (OperandContext &operandCtx : genericCtx.operands) {
        std::optional<MemorySpace> remappedMemSpace;
        for (const ChainRoot &chainRoot : operandCtx.chainRoots) {
          const auto *memrefIt = analysis.memrefs.find(chainRoot.root);
          TT_debug(memrefIt != analysis.memrefs.end());
          const MemrefValueContext &memrefCtx = memrefIt->second;

          TT_assert(memrefCtx.remappedMemSpace.has_value());
          const MemorySpace rootRemappedMemSpace = *memrefCtx.remappedMemSpace;
          if (!remappedMemSpace.has_value()) {
            remappedMemSpace = rootRemappedMemSpace;
          } else {
            TT_assert(*remappedMemSpace == rootRemappedMemSpace);
          }

          for (Operation *opOnChain : chainRoot.defChain) {
            if (!visited.insert(opOnChain).second) {
              // Assigning final memspace is idempotent, but no need to do this
              // repeatedly.
              continue;
            }
            llvm::TypeSwitch<Operation *, void>(opOnChain)
                .Case([&](memref::AllocOp op) {
                  remap(rewriter, op, rootRemappedMemSpace);
                })
                .Case([&](d2m::ViewLayoutOp op) {
                  remap(rewriter, op, rootRemappedMemSpace);
                })
                .Case([&](d2m::CompositeViewOp op) {
                  remap(rewriter, op, rootRemappedMemSpace);
                });
          }
        }
      }
    }

    return success();
  }

  void markSynchronizedBuffer(RewriterBase &rewriter, Value buffer) const {
    if (auto allocOp = buffer.getDefiningOp<memref::AllocOp>()) {
      allocOp->setAttr("d2m.synchronized_buffer",
                       rewriter.getI32IntegerAttr(numStreamBuffers));
    }
  }

  static void markSpeculativeStreamBuffer(RewriterBase &rewriter,
                                          Value buffer) {
    if (auto allocOp = buffer.getDefiningOp<memref::AllocOp>()) {
      allocOp->setAttr("d2m.speculative_stream_buffer", rewriter.getUnitAttr());
    }
  }

  static bool isIntermediateGenericOutput(const FuncAnalysisData &analysis,
                                          const OperandContext &operandCtx) {
    return llvm::any_of(operandCtx.chainRoots, [&](const ChainRoot &chainRoot) {
      const auto *memrefIt = analysis.memrefs.find(chainRoot.root);
      TT_debug(memrefIt != analysis.memrefs.end());
      const MemrefValueContext &memrefCtx = memrefIt->second;
      return memrefCtx.usedForOutput && memrefCtx.genericUsers.size() > 1;
    });
  }

  LogicalResult materializeAliasedLoadStore(func::FuncOp funcOp,
                                            FuncAnalysisData &analysis) {
    IRRewriter rewriter(funcOp->getContext());
    auto markStreamBufferIfNeeded = [&](Value localBuffer) {
      auto allocOp = localBuffer.getDefiningOp<memref::AllocOp>();
      if (!allocOp || allocOp->getAttr("d2m.scratch_buffer") ||
          allocOp->getAttr("d2m.synchronized_buffer")) {
        return;
      }
      allocOp->setAttr("d2m.synchronized_buffer",
                       rewriter.getI32IntegerAttr(numStreamBuffers));
    };

    for (const auto &[genericOp, genericCtx] : analysis.generics) {
      const auto &genericOpRef = genericOp;
      for (const OperandContext &operandCtx : genericCtx.operands) {
        genericOpRef->walk([&](RemoteLoadOp remoteLoadOp) {
          if (remoteLoadOp.getMemref() == operandCtx.operand->get() &&
              isAliasedLoad(remoteLoadOp)) {
            if (allowL1OutputSpilling &&
                isIntermediateGenericOutput(analysis, operandCtx)) {
              markSynchronizedBuffer(rewriter, remoteLoadOp.getLocalBuffer());
              markSpeculativeStreamBuffer(rewriter,
                                          remoteLoadOp.getLocalBuffer());
              return;
            }
            // Replace memref.alloc with operand alias op
            auto *allocOp = remoteLoadOp.getLocalBuffer().getDefiningOp();
            rewriter.setInsertionPoint(allocOp);
            rewriter.replaceOpWithNewOp<d2m::OperandAliasOp>(
                allocOp, allocOp->getResultTypes(), remoteLoadOp.getMemref());
          }
        });
      }
    }

    for (const auto &[genericOp, genericCtx] : analysis.generics) {
      const auto &genericOpRef = genericOp;
      for (const OperandContext &operandCtx : genericCtx.operands) {
        genericOpRef->walk([&](RemoteStoreOp remoteStoreOp) {
          //  Check we don't already have aliased load since we can't alias
          //  DMA on both sides
          if (mlir::isa<d2m::OperandAliasOp>(
                  remoteStoreOp.getLocalBuffer().getDefiningOp())) {
            return WalkResult::advance();
          }

          if (remoteStoreOp.getMemref() == operandCtx.operand->get() &&
              isAliasedStore(remoteStoreOp)) {
            if (allowL1OutputSpilling &&
                isIntermediateGenericOutput(analysis, operandCtx)) {
              markSynchronizedBuffer(rewriter, remoteStoreOp.getLocalBuffer());
              markSpeculativeStreamBuffer(rewriter,
                                          remoteStoreOp.getLocalBuffer());
              return WalkResult::advance();
            }
            auto *allocOp = remoteStoreOp.getLocalBuffer().getDefiningOp();
            TT_assertv(mlir::isa<memref::AllocOp>(allocOp),
                       "Expected memref::AllocOp");
            rewriter.setInsertionPoint(allocOp);
            rewriter.replaceOpWithNewOp<d2m::OperandAliasOp>(
                allocOp, allocOp->getResultTypes(), remoteStoreOp.getMemref());
          }
          return WalkResult::advance();
        });
      }
    }

    // Any remaining remote load/store local buffers could not be aliased (for
    // example, after reblocking introduced operand views), so they are real
    // stream buffers that must be visible to the allocator.
    for (const auto &[genericOp, _] : analysis.generics) {
      genericOp->walk([&](RemoteLoadOp remoteLoadOp) {
        markStreamBufferIfNeeded(remoteLoadOp.getLocalBuffer());
      });
      genericOp->walk([&](RemoteStoreOp remoteStoreOp) {
        markStreamBufferIfNeeded(remoteStoreOp.getLocalBuffer());
      });
    }

    // Remove in-generic allocs that are not used
    for (const auto &[genericOp, genericCtx] : analysis.generics) {
      genericOp->walk([&](memref::AllocOp allocOp) {
        if (allocOp.getResult().getUsers().empty()) {
          rewriter.eraseOp(allocOp);
        }
      });
    }

    return success();
  }

  LogicalResult
  materializeUnspilledIntermediateAliasedLoadStore(func::FuncOp funcOp,
                                                   FuncAnalysisData &analysis) {
    // After the planner has made the decisions, alias speculative stream
    // buffers whose root intermediate stayed in L1.
    IRRewriter rewriter(funcOp->getContext());

    for (const auto &[genericOp, genericCtx] : analysis.generics) {
      genericOp->walk([&](RemoteLoadOp remoteLoadOp) {
        auto allocOp =
            remoteLoadOp.getLocalBuffer().getDefiningOp<memref::AllocOp>();
        if (!allocOp) {
          return;
        }
        auto *it = analysis.memrefs.find(allocOp.getResult());
        if (it == analysis.memrefs.end() ||
            !it->second.speculativeStreamBuffer) {
          return;
        }

        Value rootMemref = remoteLoadOp.getMemref();
        auto *rootIt = analysis.memrefs.find(rootMemref);
        if (rootIt != analysis.memrefs.end() &&
            rootIt->second.remappedMemSpace == MemorySpace::DeviceDRAM) {
          return;
        }

        rewriter.setInsertionPoint(allocOp);
        analysis.memrefs.erase(allocOp.getResult());
        rewriter.replaceOpWithNewOp<d2m::OperandAliasOp>(
            allocOp, allocOp->getResultTypes(), remoteLoadOp.getMemref());
      });
    }

    for (const auto &[genericOp, genericCtx] : analysis.generics) {
      genericOp->walk([&](RemoteStoreOp remoteStoreOp) {
        if (mlir::isa<d2m::OperandAliasOp>(
                remoteStoreOp.getLocalBuffer().getDefiningOp())) {
          return WalkResult::advance();
        }

        auto allocOp =
            remoteStoreOp.getLocalBuffer().getDefiningOp<memref::AllocOp>();
        if (!allocOp) {
          return WalkResult::advance();
        }
        auto *it = analysis.memrefs.find(allocOp.getResult());
        if (it == analysis.memrefs.end() ||
            !it->second.speculativeStreamBuffer) {
          return WalkResult::advance();
        }

        Value rootMemref = remoteStoreOp.getMemref();
        auto *rootIt = analysis.memrefs.find(rootMemref);
        if (rootIt != analysis.memrefs.end() &&
            rootIt->second.remappedMemSpace == MemorySpace::DeviceDRAM) {
          return WalkResult::advance();
        }

        rewriter.setInsertionPoint(allocOp);
        analysis.memrefs.erase(allocOp.getResult());
        rewriter.replaceOpWithNewOp<d2m::OperandAliasOp>(
            allocOp, allocOp->getResultTypes(), remoteStoreOp.getMemref());
        return WalkResult::advance();
      });
    }

    return success();
  }

  /// memref.allocs inserted for added streams have been matched with deallocs
  /// at the point of stream insertion by an earlier step; we also need
  /// to do the same for all memref.allocs that define Values collected into
  /// `analysis.memrefs`.
  ///
  LogicalResult insertDeallocs(func::FuncOp funcOp,
                               const FuncAnalysisData &analysis) {
    IRRewriter rewriter(funcOp->getContext());

    for (auto &[memref, memrefCtx] : analysis.memrefs) {
      // In-generic allocs are bounded by the enclosing region, not
      // the func body — skip func-level dealloc insertion.
      // NB: must check before getDefiningOp because insertOperandStreams
      // may have erased the original alloc and replaced it with a new one.
      if (memrefCtx.isInsideGeneric) {
        continue;
      }
      memref::AllocOp allocOp = memref.getDefiningOp<memref::AllocOp>();
      if (!allocOp) {
        continue;
      }
      if (!ttcore::isDeviceMemorySpace(
              ttcore::getMemorySpace(memrefCtx.type, MemorySpace::System))) {
        continue;
      }

      insertDealloc(rewriter, allocOp, memrefCtx.live.last,
                    analysis.sequencing);
    }

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

  /// @return `true` iff the operand's defining chain includes a non-identity
  /// view remapping.
  static bool isNonTrivialView(const Value &operand) {
    return allocation::hasNonTrivialView(operand);
  }

  /// @return `true` if the remote_load/store for this operand does not require
  /// real DMA. (i.e. can be aliased with it's corresponding operand). This is
  /// the case when the operand:
  ///  - does not have a non-trivial view layout
  ///  - is in L1 and not DRAM
  ///  - is not in an explicit datamovement form and the dimensions
  ///  corresponding to the load/store indices
  ///    are not broadcast or reduction.
  /// This is a common function shared by isAliasedLoad and isAliasedStore.
  bool canAliasOperand(d2m::GenericOp genericOp, Value genericOperand) const {
    // Check if operand requires aliasing
    bool isOutput = llvm::find_if(genericOp.getOutputs(), [&](Value operand) {
                      return operand == genericOperand;
                    }) != genericOp.getOutputs().end();
    auto memspace = ttcore::getMemorySpace(
        mlir::cast<MemRefType>(genericOperand.getType()));
    if (!isNonTrivialView(genericOperand) && isOutput &&
        !allowL1OutputSpilling && memspace != MemorySpace::DeviceDRAM) {
      return true;
    }

    if (useAlwaysStreamPolicy()) {
      return false;
    }

    // Non-trivial views need a stream to represent the implied data movement.
    if (isNonTrivialView(genericOperand)) {
      return false;
    }

    // DRAM operands always need streams because data must physically
    // move between DRAM and L1 circular buffers.
    if (memspace == MemorySpace::DeviceDRAM) {
      return false;
    }

    // Early out if in explicit datamovement form (no indexing map info
    // available).
    if (genericOp.isExplicitDatamovementForm()) {
      return false;
    }

    auto operandIndex = genericOp.getOperandIndex(genericOperand);
    const AffineMap indexingMap = genericOp.getIndexingMap(operandIndex);
    const auto broadcastDims = indexingMap.getBroadcastDims();
    const auto iteratorTypes = genericOp.getIteratorTypesValue();

    for (std::size_t resultIndex = 0; resultIndex < indexingMap.getNumResults();
         ++resultIndex) {
      if (llvm::is_contained(broadcastDims, resultIndex)) {
        return false;
      }
      if (iteratorTypes[indexingMap.getDimPosition(resultIndex)] ==
          ttcore::IteratorType::Reduction) {
        return false;
      }
    };

    return true;
  }

  bool isAliasedLoad(RemoteLoadOp loadOp) {
    return canAliasOperand(loadOp->getParentOfType<d2m::GenericOp>(),
                           loadOp.getMemref()) &&
           !loadOp.isMcast();
  }

  bool isAliasedStore(RemoteStoreOp storeOp) {
    return canAliasOperand(storeOp->getParentOfType<d2m::GenericOp>(),
                           storeOp.getMemref()) &&
           storeOp.getStartDevice().empty();
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

  /// Walk the chain of ops starting with the one defining `operand` and
  /// terminating in either a `memref::AllocOp` (that backs the operand's
  /// tensor), a block argument (if the tensor comes from an outside scope),
  /// or a ttnn bridge cast. The walk may involve going through a chain of
  /// `view_layout`s.
  ///
  /// Besides the above, determine a few more things:
  ///  1. the ChainRoot values that terminates each chain (used as a unique key
  ///  for the chain);
  ///  2. the "effective memref type" to associate with the ChainRoot (either
  ///  the actual type of its Value or the type it is being cast to by a ttnn
  ///  bridge cast).
  ///
  static SmallVector<ChainRoot> analyzeOperandDefChain(d2m::GenericOp genericOp,
                                                       Value operand) {
    [[maybe_unused]] AsOperandPrinter asOperand{genericOp->getParentOp()};

    OperandDefChain chain;
    MemRefType type = nullptr;

    Value value = operand;
    Operation *definingOp = value.getDefiningOp();

    while (definingOp != nullptr) {
      chain.emplace_back(definingOp);

      if (auto op = llvm::dyn_cast<memref::AllocOp>(definingOp)) {
        type = mlir::cast<MemRefType>(op->getResultTypes().front());
        return {{value, type, chain}};
      }
      if (auto op = llvm::dyn_cast<ttir::TTNNMetalLayoutCastOp>(definingOp)) {
        value = op.getInput();
        type = mlir::cast<MemRefType>(op->getResultTypes().front());
        return {{value, type, chain}};
      }

      if (auto op = llvm::dyn_cast<d2m::CompositeViewOp>(definingOp)) {
        // Recurse into each input of the composite view to collect all the
        // chains & roots. Prefix each child chain with the current chain to
        // preserve per-root context.
        SmallVector<ChainRoot> allRoots;
        for (Value input : op.getCompositeInputs()) {
          SmallVector<ChainRoot> inputRoots =
              analyzeOperandDefChain(genericOp, input);
          for (ChainRoot &inputRoot : inputRoots) {
            OperandDefChain prefixedChain;
            prefixedChain.reserve(chain.size() + inputRoot.defChain.size());
            prefixedChain.append(chain.begin(), chain.end());
            prefixedChain.append(inputRoot.defChain.begin(),
                                 inputRoot.defChain.end());
            inputRoot.defChain = std::move(prefixedChain);

            allRoots.push_back(std::move(inputRoot));
          }
        }
        return allRoots;
      }

      if (auto op = llvm::dyn_cast_or_null<d2m::ViewOpInterface>(definingOp)) {
        value = op.getInput();
      } else if (auto op =
                     llvm::dyn_cast<d2m::CreateGlobalSemaphoreOp>(definingOp)) {
        value = op.getInput();
      } else {
        TT_assertv(false,
                   "unexpected op '{}' in the def chain for operand '{}'",
                   definingOp->getName(), asOperand(operand));
      }

      definingOp = value.getDefiningOp();
    }

    // If `type` was not discovered above it means the walk ended in a block
    // arg.
    if (type == nullptr) {
      const auto arg = mlir::dyn_cast<BlockArgument>(value);
      TT_assertv(arg != nullptr, "expected to reach a known cast or block arg");
      type = mlir::cast<MemRefType>(arg.getType());
    }

    return {{value, type, chain}};
  }

  // Factor out defaults passed into DeviceAttr::getMemrefSizeBytes()
  // for operand memrefs.
  static int64_t getMemrefSizeBytes(MemRefType bufferType,
                                    ttcore::DeviceAttr device) {
    // A tighter size calculation is possible for memrefs that don't map to
    // CBs which we don't attempt here except to ignore `buffers` multiplier.
    return device.getMemrefSizeBytes(bufferType, 0, false);
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
      MemRefType remapped = remap(rewriter, type, memspace);
      int64_t sizeBytes = getMemrefSizeBytes(remapped, device);

      // This function returns the aligned allocation size on each core.
      // - For block-sharded L1 tensors it's the size of a single shard.
      // - For DRAM tensors that live on the 1D DRAM core grid we should return
      //   the per-bank memory footprint. Since shards are distributed to the
      //   DRAM banks in the round-robin pattern, we align up the number of
      //   shards to the number of DRAM banks (bank-aligned padding).
      if (memspace == MemorySpace::DeviceDRAM) {
        if (auto layout =
                mlir::dyn_cast<ttcore::ShardLayoutAttr>(remapped.getLayout())) {
          const int64_t nShards =
              ttmlir::utils::volume(layout.getGridShape(remapped));
          const int64_t nDramBanks = device.getDramGrid().getGridVolume();
          const int64_t shardsPerBank =
              ttmlir::utils::alignUp(nShards, nDramBanks) / nDramBanks;
          sizeBytes *= shardsPerBank;
        }
      }

      sizes[ordinal(placement)] = ttmlir::utils::alignUp(
          sizeBytes, memSpaces[ordinal(memspace)].alignment);
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

  /// @return 'op' with given memory space override
  static void remap(RewriterBase &rewriter, d2m::CompositeViewOp op,
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
        if (llvm::isa<d2m::ViewLayoutOp, d2m::CompositeViewOp,
                      d2m::CreateGlobalSemaphoreOp>(user)) {
          last = std::max(last, resolve(user, graph));
        }
      }
    }

    return last;
  }

  static MemorySpaces
  getMemorySpaces(ttcore::ChipDescAttr chipDesc,
                  SmallVector<AllocSizeT> l1AddrRangeOverride,
                  AllocSizeT l1CapacityOverride) {
    MemorySpaces info;
    {
      // Currently, we only need L1 and DRAM slots in 'info'.

      const bool L1AddrRangeOverrideSet = l1AddrRangeOverride.size() == 2;
      const bool l1CapacityOverrideSet = l1CapacityOverride > 0;

      TT_assertv(
          !(L1AddrRangeOverrideSet && l1CapacityOverrideSet),
          "overriding both L1 addr range and L1 capacity is not allowed");

      if (L1AddrRangeOverrideSet) {
        TT_assert(l1AddrRangeOverride[0] >= chipDesc.getL1UnreservedBase());
        TT_assert(l1AddrRangeOverride[1] <= chipDesc.getL1Size());

        info[ordinal(MemorySpace::DeviceL1)] =
            MemorySpaceInfo(l1AddrRangeOverride[0], l1AddrRangeOverride[1],
                            chipDesc.getNocL1AddressAlignBytes());
      } else {
        const AllocSizeT l1AddrLimit =
            l1CapacityOverrideSet
                ? (chipDesc.getL1UnreservedBase() + l1CapacityOverride)
                : chipDesc.getL1Size();

        info[ordinal(MemorySpace::DeviceL1)] =
            MemorySpaceInfo(chipDesc.getL1UnreservedBase(), l1AddrLimit,
                            chipDesc.getNocL1AddressAlignBytes());
      }

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
