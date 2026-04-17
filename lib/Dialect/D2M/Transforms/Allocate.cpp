// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Analysis/Allocation/Planner.h"
#include "ttmlir/Dialect/D2M/Analysis/Allocation/Utils.h"
#include "ttmlir/Dialect/D2M/Analysis/BlockFactorAnalysis.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
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
};

using OperandDefChain = llvm::SmallVector<Operation *, 4>;

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
  // To be able to plan possible pressure on L1, this precomputes
  // the type of the circular buffer this operand would have.
  MemRefType bufferType;
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
  using BufferSizePolicy = BlockFactorAnalysis::BufferSizePolicy;

  MemorySpaces memSpaces;
  ttcore::MemorySpaceAttr L1Attr = nullptr;
  ttcore::MemorySpaceAttr DRAMAttr = nullptr;
  BufferSizePolicy bufferSizePolicy = BufferSizePolicy::Auto;

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
    s << "\ttest-buffer-size-policy: " << obj.testBufferSizePolicy << "\n";
    s << "}";
    return s.str();
  }

  bool useAlwaysStreamPolicy() const {
    return (streamInsertPolicy == "always");
  }

  static std::optional<BufferSizePolicy>
  parseBufferSizePolicy(StringRef policy) {
    return llvm::StringSwitch<std::optional<BufferSizePolicy>>(policy)
        .Case("auto", BufferSizePolicy::Auto)
        .Case("min", BufferSizePolicy::Min)
        .Case("max", BufferSizePolicy::Max)
        .Default(std::nullopt);
  }

  void runOnOperation() override {
    TT_ALLOC_DEBUG("configured with options: {}", to_string(*this));

    // Set some instance state:

    ModuleOp moduleOp = getOperation();

    const std::optional<BufferSizePolicy> parsedBufferSizePolicy =
        parseBufferSizePolicy(testBufferSizePolicy);
    if (!parsedBufferSizePolicy.has_value()) {
      moduleOp.emitOpError()
          << "invalid test-buffer-size-policy '" << testBufferSizePolicy
          << "' (expected one of: auto, min, max)";
      return signalPassFailure();
    }
    bufferSizePolicy = *parsedBufferSizePolicy;

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
    // memrefs or views/streams of those).
    //
    // The IR is allowed to contain "standalone" allocs that don't feed into
    // generic ops (TODO(vroubtsov) these won't become CBs, so can this
    // assumption be removed?). Conversely, generic ops are allowed to have
    // their operands rooted at memrefs that are not allocated within `funcOp`,
    // e.g. passed in as func arguments. Therefore, the two
    // sets of memref values, (a) those allocated within `funcOp` and (b) those
    // defining generic op operands are incomparable (neither is a subset of the
    // other). We try to track this carefully.

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

    if (failed(analyzeGenericRegionAllocs(funcOp, analysis))) {
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

    if (failed(reblockGenerics(funcOp, analysis))) {
      return failure();
    }

    if (failed(insertOperandStreams(funcOp, analysis))) {
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
              // Update result types for operations that produce values
              rewriter.modifyOpInPlace(op, [&]() {
                for (OpResult result : op->getResults()) {
                  // Only update if the result is a memref type
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

    funcBody.walk([&](d2m::GenericOp genericOp) {
      SequenceT genericSeqPos = analysis.sequencing[genericOp];
      auto *genericIt = analysis.generics.find(genericOp);

      // Register in-generic allocs that back streamed operands.
      // The internal alloc needs a planner-assigned L1 address and will be
      // stamped with CBLayoutAttr, even for explicit generics, so that it can
      // be hoisted correctly as a CB later in the HoistCBAllocs pass.
      if (genericIt != analysis.generics.end()) {
        for (Region &region : genericOp->getRegions()) {
          for (const OperandContext &operandCtx : genericIt->second.operands) {
            if (!operandCtx.bufferType) {
              continue;
            }
            auto operandMemSpace =
                ttcore::getMemorySpace(operandCtx.operand->get().getType());
            if (!requiresCBAllocation(genericOp, genericIt->second, operandCtx,
                                      operandMemSpace)) {
              continue;
            }
            Value operandAlloc = d2m::GenericOp::getOperandAlloc(
                region, operandCtx.operandIndex());
            // If getOperandAlloc fails but requiresCBAllocation is true,
            // assume that is because the local buffer associated with this
            // operand might be shared by a pair of remote load and store ops
            // (buffer is not exclusive to a single operand).
            if (!operandAlloc) {
              operandAlloc = findSharedOutputBuffer(genericOp, operandCtx);
            }
            if (!operandAlloc) {
              continue;
            }
            auto allocOp = operandAlloc.getDefiningOp<memref::AllocOp>();
            if (!allocOp) {
              continue;
            }
            auto memrefType = allocOp.getType();
            MemrefValueContext &ctx = addMemrefValueContext(
                rewriter, analysis, allocOp.getResult(), memrefType, device);
            ctx.live = {genericSeqPos, genericSeqPos};
            ctx.isInsideGeneric = true;
            ctx.isMemspaceBound = true;
            ctx.allocSize[ordinal(asPlannerSpace(MemorySpace::DeviceL1))] =
                ttmlir::utils::alignUp(
                    getCBBufferSizeBytes(operandCtx.bufferType, device),
                    L1memInfo.alignment);
          }
        }
      }

      // Register all in-generic CBLayoutAttr allocs that were not already
      // registered above (e.g. scratch buffers created by
      // InsertScratchBuffers).
      for (Region &region : genericOp->getRegions()) {
        region.walk([&](memref::AllocOp allocOp) {
          auto memrefType = allocOp.getType();
          if (!mlir::isa<ttcore::CBLayoutAttr>(memrefType.getLayout())) {
            return;
          }
          // Skip if already registered (e.g. operand-backed allocs above).
          if (analysis.memrefs.count(allocOp.getResult())) {
            return;
          }
          MemrefValueContext &ctx = addMemrefValueContext(
              rewriter, analysis, allocOp.getResult(), memrefType, device);
          ctx.live = {genericSeqPos, genericSeqPos};
          ctx.isInsideGeneric = true;
          ctx.isMemspaceBound = true;
          // Total CB size = shape[0] * stride[0] (row-major, stride
          // includes element size).
          auto cbLayout =
              mlir::cast<ttcore::CBLayoutAttr>(memrefType.getLayout());
          int64_t totalSizeBytes =
              memrefType.getShape().front() * cbLayout.getStride().front();
          ctx.allocSize[ordinal(asPlannerSpace(MemorySpace::DeviceL1))] =
              ttmlir::utils::alignUp(totalSizeBytes, L1memInfo.alignment);
        });
      }
    });

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

  static bool
  shouldApplyAutoReblocking(d2m::GenericOp genericOp,
                            ArrayRef<AffineMap> indexingMaps,
                            ArrayRef<ttcore::IteratorType> iteratorTypes) {
    const std::optional<std::size_t> reductionDim =
        allocation::getSingleReductionDim(iteratorTypes);
    if (!reductionDim.has_value()) {
      // Parallel-only ops: trust BlockFactorAnalysis, which only selects
      // non-trivial factors when the baseline CB footprint exceeds the L1
      // budget.
      return true;
    }

    int64_t scalableInputCount = 0;
    for (auto [operandIndex, indexingMap] : llvm::enumerate(indexingMaps)) {
      if (genericOp.isOutputOperandIdx(operandIndex)) {
        continue;
      }
      if (indexingMap.isFunctionOfDim(*reductionDim)) {
        ++scalableInputCount;
      }
    }

    return scalableInputCount >= 2;
  }

  // Internal helper used by `analyzeGenericOps()` to create analysis entries
  // for each operand of `genericOp`.
  void createOperandContexts(FuncAnalysisData &analysis,
                             d2m::GenericOp genericOp,
                             GenericOpContext &genericCtx,
                             const BlockFactorAnalysis &blockFactorAnalysis) {
    [[maybe_unused]] AsOperandPrinter asOperand{genericOp->getParentOp()};
    [[maybe_unused]] ttcore::DeviceAttr device =
        ttcore::lookupDevice(genericOp);

    IRRewriter rewriter(genericOp->getContext());

    const bool haveIterationSpaceInfo = !genericCtx.isExplicitDatamovement;

    using OperationSet = llvm::SmallPtrSet<Operation *, 4>;

    // This is temp state to help set `MemrefValueContext::isMemspaceBound`.
    // This maps every `memref::AllocOp` to a union set of `Operation`s
    // that are seen on the use/def paths leading to their downstream
    // `d2m::GenericOp`s. Later, these sets will be intersected
    // with `memref::AllocOp->getUsers()` to detect if there are
    // any user not contained within the union sets.
    llvm::DenseMap<memref::AllocOp, OperationSet> genericUseClosure;

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

      std::tie(gridExtents, shardExtents) = getGridAndShardExtents(genericOp);
      std::tie(inputTileFactors, outputTileFactors) =
          getOperandTileShapes(genericOp);

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

      // Look up pre-computed execution blocking from the block factor
      // analysis.
      if (const auto *bfResult = blockFactorAnalysis.lookup(genericOp)) {
        blockFactors = bfResult->reblockedFactors;
        if (bufferSizePolicy == BufferSizePolicy::Auto &&
            !shouldApplyAutoReblocking(genericOp, indexingMaps,
                                       iteratorTypes)) {
          blockFactors = originalBlockFactors;
        }
      }

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

      // Right now it's OK to use the 1st root's memref because we assert that
      // all roots must be in the same memory space.
      auto memrefType = chainRoots.front().type;
      if (isOperandExemptFromStreaming(operandCtx,
                                       ttcore::getMemorySpace(memrefType))) {
        // For now, disabled `allow-l1-output-spilling` also means
        // "don't insert streams but allow them in the incoming IR".
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

      if (haveIterationSpaceInfo) {
        // To know the exact L1 memory pressure, we need to know the type/size
        // of this operand's stream if one were to be inserted.

        const AffineMap &indexingMap = indexingMaps[operandIndex];
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

        operandCtx.bufferType = getCBBufferType(
            gridShapeRescaled, shardShapeRescaled, operandType.getElementType(),
            L1Attr, numStreamBuffers);
        TT_ALLOC_TRACE("\t[operand #{}], would-be buffer "
                       "type ({} byte(s)): {}",
                       operandIndex,
                       getCBBufferSizeBytes(operandCtx.bufferType, device),
                       operandCtx.bufferType);
        TT_debug(getCBBufferSizeBytes(operandCtx.bufferType, device) > 0);
      } else {
        // If no iteration info is available, generic op should be classified as
        // explicit datamovement form.
        TT_assert(genericCtx.isExplicitDatamovement);

        // For explicit datamovement ops, the grid and shard shapes aren't
        // reblockable, so use operand device shape as-is for CB buffer type
        // computation.
        auto operandValue = genericOp->getOperand(operandIndex);
        auto gridShape = ttcore::getGridShape(operandValue);
        auto shardShape = ttcore::getShardShape(operandValue);

        const auto operandType =
            mlir::cast<MemRefType>(operand.get().getType());

        operandCtx.bufferType =
            getCBBufferType(gridShape, shardShape, operandType.getElementType(),
                            L1Attr, numStreamBuffers);
      }

      // Finally, insert `operandCtx` into `genericCtx`.
      // Even for composite view it's one OperandContext per GenericOp operand.
      genericCtx.operands.push_back(std::move(operandCtx));
    }
    TT_assert(genericCtx.operands.size() ==
              genericOp.getInputsAndOutputs().size());

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
  ///   - Each operand's context memoizes the buffer type which determines
  ///     the allocation size and shard shape for the in-generic CB alloc.
  ///
  /// Note that each decision to spill a memref alloc is binary while the stream
  /// buffer sizing decision is in theory k-ary. `testBufferSizePolicy`
  /// selects between the `min` / `max` / `auto` policies.
  ///
  LogicalResult analyzeGenericOps(func::FuncOp funcOp,
                                  FuncAnalysisData &analysis) {

    [[maybe_unused]] AsOperandPrinter asOperand{funcOp};
    [[maybe_unused]] ttcore::DeviceAttr device = ttcore::lookupDevice(funcOp);

    Block &funcBody = funcOp.getBody().front();
    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);

    // Select execution blocking for would-be operand streams.
    // `max` preserves the original blocking,
    // `min` shrinks all non-participating dims
    // `auto` considers legal divisors of the reduction shard factor and
    // rejects candidates that shrink tuned-input shards below 4 tiles.
    BlockFactorAnalysis::Options bfOpts;
    bfOpts.policy = bufferSizePolicy;
    bfOpts.numBuffers = numStreamBuffers;
    BlockFactorAnalysis blockFactorAnalysis(funcOp, bfOpts);

    [[maybe_unused]] int32_t genericsInExplicitDatamovementForm = 0;

    funcBody.walk([&](d2m::GenericOp genericOp) {
      GenericOpContext &genericCtx = createGenericContext(analysis, genericOp);

      genericsInExplicitDatamovementForm += genericCtx.isExplicitDatamovement;

      createOperandContexts(analysis, genericOp, genericCtx,
                            blockFactorAnalysis);
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
            //  - if it is the output of a generic op and the enabled pass
            //    options do not allow output spilling;
            //  - (edge case) if it has zero generic op users;
            const bool bound =
                (memspace == MemorySpace::DeviceDRAM) ||
                memrefCtx.isMemspaceBound ||
                (memrefCtx.usedForOutput && !allowL1OutputSpilling) ||
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

  /// Rebuild generic ops using the planned block factors.
  LogicalResult reblockGenerics(func::FuncOp funcOp,
                                FuncAnalysisData &analysis) {
    IRRewriter rewriter(funcOp->getContext());
    llvm::MapVector<d2m::GenericOp, GenericOpContext> updatedGenerics;

    for (auto &[genericOp, genericCtx] : analysis.generics) {
      d2m::GenericOp oldGenericOp = genericOp;
      // Skip generics whose execution shape is already final.
      if (genericCtx.isExplicitDatamovement ||
          oldGenericOp.getBlockFactorsValue() == genericCtx.reblockedFactors) {
        updatedGenerics.insert({oldGenericOp, std::move(genericCtx)});
        continue;
      }

      // Rebuild the generic so its types match allocator-chosen factors.
      rewriter.setInsertionPoint(oldGenericOp);
      FailureOr<d2m::ParallelizedGeneric> reblocked =
          oldGenericOp.withParallelization(rewriter, std::nullopt,
                                           genericCtx.reblockedFactors,
                                           /*generateReturnView=*/true);
      if (failed(reblocked)) {
        oldGenericOp.emitOpError()
            << "Allocator failed to rebuild generic op with updated block "
               "factors";
        return failure();
      }

      TT_assertv(oldGenericOp.getOutputs().size() == 1u,
                 "Allocator reblocking expects a single output operand");
      Operation *sequenceAnchor = reblocked->returnView.getOperation();
      Value newOutput = reblocked->returnView.getResult();

      // Move sequencing metadata to the new anchor op produced by the rewrite.
      SequenceT sequencePosition = analysis.sequencing[oldGenericOp];
      analysis.sequencing.positionMap[sequencePosition] = sequenceAnchor;
      analysis.sequencing.operationMap.erase(oldGenericOp.getOperation());
      analysis.sequencing.operationMap[sequenceAnchor] = sequencePosition;

      // Redirect the single externally visible output to the rebuilt view.
      if (oldGenericOp.getNumResults() > 0) {
        TT_assert(oldGenericOp.getNumResults() == 1u);
        oldGenericOp.getResult(0).replaceAllUsesWith(newOutput);
      } else {
        auto getContainingOpInBlock = [&](Operation *op) -> Operation * {
          Operation *current = op;
          while (current && current->getBlock() != sequenceAnchor->getBlock()) {
            current = current->getParentOp();
          }
          return current;
        };
        // Update nested uses inside regions of later ops in the same block.
        oldGenericOp.getOutputs().front().replaceUsesWithIf(
            newOutput, [&](OpOperand &use) {
              Operation *ownerInBlock = getContainingOpInBlock(use.getOwner());
              return ownerInBlock &&
                     ownerInBlock != oldGenericOp.getOperation() &&
                     sequenceAnchor->isBeforeInBlock(ownerInBlock);
            });
      }

      // Recompute operand def-chains against the rebuilt generic operands.
      OperandContextList oldOperandContexts = genericCtx.operands;
      GenericOpContext updatedCtx = std::move(genericCtx);
      updatedCtx.operands.clear();
      updatedCtx.operands.reserve(oldOperandContexts.size());

      MutableArrayRef<OpOperand> newOperands =
          reblocked->genericOp.getInputsAndOutputsMutable();
      TT_assert(newOperands.size() == oldOperandContexts.size());
      for (auto [operandIndex, operand] : llvm::enumerate(newOperands)) {
        OperandContext operandCtx = oldOperandContexts[operandIndex];
        operandCtx.operand = &operand;
        operandCtx.chainRoots.clear();

        SmallVector<ChainRoot> chainRoots =
            analyzeOperandDefChain(reblocked->genericOp, operand.get());
        operandCtx.chainRoots = chainRoots;
        operandCtx.primaryRoot = chainRoots.front().root;

        updatedCtx.operands.push_back(std::move(operandCtx));
      }

      // Replace the old generic entry in analysis with the rebuilt one.
      updatedGenerics.insert({reblocked->genericOp, std::move(updatedCtx)});
      rewriter.eraseOp(oldGenericOp);
    }

    analysis.generics = std::move(updatedGenerics);
    return success();
  }

  /// Sweep through all collected generic ops and make several in-place
  /// modifications:
  ///  - modify root alloc ops and any view layout ops to be in the final
  ///    memspace decided by the planner;
  ///  - create local CB allocs inside the generic for streamed operands.
  ///
  LogicalResult insertOperandStreams(func::FuncOp funcOp,
                                     const FuncAnalysisData &analysis) {
    IRRewriter rewriter(funcOp->getContext());

    const auto &L1memInfo = memSpaces[ordinal(MemorySpace::DeviceL1)];

    llvm::DenseSet<Operation *> visited;
    for (const auto &[genericOp, genericCtx] : analysis.generics) {

      // Map every pre-stream alias back to its operand index so nested
      // remote ops can be retargeted even if they still reference an older
      // def-chain value such as the root alloc.
      llvm::DenseMap<Value, int32_t> preStreamOperandIndices;

      for (const OperandContext &operandCtx : genericCtx.operands) {
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
        TT_assert(remappedMemSpace.has_value());
        const MemorySpace finalRemappedMemSpace = *remappedMemSpace;
        if (requiresCBAllocation(genericOp, genericCtx, operandCtx,
                                 *remappedMemSpace)) {

          // Record all aliases along the operand def-chain.
          // This is needed because a generic operand may have several aliases
          // along the def-chain (e.g. views, root alloc, etc.) and nested
          // remote ops may reference any of them.
          preStreamOperandIndices[operandCtx.operand->get()] =
              operandCtx.operandIndex();
          for (const ChainRoot &chainRoot : operandCtx.chainRoots) {
            preStreamOperandIndices[chainRoot.root] = operandCtx.operandIndex();
            for (Operation *opOnChain : chainRoot.defChain) {
              if (opOnChain->getNumResults() == 1) {
                preStreamOperandIndices[opOnChain->getResult(0)] =
                    operandCtx.operandIndex();
              }
            }
          }

          auto &operand = *operandCtx.operand;

          if (failed(insertStream(
                  rewriter, operand, genericOp, operandCtx,
                  (finalRemappedMemSpace == MemorySpace::DeviceDRAM ? DRAMAttr
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

      // Post stream insertion and reblocking, remote_load
      // and remote_store ops must be updated to reference
      // updated operands and CB result types.
      llvm::DenseMap<int32_t, Value> operandValueByIndex;
      llvm::DenseMap<int32_t, Type> operandCBTypeByIndex;
      for (const OperandContext &operandCtx : genericCtx.operands) {

        const auto operandIndex = operandCtx.operand->getOperandNumber();

        const auto *memrefIt2 = analysis.memrefs.find(operandCtx.primaryRoot);
        TT_debug(memrefIt2 != analysis.memrefs.end());
        const MemorySpace operandMemSpace = *memrefIt2->second.remappedMemSpace;

        if (requiresCBAllocation(genericOp, genericCtx, operandCtx,
                                 operandMemSpace)) {
          Type cbUnderlyingType;
          TT_assert(!genericOp->getRegions().empty());
          Region &region = genericOp->getRegions().front();
          TT_assert(region.hasOneBlock());
          Value operandAlloc =
              d2m::GenericOp::getOperandAlloc(region, operandIndex);
          if (!operandAlloc) {
            operandAlloc = findSharedOutputBuffer(genericOp, operandCtx);
          }
          if (operandAlloc) {
            cbUnderlyingType = operandAlloc.getType();
            if (auto cbType = mlir::dyn_cast<d2m::CBType>(cbUnderlyingType)) {
              cbUnderlyingType = cbType.getUnderlying();
            }
          }
          operandValueByIndex[operandCtx.operandIndex()] =
              operandCtx.operand->get();
          if (cbUnderlyingType) {
            operandCBTypeByIndex[operandCtx.operandIndex()] = cbUnderlyingType;
          }
        }
      }

      // Helper to update localBuffer's defining op type.
      auto updateLocalBufferType = [](d2m::RemoteLoadOp op, Type newType) {
        if (Value localBuffer = op.getLocalBuffer()) {
          if (Operation *defOp = localBuffer.getDefiningOp()) {
            if (defOp->getNumResults() == 1) {
              defOp->getResult(0).setType(newType);
            }
          }
        }
      };

      // Rewrite remote load/store ops result types and remote memrefs
      const GenericOpContext *genericCtxPtr = &genericCtx;
      for (Region &region : genericOp->getRegions()) {
        TT_assert(region.hasOneBlock());
        Block &block = region.getBlocks().front();

        block.walk([&](Operation *blockOp) {
          llvm::TypeSwitch<Operation *, void>(blockOp)
              .Case([&](d2m::RemoteLoadOp op) {
                Value oldMemref = op.getMemref();
                std::optional<int32_t> operandIndex;
                // First try pre-stream operand indices to find the new stream
                // value.
                auto aliasIt = preStreamOperandIndices.find(oldMemref);
                if (aliasIt != preStreamOperandIndices.end()) {
                  operandIndex = aliasIt->second;
                } else {
                  // Try matching the current operand value directly.
                  for (const OperandContext &operandCtx :
                       genericCtxPtr->operands) {
                    if (operandCtx.operand->get() == oldMemref) {
                      operandIndex = operandCtx.operandIndex();
                      break;
                    }
                  }
                }

                if (operandIndex) {
                  // Rewrite the memref to the current operand value.
                  if (auto valueIt = operandValueByIndex.find(*operandIndex);
                      valueIt != operandValueByIndex.end()) {
                    op.setMemRef(valueIt->second);
                  }
                }

                if (operandIndex) {
                  // Update the result type so the load result matches the
                  // stream type now associated with the operand.
                  auto typeIt = operandCBTypeByIndex.find(*operandIndex);
                  if (typeIt != operandCBTypeByIndex.end()) {
                    Type newShardType = typeIt->second;
                    op.getResult().setType(newShardType);
                    updateLocalBufferType(op, newShardType);
                    return;
                  }
                }

                if (Value localBuffer = op.getLocalBuffer()) {
                  Type localBufferType = localBuffer.getType();
                  if (localBufferType != op.getResult().getType()) {
                    op.getResult().setType(localBufferType);
                    updateLocalBufferType(op, localBufferType);
                    return;
                  }
                }

                if (op.isImplicitForm()) {
                  // Fallback: compute shard shape from device layout when not
                  // in the CB type map.
                  Value memref = op.getMemref();
                  auto deviceLayout = ttcore::getDeviceLayout(memref);
                  if (deviceLayout) {
                    auto shapedType = mlir::cast<ShapedType>(memref.getType());
                    auto shardShape = deviceLayout.getShardShape(shapedType);
                    MemRefType newShardType = MemRefType::get(
                        shardShape, shapedType.getElementType(), nullptr,
                        rewriter.getAttr<ttcore::MemorySpaceAttr>(
                            ttcore::MemorySpace::DeviceL1));
                    op.getResult().setType(newShardType);
                    updateLocalBufferType(op, newShardType);
                  }
                }
              })
              .Case([&](d2m::RemoteStoreOp op) {
                Value oldMemref = op.getMemref();
                std::optional<int32_t> operandIndex;
                // First try pre-stream operand indices to find the new value.
                auto aliasIt = preStreamOperandIndices.find(oldMemref);
                if (aliasIt != preStreamOperandIndices.end()) {
                  operandIndex = aliasIt->second;
                } else {
                  // Try matching the current operand value directly.
                  for (const OperandContext &operandCtx :
                       genericCtxPtr->operands) {
                    if (operandCtx.operand->get() == oldMemref) {
                      operandIndex = operandCtx.operandIndex();
                      break;
                    }
                  }
                }

                if (!operandIndex) {
                  return;
                }

                // Rewrite the memref to the current operand value.
                auto valueIt = operandValueByIndex.find(*operandIndex);
                if (valueIt != operandValueByIndex.end()) {
                  op.setMemRef(valueIt->second);
                }
              });
        });
      }
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

  LogicalResult insertStream(RewriterBase &rewriter, OpOperand &operand,
                             d2m::GenericOp op,
                             const OperandContext &operandCtx,
                             ttcore::MemorySpaceAttr remappedMemspace,
                             const MemorySpaceInfo &info,
                             const SequenceMapping &sequencing) {
    const MemRefType bufferType = operandCtx.bufferType;
    TT_debug(bufferType != nullptr);

    // Compute the shard shape for the CB alloc inside the generic region.
    auto shardShape =
        mlir::cast<ttcore::DeviceLayoutInterface>(bufferType.getLayout())
            .getShardShape(bufferType);
    Type elementType = bufferType.getElementType();

    // Replace the in-generic alloc with a CBLayoutAttr-tagged alloc that
    // represents the circular buffer for this operand.
    rewriter.startOpModification(op);
    {
      for (Region &region : op->getRegions()) {
        TT_assert(region.hasOneBlock());

        const auto operandIndex = operandCtx.operand->getOperandNumber();
        Value oldTensor = d2m::GenericOp::getOperandAlloc(region, operandIndex);
        if (!oldTensor) {
          oldTensor = findSharedOutputBuffer(op, operandCtx);
        }
        if (oldTensor && oldTensor.getDefiningOp()) {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(oldTensor.getDefiningOp());
          if (auto getCbOp =
                  mlir::dyn_cast<d2m::GetCBOp>(oldTensor.getDefiningOp())) {
            // Update the get_cb op's result type to reflect the new shard
            // shape, preserving it as a CBType.
            auto oldCbType =
                mlir::cast<d2m::CBType>(getCbOp.getResult().getType());
            auto oldUnderlying =
                oldCbType.template getUnderlyingAs<MemRefType>();
            auto newUnderlying =
                MemRefType::get(shardShape, elementType,
                                /*layout=*/MemRefLayoutAttrInterface{},
                                oldUnderlying.getMemorySpace());
            getCbOp.getResult().setType(
                d2m::CBType::get(getCbOp.getContext(), newUnderlying));
          } else {
            Value newValue;
            if (auto oldAllocOp = mlir::dyn_cast<memref::AllocOp>(
                    oldTensor.getDefiningOp())) {
              auto oldMemRefType = mlir::cast<MemRefType>(oldTensor.getType());
              auto cbLayout = ttcore::CBLayoutAttr::get(
                  bufferType.getContext(), shardShape,
                  ttcore::getElementSizeBytes(elementType), numStreamBuffers);
              auto newAllocOp = rewriter.create<memref::AllocOp>(
                  oldTensor.getLoc(),
                  MemRefType::get(shardShape, elementType, cbLayout,
                                  oldMemRefType.getMemorySpace()));
              // Transfer address and alignment from the old alloc (assigned
              // by the planner in assignAllocAddresses).
              if (auto addrAttr = oldAllocOp->getAttr("address")) {
                newAllocOp->setAttr("address", addrAttr);
              }
              if (auto alignAttr = oldAllocOp.getAlignmentAttr()) {
                newAllocOp.setAlignmentAttr(alignAttr);
              }
              newValue = newAllocOp.getResult();
            } else {
              auto newEmptyOp = rewriter.create<mlir::tensor::EmptyOp>(
                  oldTensor.getLoc(), shardShape, elementType);
              newValue = newEmptyOp.getResult();
            }
            rewriter.replaceAllUsesWith(oldTensor, newValue);
            rewriter.eraseOp(oldTensor.getDefiningOp());
          }
        }
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

  /// Walk the operand's def chain and check if any ViewLayoutOp has a
  /// non-identity affine map.
  /// @return `true` if any ViewLayoutOp in the chain has a non-identity map.
  static bool isNonTrivialView(const OperandContext &operandCtx) {
    for (ChainRoot chainRoot : operandCtx.chainRoots) {
      for (Operation *op : chainRoot.defChain) {
        if (auto view = llvm::dyn_cast<d2m::ViewLayoutOp>(op)) {
          if (!view.getRemapping().isIdentity()) {
            return true;
          }
        }
      }
    }
    return false;
  }

  struct SharedLoadStoreInfo {
    Value localBuffer;
    Value loadMemref;
    Value storeMemref;
  };

  static std::optional<SharedLoadStoreInfo>
  getSharedLoadStoreInfo(Value localBuffer) {
    if (!localBuffer) {
      return std::nullopt;
    }

    SharedLoadStoreInfo info{localBuffer, Value(), Value()};
    for (Operation *userOp : localBuffer.getUsers()) {
      if (auto loadOp = mlir::dyn_cast<d2m::RemoteLoadOp>(userOp)) {
        if (loadOp.getLocalBuffer() == localBuffer) {
          info.loadMemref = loadOp.getMemref();
          continue;
        }
      }
      if (auto storeOp = mlir::dyn_cast<d2m::RemoteStoreOp>(userOp)) {
        if (storeOp.getLocalBuffer() == localBuffer) {
          info.storeMemref = storeOp.getMemref();
        }
      }
    }

    if (!info.loadMemref || !info.storeMemref ||
        info.loadMemref == info.storeMemref) {
      return std::nullopt;
    }

    return info;
  }

  static bool isOperandAliasOfValue(const OperandContext &operandCtx,
                                    Value value) {
    if (!value) {
      return false;
    }
    if (operandCtx.operand->get() == value) {
      return true;
    }
    for (const ChainRoot &chainRoot : operandCtx.chainRoots) {
      if (chainRoot.root == value) {
        return true;
      }
      for (Operation *opOnChain : chainRoot.defChain) {
        if (opOnChain->getNumResults() == 1 &&
            opOnChain->getResult(0) == value) {
          return true;
        }
      }
    }
    return false;
  }

  static const OperandContext *
  findOperandContextForAlias(const GenericOpContext &genericCtx, Value alias,
                             const OperandContext &excludeOperandCtx) {
    for (const OperandContext &candidateCtx : genericCtx.operands) {
      if (&candidateCtx == &excludeOperandCtx) {
        continue;
      }
      if (isOperandAliasOfValue(candidateCtx, alias)) {
        return &candidateCtx;
      }
    }
    return nullptr;
  }

  static const OperandContext *
  findSharedPeerOperandContext(d2m::GenericOp genericOp,
                               const GenericOpContext &genericCtx,
                               const OperandContext &operandCtx) {
    Value operandValue = operandCtx.operand->get();
    for (Region &region : genericOp->getRegions()) {
      TT_assert(region.hasOneBlock());
      const OperandContext *peerCtx = nullptr;
      WalkResult result = region.front().walk([&](Operation *op) {
        Value localBuffer;
        if (auto loadOp = mlir::dyn_cast<d2m::RemoteLoadOp>(op)) {
          if (loadOp.getMemref() != operandValue) {
            return WalkResult::advance();
          }
          localBuffer = loadOp.getLocalBuffer();
        } else if (auto storeOp = mlir::dyn_cast<d2m::RemoteStoreOp>(op)) {
          if (storeOp.getMemref() != operandValue) {
            return WalkResult::advance();
          }
          localBuffer = storeOp.getLocalBuffer();
        } else {
          return WalkResult::advance();
        }

        std::optional<SharedLoadStoreInfo> sharedInfo =
            getSharedLoadStoreInfo(localBuffer);
        if (!sharedInfo) {
          return WalkResult::advance();
        }

        Value peerAlias = sharedInfo->loadMemref == operandValue
                              ? sharedInfo->storeMemref
                              : sharedInfo->loadMemref;
        peerCtx = findOperandContextForAlias(genericCtx, peerAlias, operandCtx);
        if (!peerCtx) {
          return WalkResult::advance();
        }

        return WalkResult::interrupt();
      });
      (void)result;
      if (peerCtx) {
        return peerCtx;
      }
    }

    return nullptr;
  }

  static Value findSharedOutputBuffer(d2m::GenericOp genericOp,
                                      const OperandContext &operandCtx) {
    if (!operandCtx.isOutput) {
      return Value();
    }

    Value operandValue = operandCtx.operand->get();
    for (Region &region : genericOp->getRegions()) {
      TT_assert(region.hasOneBlock());
      Value sharedBuffer;
      WalkResult result = region.front().walk([&](d2m::RemoteStoreOp storeOp) {
        if (storeOp.getMemref() != operandValue) {
          return WalkResult::advance();
        }
        Value localBuffer = storeOp.getLocalBuffer();
        if (getSharedLoadStoreInfo(localBuffer)) {
          sharedBuffer = localBuffer;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      (void)result;
      if (sharedBuffer) {
        return sharedBuffer;
      }
    }

    return Value();
  }

  bool requiresCBAllocation(d2m::GenericOp genericOp,
                            const GenericOpContext &genericCtx,
                            const OperandContext &operandCtx,
                            MemorySpace memspace) const {

    if (isOperandExemptFromStreaming(operandCtx, memspace)) {
      return false;
    }

    return inferStreamRequirement(genericOp, genericCtx, operandCtx, memspace);
  }

  /// @return `true` if `operandCtx` is an output that is exempt from stream
  /// insertion. Currently, this is true for outputs when L1 output spilling is
  /// disabled and the output is not a non-trivial view.
  bool isOperandExemptFromStreaming(const OperandContext &operandCtx,
                                    MemorySpace memspace) const {
    if (isNonTrivialView(operandCtx)) {
      return false;
    }
    return operandCtx.isOutput && !allowL1OutputSpilling &&
           memspace != MemorySpace::DeviceDRAM;
  }

  /// @return `true` if `genericOp` requires a stream
  /// for operand @`operandIndex` based on the available indexing space
  /// information
  bool inferBaseStreamRequirement(d2m::GenericOp genericOp,
                                  const OperandContext &operandCtx,
                                  MemorySpace memspace) const {
    if (useAlwaysStreamPolicy()) {
      return true;
    }

    // Non-trivial views need a stream to represent the implied data movement.
    if (isNonTrivialView(operandCtx)) {
      return true;
    }

    const uint32_t operandIndex = operandCtx.operandIndex();

    // DRAM operands always need streams because data must physically
    // move between DRAM and L1 circular buffers.
    if (memspace == MemorySpace::DeviceDRAM) {
      return true;
    }

    // Early out if in explicit datamovement form (no indexing map info
    // available).
    if (genericOp.isExplicitDatamovementForm()) {
      return false;
    }

    const AffineMap indexingMap = genericOp.getIndexingMap(operandIndex);

    const auto broadcastDims = indexingMap.getBroadcastDims();
    const auto iteratorTypes = genericOp.getIteratorTypesValue();

    for (std::size_t resultIndex = 0; resultIndex < indexingMap.getNumResults();
         ++resultIndex) {
      if (llvm::is_contained(broadcastDims, resultIndex)) {
        return true;
      }
      if (iteratorTypes[indexingMap.getDimPosition(resultIndex)] ==
          ttcore::IteratorType::Reduction) {
        return true;
      }
    };

    return false;
  }

  bool inferStreamRequirement(d2m::GenericOp genericOp,
                              const GenericOpContext &genericCtx,
                              const OperandContext &operandCtx,
                              MemorySpace memspace) const {
    const bool thisOperandNeedsStream =
        inferBaseStreamRequirement(genericOp, operandCtx, memspace);
    if (!thisOperandNeedsStream) {
      return false;
    }

    const OperandContext *sharedPeerCtx =
        findSharedPeerOperandContext(genericOp, genericCtx, operandCtx);
    if (!sharedPeerCtx) {
      return true;
    }

    const MemorySpace peerMemspace =
        ttcore::getMemorySpace(sharedPeerCtx->operand->get().getType());
    return inferBaseStreamRequirement(genericOp, *sharedPeerCtx, peerMemspace);
  }

  static std::tuple</* input */ SmallVector<int64_t>,
                    /* output */ SmallVector<int64_t>>
  getOperandTileShapes(d2m::GenericOp genericOp) {
    const Type inputElementType =
        mlir::cast<MemRefType>(
            genericOp.getInputsAndOutputs().front().getType())
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
        mlir::cast<MemRefType>(genericOp.getInputsAndOutputs().back().getType())
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
