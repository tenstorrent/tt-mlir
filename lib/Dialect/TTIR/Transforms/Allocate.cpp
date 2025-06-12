// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Analysis/AllocationPlanner.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallSet.h"

#include <algorithm>

#include <unordered_map>
#include <vector>

// ----------------------------------------------------------------------------
namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRFAKE

#define GEN_PASS_DEF_TTIRALLOCATE
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Helper definitions.
//===----------------------------------------------------------------------===//

#define TT_assert(condition, msg) assert((condition) && msg)

#define TTMLIR_INFO(/* fmt, args */...)                                        \
  llvm::outs() << llvm::formatv(__VA_ARGS__) << "\n"

#define TT_ALLOC_DEBUG(/* fmt, args */...)                                     \
  TTMLIR_DEBUG(ttmlir::LogComponent::Allocator, __VA_ARGS__)

#define TT_ALLOC_TRACE(/* fmt, args */...)                                     \
  TTMLIR_TRACE(ttmlir::LogComponent::Allocator, __VA_ARGS__)

inline MemorySpace getMemorySpace(MemRefType memref, MemorySpace dflt) {
  auto memSpace = memref.getMemorySpace();
  return memSpace
             ? mlir::cast<MemorySpaceAttr>(memref.getMemorySpace()).getValue()
             : dflt;
}

// ----------------------------------------------------------------------------
// FAKE

namespace {
class TTIRFake final : public impl::TTIRFakeBase<TTIRFake> {
  using Base = impl::TTIRFakeBase<TTIRFake>;

  using Base::Base;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // (1) Create streams (with their backing buffers) where needed.
    if (failed(runOnFuncs(moduleOp))) {
      signalPassFailure();
      return;
    }
  }

  struct tensor_info {
    std::string m_name;
    std::size_t m_size;
    std::int32_t m_first;
    std::int32_t m_last;
    MemorySpace m_mem_space;
  };

  LogicalResult runOnFuncs(ModuleOp moduleOp) {
    moduleOp->walk([&](func::FuncOp func) {
      if (func.isDeclaration()) {
        return WalkResult::skip();
      }
      Block &b = func.getBody().front();

      mlir::AsmState state{func};
      std::vector<tensor_info> buffers{};
      llvm::DenseMap<Value, std::int32_t> value_map{};

      for (auto arg : b.getArguments()) {
        auto arg_as_tensor = is_tensor(arg.getType());
        if (std::get<0>(arg_as_tensor)) {
          auto i = value_map.find(arg);
          TT_assert(i == value_map.end(),
                    "block args not mapped at this point");
          tensor_info ti{};
          {
            ti.m_name = as_operand_str(arg, state);
            ti.m_size = std::get<1>(arg_as_tensor);
            ti.m_first = static_cast<int32_t>(buffers.size());
            ti.m_last = -1;
            ti.m_mem_space = MemorySpace::System;
          }

          buffers.emplace_back(ti);
          value_map[arg] = ti.m_first;
        }
      }

      b.walk<WalkOrder::PreOrder>([&](Operation *op) {
        // TTMLIR_INFO("OP: {}", (*op));

        int32_t const position = static_cast<int32_t>(buffers.size());

        for (auto v : op->getOperands()) {
          auto v_as_tensor = is_tensor(v.getType());
          if (std::get<0>(v_as_tensor)) {
            auto i = value_map.find(v);
            TT_assert(i != value_map.end(), "unexpected kill");
            tensor_info &ti = buffers[i->second];
            ti.m_last = std::max(ti.m_last, position);

            TT_ALLOC_TRACE("KILLED: [{}, {}] {}", ti.m_first, ti.m_last,
                           ti.m_mem_space);
          }
        }
        for (auto r : op->getResults()) {
          auto r_as_tensor = is_tensor(r.getType());
          if (std::get<0>(r_as_tensor)) {
            // TTMLIR_INFO("\toperand size {}", std::get<1>(r_as_tensor));
            auto i = value_map.find(r);
            if (i == value_map.end()) {
              tensor_info ti{};
              {
                ti.m_name = as_operand_str(r, state);
                ti.m_size = std::get<1>(r_as_tensor);
                ti.m_first = position;
                ti.m_last = -1;
                ti.m_mem_space = MemorySpace::DeviceL1;
              }

              buffers.emplace_back(ti);
              value_map[r] = ti.m_first;
            } else {
              TTMLIR_INFO("unexpected op result mapping: {}", r);
              TT_assert(false, "unexpected op result mapping");
            }
          }
        }
      });
      TTMLIR_INFO("info.size = {}", value_map.size());

      {
        std::error_code ec;
        llvm::raw_fd_ostream out("dump.csv", ec);
        TT_assert(!ec, "couldn't open dump file");

        for (auto const &ti : buffers) {
          out << ti.m_name << ", " << ti.m_mem_space << ", " << ti.m_size
              << ", " << (ti.m_last >= 0 ? (ti.m_last - ti.m_first + 1) : -1)
              << ", " << ti.m_first << ", " << ti.m_last << "\n";
        }
      }

      return WalkResult::advance();
    });

    return llvm::success();
  }

  static std::tuple<bool, std::size_t> is_tensor(Type t) {
    if (mlir::isa<mlir::RankedTensorType>(t)) {
      return {true, tensor_size(mlir::cast<mlir::RankedTensorType>(t))};
    }

    return {false, 0};
  }

  static std::size_t tensor_size(mlir::RankedTensorType t) {
    auto shape = t.getShape();
    // llvm::outs() << "el type " << t.getElementType() << ", el size = "
    //              << tt::getElementSizeBytes(t.getElementType()) << "\n";
    const std::size_t elsize =
        std::max<std::size_t>(1, tt::getElementSizeBytes(t.getElementType()));
    return std::accumulate(shape.begin(), shape.end(), elsize,
                           std::multiplies<int64_t>());
  }

  std::string as_operand_str(Value v, mlir::AsmState &state) {
    std::string s{};
    llvm::raw_string_ostream out{s};
    v.printAsOperand(out, state);
    return s;
  }
};
} // namespace
// FAKE
// ----------------------------------------------------------------------------

//===----------------------------------------------------------------------===//
// Helper classes.
//===----------------------------------------------------------------------===//
namespace {

using AllocSizeT = AllocationPlanner::AllocSizeT;
using SequenceT = AllocationPlanner::SequenceT;

struct MemorySpaceInfo {

  MemorySpaceInfo() = default;
  MemorySpaceInfo(AllocSizeT baseAddress, AllocSizeT size, AllocSizeT alignment)
      : baseAddress(baseAddress), size(size), alignment(alignment) {
    assert(baseAddress % alignment == 0 && "expected aligned base address");
  }

  AllocSizeT baseAddress = 0;
  AllocSizeT size = 0;
  AllocSizeT alignment = 0;

  static constexpr std::size_t kMaxEnumValForMemorySpace =
      (getMaxEnumValForMemorySpace() + 1);
};

using MemorySpaces =
    std::array<MemorySpaceInfo, MemorySpaceInfo::kMaxEnumValForMemorySpace>;

struct FuncAnalysisData final : public AllocationPlanner::Context {

  using Base = AllocationPlanner::Context;

  using Base::Base;

  void add(AllocSizeT size, SequenceT first, SequenceT last,
           memref::AllocOp alloc) {
    Base::add(size, first, last);
    allocs.emplace_back(alloc);
  }

  // A list of alloc ops, parallel to `Base::records`
  std::vector<memref::AllocOp> allocs;

  // Within a func body scope, maps logical time positions (in preorder)
  // to their `Operation`s.
  std::vector<Operation *> positionMap;
  // Inverse of `positionMap`.
  DenseMap<Operation *, SequenceT> operationMap;
};

struct ModuleAnalysisData {

  ModuleAnalysisData(MemorySpaces memSpaces) : memSpaces(memSpaces) {}

  MemorySpaces memSpaces;
  DenseMap<func::FuncOp, FuncAnalysisData> funcAnalysis;
};

} // namespace
//===----------------------------------------------------------------------===//
// Pass implementation.
//===----------------------------------------------------------------------===//
namespace {
class TTIRAllocateStreams final : public OpRewritePattern<ttir::GenericOp> {
  using Base = OpRewritePattern<ttir::GenericOp>;

  using Base::Base;

  LogicalResult matchAndRewrite(ttir::GenericOp op,
                                PatternRewriter &rewriter) const final {
    bool modified = false;
    unsigned outputOperandsIndex = op.getOutputs().getBeginOperandIndex();
    ArrayAttr iteratorTypes = op.getIteratorTypes();
    for (OpOperand &operand : op->getOpOperands()) {
      bool isOutput = operand.getOperandNumber() >= outputOperandsIndex;
      AffineMap operandIndexingMap =
          mlir::cast<AffineMapAttr>(
              op.getIndexingMaps()[operand.getOperandNumber()])
              .getValue();

      if (!needsStream(operand.get(), isOutput, operandIndexingMap,
                       iteratorTypes)) {
        continue;
      }

      insertStream(rewriter, operand, op);
      modified = true;
    }

    return success(modified);
  }

  static bool needsStream(Value operand, bool isOutput,
                          AffineMap operandIndexingMap,
                          ArrayAttr iteratorTypes) {

    Operation *definingOp = operand.getDefiningOp();

    // Greedy driver fixed point reached?
    if (mlir::isa_and_nonnull<ttir::StreamLayoutOp>(definingOp)) {
      return false;
    }

    // No stream (NOC ops) will be needed if 'operand' is already
    // allocated in L1 ("alias" mode), which is currently guaranteed
    // to be the case for outputs.
    if (isOutput) {
      return false;
    }

    // A core participating in a reduction dim necessarily requires
    // non-local data movement unless it is the only core involved
    // in that dim.
    //
    // Similar logic applies to a broadcast dim.
    //
    // TODO(vroubtsov) we are currently fixing the core grid shape to be
    // equal to the output shape, hence could we not infer the *exact*
    // pattern of data movement that's not local to any core by walking
    // the operand/output affine maps?

    const auto bcastDims = operandIndexingMap.getBroadcastDims();
    const llvm::SmallSet<unsigned, 4> bcastDimIndex(bcastDims.begin(),
                                                    bcastDims.end());

    const bool operandNeedsDataMovement = llvm::any_of(
        llvm::seq(operandIndexingMap.getNumResults()),
        [&](unsigned resultIndex) {
          if (bcastDimIndex.contains(resultIndex)) {
            return true;
          }
          const auto dimPosition =
              operandIndexingMap.getDimPosition(resultIndex);
          IteratorType iteratorType =
              mlir::cast<IteratorTypeAttr>(iteratorTypes[dimPosition])
                  .getValue();
          return (iteratorType == IteratorType::Reduction);
        });
    return operandNeedsDataMovement;
  }

  static void insertStream(PatternRewriter &rewriter, OpOperand &operand,
                           ttir::GenericOp op) {
    auto memref = mlir::cast<MemRefType>(operand.get().getType());
    auto streamAttr = rewriter.getAttr<ViewLayoutAttr>(
        rewriter.getMultiDimIdentityMap(memref.getRank()));
    auto streamMemref =
        MemRefType::get(memref.getShape(), memref.getElementType(), streamAttr,
                        memref.getMemorySpace());
    auto storageAttr = ShardLayoutAttr::get(memref, /*buffers=*/1);
    auto storageMemref =
        MemRefType::get(memref.getShape(), memref.getElementType(), storageAttr,
                        memref.getMemorySpace());
    auto storage = rewriter.create<memref::AllocOp>(op.getLoc(), storageMemref);
    auto streamLayout = rewriter.create<ttir::StreamLayoutOp>(
        op.getLoc(), streamMemref, operand.get(), storage);
    rewriter.modifyOpInPlace(
        op, [&]() { operand.assign(streamLayout.getResult()); });
  }
};
} // namespace

namespace {
class TTIRAllocate final : public impl::TTIRAllocateBase<TTIRAllocate> {
  using Base = impl::TTIRAllocateBase<TTIRAllocate>;

  using Base::Base;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // (1) Create streams (with their backing buffers) where needed.
    if (failed(runAllocateStreams(moduleOp))) {
      signalPassFailure();
      return;
    }

    // (2) Solve static buffer allocation problem.
    FailureOr<ModuleAnalysisData> analysis = runAnalyzeBuffers(moduleOp);
    if (failed(analysis)) {
      signalPassFailure();
      return;
    }

    // (3) Annotate buffers with addresses and pair allocs with their deallocs.
    if (failed(runAllocateBuffers(moduleOp, *analysis))) {
      signalPassFailure();
      return;
    }
  }

  // Create/allocate streams within a module.
  LogicalResult runAllocateStreams(ModuleOp moduleOp) {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRAllocateStreams>(&getContext());
    return mlir::applyPatternsGreedily(getOperation(), std::move(patterns));
  }

  // Analyze buffer allocation needs for a module.
  FailureOr<ModuleAnalysisData> runAnalyzeBuffers(ModuleOp moduleOp) {

    SystemDescAttr systemDesc = getCurrentScopeSystemDesc(moduleOp);
    ChipDescAttr chipDesc = systemDesc.getChipDescs().front();

    MemorySpaces memSpaces = getMemorySpaces(chipDesc);
    ModuleAnalysisData moduleAnalysis(memSpaces);

    moduleOp->walk([&](func::FuncOp func) {
      if (func.isDeclaration()) {
        return WalkResult::skip();
      }

      FailureOr<FuncAnalysisData> funcAnalysis =
          runAnalyzeBuffers(func, memSpaces);
      if (failed(funcAnalysis)) {
        return WalkResult::interrupt();
      }

      moduleAnalysis.funcAnalysis[func] = std::move(*funcAnalysis);
      return WalkResult::advance();
    });

    return moduleAnalysis;
  }

  struct LivenessClosure {
    Operation *lastOp;
    SequenceT first;
    SequenceT maxLast;
  };

  // Analyze and plan buffer allocation for a func.
  FailureOr<FuncAnalysisData> runAnalyzeBuffers(func::FuncOp func,
                                                const MemorySpaces &memSpaces) {
    DeviceAttr device = lookupDevice(func);
    Block &funcBody = func.getBody().front();

    // Start with SSA liveness for `func`.

    Liveness liveness(func.getOperation());
    const LivenessBlockInfo *li = liveness.getLiveness(&funcBody);

    FuncAnalysisData analysis;

    //  (a) Build `Operation` <-> preorder position mappings for all `func` ops.
    //  (b) Collect a separate set of "ops of interest", which are
    //  `memref.alloc`s as well as certain ops that we imbue with semantics
    //   of extending liveness of their memref operands.

    llvm::DenseMap<Operation *, LivenessClosure> livenessJoinGraph;

    funcBody.walk<WalkOrder::PreOrder>([&](Operation *op) {
      const SequenceT position = analysis.positionMap.size();
      TT_ALLOC_TRACE("preorder visit @{}: {}", position, *op);

      analysis.operationMap[op] = position;
      analysis.positionMap.emplace_back(op);

      if (llvm::isa<memref::AllocOp, ttir::ViewLayoutOp, ttir::StreamLayoutOp>(
              op)) {
        TT_assert(op->getNumResults() == 1, "expected a single-result op");
        Value result = op->getResult(0);

        Operation *firstOp = li->getStartOperation(result);
        Operation *lastOp = li->getEndOperation(result, firstOp);

        livenessJoinGraph[op] = {lastOp, position, -1};
      }
    });

    // Ops in `livenessJoinGraph` form a graph of Values and their users where
    // some Values have their original SSA liveness "extended" by stream op
    // users (ttir.view_layout, ttir.stream_layout).
    //
    // We calculate the "last use position" by computing for each value
    // the max over its users over a traversal through this graph.

    for (auto &[op, ctx] : livenessJoinGraph) {
      // Initial maxLast values are from the SSA liveness calculation.
      auto i = analysis.operationMap.find(ctx.lastOp);
      TT_assert(i != analysis.operationMap.end(),
                "couldn't map the starting lastOp");
      ctx.maxLast = i->second;
    }

    for (auto &[op, ctx] : livenessJoinGraph) {
      const SequenceT maxLast = resolve(op, livenessJoinGraph);
      TT_ALLOC_DEBUG("last use of @{} extended from {} to {}", ctx.first,
                     ctx.maxLast, maxLast);
      ctx.maxLast = maxLast;
    }

    // Finish building the allocation planner context by computing
    // (aligned) sizes of all buffers under consideration.

    for (auto &[op, ctx] : livenessJoinGraph) {
      if (memref::AllocOp alloc = llvm::dyn_cast<memref::AllocOp>(op)) {
        MemRefType memrefTy = alloc.getType();
        MemorySpace memorySpace = getMemorySpace(
            memrefTy, MemorySpace::System); // Interpret unset as "host memory".

        if (!isL1MemorySpace(memorySpace)) {
          continue; // Only handling L1 space at the moment.
        }

        const AllocSizeT alignment =
            memSpaces[llvm::to_underlying(memorySpace)].alignment;
        const AllocSizeT sizeBytes = device.getMemrefSizeBytes(memrefTy, 0);
        const AllocSizeT alignedSize =
            ttmlir::utils::alignUp(sizeBytes, alignment);

        analysis.add(alignedSize, ctx.first, ctx.maxLast, alloc);
      }
    }

    const AllocationPlanner::Stats stats =
        AllocationPlanner::allocate(analysis);

    // TODO(#3378) dump this instead (usageRatio() is useful) in "debug" mode:
    // AllocationPlanner::Stats stats = AllocationPlanner::verify(analysis);
    TT_ALLOC_DEBUG("allocation planning outcome: {}", stats);

    const auto memSizeL1 =
        memSpaces[llvm::to_underlying(MemorySpace::DeviceL1)].size;
    if (stats.memUsage > memSizeL1) {
      return func.emitOpError() << "required memory usage " << stats.memUsage
                                << " exceeds memory size " << memSizeL1;
    }

    return analysis;
  }

  // Apply buffer allocation `analysis` to `moduleOp`.
  LogicalResult runAllocateBuffers(ModuleOp moduleOp,
                                   const ModuleAnalysisData &analysis) {
    auto result = moduleOp->walk([&](func::FuncOp func) {
      auto funcAnalysis = analysis.funcAnalysis.find(func);
      if (funcAnalysis == analysis.funcAnalysis.end()) {
        return WalkResult::skip();
      }

      if (failed(runAllocateBuffers(func, funcAnalysis->second,
                                    analysis.memSpaces))) {
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    return success(!result.wasInterrupted());
  }

  // Apply buffer allocation `analysis` to `func`.
  LogicalResult runAllocateBuffers(func::FuncOp func,
                                   const FuncAnalysisData &analysis,
                                   const MemorySpaces &memSpaces) {
    assert(func.getBody().hasOneBlock() &&
           "found func that didn't have one block!");

    // Augment all 'memref.alloc's in device memory with allocated addresses and
    // correct alignments.

    IRRewriter rewriter(&getContext());

    for (std::size_t t = 0; t < analysis.size(); ++t) {
      const AllocationPlanner::Record &record = analysis[t];
      memref::AllocOp alloc = analysis.allocs[t];

      MemRefType memrefTy = alloc.getType();
      MemorySpace memorySpace = getMemorySpace(
          memrefTy, MemorySpace::System); // Interpret unset as "host memory".

      if (!isL1MemorySpace(memorySpace)) {
        continue; // Only handling L1 space at the moment.
      }

      const auto &info = memSpaces[llvm::to_underlying(memorySpace)];

      const AllocSizeT alignment = info.alignment;
      const AllocSizeT address = info.baseAddress + record.offset;

      rewriter.startOpModification(alloc);
      {
        alloc.setAlignment(alignment);
        alloc->setAttr("address", rewriter.getI64IntegerAttr(address));
      };
      rewriter.finalizeOpModification(alloc);

      Operation *lastOp = analysis.positionMap[record.last];
      if (!llvm::isa<func::ReturnOp>(lastOp)) {
        rewriter.setInsertionPointAfter(lastOp);
        rewriter.create<memref::DeallocOp>(lastOp->getLoc(), alloc.getResult());
      }
    }

    return success();
  }

  // Recursive helper for `runAnalyzeBuffers(func::FuncOp func...)`.
  // Note: the overall traversal cost can be reduced by memoizing
  // final maxLast values and/or visiting Values in a reverse topological
  // sort order. This is not done at the moment.
  static SequenceT
  resolve(Operation *op,
          const llvm::DenseMap<Operation *, LivenessClosure> &graph) {

    auto opClosure = graph.find(op);
    TT_assert(opClosure != graph.end(), "malformed liveness closure graph");
    SequenceT maxLast = opClosure->second.maxLast;

    for (Operation *user : op->getResult(0).getUsers()) {
      if (graph.contains(user)) {
        if (llvm::isa<ttir::ViewLayoutOp, ttir::StreamLayoutOp>(user)) {
          maxLast = std::max(maxLast, resolve(user, graph));
        }
      }
    }

    return maxLast;
  }

  static MemorySpaces getMemorySpaces(ChipDescAttr chipDesc) {
    std::array<MemorySpaceInfo, MemorySpaceInfo::kMaxEnumValForMemorySpace>
        info;
    // Currently, we only need some slots in 'info'.
    {
      info[llvm::to_underlying(MemorySpace::DeviceL1)] = MemorySpaceInfo(
          chipDesc.getL1UnreservedBase(),
          chipDesc.getL1Size() - chipDesc.getScratchL1RegionSize(),
          chipDesc.getNocL1AddressAlignBytes());

      info[llvm::to_underlying(MemorySpace::DeviceDRAM)] = MemorySpaceInfo(
          chipDesc.getDramUnreservedBase(), chipDesc.getDramChannelSize(),
          chipDesc.getNocDRAMAddressAlignBytes());
    }
    return info;
  }
};
} // namespace

} // namespace mlir::tt::ttir
// ----------------------------------------------------------------------------
