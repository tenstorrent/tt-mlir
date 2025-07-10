// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
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
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <variant>

// ----------------------------------------------------------------------------
namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRALLOCATE
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Helper definitions.
//===----------------------------------------------------------------------===//

#define TT_ALLOC_DEBUG(/* fmt, args */...)                                     \
  TTMLIR_DEBUG(ttmlir::LogComponent::Allocator, __VA_ARGS__)

#define TT_ALLOC_TRACE(/* fmt, args */...)                                     \
  TTMLIR_TRACE(ttmlir::LogComponent::Allocator, __VA_ARGS__)

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

using AllocSizeT = AllocationPlanner::AllocSizeT;
using SequenceT = AllocationPlanner::SequenceT;

struct MemorySpaceInfo {

  MemorySpaceInfo() = default;
  MemorySpaceInfo(AllocSizeT baseAddress, AllocSizeT maxAddress,
                  AllocSizeT alignment)
      : baseAddress(baseAddress), maxAddress(maxAddress), alignment(alignment) {
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

struct MemorySpaceContext final : public AllocationPlanner::Context {

  using Base = AllocationPlanner::Context;

  void add(AllocSizeT size, SequenceT first, SequenceT last,
           memref::AllocOp alloc) {
    Base::add(size, first, last);
    allocs.emplace_back(alloc);
  }

  // A list of alloc ops mapping to a given memspace, parallel
  // to 'Base::records'.
  std::vector<memref::AllocOp> allocs;
};

using MemorySpaceContexts =
    std::array<MemorySpaceContext, MemorySpaceInfo::kMaxEnumValForMemorySpace>;

struct FuncAnalysisData final {

  void add(AllocSizeT size, SequenceT first, SequenceT last,
           memref::AllocOp alloc, ttcore::MemorySpace memorySpace) {
    contexts[llvm::to_underlying(memorySpace)].add(size, first, last, alloc);
  }

  // Memory planner contexts, one per mem space.
  MemorySpaceContexts contexts;

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

struct FuncAnalysisData2 final : public AllocationPlanner::Context {};

struct ModuleAnalysisData2 {

  DenseMap<func::FuncOp, FuncAnalysisData2> funcAnalysis;
};

} // namespace
//===----------------------------------------------------------------------===//
// Pass implementation.
//===----------------------------------------------------------------------===//
namespace {
class TTIRAllocateStreams final : public OpRewritePattern<ttir::GenericOp> {
public:
  TTIRAllocateStreams(MLIRContext *context, unsigned numStreamBuffers)
      : OpRewritePattern<ttir::GenericOp>(context),
        numStreamBuffers(numStreamBuffers) {}

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
          ttcore::IteratorType iteratorType =
              mlir::cast<ttcore::IteratorTypeAttr>(iteratorTypes[dimPosition])
                  .getValue();
          return (iteratorType == ttcore::IteratorType::Reduction);
        });
    return operandNeedsDataMovement;
  }

  void insertStream(PatternRewriter &rewriter, OpOperand &operand,
                    ttir::GenericOp op) const {
    auto memref = mlir::cast<MemRefType>(operand.get().getType());
    auto streamAttr = rewriter.getAttr<ttcore::ViewLayoutAttr>(
        rewriter.getMultiDimIdentityMap(memref.getRank()));
    auto streamMemref =
        MemRefType::get(memref.getShape(), memref.getElementType(), streamAttr,
                        memref.getMemorySpace());
    auto storageAttr =
        ttcore::ShardLayoutAttr::get(memref, /*buffers=*/numStreamBuffers);
    auto storageMemref =
        MemRefType::get(memref.getShape(), memref.getElementType(), storageAttr,
                        memref.getMemorySpace());
    auto storage = rewriter.create<memref::AllocOp>(op.getLoc(), storageMemref);
    auto streamLayout = rewriter.create<ttir::StreamLayoutOp>(
        op.getLoc(), streamMemref, operand.get(), storage);
    rewriter.modifyOpInPlace(
        op, [&]() { operand.assign(streamLayout.getResult()); });
  }

  unsigned numStreamBuffers;
};
} // namespace

namespace {
class TTIRAllocate final : public impl::TTIRAllocateBase<TTIRAllocate> {
  using Base = impl::TTIRAllocateBase<TTIRAllocate>;

  using Base::Base;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // (0)
    if (failed(runAnalyzeOperands(moduleOp))) {
      signalPassFailure();
      return;
    }

    // // (1) Create streams (with their backing buffers) where needed.
    // if (failed(runAllocateStreams(moduleOp))) {
    //   signalPassFailure();
    //   return;
    // }

    // // (2) Solve static buffer allocation problem.
    // FailureOr<ModuleAnalysisData> analysis = runAnalyzeBuffers(moduleOp);
    // if (failed(analysis)) {
    //   signalPassFailure();
    //   return;
    // }

    // // (3) Annotate buffers with addresses and pair allocs with their
    // deallocs. if (failed(runAllocateBuffers(moduleOp, *analysis))) {
    //   signalPassFailure();
    //   return;
    // }
  }

  // ----------------------

  // Create/allocate streams within a module.
  FailureOr<ModuleAnalysisData2> runAnalyzeOperands(ModuleOp moduleOp) {
    ModuleAnalysisData2 moduleAnalysis;
    if (moduleOp
            ->walk([&](func::FuncOp funcOp) {
              if (funcOp.isDeclaration()) {
                return WalkResult::skip();
              }

              FailureOr<FuncAnalysisData2> funcAnalysis =
                  runAnalyzeOperands(funcOp);
              if (failed(funcAnalysis)) {
                return WalkResult::interrupt();
              }

              moduleAnalysis.funcAnalysis[funcOp] = std::move(*funcAnalysis);
              return WalkResult::advance();
            })
            .wasInterrupted()) {
      return failure();
    }

    return moduleAnalysis;
  }

  struct AliasMode {
    AllocSizeT alignedSize;
    SequenceT first;
    SequenceT last;
  };

  struct StreamMode {
    AllocSizeT bufferSize; // TODO this will become a set of choices
    SequenceT first;
    SequenceT last;
    bool isL1Stream;
  };

  FailureOr<FuncAnalysisData2> runAnalyzeOperands(func::FuncOp funcOp) {
    Block &funcBody = funcOp.getBody().front();

    FuncAnalysisData2 analysis;

    // Collect all memref.allocs that are root defs of all generic operands.
    // (For streams and views, traverse through the input.) Note that such
    // a memref.alloc can also have non-generic op users; presence of those
    // will make the alloc ineligible for memspace remapping.

    llvm::DenseMap<memref::AllocOp, llvm::SmallPtrSet<Operation *, 4>> allocOps;

    funcBody.walk([&](ttir::GenericOp genericOp) {
      for (Value operand : genericOp->getOperands()) {
        llvm::SmallVector<Operation *> path;
        memref::AllocOp allocOp = findRootAlloc(operand, path);
        if (allocOp) {
          llvm::outs() << "path length " << path.size() << "\n";
          allocOps[allocOp].insert(path.begin(), path.end());
        }
      }
    });
    llvm::outs() << "found " << allocOps.size() << " root alloc(s)\n";

    // Check 'allocOp' immediate users against the set seen in the paths
    // leading to generic op operands.

    for (auto &[allocOp, pathSet] : allocOps) {
      for (Operation *user : allocOp->getUsers()) {
        if (!pathSet.contains(user)) {
          llvm::outs() << "user not in pathset: " << (*user) << "\n";
          // TODO remove it/mark ineligible for remapping
        }
      }
    }

    // Find all alloc sites and analyse sets of their possible placements.

    funcBody.walk<WalkOrder::PreOrder>([&](memref::AllocOp allocOp) {

    });

    return analysis;
  }

  LogicalResult runRemapOperands(func::FuncOp funcOp) {
    Block &funcBody = funcOp.getBody().front();
    IRRewriter rewriter(funcOp->getContext());

    funcBody.walk([&](ttir::GenericOp genericOp) {
      for (OpOperand &operand : genericOp->getOpOperands()) {
        walkAndRewrite(rewriter, operand.get(),
                       ttcore::MemorySpace::DeviceDRAM);
        insertStream2(rewriter, operand, genericOp);
        break; // TODO doing only the 1st operand as practice
      }
      return WalkResult::interrupt(); // TODO doing only the 1st generic as
                                      // practice
    });

    // funcBody.walk([&](ttir::GenericOp genericOp) {
    //   for (OpOperand &operand : genericOp->getOpOperands()) {
    //     insertStream2(rewriter, operand, genericOp);
    //     break;
    //   }
    //   return WalkResult::interrupt();
    // });

    // llvm::DenseSet<memref::AllocOp> allocOps;
    // llvm::DenseSet<ttir::GenericOp> genericOps;

    // funcBody.walk([&](Operation *op) {
    //   llvm::TypeSwitch<Operation *, void>(op)
    //       .Case([&](memref::AllocOp op) { allocOps.insert(op); })
    //       .Case([&](ttir::GenericOp op) { genericOps.insert(op); });
    // });
    // llvm::outs() << "found " << allocOps.size() << " alloc(s), "
    //              << genericOps.size() << " generic(s)\n";
    //
    // for (ttir::GenericOp genericOp : genericOps) {
    //   // Note: this covers the results (outs).
    //   for (Value operand : genericOp->getOperands()) {
    //     Operation *srcAllocOP = walkUseDef(operand);
    //     llvm::outs() << genericOp->getName() << " operand <- "
    //                  << srcAllocOP->getLoc() << "\n";
    //   }
    // }

    return success();
  }

  static MemRefType remap(RewriterBase &rewriter, MemRefType memrefType,
                          ttcore::MemorySpace space) {
    return MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                           memrefType.getLayout(),
                           rewriter.getAttr<ttcore::MemorySpaceAttr>(space));
  }
  static void remap(RewriterBase &rewriter, memref::AllocOp op,
                    ttcore::MemorySpace space) {
    auto memref = op.getMemref();
    MemRefType memrefType = memref.getType();
    MemRefType newType = remap(rewriter, memrefType, space);

    rewriter.modifyOpInPlace(op, [&]() { memref.setType(newType); });
  }

  static void remap(RewriterBase &rewriter, ttir::ViewLayoutOp op,
                    ttcore::MemorySpace space) {
    auto memref = op->getResult(0); // TODO name
    MemRefType memrefType = llvm::cast<MemRefType>(memref.getType());
    MemRefType newType = remap(rewriter, memrefType, space);

    rewriter.modifyOpInPlace(op, [&]() { memref.setType(newType); });
  }

  static void walkAndRewrite(RewriterBase &rewriter, Value v,
                             ttcore::MemorySpace space) {
    llvm::TypeSwitch<Operation *, void>(v.getDefiningOp())
        .Case([&](memref::AllocOp op) { remap(rewriter, op, space); })
        .Case([&](ttir::ViewLayoutOp op) {
          remap(rewriter, op, space);
          walkAndRewrite(rewriter, op.getInput(), space);
        })
        .Case([&](ttir::StreamLayoutOp op) {
          // TODO correct handling here
          walkAndRewrite(rewriter, op.getInput(), space);
        });
  }

  static void insertStream2(RewriterBase &rewriter, OpOperand &operand,
                            ttir::GenericOp op) {
    auto memref = mlir::cast<MemRefType>(operand.get().getType());

    OpBuilder::InsertionGuard guard(rewriter);
    {
      // By design, must insert just before the generic op.
      rewriter.setInsertionPoint(op);

      auto streamAttr = rewriter.getAttr<ttcore::ViewLayoutAttr>(
          rewriter.getMultiDimIdentityMap(memref.getRank()));
      auto streamMemref =
          MemRefType::get(memref.getShape(), memref.getElementType(),
                          streamAttr, memref.getMemorySpace());

      auto bufferLayout = ttcore::ShardLayoutAttr::get(memref, /*buffers=*/1);
      // TODO this needs re-shaping/re-sizing:
      auto bufferMemref = MemRefType::get(
          memref.getShape(), memref.getElementType(), bufferLayout,
          rewriter.getAttr<ttcore::MemorySpaceAttr>(
              ttcore::MemorySpace::DeviceL1));
      auto buffer = rewriter.create<memref::AllocOp>(op.getLoc(), bufferMemref);

      auto stream = rewriter.create<ttir::StreamLayoutOp>(
          op.getLoc(), streamMemref, operand.get(), buffer);

      rewriter.modifyOpInPlace(op,
                               [&]() { operand.assign(stream.getResult()); });
    }
  }

  static memref::AllocOp findRootAlloc(Value v,
                                       llvm::SmallVector<Operation *> &path) {
    // A canonicalizer pass would collapse all view_layout chains but don't
    // rely on that here.
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

  // ----------------------

  // Create/allocate streams within a module.
  LogicalResult runAllocateStreams(ModuleOp moduleOp) {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRAllocateStreams>(&getContext(), numStreamBuffers);
    return mlir::applyPatternsGreedily(getOperation(), std::move(patterns));
  }

  // Analyze buffer allocation needs for a module.
  FailureOr<ModuleAnalysisData> runAnalyzeBuffers(ModuleOp moduleOp) {

    ttcore::SystemDescAttr systemDesc =
        ttcore::getCurrentScopeSystemDesc(moduleOp);
    ttcore::ChipDescAttr chipDesc = systemDesc.getChipDescs().front();

    MemorySpaces memSpaces = getMemorySpaces(chipDesc);
    ModuleAnalysisData moduleAnalysis(memSpaces);

    if (moduleOp
            ->walk([&](func::FuncOp funcOp) {
              if (funcOp.isDeclaration()) {
                return WalkResult::skip();
              }

              FailureOr<FuncAnalysisData> funcAnalysis =
                  runAnalyzeBuffers(funcOp, memSpaces);
              if (failed(funcAnalysis)) {
                return WalkResult::interrupt();
              }

              moduleAnalysis.funcAnalysis[funcOp] = std::move(*funcAnalysis);
              return WalkResult::advance();
            })
            .wasInterrupted()) {
      return failure();
    }

    return moduleAnalysis;
  }

  struct LivenessClosure {
    Operation *lastOp;
    SequenceT first;
    SequenceT maxLast;
  };

  // Analyze and plan buffer allocation for a func.
  FailureOr<FuncAnalysisData> runAnalyzeBuffers(func::FuncOp funcOp,
                                                const MemorySpaces &memSpaces) {
    ttcore::DeviceAttr device = ttcore::lookupDevice(funcOp);
    Block &funcBody = funcOp.getBody().front();

    // Start with SSA liveness for `funcOp`.

    Liveness liveness(funcOp.getOperation());
    const LivenessBlockInfo *li = liveness.getLiveness(&funcBody);

    FuncAnalysisData analysis;

    //  (a) Build `Operation` <-> preorder position mappings for all `funcOp`
    //  ops. (b) Collect a separate set of "ops of interest", which are
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
        TT_assertv(op->getNumResults() == 1l, "for {}", *op);
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
      TT_assert(i != analysis.operationMap.end());
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
        ttcore::MemorySpace memorySpace = getMemorySpace(
            memrefTy,
            ttcore::MemorySpace::System); // Interpret unset as "host memory".

        if (!isDeviceMemorySpace(memorySpace)) {
          continue;
        }

        const AllocSizeT alignment =
            memSpaces[llvm::to_underlying(memorySpace)].alignment;
        const AllocSizeT sizeBytes =
            device.getMemrefSizeBytes(memrefTy, 0, true);
        const AllocSizeT alignedSize =
            ttmlir::utils::alignUp(sizeBytes, alignment);

        analysis.add(alignedSize, ctx.first, ctx.maxLast, alloc, memorySpace);
      }
    }

    for (ttcore::MemorySpace memorySpace :
         {ttcore::MemorySpace::DeviceL1, ttcore::MemorySpace::DeviceDRAM}) {
      const AllocationPlanner::Stats stats = AllocationPlanner::allocate(
          analysis.contexts[llvm::to_underlying(memorySpace)]);

      // TODO(#3378) dump this instead (usageRatio() is useful) in "debug" mode:
      // AllocationPlanner::Stats stats = AllocationPlanner::verify(analysis);
      TT_ALLOC_DEBUG("{} allocation planning outcome: {}", memorySpace, stats);

      const auto &info = memSpaces[llvm::to_underlying(memorySpace)];

      const auto memCapacity = info.maxAddress - info.baseAddress;
      if (stats.memUsage > memCapacity) {
        return funcOp.emitOpError()
               << "required " << stringifyEnum(memorySpace) << " memory usage "
               << stats.memUsage << " exceeds memory capacity " << memCapacity
               << " (usable space is [" << info.baseAddress << ", "
               << info.maxAddress << "))";
      }
    }

    return analysis;
  }

  // Apply buffer allocation `analysis` to `moduleOp`.
  LogicalResult runAllocateBuffers(ModuleOp moduleOp,
                                   const ModuleAnalysisData &analysis) {
    auto result = moduleOp->walk([&](func::FuncOp funcOp) {
      auto funcAnalysis = analysis.funcAnalysis.find(funcOp);
      if (funcAnalysis == analysis.funcAnalysis.end()) {
        return WalkResult::skip();
      }

      if (failed(runAllocateBuffers(funcOp, funcAnalysis->second,
                                    analysis.memSpaces))) {
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    return success(!result.wasInterrupted());
  }

  // Apply buffer allocation `analysis` to `funcOp`.
  LogicalResult runAllocateBuffers(func::FuncOp funcOp,
                                   const FuncAnalysisData &analysis,
                                   const MemorySpaces &memSpaces) {
    TT_assert(funcOp.getBody().hasOneBlock());

    // Augment all 'memref.alloc's in device memory with allocated addresses and
    // correct alignments.

    IRRewriter rewriter(&getContext());

    for (ttcore::MemorySpace memorySpace :
         {ttcore::MemorySpace::DeviceL1, ttcore::MemorySpace::DeviceDRAM}) {
      const MemorySpaceContext &context =
          analysis.contexts[llvm::to_underlying(memorySpace)];

      for (std::size_t t = 0; t < context.size(); ++t) {
        const AllocationPlanner::Record &record = context[t];
        memref::AllocOp alloc = context.allocs[t];

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
          rewriter.create<memref::DeallocOp>(lastOp->getLoc(),
                                             alloc.getResult());
        }
      }
    }

    return success();
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

  static MemorySpaces getMemorySpaces(ttcore::ChipDescAttr chipDesc) {
    std::array<MemorySpaceInfo, MemorySpaceInfo::kMaxEnumValForMemorySpace>
        info;
    // Currently, we only need some slots in 'info'.
    {
      info[llvm::to_underlying(ttcore::MemorySpace::DeviceL1)] =
          MemorySpaceInfo(chipDesc.getL1UnreservedBase(), chipDesc.getL1Size(),
                          chipDesc.getNocL1AddressAlignBytes());

      info[llvm::to_underlying(ttcore::MemorySpace::DeviceDRAM)] =
          MemorySpaceInfo(chipDesc.getDramUnreservedBase(),
                          chipDesc.getDramChannelSize(),
                          chipDesc.getNocDRAMAddressAlignBytes());
    }
    return info;
  }
};
} // namespace

} // namespace mlir::tt::ttir
// ----------------------------------------------------------------------------
