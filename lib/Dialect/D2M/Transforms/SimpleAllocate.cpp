#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/Analysis/Allocation/Planner.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseSet.h"

namespace mlir::tt::d2m {

#define GEN_PASS_DEF_D2MSIMPLEALLOCATE
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

using namespace mlir::tt::d2m;
using Planner = allocation::Planner;
using AllocSizeT = Planner::AllocSizeT;
using PlannerSpace = Planner::Space;
using SequenceT = Planner::SequenceT;
using LiveRange = Planner::LiveRange;
using ttcore::MemorySpace;

namespace {

struct AllocInfo {
  memref::AllocOp op;
  MemorySpace memSpace;
  AllocSizeT size;
  LiveRange range;
  int32_t varIndex = -1;
};

class D2MSimpleAllocate : public impl::D2MSimpleAllocateBase<D2MSimpleAllocate> {
public:
  using impl::D2MSimpleAllocateBase<D2MSimpleAllocate>::D2MSimpleAllocateBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (funcOp.isDeclaration())
      return;

    // Get device info
    ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
    ttcore::SystemDescAttr systemDesc =
        ttcore::getCurrentScopeSystemDesc(moduleOp);
    ttcore::ChipDescAttr chipDesc = systemDesc.getChipDescs().front();

    // Collect allocations
    SmallVector<AllocInfo> allocs;
    if (failed(collectAllocs(funcOp, chipDesc, allocs))) {
      return signalPassFailure();
    }

    // Allocate L1 addresses
    if (failed(allocateL1(funcOp, chipDesc, allocs))) {
      return signalPassFailure();
    }

    // Assign addresses to ops
    if (failed(assignAddresses(funcOp, allocs))) {
      return signalPassFailure();
    }

    // Insert deallocs (simple version - at function end)
    insertDeallocs(funcOp, allocs);
  }

private:
  LogicalResult collectAllocs(func::FuncOp funcOp,
                               ttcore::ChipDescAttr chipDesc,
                               SmallVector<AllocInfo> &allocs) {
    mlir::Liveness liveness(funcOp);
    Block &body = funcOp.getBody().front();

    // Build sequence map
    llvm::DenseMap<Operation *, SequenceT> opToSeq;
    SequenceT seq = 0;
    body.walk<WalkOrder::PreOrder>([&](Operation *op) { opToSeq[op] = seq++; });

    // Collect allocs with liveness
    funcOp.walk([&](memref::AllocOp allocOp) {
      MemRefType type = allocOp.getType();
      MemorySpace memSpace =
          ttcore::getMemorySpace(type, MemorySpace::System);

      // Only handle device memory
      if (!ttcore::isDeviceMemorySpace(memSpace))
        return;

      // Compute size
      ttcore::DeviceAttr device = ttcore::lookupDevice(funcOp);
      int64_t sizeBytes = device.getMemrefSizeBytes(type, 0, false);

      // Get memory space alignment
      AllocSizeT alignment;
      if (memSpace == MemorySpace::DeviceL1) {
        alignment = chipDesc.getNocL1AddressAlignBytes();
      } else {
        alignment = chipDesc.getNocDRAMAddressAlignBytes();
      }
      AllocSizeT size = ttmlir::utils::alignUp(sizeBytes, alignment);

      // Compute live range
      Value result = allocOp.getResult();
      const mlir::LivenessBlockInfo *li = liveness.getLiveness(&body);
      Operation *startOp = li->getStartOperation(result);
      Operation *endOp = li->getEndOperation(result, startOp);

      // Extend liveness through view/stream users
      endOp = extendLiveness(result, endOp, opToSeq);

      LiveRange range = {opToSeq[startOp], opToSeq[endOp]};

      allocs.push_back(
          {allocOp, memSpace, size, range, -1});
    });

    return success();
  }

  Operation *extendLiveness(Value val, Operation *currentEnd,
                            const llvm::DenseMap<Operation *, SequenceT> &opToSeq) {
    llvm::DenseSet<Value> visited;
    return extendLivenessImpl(val, currentEnd, opToSeq, visited);
  }

  Operation *extendLivenessImpl(
      Value val, Operation *currentEnd,
      const llvm::DenseMap<Operation *, SequenceT> &opToSeq,
      llvm::DenseSet<Value> &visited) {
    // Prevent infinite recursion
    if (!visited.insert(val).second)
      return currentEnd;

    SequenceT maxSeq = opToSeq.lookup(currentEnd);

    for (Operation *user : val.getUsers()) {
      auto it = opToSeq.find(user);
      if (it != opToSeq.end()) {
        maxSeq = std::max(maxSeq, it->second);
      }

      // Trace through view/stream ops
      if (auto viewOp = dyn_cast<d2m::ViewLayoutOp>(user)) {
        Operation *extendedEnd = extendLivenessImpl(
            viewOp.getResult(), currentEnd, opToSeq, visited);
        if (opToSeq.lookup(extendedEnd) > maxSeq)
          currentEnd = extendedEnd;
      } else if (auto streamOp = dyn_cast<d2m::StreamLayoutOp>(user)) {
        Operation *extendedEnd = extendLivenessImpl(
            streamOp.getResult(), currentEnd, opToSeq, visited);
        if (opToSeq.lookup(extendedEnd) > maxSeq)
          currentEnd = extendedEnd;
      }
    }

    // Find op with maxSeq
    for (auto &[op, s] : opToSeq) {
      if (s == maxSeq)
        return op;
    }
    return currentEnd;
  }

  LogicalResult allocateL1(func::FuncOp funcOp, ttcore::ChipDescAttr chipDesc,
                            SmallVector<AllocInfo> &allocs) {
    // Setup L1 capacity
    AllocSizeT l1Base = chipDesc.getL1UnreservedBase();
    AllocSizeT l1Size = testAssumeL1Capacity > 0
                            ? (l1Base + testAssumeL1Capacity)
                            : chipDesc.getL1Size();
    AllocSizeT l1Capacity = l1Size - l1Base;

    // Build allocation problem for L1
    Planner::Problem problem;

    for (auto &info : allocs) {
      if (info.memSpace == MemorySpace::DeviceL1) {
        info.varIndex = problem.def([&](Planner::VariableBuilder &b) {
          b.request(PlannerSpace::Scratch, info.size, info.range.first,
                    info.range.last);
        });
      }
    }

    // Solve with simple allocation (no spilling for now)
    auto stats = Planner::allocate(problem);

    if (stats.memUsage > l1Capacity) {
      return funcOp.emitError("L1 capacity exceeded: ")
             << stats.memUsage << " bytes required, " << l1Capacity
             << " bytes available";
    }

    // Store offsets for later
    for (auto &info : allocs) {
      if (info.varIndex >= 0) {
        const auto &var = problem.variable(info.varIndex);
        // Get requests from the Scratch space in the variable's domain
        const auto &scratchRequests = var.domain[allocation::ordinal(PlannerSpace::Scratch)];
        if (!scratchRequests.empty()) {
          // Get the first request index and look it up
          auto reqIndex = *scratchRequests.begin();
          // Store offset in the AllocInfo (we'll add base address later)
          info.size = problem.request(reqIndex).offset;
        }
      }
    }

    return success();
  }

  LogicalResult assignAddresses(func::FuncOp funcOp,
                                 SmallVector<AllocInfo> &allocs) {
    ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
    ttcore::SystemDescAttr systemDesc =
        ttcore::getCurrentScopeSystemDesc(moduleOp);
    ttcore::ChipDescAttr chipDesc = systemDesc.getChipDescs().front();

    IRRewriter rewriter(funcOp.getContext());

    for (auto &info : allocs) {
      if (info.memSpace == MemorySpace::DeviceL1) {
        AllocSizeT base = chipDesc.getL1UnreservedBase();
        AllocSizeT alignment = chipDesc.getNocL1AddressAlignBytes();

        // info.size contains the offset from the planner
        AllocSizeT address = base + info.size;

        rewriter.startOpModification(info.op);
        info.op.setAlignment(alignment);
        info.op->setAttr("address", rewriter.getI64IntegerAttr(address));
        rewriter.finalizeOpModification(info.op);
      } else if (info.memSpace == MemorySpace::DeviceDRAM) {
        // For DRAM, just set alignment (runtime handles addresses)
        AllocSizeT alignment = chipDesc.getNocDRAMAddressAlignBytes();
        rewriter.startOpModification(info.op);
        info.op.setAlignment(alignment);
        rewriter.finalizeOpModification(info.op);
      }
    }

    return success();
  }

  void insertDeallocs(func::FuncOp funcOp, SmallVector<AllocInfo> &allocs) {
    // Simple strategy: insert all deallocs before the return
    // This is conservative but safe - avoids all aliasing issues
    IRRewriter rewriter(funcOp.getContext());

    funcOp.walk([&](func::ReturnOp returnOp) {
      rewriter.setInsertionPoint(returnOp);
      for (auto &info : allocs) {
        rewriter.create<memref::DeallocOp>(returnOp.getLoc(),
                                            info.op.getResult());
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::d2m
