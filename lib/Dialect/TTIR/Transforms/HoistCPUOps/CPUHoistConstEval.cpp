// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/HoistCPUOps/HoistCPUOps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_CPUHOISTCONSTEVALTRANSFORM
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Check if an op is "transparent" - it doesn't change semantic meaning,
// just format/type.
static bool isTransparentOp(mlir::Operation *op) {
  return mlir::isa<ReshapeOp, TypecastOp>(op);
}

// Walk backward from a value through transparent ops in a single traversal.
// If the chain terminates at a creation skippable op, return it.
static llvm::SmallVector<mlir::Operation *> traceCreationOpChain(Value v) {
  llvm::SmallVector<mlir::Operation *> chain;

  while (Operation *defOp = v.getDefiningOp()) {
    if (defOp->hasTrait<ttcore::Trait::TTCoreCreationOpTrait>()) {
      chain.push_back(defOp);
      return chain;
    }

    if (isTransparentOp(defOp)) {
      chain.push_back(defOp);
      v = defOp->getOperand(0);
      continue;
    }

    // Non-transparent, non-creation: chain is not skippable.
    break;
  }

  return {};
}

// Analyze a const-eval function for ops to hoist as a whole.
//
// Motivation for CPU-hoisting const-eval ops:
// - CPU-hoisted ops operate on 32-bit integers/floats, which should result in
//   more precise calculations compared to device execution.
// - Peak DRAM/L1 usage should be reduced, since intermediate tensors are stored
//   in host memory. This is especially beneficial for tensors which would take
//   up significantly more L1 if tilized (e.g. tensor<1024x1024x1x1).
static llvm::SmallVector<CPUHoistedOpsDescriptor>
analyzeConstEval(func::FuncOp funcOp) {
  if (!ttmlir::utils::isConstEvalFunc(funcOp)) {
    return {};
  }

  CPUHoistedOpsDescriptor descriptor({}, {}, llvm::StringRef("const_eval"));

  // Check if it is possible to CPU-hoist this const-eval function.
  auto walkResult = funcOp.walk([&](mlir::Operation *nestedOp) {
    // If there is already a CPU-hoisted call inside the const-eval
    // subgraph, skip CPU hoisting altogether to avoid nested hoisting.
    if (nestedOp->hasAttr(ttir::CPUHoistedCallAttr::name)) {
      return WalkResult::interrupt();
    }

    if (auto meshShardOp =
            mlir::dyn_cast<mlir::tt::ttir::MeshShardOp>(nestedOp)) {
      // If there is a non-identity TTIR MeshShardOp, skip CPU hoisting
      // altogether.
      // TODO(dmilinkovic) - issue #6709,
      if (meshShardOp.getShardType() != ttcore::MeshShardType::Identity) {
        return WalkResult::interrupt();
      }
    }

    // If there is any CCL op, skip CPU hoisting altogether.
    // TODO(dmilinkovic) - issue #6709
    if (mlir::isa<mlir::tt::ttir::AllGatherOp, mlir::tt::ttir::AllReduceOp,
                  mlir::tt::ttir::ReduceScatterOp,
                  mlir::tt::ttir::CollectivePermuteOp,
                  mlir::tt::ttir::AllToAllOp,
                  mlir::tt::ttir::CollectiveBroadcastOp,
                  mlir::tt::ttir::MeshPartitionOp>(nestedOp)) {
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    return {};
  }

  auto returnOp =
      llvm::cast<func::ReturnOp>(funcOp.getBody().front().getTerminator());

  llvm::SmallPtrSet<mlir::Operation *, 8> opsToSkip;

  // Skip chains of creation ops and transparent ops leading to
  // them. This is done because:
  // 1. Downstream passes might try to extract constant values from these ops,
  //    which isn't possible if these are moved to the CPU-module.
  // 2. CPU-hoisting creation ops which are results of const-eval doesn't
  //    improve PCC nor peak DRAM/L1 usage.
  for (Value retVal : returnOp.getOperands()) {
    auto chain = traceCreationOpChain(retVal);
    if (chain.empty()) {
      descriptor.outputValues.push_back(retVal);
    } else {
      opsToSkip.insert(chain.begin(), chain.end());
    }
  }

  // Skip identity MeshShard ops.
  // These ops are just semantic decorators, and are no-ops
  // from the runtime perspective.
  for (auto nestedOp : funcOp.getOps<mlir::tt::ttir::MeshShardOp>()) {
    if (nestedOp.getShardType() == ttcore::MeshShardType::Identity) {
      opsToSkip.insert(nestedOp);
    }
  }

  // Collect all ops that are not skipped.
  funcOp.walk([&](mlir::Operation *nestedOp) {
    if (llvm::isa<func::FuncOp, func::ReturnOp>(nestedOp)) {
      return;
    }

    if (opsToSkip.contains(nestedOp)) {
      return;
    }

    descriptor.operations.push_back(nestedOp);
  });

  if (descriptor.operations.empty()) {
    return {};
  }

  return {descriptor};
}

// Transform pass to hoist const-eval subgraphs as a whole.
class CPUHoistConstEvalTransform
    : public impl::CPUHoistConstEvalTransformBase<CPUHoistConstEvalTransform> {
public:
  using CPUHoistConstEvalTransformBase::CPUHoistConstEvalTransformBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    mlir::ModuleOp deviceInnerModule = getDeviceInnerModule(rootModule);
    if (!deviceInnerModule) {
      return;
    }

    llvm::SmallVector<CPUHoistedOpsDescriptor> descriptors;
    deviceInnerModule.walk([&](func::FuncOp funcOp) {
      auto result = analyzeConstEval(funcOp);
      descriptors.append(std::make_move_iterator(result.begin()),
                         std::make_move_iterator(result.end()));
    });

    runCPUHoist(rootModule, std::move(descriptors));
  }
};

} // namespace
} // namespace mlir::tt::ttir
