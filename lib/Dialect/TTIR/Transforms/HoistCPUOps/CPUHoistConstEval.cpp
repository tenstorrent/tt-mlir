// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/HoistCPUOps/HoistCPUOps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/FunctionTypes.h"
#include "ttmlir/Utils.h"

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

// Check if an op only rearranges or retypes data without performing any
// arithmetic. This covers tensor-manipulation ops (transpose, reshape,
// permute, rearrange) and typecast. For such ops f32 execution yields no
// precision benefit, since a permutation/retype is exact regardless of the
// element type.
static bool isPureDataMovementOp(mlir::Operation *op) {
  return op->hasTrait<TensorManipulation::Trait>() || mlir::isa<TypecastOp>(op);
}

// Check if an op rearranges data layout/shape (transpose, reshape, permute,
// rearrange). Unlike a bare typecast, such an op produces a differently-shaped
// tensor that downstream device passes may want to handle (e.g. folding a
// weight transpose into a matmul's transpose_b).
static bool isTensorManipulationOp(mlir::Operation *op) {
  return op->hasTrait<TensorManipulation::Trait>();
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

// Analyze a const-eval function for ops to perform CPU-hoisting on.
//
// Motivation for CPU-hoisting const-eval ops:
// - CPU-hoisted ops operate on 32-bit integers/floats, which should result in
//   more precise calculations compared to device execution.
// - Peak DRAM/L1 usage should be reduced, since intermediate tensors are stored
//   in host memory. This is especially beneficial for tensors which would take
//   up significantly more L1 if tilized (e.g. tensor<1024x1024x1x1).
static llvm::SmallVector<CPUHoistedOpsDescriptor>
analyzeConstEval(func::FuncOp funcOp, bool dataMovementF32) {
  if (!ttmlir::utils::isConstEvalFunc(funcOp)) {
    return {};
  }

  // Skip CPU-hoisting for multi-device graphs.
  // TODO(dmilinkovic) - issue #6709.
  auto deviceAttr = ttcore::lookupDevice(funcOp);
  if (deviceAttr && ttmlir::utils::volume(deviceAttr.getMeshShape()) > 1) {
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

  // Hoist const-eval subgraphs that perform no arithmetic and include at least
  // one shape/layout-changing op - i.e. pure data-movement graphs (transpose,
  // permute, reshape; optionally interleaved with typecast) - while preserving
  // their native element type. The precision motivation for f32 CPU execution
  // does not apply: a permutation/retype is exact regardless of element type,
  // so forcing f32 only adds a bf16->f32->bf16 typecast round-trip plus host
  // round-trips. Keeping the original dtype avoids those conversions entirely
  // (the subgraph still runs on host, which sidesteps on-device reshape/
  // transpose issues).
  //
  // Note: a subgraph that is *only* typecast(s) is left on the f32 path. There
  // the conversion can produce a narrower-dtype tensor on host, so the peak
  // DRAM/L1 motivation still applies.
  //
  // The data-movement-f32 option restores the legacy behavior (force f32 for
  // these too) as an escape hatch for A/B comparison.
  if (!dataMovementF32 &&
      llvm::all_of(descriptor.operations, isPureDataMovementOp) &&
      llvm::any_of(descriptor.operations, isTensorManipulationOp)) {
    descriptor.preserveElementType = true;
  }

  // Verify all ops in the descriptor can be lowered to Linalg.
  // If any op fails, skip CPU-hoisting altogether.
  for (auto *op : descriptor.operations) {
    if (!canLowerTTIRToLinalg(op)) {
      op->emitWarning("Skipping CPU hoisting of const-eval "
                      "subgraph: op cannot be lowered to Linalg");
      return {};
    }
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
      auto result = analyzeConstEval(funcOp, dataMovementF32);
      descriptors.append(std::make_move_iterator(result.begin()),
                         std::make_move_iterator(result.end()));
    });

    runCPUHoist(rootModule, std::move(descriptors));
  }
};

} // namespace
} // namespace mlir::tt::ttir
