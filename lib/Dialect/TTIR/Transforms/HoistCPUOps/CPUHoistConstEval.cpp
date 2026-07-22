// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// CPU-hoist const-eval pass.
//
// Hoists compute ops from const-eval functions to the CPU module, splitting
// the subgraph into segments around CCL barriers. Each segment becomes a
// separate CPU-hoisted function call, while CCL ops remain on device.
//
// Motivation:
// - CPU-hoisted ops operate on 32-bit integers/floats, which should result in
//   more precise calculations compared to device execution.
// - Peak DRAM/L1 usage should be reduced, since intermediate tensors are stored
//   in host memory. This is especially beneficial for tensors which would take
//   up significantly more L1 if tilized (e.g. tensor<1024x1024x1x1).
//
// Example — a const-eval function with one all_gather (types are bf16):
//
//   func @const_eval(%x: bf16, %y: bf16) {
//     %a = add(%x, %y)           // segment 0
//     %b = multiply(%a, %x)      // segment 0
//     %g = all_gather(%b)         // barrier (stays on device)
//     %c = subtract(%g, %g)      // segment 1
//     return %c
//   }
//
// After this pass, the device module function becomes:
//
//   func @const_eval(%x: bf16, %y: bf16) {
//     // --- segment 0: bf16 -> f32, compute on CPU, f32 -> bf16 ---
//     %x_f32 = to_layout(%x) : bf16 -> f32      // typecast + from_device
//     %y_f32 = to_layout(%y) : bf16 -> f32      // typecast + from_device
//     %b_f32 = call @cpu_hoisted_0(%x_f32, %y_f32)  // executes on host
//     %b     = to_layout(%b_f32) : f32 -> bf16  // typecast + to_device
//
//     // --- barrier: stays on device ---
//     %g = all_gather(%b)
//
//     // --- segment 1: bf16 → f32, compute on CPU, f32 → bf16 ---
//     %g_f32 = to_layout(%g)  : bf16 -> f32
//     %c_f32 = call @cpu_hoisted_1(%g_f32, %g_f32)
//     %c     = to_layout(%c_f32) : f32 -> bf16
//     return %c
//   }
//
// And the CPU module gets the hoisted function definitions:
//
//   cpu_module {
//     func @cpu_hoisted_0(%x: f32, %y: f32) -> f32 {
//       %a = add(%x, %y)
//       %b = multiply(%a, %x)
//       return %b
//     }
//     func @cpu_hoisted_1(%g0: f32, %g1: f32) -> f32 {
//       %c = subtract(%g0, %g1)
//       return %c
//     }
//   }
//
//===----------------------------------------------------------------------===//

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
#include "llvm/ADT/SmallString.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_CPUHOISTCONSTEVALTRANSFORM
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Check if an op is "transparent" - it doesn't change semantic meaning,
// just format/type.
static bool isTransparentOp(mlir::Operation *op) {
  return mlir::isa<ReshapeOp, TypecastOp>(op);
}

// Check if an op is a barrier for CPU hoisting - CCL and MeshShard ops must
// remain on device and split the subgraph into segments.
static bool isBarrierOp(mlir::Operation *op) {
  return mlir::isa<CCL>(op) || mlir::isa<MeshShardOp, MeshPartitionOp>(op);
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
// See file-level comment for motivation and an illustrative example.
static llvm::SmallVector<CPUHoistedOpsDescriptor>
analyzeConstEval(func::FuncOp funcOp) {
  if (!ttmlir::utils::isConstEvalFunc(funcOp)) {
    return {};
  }

  // Skip if there is already a CPU-hoisted call.
  if (llvm::any_of(funcOp.getOps(), [](Operation &op) {
        return op.hasAttr(ttir::CPUHoistedCallAttr::name);
      })) {
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
    if (!chain.empty()) {
      opsToSkip.insert(chain.begin(), chain.end());
    }
  }

  // Partition ops into segments separated by barrier ops.
  // Each segment contains consecutive non-barrier, non-skip ops.
  llvm::SmallVector<OpsVectorType> segments;
  OpsVectorType currentSegment;

  funcOp.walk([&](mlir::Operation *nestedOp) {
    if (llvm::isa<func::FuncOp, func::ReturnOp>(nestedOp)) {
      return;
    }

    if (opsToSkip.contains(nestedOp)) {
      return;
    }

    // Start a new segment at barrier ops.
    if (isBarrierOp(nestedOp)) {
      if (!currentSegment.empty()) {
        segments.push_back(std::move(currentSegment));
        currentSegment = {};
      }
      return;
    }

    currentSegment.push_back(nestedOp);
  });

  if (!currentSegment.empty()) {
    segments.push_back(std::move(currentSegment));
  }

  // Build a CPUHoistedOpsDescriptor descriptor for each segment.
  llvm::SmallVector<CPUHoistedOpsDescriptor> descriptors;
  for (const auto &segment : segments) {
    // Verify all ops can be lowered to Linalg. If any op fails, skip
    // CPU-hoisting this segment.
    auto *unlowerableOp = llvm::find_if(segment, [&](mlir::Operation *op) {
      return !canLowerTTIRToLinalg(op);
    });

    if (unlowerableOp != segment.end()) {
      (*unlowerableOp)
          ->emitWarning("Skipping CPU hoisting of const-eval "
                        "subgraph: op cannot be lowered to Linalg");
      continue;
    }

    // Output values of this CPU-hoisted segment are op results used by ops
    // outside the segment.
    ValuesVectorType outputValues;
    for (auto *op : segment) {
      for (auto result : op->getResults()) {
        bool usedOutsideSegment =
            llvm::any_of(result.getUsers(), [&](Operation *user) {
              return !llvm::is_contained(segment, user);
            });

        if (usedOutsideSegment) {
          outputValues.push_back(result);
        }
      }
    }

    descriptors.emplace_back(segment, outputValues,
                             llvm::SmallString<64>("const_eval"));
  }

  return descriptors;
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
