// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/FunctionTypes.h"
#include "ttmlir/Utils.h"

#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNCPUHOISTEDINPUTSTOSYSTEMMEMORY
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Helper function to create a system memory layout attribute for the given
// tensor type.
//
static TTNNLayoutAttr createSystemMemoryLayoutAttr(RankedTensorType type) {
  auto currentLayout = mlir::cast<TTNNLayoutAttr>(type.getEncoding());
  return currentLayout.withBufferType(BufferType::SystemMemory)
      .withLayout(Layout::RowMajor, type.getShape());
}

// Helper function to convert a tensor type to system memory type.
//
static RankedTensorType toSystemMemoryType(RankedTensorType ty) {
  TTNNLayoutAttr newLayout = createSystemMemoryLayoutAttr(ty);
  return RankedTensorType::get(ty.getShape(), ty.getElementType(), newLayout);
}

// Helper function to determine whether the given tensor type is in system
// memory.
//
static bool isSystemMemory(RankedTensorType ty) {
  auto layout = mlir::cast<TTNNLayoutAttr>(ty.getEncoding());
  return layout.isSystemBufferType();
}

// Returns true if the given block argument is only used by CPU-hoisted calls
// via to_layout ops converting to system memory.
//
// The expected pattern is:
//   blockArg -> ttnn.to_layout (to system memory) -> func.call (cpu-hoisted)
//
static bool isOnlyUsedByCPUHoistedCalls(BlockArgument blockArgument) {
  for (Operation *userOp : blockArgument.getUsers()) {
    // Each user must be a ToLayoutOp converting to system memory.
    auto toLayoutOp = mlir::dyn_cast<ttnn::ToLayoutOp>(userOp);
    if (!toLayoutOp) {
      return false;
    }

    auto memConfig = toLayoutOp.getMemoryConfig();
    if (!memConfig ||
        memConfig->getBufferType().getValue() != BufferType::SystemMemory) {
      return false;
    }

    // Every user of the ToLayoutOp result must be a CPU-hoisted func.call.
    for (Operation *toLayoutUser : toLayoutOp.getResult().getUsers()) {
      auto callOp = mlir::dyn_cast<func::CallOp>(toLayoutUser);
      if (!callOp ||
          !callOp->hasAttr(ttmlir::utils::g_cpuHoistFuncCallAttrName)) {
        return false;
      }
    }
  }

  return true;
}

// Converts the arguments of the forward function which are only consumed by
// CPU-hoisted calls to system memory.
//
static void moveCPUHoistedArgsToHost(func::FuncOp funcOp) {
  SmallVector<Type> argumentTypes;
  bool changed = false;

  for (auto blockArgument : funcOp.getRegion().getArguments()) {
    auto tensorType = mlir::dyn_cast<RankedTensorType>(blockArgument.getType());

    // Skip non-tensor arguments.
    if (!tensorType) {
      argumentTypes.push_back(blockArgument.getType());
      continue;
    }

    // If the argument is already in system memory, skip it.
    if (isSystemMemory(tensorType)) {
      argumentTypes.push_back(tensorType);
      continue;
    }

    // If the block argument has no uses, skip it.
    if (blockArgument.getNumUses() == 0) {
      argumentTypes.push_back(tensorType);
      continue;
    }

    // Check if all users are to_layout ops converting to system memory whose
    // results are exclusively consumed by CPU-hoisted calls.
    if (!isOnlyUsedByCPUHoistedCalls(blockArgument)) {
      argumentTypes.push_back(tensorType);
      continue;
    }

    // Convert the argument to system memory type.
    auto systemMemoryTensorType = toSystemMemoryType(tensorType);

    blockArgument.setType(systemMemoryTensorType);
    argumentTypes.push_back(systemMemoryTensorType);
    changed = true;
  }

  // Update function type with new argument types only if necessary.
  if (changed) {
    auto newFunctionType =
        mlir::FunctionType::get(funcOp->getContext(), argumentTypes,
                                funcOp.getFunctionType().getResults());
    funcOp.setFunctionType(newFunctionType);
  }
}

class TTNNCPUHoistedInputsToSystemMemory
    : public impl::TTNNCPUHoistedInputsToSystemMemoryBase<
          TTNNCPUHoistedInputsToSystemMemory> {
public:
  using impl::TTNNCPUHoistedInputsToSystemMemoryBase<
      TTNNCPUHoistedInputsToSystemMemory>::
      TTNNCPUHoistedInputsToSystemMemoryBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    moduleOp->walk([&](func::FuncOp funcOp) {
      if (!ttmlir::utils::isForwardDeviceFunc(funcOp)) {
        return;
      }

      moveCPUHoistedArgsToHost(funcOp);
    });
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttnn::TTNNDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }
};
} // namespace

} // namespace mlir::tt::ttnn
