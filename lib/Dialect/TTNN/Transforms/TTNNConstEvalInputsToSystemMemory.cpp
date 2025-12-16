// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNCONSTEVALINPUTSTOSYSTEMMEMORY
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

// Helper function to determine whether to transfer the const-eval argument to
// device memory.
//
// The argument should NOT be transferred to device memory only if its sole user
// is a ttnn.to_layout op which already transfers it to system memory - this
// happens if the const-eval function is CPU-hoisted.
//
static bool shouldTransferArgumentToDevice(BlockArgument blockArgument) {
  if (blockArgument.getNumUses() != 1) {
    return true;
  }

  auto toLayoutOp =
      mlir::dyn_cast<ttnn::ToLayoutOp>(*blockArgument.getUsers().begin());

  if (!toLayoutOp) {
    return true;
  }

  return toLayoutOp.getMemoryConfig()->getBufferType().getValue() !=
         BufferType::SystemMemory;
}

// Converts the arguments of the forward function which are const-eval inputs
// to system memory. Returns the list of converted arguments.
//
// An argument is considered to be a const-eval input if it is
// consumed only by LoadCachedOps.
//
static SmallVector<BlockArgument> moveConstEvalArgsToHost(func::FuncOp funcOp) {
  SmallVector<Type> argumentTypes;
  SmallVector<BlockArgument> convertedArguments;

  // Find relevant const-eval inputs and convert them to system
  // memory.
  //
  for (auto blockArgument : funcOp.getRegion().getArguments()) {
    auto tensorType = mlir::cast<RankedTensorType>(blockArgument.getType());

    // If the argument is already in the system memory, skip it.
    //
    if (isSystemMemory(tensorType)) {
      argumentTypes.push_back(tensorType);
      continue;
    }

    // If the block argument has no uses, skip it.
    //
    if (blockArgument.getNumUses() == 0) {
      argumentTypes.push_back(tensorType);
      continue;
    }

    // If the block argument has a user which is not a LoadCachedOp, skip it.
    //
    if (llvm::any_of(blockArgument.getUsers(), [](Operation *userOp) {
          return !mlir::isa<ttcore::LoadCachedOp>(userOp);
        })) {
      argumentTypes.push_back(tensorType);
      continue;
    }

    // Convert the argument to system memory type.
    //
    auto systemMemoryTensorType = toSystemMemoryType(tensorType);

    blockArgument.setType(systemMemoryTensorType);
    argumentTypes.push_back(systemMemoryTensorType);
    convertedArguments.push_back(blockArgument);
  }

  // Update function type with new argument types only if necessary.
  //
  if (!convertedArguments.empty()) {
    auto newFunctionType =
        mlir::FunctionType::get(funcOp->getContext(), argumentTypes,
                                funcOp.getFunctionType().getResults());
    funcOp.setFunctionType(newFunctionType);
  }

  return convertedArguments;
}

// Updates the const-eval function to have the specified argument index
// in system memory.
//
static void convertArgumentOfConstEvalFunc(func::FuncOp constEvalFuncOp,
                                           size_t argumentIndex,
                                           RankedTensorType systemMemoryType,
                                           ttcore::GridAttr deviceGrid) {
  SmallVector<Type> constEvalArgumentTypes(
      constEvalFuncOp.getFunctionType().getInputs());

  // If the argument is already converted to system memory, skip it.
  //
  if (constEvalArgumentTypes[argumentIndex] == systemMemoryType) {
    return;
  }

  auto blockArgument = constEvalFuncOp.getArgument(argumentIndex);

  // Check whether the argument is expected to be transferred to device
  // memory.
  //
  if (shouldTransferArgumentToDevice(blockArgument)) {
    // We need to insert a to_layout op which converts the argument to the
    // original layout on device.
    //
    mlir::OpBuilder builder(constEvalFuncOp.getRegion());

    // If there is already a ttnn.get_device op, we need to
    // set the insertion point after it.
    //
    constEvalFuncOp.walk([&builder](ttnn::GetDeviceOp getDeviceOp) {
      builder.setInsertionPointAfter(getDeviceOp);
    });

    auto deviceTensorType =
        mlir::cast<RankedTensorType>(blockArgument.getType());
    auto deviceTensorLayout =
        mlir::cast<TTNNLayoutAttr>(deviceTensorType.getEncoding());

    // Create to_layout op to convert the argument to the original layout.
    //
    auto originalDataTypeAttr = mlir::tt::ttcore::DataTypeAttr::get(
        constEvalFuncOp.getContext(), deviceTensorLayout.getDataType());

    auto toLayoutOp = builder.create<ttnn::ToLayoutOp>(
        blockArgument.getLoc(), deviceTensorType, blockArgument,
        deviceTensorLayout.getLayout(), originalDataTypeAttr,
        MemoryConfigAttr::get(deviceTensorLayout, deviceGrid));

    // Replace the argument usages with the to_layout op result.
    //
    blockArgument.replaceAllUsesExcept(toLayoutOp.getResult(), toLayoutOp);
  }

  // Finally, update the block argument type and the function type.
  //
  blockArgument.setType(systemMemoryType);

  constEvalArgumentTypes[argumentIndex] = systemMemoryType;
  auto newConstEvalFunctionType = mlir::FunctionType::get(
      constEvalFuncOp->getContext(), constEvalArgumentTypes,
      constEvalFuncOp.getFunctionType().getResults());
  constEvalFuncOp.setFunctionType(newConstEvalFunctionType);
}

class TTNNConstEvalInputsToSystemMemory
    : public impl::TTNNConstEvalInputsToSystemMemoryBase<
          TTNNConstEvalInputsToSystemMemory> {
public:
  using impl::TTNNConstEvalInputsToSystemMemoryBase<
      TTNNConstEvalInputsToSystemMemory>::TTNNConstEvalInputsToSystemMemoryBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    ttcore::DeviceAttr device = ttcore::lookupDevice(moduleOp);
    TT_assertv(device, "Device not found");

    moduleOp->walk([&](func::FuncOp funcOp) {
      // We only want to process forward functions.
      //
      if (funcOp.isDeclaration() || ttmlir::utils::isConstEvalFunc(funcOp)) {
        return;
      }

      // First, convert the arguments of the forward function which are
      // const-eval inputs to system memory.
      //
      SmallVector<BlockArgument> convertedArguments =
          moveConstEvalArgsToHost(funcOp);

      // Next, we need to update all const-eval functions accordingly.
      //
      for (auto convertedArgument : convertedArguments) {
        // Find the LoadCachedOps which use the converted argument.
        //
        llvm::SmallVector<ttcore::LoadCachedOp> loadCachedOps;
        for (auto *userOp : convertedArgument.getUsers()) {
          TT_assertv(mlir::isa<ttcore::LoadCachedOp>(userOp),
                     "Expected load cached op");
          loadCachedOps.push_back(mlir::cast<ttcore::LoadCachedOp>(userOp));
        }

        for (auto loadCachedOp : loadCachedOps) {
          // Look up the corresponding const-eval function.
          //
          auto constEvalFuncOp = mlir::cast<func::FuncOp>(
              mlir::SymbolTable::lookupNearestSymbolFrom(
                  loadCachedOp, loadCachedOp.getCalleeAttr()));

          // Find the argument index in the LoadCachedOp. This corresponds to
          // the argument index in the const-eval function which we need to
          // update.
          //
          size_t argumentIndex = std::distance(
              loadCachedOp.getOperands().begin(),
              llvm::find(loadCachedOp.getOperands(), convertedArgument));

          TT_assertv(
              argumentIndex < constEvalFuncOp.getNumArguments(),
              "Argument index out of bounds while updating LoadCachedOp.");

          // Finally, update the const-eval function so that the argument is in
          // system memory.
          //
          const auto convertedType =
              mlir::cast<RankedTensorType>(convertedArgument.getType());

          convertArgumentOfConstEvalFunc(constEvalFuncOp, argumentIndex,
                                         convertedType, device.getWorkerGrid());
        }
      }
    });
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttnn::TTNNDialect>();
    registry.insert<mlir::tt::ttcore::TTCoreDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }
};
} // namespace

} // namespace mlir::tt::ttnn
