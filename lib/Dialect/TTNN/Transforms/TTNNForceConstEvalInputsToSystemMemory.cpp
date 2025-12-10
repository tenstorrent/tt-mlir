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
#define GEN_PASS_DEF_TTNNFORCECONSTEVALINPUTSTOSYSTEMMEMORY
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Helper function to create a system memory layout attribute for the given
// tensor type.
//
static TTNNLayoutAttr createSystemMemoryLayoutAttr(RankedTensorType type) {
  static const std::array<std::pair<int64_t, int64_t>, 1> defaultCollapseDims =
      {{{0, -1}}};

  auto currentLayout = mlir::cast<TTNNLayoutAttr>(type.getEncoding());
  auto defaultGrid = ttcore::GridAttr::get(type.getContext());

  return TTNNLayoutAttr::get(
      type.getContext(), type.getShape(), type.getElementType(),
      BufferType::SystemMemory, defaultGrid, TensorMemoryLayoutAttr{},
      currentLayout.getTensorMesh(), defaultCollapseDims);
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
  return layout.getBufferType() == BufferType::SystemMemory;
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

class TTNNForceConstEvalInputsToSystemMemory
    : public impl::TTNNForceConstEvalInputsToSystemMemoryBase<
          TTNNForceConstEvalInputsToSystemMemory> {
public:
  using impl::TTNNForceConstEvalInputsToSystemMemoryBase<
      TTNNForceConstEvalInputsToSystemMemory>::
      TTNNForceConstEvalInputsToSystemMemoryBase;

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
          convertArgumentsOfForwardFunc(funcOp);

      // Next, we need to update all const-eval functions accordingly.
      //
      for (auto convertedArgument : convertedArguments) {
        // Find the LoadCachedOps which use the converted argument.
        //
        llvm::SmallVector<ttcore::LoadCachedOp> loadCachedOps;
        for (auto *userOp : convertedArgument.getUsers()) {
          if (auto op = llvm::dyn_cast<ttcore::LoadCachedOp>(userOp)) {
            loadCachedOps.push_back(op);
          }
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

private:
  // Converts the arguments of the forward function which are const-eval inputs
  // to system memory. Returns the list of converted arguments.
  //
  // An argument is considered to be a const-eval input if it is only
  // consumed by LoadCachedOps
  //
  SmallVector<BlockArgument>
  convertArgumentsOfForwardFunc(func::FuncOp funcOp) {
    SmallVector<Type> argumentTypes;
    SmallVector<BlockArgument> convertedArguments;

    // Find relevant const-eval inputs and convert them to system
    // memory.
    //
    for (auto blockArgument : funcOp.getRegion().getArguments()) {
      auto tensorType =
          mlir::dyn_cast<RankedTensorType>(blockArgument.getType());

      // If the argument is not a tensor or is already in system memory,
      // skip it.
      //
      if (!tensorType || isSystemMemory(tensorType)) {
        argumentTypes.push_back(blockArgument.getType());
        continue;
      }

      // If the block argument has no uses, skip it.
      //
      if (blockArgument.getNumUses() == 0) {
        argumentTypes.push_back(blockArgument.getType());
        continue;
      }

      // If the block argument has a user which is not a LoadCachedOp, skip it.
      //
      if (llvm::any_of(blockArgument.getUsers(), [](Operation *userOp) {
            return !mlir::isa<ttcore::LoadCachedOp>(userOp);
          })) {
        argumentTypes.push_back(blockArgument.getType());
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
  void convertArgumentOfConstEvalFunc(func::FuncOp constEvalFuncOp,
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

    constEvalArgumentTypes[argumentIndex] = systemMemoryType;

    // Update the function type.
    auto newConstEvalFunctionType = mlir::FunctionType::get(
        constEvalFuncOp->getContext(), constEvalArgumentTypes,
        constEvalFuncOp.getFunctionType().getResults());

    constEvalFuncOp.setFunctionType(newConstEvalFunctionType);

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
    } else {
      // Otherwise, the argument is already being used only as a system memory
      // tensor.
      //
      // We need to check if the only purpose of the ttnn.to_layout op
      // consuming this argument is to transfer it to system memory.
      // If so, we should remove this to_layout op altogether, in order to
      // avoid TTNNDecomposeLayouts pass failing because of a redundant
      // to_layout op, as the argument is already in system memory.
      //
      auto toLayoutOp =
          mlir::dyn_cast<ttnn::ToLayoutOp>(*blockArgument.getUsers().begin());

      TT_assertv(toLayoutOp, "Expected to_layout op as the only user.");

      // If the data type and layout of the to_layout op matches the argument
      // type, we can remove it.
      //
      auto toLayoutOpResultType =
          mlir::cast<RankedTensorType>(toLayoutOp.getResult().getType());

      if (toLayoutOpResultType.getEncoding() ==
              systemMemoryType.getEncoding() &&
          toLayoutOpResultType.getElementType() ==
              systemMemoryType.getElementType()) {
        toLayoutOp.getResult().replaceAllUsesWith(blockArgument);
        toLayoutOp.erase();
      }
    }

    // Finally, update the block argument type.
    //
    blockArgument.setType(systemMemoryType);
  }
};
} // namespace

} // namespace mlir::tt::ttnn
