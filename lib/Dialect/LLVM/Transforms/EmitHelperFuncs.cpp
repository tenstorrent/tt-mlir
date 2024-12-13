// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h" // For LLVM Dialect definitions
#include "mlir/Dialect/LLVMIR/LLVMTypes.h" // For LLVM Type support (e.g., LLVMStructType, LLVMPointerType)

#include "llvm/ADT/ArrayRef.h"    // For ArrayRef
#include "llvm/ADT/SmallVector.h" // For SmallVector
#include "llvm/Support/Casting.h" // For dyn_cast

namespace mlir::tt::llvm_util {

#define GEN_PASS_DEF_EMITHELPERFUNCS
void generateLLVMHelpersForArgRanks(mlir::ModuleOp moduleOp) {
  auto *context = moduleOp.getContext();
  OpBuilder builder(context);

  for (auto func : moduleOp.getOps<LLVM::LLVMFuncOp>()) {
    if (!func->hasAttr("arg_ranks"))
      continue;

    // Extract the `arg_ranks` attribute
    auto argRanksAttr = llvm::dyn_cast<ArrayAttr>(func->getAttr("arg_ranks"));
    if (!argRanksAttr)
      continue;

    // Define the helper function name and type
    std::string helperName = func.getName().str() + "_helper";
    auto llvmFuncType = func.getFunctionType();

    // Create the helper function
    auto helperFuncType = LLVM::LLVMFunctionType::get(
        llvmFuncType.getReturnType(),
        {LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getLiteral(
            context,
            {
                LLVM::LLVMPointerType::get(context), // start
                LLVM::LLVMPointerType::get(context), // aligned_start
                builder.getI64Type(),                // start_idx
                LLVM::LLVMPointerType::get(context)  // sizes_and_strides
            }))},
        false);

    auto helperFunc = builder.create<LLVM::LLVMFuncOp>(
        func.getLoc(), helperName, helperFuncType);

    Block *entryBlock = helperFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Unpack the argument
    Value structPtr = entryBlock->getArgument(0);
    SmallVector<Value, 16> originalCallArgs;

    // Iterate over arg_ranks to unpack tensors
    int tensorIdx = 0;
    for (auto rankAttr : argRanksAttr) {
      int64_t rank = mlir::cast<IntegerAttr>(rankAttr).getInt();

      // Generate GEP and loads for each field in the struct
      Value tensorBase = builder.create<LLVM::GEPOp>(
          func.getLoc(), LLVM::LLVMPointerType::get(context), structPtr,
          builder.getI32ArrayAttr({tensorIdx, 0})); // `start`

      Value alignedBase = builder.create<LLVM::GEPOp>(
          func.getLoc(), LLVM::LLVMPointerType::get(context), structPtr,
          builder.getI32ArrayAttr({tensorIdx, 1})); // `aligned_start`

      Value startIdx = builder.create<LLVM::GEPOp>(
          func.getLoc(), builder.getI64Type(), structPtr,
          builder.getI32ArrayAttr({tensorIdx, 2})); // `start_idx`

      Value sizesAndStrides = builder.create<LLVM::GEPOp>(
          func.getLoc(), LLVM::LLVMPointerType::get(context), structPtr,
          builder.getI32ArrayAttr({tensorIdx, 3})); // `sizes_and_strides`

      originalCallArgs.push_back(tensorBase);
      originalCallArgs.push_back(alignedBase);
      originalCallArgs.push_back(startIdx);

      // Iterate over size and stride pairs
      for (int i = 0; i < 2 * rank; i++) {
        Value idx = builder.create<LLVM::ConstantOp>(
            func.getLoc(), builder.getI64Type(), builder.getI64IntegerAttr(i));
        Value strideOrSize = builder.create<LLVM::LoadOp>(
            func.getLoc(), builder.getI64Type(), sizesAndStrides, idx);
        originalCallArgs.push_back(strideOrSize);
      }

      tensorIdx++;
    }

    // Call the original function
    Value callResult = builder.create<LLVM::CallOp>(
        func.getLoc(), func.getFunctionType().getReturnType(), func.getName(),
        originalCallArgs);

    // Return the result
    builder.create<LLVM::ReturnOp>(func.getLoc(), callResult);
  }
}

class EmitHelperFuncs : public impl::EmitHelperFuncsBase<EmitHelperFuncs> {
  using impl::EmitHelperFuncsBase<EmitHelperFuncs>::EmitHelperFuncs;

  void runOnOperation() final {
    auto moduleOp = getOperation();
    // only run this on our hoisted cpu op modules
    if (!moduleOp.getAttr("ttir.cpu_module")) {
      return;
    }
    generateLLVMHelpersForArgRanks(moduleOp);

    // for every func in this module, emit a corresponding unpacker
  }
};

/// This function creates an instance of the pass.
std::unique_ptr<OperationPass<ModuleOp>> createEmitHelperFuncs() {
  return std::make_unique<EmitHelperFuncsPass>();
}
} // namespace mlir::tt::llvm_util
