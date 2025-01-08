// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include "ttmlir/Dialect/LLVM/Transforms/Passes.h"
#include "ttmlir/Dialect/TT/IR/TTOps.h"

namespace mlir::tt::llvm_util {
#define GEN_PASS_DEF_LLVMEMITHELPERFUNCS
#include "ttmlir/Dialect/LLVM/Transforms/Passes.h.inc"

void generateLLVMHelpersForArgRanks(tt::CPUModuleOp moduleOp) {
  auto *context = moduleOp.getContext();
  OpBuilder builder(context);

  for (auto func : moduleOp.getOps<LLVM::LLVMFuncOp>()) {
    if (!func->hasAttr("arg_ranks")) {
      continue;
    }

    // Extract the `arg_ranks` attribute
    auto argRanksAttr = llvm::dyn_cast<ArrayAttr>(func->getAttr("arg_ranks"));
    if (!argRanksAttr) {
      continue;
    }

    builder.setInsertionPointToEnd(&moduleOp.getBody().front());

    // Define the helper function name and type
    llvm::SmallString<32> helperName(func.getName());
    helperName.append("_helper");

    // Create the helper function
    auto helperFuncType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(context), {LLVM::LLVMPointerType::get(context)},
        false);

    auto helperFunc = builder.create<LLVM::LLVMFuncOp>(
        func.getLoc(), helperName, helperFuncType);

    Block *entryBlock = helperFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    // Unpack the argument
    Value structArrayPtr = entryBlock->getArgument(0);
    SmallVector<Value, 16> originalCallArgs;

    // Iterate over arg_ranks to unpack tensors
    int tensorIdx = 0;
    for (auto rankAttr : argRanksAttr) {
      Value tensorIndex = builder.create<LLVM::ConstantOp>(
          func.getLoc(), builder.getI32Type(),
          builder.getI32IntegerAttr(tensorIdx++));

      Value structPtr = builder.create<LLVM::GEPOp>(
          func.getLoc(), LLVM::LLVMPointerType::get(context),
          LLVM::LLVMPointerType::get(context), structArrayPtr,
          ValueRange(tensorIndex));

      int64_t rank = mlir::cast<IntegerAttr>(rankAttr).getInt();

      Value index = builder.create<LLVM::ConstantOp>(
          func.getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(0));
      // `start`
      Value tensorBase = builder.create<LLVM::GEPOp>(
          func.getLoc(), LLVM::LLVMPointerType::get(context),
          LLVM::LLVMPointerType::get(context), structPtr, ValueRange{index});

      index = builder.create<LLVM::ConstantOp>(
          func.getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(1));
      // `aligned_start`
      Value alignedBase = builder.create<LLVM::GEPOp>(
          func.getLoc(), LLVM::LLVMPointerType::get(context),
          LLVM::LLVMPointerType::get(context), structPtr, ValueRange{index});

      index = builder.create<LLVM::ConstantOp>(
          func.getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(2));
      // `start_idx`
      Value startIdxPtr = builder.create<LLVM::GEPOp>(
          func.getLoc(), LLVM::LLVMPointerType::get(context),
          builder.getI64Type(), structPtr, ValueRange{index});
      // Convert the pointer to an integer (i64)
      Value startIdx = builder.create<LLVM::PtrToIntOp>(
          func.getLoc(), builder.getI64Type(), startIdxPtr);

      index = builder.create<LLVM::ConstantOp>(
          func.getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(3));
      // `sizes_and_strides`
      Value sizesAndStrides = builder.create<LLVM::GEPOp>(
          func.getLoc(), LLVM::LLVMPointerType::get(context),
          LLVM::LLVMPointerType::get(context), structPtr, ValueRange{index});

      originalCallArgs.push_back(tensorBase);
      originalCallArgs.push_back(alignedBase);
      originalCallArgs.push_back(startIdx);

      // Iterate over size and stride pairs
      for (int i = 0; i < 2 * rank; i++) {
        // Compute the address of the i-th element
        Value idx = builder.create<LLVM::ConstantOp>(
            func.getLoc(), builder.getI64Type(), builder.getI64IntegerAttr(i));

        Value elementPtr = builder.create<LLVM::GEPOp>(
            func.getLoc(),
            LLVM::LLVMPointerType::get(context), // Pointer to i64
            builder.getI64Type(),
            sizesAndStrides, // Base pointer
            ValueRange{idx}  // Offset
        );

        // Load the value from the computed address
        Value strideOrSize = builder.create<LLVM::LoadOp>(
            func.getLoc(),
            builder.getI64Type(), // Type of the loaded value
            elementPtr            // Computed address
        );

        // Add the loaded value to the call arguments
        originalCallArgs.push_back(strideOrSize);
      }
    }

    // Call the original function
    builder.create<LLVM::CallOp>(func.getLoc(),
                                 func.getFunctionType().getReturnType(),
                                 func.getName(), originalCallArgs);

    // Return the result
    builder.create<LLVM::ReturnOp>(func.getLoc(), ValueRange());
  }
}

class LLVMEmitHelperFuncs
    : public impl::LLVMEmitHelperFuncsBase<LLVMEmitHelperFuncs> {
  using impl::LLVMEmitHelperFuncsBase<
      LLVMEmitHelperFuncs>::LLVMEmitHelperFuncsBase;
  // using impl::createLLVMEmitHelperFuncs;

  void runOnOperation() final {
    auto moduleOp = getOperation();
    // only run this on our hoisted cpu op modules
    // if (!moduleOp->getAttr("ttir.cpu_module")) {
    //   return;
    // }
    generateLLVMHelpersForArgRanks(moduleOp);

    // for every func in this module, emit a corresponding unpacker
  }
};

} // namespace mlir::tt::llvm_util
