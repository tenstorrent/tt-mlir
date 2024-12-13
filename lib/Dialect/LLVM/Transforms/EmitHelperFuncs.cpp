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

#define GEN_PASS_DEF_LLVMEmitHelperFuncs
void generateLLVMHelpersForArgRanks(mlir::ModuleOp moduleOp) {
  auto *context = moduleOp.getContext();
  OpBuilder builder(context);

  for (auto func : moduleOp.getOps<LLVM::LLVMFuncOp>()) {
    if (!func->hasAttr("arg_ranks"))
      continue;

    // Extract the `arg_ranks` attribute
    auto argRanksAttr = func->getAttr("arg_ranks").dyn_cast<ArrayAttr>();
    if (!argRanksAttr)
      continue;

    // Define the helper function name and type
    std::string helperName = func.getName().str() + "_helper";
    auto llvmFuncType = func.getType();

    // Create the helper function
    auto helperFuncType = LLVM::LLVMFunctionType::get(
        llvmFuncType.getReturnType(),
        {LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getLiteral(
            context,
            {
                LLVM::LLVMPointerType::get(builder.getF32Type()), // start
                LLVM::LLVMPointerType::get(
                    builder.getF32Type()), // aligned_start
                builder.getI64Type(),      // start_idx
                LLVM::LLVMPointerType::get(
                    builder.getI64Type()) // sizes_and_strides
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
      int64_t rank = rankAttr.cast<IntegerAttr>().getInt();

      // Generate GEP and loads for each field in the struct
      Value tensorBase = builder.create<LLVM::GEPOp>(
          func.getLoc(), LLVM::LLVMPointerType::get(builder.getF32Type()),
          structPtr, builder.getI32ArrayAttr({tensorIdx, 0})); // `start`

      Value alignedBase = builder.create<LLVM::GEPOp>(
          func.getLoc(), LLVM::LLVMPointerType::get(builder.getF32Type()),
          structPtr,
          builder.getI32ArrayAttr({tensorIdx, 1})); // `aligned_start`

      Value startIdx = builder.create<LLVM::GEPOp>(
          func.getLoc(), builder.getI64Type(), structPtr,
          builder.getI32ArrayAttr({tensorIdx, 2})); // `start_idx`

      Value sizesAndStrides = builder.create<LLVM::GEPOp>(
          func.getLoc(), LLVM::LLVMPointerType::get(builder.getI64Type()),
          structPtr,
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
        func.getLoc(), func.getFunctionType().getReturnType(),
        func.getSymbolRef(), originalCallArgs);

    // Return the result
    builder.create<LLVM::ReturnOp>(func.getLoc(), callResult);
  }
}

class LLVMEmitHelperFuncs
    : public impl::LLVMEmitHelperFuncsBase<LLVMEmitHelperFuncs> {
  using impl::LLVMEmitHelperFuncsBase<LLVMEmitHelperFuncs>::LLVMEmitHelperFuncs;

  void runOnOperation() final {
    auto moduleOp = getOperation();
    // only run this on our hoisted cpu op modules
    if (!moduleOp.getAttr("ttir.cpu_module")) {
      return;
    }
    generateLLVMHelpersForArgRanks(moduleOp);

    // for every func in this module, emit a corresponding unpacker
  }
}

/// This function creates an instance of the pass.
std::unique_ptr<mlir::Pass>
createLLVMEmitHelperFuncs() {
  return std::make_unique<LLVMEmitHelperFuncsPass>();
}
} // namespace mlir::tt::llvm_util
