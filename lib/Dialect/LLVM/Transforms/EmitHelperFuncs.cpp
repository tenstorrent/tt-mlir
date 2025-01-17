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

void generateLLVMHelpersForArgRanks(ModuleOp moduleOp) {
  auto *context = moduleOp.getContext();
  OpBuilder builder(context);

  // Define the struct type
  auto i64Ty = builder.getI64Type();
  // auto f32Ty = builder.getF32Type();
  auto ptrTy = LLVM::LLVMPointerType::get(context);

  // auto wrappedTensorTy = LLVM::LLVMStructType::getLiteral(
  //     context, {
  //                  ptrTy, // start
  //                  ptrTy, // aligned_start
  //                  i64Ty,      // start_idx
  //                  ptrTy    // sizes_and_strides
  //              });

  for (auto func : moduleOp.getOps<LLVM::LLVMFuncOp>()) {
    if (!func->hasAttr("arg_ranks")) {
      continue;
    }

    auto argRanksAttr = llvm::dyn_cast<ArrayAttr>(func->getAttr("arg_ranks"));
    if (!argRanksAttr) {
      continue;
    }

    builder.setInsertionPointToEnd(moduleOp.getBody());

    llvm::SmallString<32> helperName(func.getName());
    helperName.append("_helper");

    auto helperFuncType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(context), {LLVM::LLVMPointerType::get(context)},
        false);

    auto helperFunc = builder.create<LLVM::LLVMFuncOp>(
        func.getLoc(), helperName, helperFuncType);

    Block *entryBlock = helperFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value structArrayPtr = entryBlock->getArgument(0);
    SmallVector<Value, 16> originalCallArgs;

    // Note: we can't create typed pointer types, but we can create the struct
    // type
    auto wrappedTensorTy = LLVM::LLVMStructType::getLiteral(
        context, {
                     LLVM::LLVMPointerType::get(context), // start
                     LLVM::LLVMPointerType::get(context), // aligned_start
                     builder.getI64Type(),                // start_idx
                     LLVM::LLVMPointerType::get(context)  // sizes_and_strides
                 });

    // First get the base pointer to array of tensors (do this once)
    // Value baseStructPtr = builder.create<LLVM::LoadOp>(
    //     func.getLoc(), LLVM::LLVMPointerType::get(context), structArrayPtr);

    // Iterate over arg_ranks to unpack tensors
    int tensorIdx = 0;
    for (auto rankAttr : argRanksAttr) {
      // Compute the offset for the current tensor (as index * size of
      // wrapped_tensor)
      Value tensorIndex = builder.create<LLVM::ConstantOp>(
          func.getLoc(), builder.getI64Type(),
          builder.getI64IntegerAttr(tensorIdx++));

      // Calculate the ptr-width offset for the tensor; 3 pointers and one i64
      // = 4.
      constexpr auto wrappedTensorSize = 4;

      Value offset = builder.create<LLVM::MulOp>(
          func.getLoc(), tensorIndex,
          builder.create<LLVM::ConstantOp>(
              func.getLoc(), builder.getI64Type(),
              builder.getI64IntegerAttr(wrappedTensorSize)));

      // Get pointer to the struct for this tensor
      Value structPtr = builder.create<LLVM::GEPOp>(
          func.getLoc(), ptrTy, ptrTy, structArrayPtr, ValueRange(offset),
          /*inbounds=*/true);

      // Load the entire struct
      Value tensorStruct = builder.create<LLVM::LoadOp>(
          func.getLoc(), wrappedTensorTy, structPtr);

      // Extract fields using extractvalue
      Value tensorBase = builder.create<LLVM::ExtractValueOp>(
          func.getLoc(), ptrTy, tensorStruct,
          builder.getDenseI64ArrayAttr({0})); // start field

      Value alignedBase = builder.create<LLVM::ExtractValueOp>(
          func.getLoc(), LLVM::LLVMPointerType::get(context), tensorStruct,
          builder.getDenseI64ArrayAttr({1})); // aligned_start field

      Value startIdx = builder.create<LLVM::ExtractValueOp>(
          func.getLoc(), builder.getI64Type(), tensorStruct,
          builder.getDenseI64ArrayAttr({2})); // start_idx field

      Value sizesAndStrides = builder.create<LLVM::ExtractValueOp>(
          func.getLoc(), LLVM::LLVMPointerType::get(context), tensorStruct,
          builder.getDenseI64ArrayAttr({3})); // sizes_and_strides field

      originalCallArgs.push_back(tensorBase);
      originalCallArgs.push_back(alignedBase);
      originalCallArgs.push_back(startIdx);

      // Iterate over size and stride pairs
      int64_t rank = mlir::cast<IntegerAttr>(rankAttr).getInt();
      for (int i = 0; i < 2 * rank; i++) {
        Value idx = builder.create<LLVM::ConstantOp>(
            func.getLoc(), builder.getI64Type(), builder.getI64IntegerAttr(i));

        Value elementPtr = builder.create<LLVM::GEPOp>(
            func.getLoc(), ptrTy, ptrTy, sizesAndStrides, ValueRange{idx});

        Value strideOrSize =
            builder.create<LLVM::LoadOp>(func.getLoc(), i64Ty, elementPtr);

        originalCallArgs.push_back(strideOrSize);
      }
    }

    // Call the function
    builder.create<LLVM::CallOp>(func.getLoc(), TypeRange(), func.getName(),
                                 originalCallArgs);

    builder.create<LLVM::ReturnOp>(func.getLoc(), ValueRange());
  }

  builder.setInsertionPointToEnd(moduleOp.getBody());
}

class LLVMEmitHelperFuncs
    : public impl::LLVMEmitHelperFuncsBase<LLVMEmitHelperFuncs> {
  using impl::LLVMEmitHelperFuncsBase<
      LLVMEmitHelperFuncs>::LLVMEmitHelperFuncsBase;
  // using impl::createLLVMEmitHelperFuncs;

  void runOnOperation() final {
    llvm::outs() << "LLVMEmitHelperFuncs::runOnOperation()\n";
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
