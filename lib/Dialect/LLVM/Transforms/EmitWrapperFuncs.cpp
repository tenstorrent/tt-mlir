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
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"

namespace mlir::tt::llvm_util {
#define GEN_PASS_DEF_LLVMEMITCALLINGCONVENTIONWRAPPERFUNCS
#include "ttmlir/Dialect/LLVM/Transforms/Passes.h.inc"

// Generate wrapper func which
void generateLLVMWrappersForArgRanks(ModuleOp moduleOp) {
  auto *context = moduleOp.getContext();
  OpBuilder builder(context);

  auto ptrTy = LLVM::LLVMPointerType::get(context);

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

    auto helperFunc = LLVM::LLVMFuncOp::create(builder, func.getLoc(),
                                               helperName, helperFuncType);

    Block *entryBlock = helperFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value structArrayPtr = entryBlock->getArgument(0);
    SmallVector<Value, 16> originalCallArgs;

    // Note we can't create typed pointer types, which is annoying.
    auto wrappedTensorTy = LLVM::LLVMStructType::getLiteral(
        context, {
                     LLVM::LLVMPointerType::get(context), // start
                     LLVM::LLVMPointerType::get(context), // aligned_start
                     builder.getI64Type(),                // start_idx
                     LLVM::LLVMPointerType::get(context)  // sizes_and_strides
                 });

    // Iterate over arg_ranks to unpack tensors.
    int tensorIdx = 0;
    for (auto rankAttr : argRanksAttr) {
      // Compute the offset for the current tensor (as index * size of
      // wrapped_tensor).
      Value tensorIndex =
          LLVM::ConstantOp::create(builder, func.getLoc(), builder.getI64Type(),
                                   builder.getI64IntegerAttr(tensorIdx++));

      // Calculate the ptr-width offset for the tensor; 3 pointers and one i64
      // = 4.
      constexpr auto wrappedTensorSize = 4;

      Value offset = LLVM::MulOp::create(
          builder, func.getLoc(), tensorIndex,
          LLVM::ConstantOp::create(
              builder, func.getLoc(), builder.getI64Type(),
              builder.getI64IntegerAttr(wrappedTensorSize)));

      // Get pointer to the struct for this offset-th tensor in input array.
      Value structPtr = LLVM::GEPOp::create(
          builder, func.getLoc(), ptrTy, ptrTy, structArrayPtr,
          ValueRange(offset), LLVM::GEPNoWrapFlags::inbounds);

      // Load actual tensor object from pointer so we can extract its members.
      Value tensorStruct = LLVM::LoadOp::create(builder, func.getLoc(),
                                                wrappedTensorTy, structPtr);

      Value tensorBase = LLVM::ExtractValueOp::create(
          builder, func.getLoc(), ptrTy, tensorStruct,
          builder.getDenseI64ArrayAttr({0}));
      originalCallArgs.push_back(tensorBase);

      Value alignedBase = LLVM::ExtractValueOp::create(
          builder, func.getLoc(), LLVM::LLVMPointerType::get(context),
          tensorStruct, builder.getDenseI64ArrayAttr({1}));
      originalCallArgs.push_back(alignedBase);

      Value startIdx = LLVM::ExtractValueOp::create(
          builder, func.getLoc(), builder.getI64Type(), tensorStruct,
          builder.getDenseI64ArrayAttr({2}));
      originalCallArgs.push_back(startIdx);

      Value sizesAndStrides = LLVM::ExtractValueOp::create(
          builder, func.getLoc(), LLVM::LLVMPointerType::get(context),
          tensorStruct, builder.getDenseI64ArrayAttr({3}));
      // The sizesAndStrides field is an array itself, so we need to step into
      // it and extract elements.
      int64_t rank = mlir::cast<IntegerAttr>(rankAttr).getInt();
      for (int i = 0; i < 2 * rank; i++) {
        Value idx = LLVM::ConstantOp::create(builder, func.getLoc(),
                                             builder.getI64Type(),
                                             builder.getI64IntegerAttr(i));

        Value elementPtr =
            LLVM::GEPOp::create(builder, func.getLoc(), ptrTy, ptrTy,
                                sizesAndStrides, ValueRange{idx});

        Value strideOrSize = LLVM::LoadOp::create(
            builder, func.getLoc(), builder.getI64Type(), elementPtr);

        originalCallArgs.push_back(strideOrSize);
      }
    }

    // Call the original functions with the unpacked args.
    LLVM::CallOp::create(builder, func.getLoc(), TypeRange(), func.getName(),
                         originalCallArgs);

    LLVM::ReturnOp::create(builder, func.getLoc(), ValueRange());
  }

  builder.setInsertionPointToEnd(moduleOp.getBody());
}

class LLVMEmitCallingConventionWrapperFuncs
    : public impl::LLVMEmitCallingConventionWrapperFuncsBase<
          LLVMEmitCallingConventionWrapperFuncs> {
  using impl::LLVMEmitCallingConventionWrapperFuncsBase<
      LLVMEmitCallingConventionWrapperFuncs>::
      LLVMEmitCallingConventionWrapperFuncsBase;

  void runOnOperation() final {
    generateLLVMWrappersForArgRanks(getOperation());
  }
};

} // namespace mlir::tt::llvm_util
