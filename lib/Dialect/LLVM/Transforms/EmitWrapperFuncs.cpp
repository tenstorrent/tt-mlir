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

// Generates LLVM wrapper functions for CPU-hoisted functions that:
// - unpack WrappedTensor arguments from an array of WrappedTensors and
//   pass them as individual arguments to the original function
// - pack returned memref descriptors into WrappedTensors and return them
//   as an array of WrappedTensors
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

    auto resultRanksAttr =
        llvm::dyn_cast<ArrayAttr>(func->getAttr("result_ranks"));
    if (!resultRanksAttr) {
      continue;
    }

    builder.setInsertionPointToEnd(moduleOp.getBody());

    llvm::SmallString<32> helperName(func.getName());
    helperName.append("_helper");

    // Get result ranks from attribute (set by HoistCPUOps).
    SmallVector<int64_t, 4> resultRanks;
    for (auto attr : resultRanksAttr) {
      resultRanks.push_back(mlir::cast<IntegerAttr>(attr).getInt());
    }

    bool hasOutputs = !resultRanks.empty();

    // Helper returns ptr (to array of WrappedTensors) if there are outputs,
    // void otherwise.
    Type helperReturnType =
        hasOutputs ? Type(ptrTy) : Type(LLVM::LLVMVoidType::get(context));

    auto helperFuncType = LLVM::LLVMFunctionType::get(
        helperReturnType, {LLVM::LLVMPointerType::get(context)}, false);

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

    // CPU-hoisted functions return results as memref descriptors:
    // !llvm.struct<(ptr, ptr, i64, array<rank x i64>, array<rank x i64>)>
    // which corresponds to (basePtr, alignedBasePtr, offset, sizes, strides).
    // We need to wrap these back into WrappedTensor structs for the caller.

    if (!hasOutputs) {
      LLVM::CallOp::create(builder, func.getLoc(), TypeRange(), func.getName(),
                           originalCallArgs);
      LLVM::ReturnOp::create(builder, func.getLoc(), ValueRange());
    } else {
      // Call original function and pack results into WrappedTensors.
      auto returnType = func.getFunctionType().getReturnType();
      Value result = LLVM::CallOp::create(builder, func.getLoc(), returnType,
                                          func.getName(), originalCallArgs)
                         .getResult();

      auto mallocFunc = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>("malloc");
      auto loc = func.getLoc();
      auto i64Ty = builder.getI64Type();

      auto makeConst = [&](int64_t val) {
        return LLVM::ConstantOp::create(builder, loc, i64Ty,
                                        builder.getI64IntegerAttr(val));
      };

      // Allocate output array and sizesAndStrides buffer.
      constexpr int64_t kWrappedTensorBytes = 32; // 4 fields * 8 bytes
      int64_t numOutputs = resultRanks.size();
      int64_t totalSizesStridesBytes = 0;
      for (int64_t rank : resultRanks) {
        totalSizesStridesBytes += 2 * rank * 8;
      }

      Value outputArrayPtr =
          LLVM::CallOp::create(
              builder, loc, ptrTy, mallocFunc.getName(),
              ValueRange{makeConst(numOutputs * kWrappedTensorBytes)})
              .getResult();
      Value sizesStridesBase =
          LLVM::CallOp::create(builder, loc, ptrTy, mallocFunc.getName(),
                               ValueRange{makeConst(totalSizesStridesBytes)})
              .getResult();

      int64_t sizesStridesOffset = 0;
      for (int64_t outIdx = 0; outIdx < numOutputs; ++outIdx) {
        int64_t rank = resultRanks[outIdx];

        // For single output, result is the descriptor; otherwise extract it.
        Value desc = (numOutputs == 1)
                         ? result
                         : LLVM::ExtractValueOp::create(
                               builder, loc, result,
                               builder.getDenseI64ArrayAttr({outIdx}));

        // Extract memref descriptor fields.
        Value basePtr = LLVM::ExtractValueOp::create(
            builder, loc, ptrTy, desc, builder.getDenseI64ArrayAttr({0}));
        Value alignedPtr = LLVM::ExtractValueOp::create(
            builder, loc, ptrTy, desc, builder.getDenseI64ArrayAttr({1}));
        Value offset = LLVM::ExtractValueOp::create(
            builder, loc, i64Ty, desc, builder.getDenseI64ArrayAttr({2}));

        // Get pointer to this output's sizesAndStrides array.
        Value sizesStridesPtr = LLVM::GEPOp::create(
            builder, loc, ptrTy, builder.getI8Type(), sizesStridesBase,
            ValueRange{makeConst(sizesStridesOffset)});

        // Copy sizes and strides from descriptor to sizesAndStrides array.
        for (int64_t i = 0; i < 2 * rank; ++i) {
          int64_t structIdx = (i < rank) ? 3 : 4;
          int64_t arrayIdx = (i < rank) ? i : i - rank;
          Value val = LLVM::ExtractValueOp::create(
              builder, loc, i64Ty, desc,
              builder.getDenseI64ArrayAttr({structIdx, arrayIdx}));
          Value destPtr =
              LLVM::GEPOp::create(builder, loc, ptrTy, i64Ty, sizesStridesPtr,
                                  ValueRange{makeConst(i)});
          LLVM::StoreOp::create(builder, loc, val, destPtr);
        }

        // Build and store WrappedTensor struct.
        Value wrapped = LLVM::UndefOp::create(builder, loc, wrappedTensorTy);
        wrapped = LLVM::InsertValueOp::create(
            builder, loc, wrapped, basePtr, builder.getDenseI64ArrayAttr({0}));
        wrapped =
            LLVM::InsertValueOp::create(builder, loc, wrapped, alignedPtr,
                                        builder.getDenseI64ArrayAttr({1}));
        wrapped = LLVM::InsertValueOp::create(
            builder, loc, wrapped, offset, builder.getDenseI64ArrayAttr({2}));
        wrapped =
            LLVM::InsertValueOp::create(builder, loc, wrapped, sizesStridesPtr,
                                        builder.getDenseI64ArrayAttr({3}));

        Value outPtr =
            LLVM::GEPOp::create(builder, loc, ptrTy, wrappedTensorTy,
                                outputArrayPtr, ValueRange{makeConst(outIdx)});
        LLVM::StoreOp::create(builder, loc, wrapped, outPtr);

        sizesStridesOffset += 2 * rank * 8;
      }

      LLVM::ReturnOp::create(builder, loc, ValueRange{outputArrayPtr});
    }
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
