// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNSPLITFUNCTIONARGUMENTS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNSplitFunctionArguments
    : public impl::TTNNSplitFunctionArgumentsBase<TTNNSplitFunctionArguments> {

public:
  using impl::TTNNSplitFunctionArgumentsBase<
      TTNNSplitFunctionArguments>::TTNNSplitFunctionArgumentsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    // Walk through all functions in the module
    //
    moduleOp->walk([&](func::FuncOp funcOp) {
      // Skip private functions
      //
      if (funcOp.isPrivate()) {
        return;
      }

      // Collect arguments by their argument type
      //
      llvm::SmallVector<BlockArgument> inputArgs;
      llvm::SmallVector<BlockArgument> paramArgs;

      for (BlockArgument arg : funcOp.getArguments()) {
        if (auto typeAttr = funcOp.getArgAttrOfType<ttcore::ArgumentTypeAttr>(
                arg.getArgNumber(), ttcore::ArgumentTypeAttr::name)) {
          auto argTypeValue = typeAttr.getValue();
          if (argTypeValue == ttcore::ArgumentType::Input ||
              argTypeValue == ttcore::ArgumentType::Default) {
            inputArgs.push_back(arg);
          } else if (argTypeValue == ttcore::ArgumentType::Parameter ||
                     argTypeValue == ttcore::ArgumentType::Constant) {
            paramArgs.push_back(arg);
          }
        }
      }

      // If all arguments are already grouped correctly, skip
      //
      if (inputArgs.empty() || paramArgs.empty()) {
        return;
      }

      // Check if arguments are already in the correct order
      // (all inputs before all parameters)
      //
      bool needsReordering = false;
      size_t lastInputIdx = 0;
      size_t firstParamIdx = funcOp.getNumArguments();

      for (BlockArgument inputArg : inputArgs) {
        lastInputIdx = std::max(lastInputIdx,
                                static_cast<size_t>(inputArg.getArgNumber()));
      }

      for (BlockArgument paramArg : paramArgs) {
        firstParamIdx = std::min(firstParamIdx,
                                 static_cast<size_t>(paramArg.getArgNumber()));
      }

      if (lastInputIdx >= firstParamIdx) {
        needsReordering = true;
      }

      if (!needsReordering) {
        return;
      }

      // Create new argument order: inputs first, then parameters
      //
      llvm::SmallVector<BlockArgument> newArgOrder;
      newArgOrder.append(inputArgs.begin(), inputArgs.end());
      newArgOrder.append(paramArgs.begin(), paramArgs.end());

      // Create new function type with reordered arguments
      //
      llvm::SmallVector<Type> newArgTypes;
      llvm::SmallVector<DictionaryAttr> newArgAttrs;

      for (BlockArgument arg : newArgOrder) {
        newArgTypes.push_back(arg.getType());

        // Get existing attributes and add original argument number
        //
        auto existingAttrs = funcOp.getArgAttrDict(arg.getArgNumber());
        llvm::SmallVector<mlir::NamedAttribute> attrs;

        // Copy existing attributes
        if (existingAttrs) {
          for (auto attr : existingAttrs) {
            attrs.push_back(attr);
          }
        }

        // Add original argument number attribute
        attrs.push_back(rewriter.getNamedAttr(
            "ttcore.original_arg_num",
            rewriter.getI64IntegerAttr(arg.getArgNumber())));

        newArgAttrs.push_back(rewriter.getDictionaryAttr(attrs));
      }

      FunctionType newFuncType = FunctionType::get(
          &getContext(), newArgTypes, funcOp.getFunctionType().getResults());

      // Update the function signature
      //
      rewriter.modifyOpInPlace(funcOp, [&]() {
        funcOp.setType(newFuncType);
        funcOp.setAllArgAttrs(newArgAttrs);
      });

      // Update the entry block arguments
      //
      Block &entryBlock = funcOp.getBlocks().front();

      // Store the mapping from old block arguments to their values before we
      // modify anything
      //
      llvm::SmallVector<BlockArgument> oldArguments;
      for (BlockArgument arg : newArgOrder) {
        oldArguments.push_back(entryBlock.getArgument(arg.getArgNumber()));
      }

      // Insert new arguments at the end temporarily
      //
      size_t originalNumArgs = entryBlock.getNumArguments();
      for (size_t i = 0; i < newArgOrder.size(); ++i) {
        BlockArgument oldArg = newArgOrder[i];
        entryBlock.addArgument(oldArg.getType(), funcOp.getLoc());
      }

      // Replace uses of old arguments with new ones
      //
      for (size_t i = 0; i < oldArguments.size(); ++i) {
        BlockArgument oldArg = oldArguments[i];
        BlockArgument newArg = entryBlock.getArgument(originalNumArgs + i);
        rewriter.replaceAllUsesWith(oldArg, newArg);
      }

      // Erase the old arguments (they are at the beginning)
      //
      entryBlock.eraseArguments(0, originalNumArgs);

      // Update all call sites for this function
      //
      moduleOp->walk([&](func::CallOp callOp) {
        if (callOp.getCallee() == funcOp.getName()) {
          llvm::SmallVector<Value> newOperands;
          for (BlockArgument arg : newArgOrder) {
            newOperands.push_back(callOp.getOperand(arg.getArgNumber()));
          }
          rewriter.modifyOpInPlace(callOp, [&]() {
            callOp.getOperandsMutable().assign(newOperands);
          });
        }
      });
    });
  }
};

} // namespace mlir::tt::ttnn
