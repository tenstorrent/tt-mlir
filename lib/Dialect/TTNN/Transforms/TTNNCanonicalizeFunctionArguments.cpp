// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNCANONICALIZEFUNCTIONARGUMENTS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNCanonicalizeFunctionArguments
    : public impl::TTNNCanonicalizeFunctionArgumentsBase<
          TTNNCanonicalizeFunctionArguments> {

public:
  using impl::TTNNCanonicalizeFunctionArgumentsBase<
      TTNNCanonicalizeFunctionArguments>::TTNNCanonicalizeFunctionArgumentsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    // Walk through all functions in the module
    //
    moduleOp->walk([&](func::FuncOp funcOp) {
      // Skip private and const eval functions
      //
      if (funcOp.isPrivate() || ttmlir::utils::isConstEvalFunc(funcOp)) {
        return;
      }

      // Process each argument to ensure it has argument_type and
      // original_arg_num
      //
      llvm::SmallVector<DictionaryAttr> updatedArgAttrs;
      bool needsUpdate = false;

      for (size_t i = 0; i < funcOp.getNumArguments(); ++i) {
        auto existingAttrs = funcOp.getArgAttrDict(i);
        llvm::SmallVector<mlir::NamedAttribute> attrs;

        // Check if argument_type already exists
        bool hasArgumentType = false;
        bool hasOriginalArgNum = false;

        if (existingAttrs) {
          for (auto attr : existingAttrs) {
            if (attr.getName() == ttcore::ArgumentTypeAttr::name) {
              hasArgumentType = true;
            }
            if (attr.getName() == ttcore::OriginalArgPositionAttr::name) {
              hasOriginalArgNum = true;
            }
            attrs.push_back(attr);
          }
        }

        // Add argument_type if missing (default to Input)
        if (!hasArgumentType) {
          attrs.push_back(rewriter.getNamedAttr(
              ttcore::ArgumentTypeAttr::name,
              ttcore::ArgumentTypeAttr::get(&getContext(),
                                            ttcore::ArgumentType::Input)));
          needsUpdate = true;
        }

        // Add original_arg_num if missing
        if (!hasOriginalArgNum) {
          attrs.push_back(rewriter.getNamedAttr(
              ttcore::OriginalArgPositionAttr::name,
              ttcore::OriginalArgPositionAttr::get(&getContext(), i)));
          needsUpdate = true;
        }

        updatedArgAttrs.push_back(rewriter.getDictionaryAttr(attrs));
      }

      // Update function argument attributes if any changes were made
      if (needsUpdate) {
        rewriter.modifyOpInPlace(
            funcOp, [&]() { funcOp.setAllArgAttrs(updatedArgAttrs); });
      }
    });
  }
};

} // namespace mlir::tt::ttnn
