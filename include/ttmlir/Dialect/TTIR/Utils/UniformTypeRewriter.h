// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_UTILS_UNIFORMTYPEREWRITER_H
#define TTMLIR_DIALECT_TTIR_UTILS_UNIFORMTYPEREWRITER_H

#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include <stablehlo/dialect/StablehloOps.h>

namespace mlir::tt::ttir {
class UniformTypeRewriter : public RewritePattern {
public:
  UniformTypeRewriter(const TypeConverter &converter, MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
        converter(converter) {}

  template <typename ValueRange>
  bool convertTypes(ValueRange valueRange, SmallVector<Type> &newTypes) const {
    // for (Value val : valueRange) {
    //   for (Operation *user : val.getUsers()) {
    //     if (isa<mlir::stablehlo::StablehloDialect>(user->getDialect())) {
    //       return false;
    //     }
    //   }

    //   if (Operation *op = val.getDefiningOp()) {
    //     if (isa<mlir::stablehlo::StablehloDialect>(op->getDialect())) {
    //       return false;
    //     }
    //   }
    // }

    bool updated = false;
    for (Value val : valueRange) {
      bool doConversion = true;
      for (Operation *user : val.getUsers()) {
        if (isa<mlir::stablehlo::StablehloDialect>(user->getDialect())) {
          doConversion = false;
        }
      }

      if (Operation *op = val.getDefiningOp()) {
        if (isa<mlir::stablehlo::StablehloDialect>(op->getDialect())) {
          doConversion = false;
        }
      }

      if (!doConversion) {
        newTypes.push_back(val.getType());
        continue;
      }

      Type newType = converter.convertType(val.getType());
      if (!newType) {
        return false;
      }
      newTypes.push_back(newType);
      if (val.getType() == newType) {
        continue;
      }
      val.setType(newType);
      updated = true;
      // for (auto [operand, newType] : llvm::zip_equal(valueRange, newTypes)) {
      //   if (operand.getType() == newType) {
      //     continue;
      //   }
      //   operand.setType(newType);
      //   updated = true;
      // }
    }
    return updated;
  }

  bool convertFuncType(Operation *op, PatternRewriter &rewriter) const {
    auto funcOp = dyn_cast<func::FuncOp>(op);
    if (!funcOp) {
      return false;
    }
    SmallVector<Type> inputTypes; //(funcOp.getArgumentTypes());
    SmallVector<Type> outputTypes(funcOp.getResultTypes());
    // for (Type &ty : inputTypes) {
    //   ty = converter.convertType(ty);
    // }
    for (Type &ty : outputTypes) {
      ty = converter.convertType(ty);
    }

    convertTypes(funcOp.getArguments(), inputTypes);

    auto newType = rewriter.getType<FunctionType>(inputTypes, outputTypes);
    if (funcOp.getFunctionType() == newType) {
      return false;
    }
    funcOp.setFunctionType(newType);

    if (funcOp.isDeclaration()) {
      return true;
    }

    Block &entryBlock = funcOp.getBody().front();
    for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
      entryBlock.getArgument(i).setType(inputTypes[i]);
    }

    return true;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Skip if we're inside a GenericOp.
    if (mlir::isa<GenericOp>(op->getParentOp()) ||
        isa<mlir::stablehlo::StablehloDialect>(op->getDialect())) {
      return failure();
    }
    bool updated = false;
    SmallVector<Type> operands;
    SmallVector<Type> results;
    rewriter.startOpModification(op);
    updated |= convertTypes(op->getOperands(), operands);
    updated |= convertTypes(op->getResults(), results);
    updated |= convertFuncType(op, rewriter);
    if (!updated) {
      rewriter.cancelOpModification(op);
      return failure();
    }
    rewriter.finalizeOpModification(op);
    return success();
  }

  TypeConverter converter;
};
} // namespace mlir::tt::ttir

#endif
