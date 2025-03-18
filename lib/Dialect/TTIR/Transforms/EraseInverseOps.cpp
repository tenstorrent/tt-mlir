// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Utils.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRERASEINVERSEOPS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
class TTIRCommuteTmsAboveElementwiseRewriter : public RewritePattern {
public:
  TTIRCommuteTmsAboveElementwiseRewriter(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult match(Operation *op) const override {
    if (failed(checkTrait(op))) {
      // The op should support implicit broadcast to fold them.
      return failure();
    }
    return shouldCommute(op);
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    SmallVector<Operation *> users(op->getUsers());
    if (isa<ttir::TransposeOp>(users[0])) {
      commuteTmsThroughEltwise<ttir::TransposeOp>(op, users, op->getOperands(),
                                                  rewriter);
    } else if (isa<ttir::PermuteOp>(users[0])) {
      commuteTmsThroughEltwise<ttir::PermuteOp>(op, users, op->getOperands(),
                                                rewriter);
    } else if (isa<ttir::ReshapeOp>(users[0])) {
      commuteTmsThroughEltwise<ttir::ReshapeOp>(op, users, op->getOperands(),
                                                rewriter);
    } else {
      llvm_unreachable("users[0] must be one of ttir::TransposeOp, "
                       "ttir::PermuteOp, ttir::ReshapeOp");
    }
  }

  LogicalResult
  checkAllUsersAreIdenticalTms(SmallVector<Operation *> users) const {
    Operation *firstUser = users[0];
    for (auto *user : users) {
      if (user->getAttrDictionary() != firstUser->getAttrDictionary()) {
        return failure();
      }
    }
    return success(
        isa<ttir::TransposeOp, ttir::PermuteOp, ttir::ReshapeOp>(firstUser));
  }

  LogicalResult checkAllOperandsHaveSameShape(ValueRange operands) const {
    RankedTensorType firstOperandType =
        cast<RankedTensorType>(operands[0].getType());
    for (Value operand : operands) {
      RankedTensorType operandType = cast<RankedTensorType>(operand.getType());
      if (operandType.getShape() != firstOperandType.getShape()) {
        return failure();
      }
    }
    return success();
  }

private:
  LogicalResult virtual shouldCommute(Operation *op) const {
    llvm_unreachable("shouldCommute must be overridden");
  };

  LogicalResult virtual checkTrait(Operation *op) const {
    llvm_unreachable("checkTrait must be overridden");
  };

  template <typename TMOpType>
  void commuteTmsThroughEltwise(Operation *op, SmallVector<Operation *> users,
                                ValueRange operands,
                                PatternRewriter &rewriter) const {
    Operation *user = users[0];
    auto newEltwiseType = cast<RankedTensorType>(user->getResult(0).getType());

    SmallVector<mlir::tensor::EmptyOp> newTransposeDPSOperands;
    SmallVector<TMOpType> newTransposes;
    for (uint32_t operandIdx = 0; operandIdx < op->getNumOperands() - 1;
         operandIdx++) {
      newTransposeDPSOperands.push_back(rewriter.create<tensor::EmptyOp>(
          op->getLoc(), newEltwiseType.getShape(),
          newEltwiseType.getElementType()));

      TMOpType newTranspose = cast<TMOpType>(rewriter.clone(*user));
      newTranspose->setOperand(newTranspose->getNumOperands() - 1,
                               newTransposeDPSOperands[operandIdx]);
      newTransposes.push_back(newTranspose);
    }

    mlir::tensor::EmptyOp newEltwiseDPS = rewriter.create<tensor::EmptyOp>(
        op->getLoc(), newEltwiseType.getShape(),
        newEltwiseType.getElementType());
    Operation *newEltwise = rewriter.clone(*op);

    // Do not want to put a clone on the DPS operand
    for (uint32_t operandIdx = 0; operandIdx < newEltwise->getNumOperands() - 1;
         operandIdx++) {
      newTransposes[operandIdx]->setOperand(0, operands[operandIdx]);
      newTransposes[operandIdx]->setOperand(
          1, newTransposeDPSOperands[operandIdx]);
      newEltwise->setOperand(operandIdx,
                             newTransposes[operandIdx]->getResult(0));
    }
    newEltwise->setOperand(newEltwise->getNumOperands() - 1,
                           newEltwiseDPS->getResult(0));
    newEltwise->getResult(0).setType(newEltwiseType);

    for (auto *user : users) {
      rewriter.replaceOp(user, newEltwise);
    }
  }
};

class TTIRCommuteTmsAboveElementwiseUnaryRewriter
    : public TTIRCommuteTmsAboveElementwiseRewriter {
public:
  TTIRCommuteTmsAboveElementwiseUnaryRewriter(MLIRContext *ctx)
      : TTIRCommuteTmsAboveElementwiseRewriter(ctx) {}

private:
  LogicalResult checkTrait(Operation *op) const override {
    return success(op->hasTrait<ElementwiseUnary::Trait>());
  }

  LogicalResult shouldCommute(Operation *op) const override {
    // For now we always want to commute through unary elementwise ops if all
    // the users are identical
    SmallVector<Operation *> users(op->getUsers());
    return success(succeeded(checkAllUsersAreIdenticalTms(users)) &&
                   succeeded(checkAllOperandsHaveSameShape(op->getOperands())));
  };
};

class TTIRCommuteTmsAboveElementwiseBinaryRewriter
    : public TTIRCommuteTmsAboveElementwiseRewriter {
public:
  TTIRCommuteTmsAboveElementwiseBinaryRewriter(MLIRContext *ctx)
      : TTIRCommuteTmsAboveElementwiseRewriter(ctx) {}

private:
  LogicalResult checkTrait(Operation *op) const override {
    return success(op->hasTrait<ElementwiseBinary::Trait>());
  }

  LogicalResult shouldCommute(Operation *op) const override {
    // For now we always want to commute through unary elementwise ops if all
    // the users are identical
    SmallVector<Operation *> users(op->getUsers());
    return success(succeeded(checkAllUsersAreIdenticalTms(users)) &&
                   succeeded(checkAllOperandsHaveSameShape(op->getOperands())));
  };
};
} // namespace

namespace {
class TTIREraseInverseTransposes : public OpRewritePattern<ttir::TransposeOp> {
public:
  using OpRewritePattern<ttir::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::TransposeOp op,
                                PatternRewriter &rewriter) const override {

    // Erase with operand
    if (!op->getOperand(0).getDefiningOp()) {
      return failure();
    }
    ttir::TransposeOp operand =
        dyn_cast<ttir::TransposeOp>(op->getOperand(0).getDefiningOp());
    if (!operand) {
      return failure();
    }

    auto opDim0 = op.getDim0();
    auto opDim1 = op.getDim1();
    auto operandDim0 = operand.getDim0();
    auto operandDim1 = operand.getDim1();

    if ((opDim0 == operandDim1 && opDim1 == operandDim0) ||
        (opDim0 == operandDim0 && opDim1 == operandDim1)) {
      rewriter.replaceOp(op, operand->getOperand(0));
      return success();
    }
    return failure();
  }
};
} // namespace

namespace {
class TTIREraseInversePermutations : public OpRewritePattern<ttir::PermuteOp> {
public:
  using OpRewritePattern<ttir::PermuteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::PermuteOp op,
                                PatternRewriter &rewriter) const override {

    // Erase with operand
    if (!op->getOperand(0).getDefiningOp()) {
      return failure();
    }
    ttir::PermuteOp operand =
        dyn_cast<ttir::PermuteOp>(op->getOperand(0).getDefiningOp());
    if (!operand) {
      return failure();
    }

    // Apply the permutation of this op to the permuatation of the operand
    // If the result is the identity permutation, erase the ops
    ArrayRef<int64_t> opPemutation = op.getPermutation();
    ArrayRef<int64_t> operandPermutation = operand.getPermutation();

    SmallVector<int64_t> newPermutation;
    for (int64_t i = 0; i < static_cast<int64_t>(opPemutation.size()); i++) {
      if (operandPermutation[opPemutation[i]] != i) {
        return failure();
      };
    }

    rewriter.replaceOp(op, operand->getOperand(0));
    return success();
  }
};
} // namespace

namespace {
class TTIREraseInverseReshapes : public OpRewritePattern<ttir::ReshapeOp> {
public:
  using OpRewritePattern<ttir::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::ReshapeOp op,
                                PatternRewriter &rewriter) const override {

    // Erase with operand
    if (!op->getOperand(0).getDefiningOp()) {
      return failure();
    }
    ttir::ReshapeOp operand =
        dyn_cast<ttir::ReshapeOp>(op->getOperand(0).getDefiningOp());
    if (!operand) {
      return failure();
    }

    // if the input shape of the operand is the same as the output shape of this
    // op, erase the ops
    auto opShape =
        cast<RankedTensorType>(op->getResult(0).getType()).getShape();
    auto inputShape =
        cast<RankedTensorType>(operand->getOperand(0).getType()).getShape();

    if (opShape != inputShape) {
      return failure();
    }
    rewriter.replaceOp(op, operand->getOperand(0));
    return success();
  }
};
} // namespace

namespace {

ttir::ReshapeOp
generateTTIRReshape(mlir::TypedValue<mlir::RankedTensorType> input,
                    ArrayRef<int64_t> newShape, PatternRewriter &rewriter) {
  // With reshape op, the output layout changes due to new output shape, hence
  // we need to create a new output layout attribute with the new shape.
  RankedTensorType inputType = input.getType();

  // Create a new output type for reshape operation with new shape and new
  // output layout.
  RankedTensorType outputType =
      RankedTensorType::get(newShape, inputType.getElementType());

  llvm::SmallVector<int32_t> newShapeI32(newShape.begin(), newShape.end());
  auto reshapeDPS = rewriter.create<tensor::EmptyOp>(
      input.getLoc(), outputType.getShape(), outputType.getElementType());

  return rewriter.create<ttir::ReshapeOp>(
      input.getLoc(), outputType, input, reshapeDPS,
      rewriter.getI32ArrayAttr(newShapeI32));
}

ttir::ReshapeOp
generateTTIRNHWFlatten(mlir::TypedValue<mlir::RankedTensorType> input,
                       PatternRewriter &rewriter) {
  llvm::ArrayRef<int64_t> shape = input.getType().getShape();

  assert(shape.size() == 4 && "Must have 4-dim tensor as conv2d input");

  llvm::SmallVector<int64_t> newShape = {1, 1, shape[0] * shape[1] * shape[2],
                                         shape[3]};
  return generateTTIRReshape(input, newShape, rewriter);
}
class ConvertToOptimizedFlattenedConv2dPattern
    : public OpConversionPattern<ttir::Conv2dOp> {
public:
  using OpConversionPattern<ttir::Conv2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::Conv2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto inputTy = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    auto kernelTy = mlir::cast<RankedTensorType>(adaptor.getWeight().getType());
    auto outputTy = op.getResult().getType();

    auto batchSizeAttr = rewriter.getI32IntegerAttr(inputTy.getDimSize(0));
    auto inputHeightAttr = rewriter.getI32IntegerAttr(inputTy.getDimSize(1));
    auto inputWidthAttr = rewriter.getI32IntegerAttr(inputTy.getDimSize(2));
    auto inChannelsAttr = rewriter.getI32IntegerAttr(inputTy.getDimSize(3));
    auto outChannelsAttr = rewriter.getI32IntegerAttr(outputTy.getDimSize(3));

    auto kernelSizeAttr = rewriter.getDenseI32ArrayAttr(
        {static_cast<int32_t>(kernelTy.getDimSize(2)),
         static_cast<int32_t>(kernelTy.getDimSize(3))});

    auto strideAttr = attrToDenseI32ArrayAttr(adaptor.getStride(), rewriter);
    if (auto error = strideAttr.takeError()) {
      return rewriter.notifyMatchFailure(op, llvm::toString(std::move(error)));
    }

    auto paddingAttr =
        attrToDenseI32ArrayAttr(adaptor.getPadding(), rewriter, 4);
    if (auto error = paddingAttr.takeError()) {
      return rewriter.notifyMatchFailure(op, llvm::toString(std::move(error)));
    }

    auto paddingArrayRef = paddingAttr->asArrayRef();
    if (paddingArrayRef[0] != paddingArrayRef[2] ||
        paddingArrayRef[1] != paddingArrayRef[3]) {
      return rewriter.notifyMatchFailure(
          op,
          "TTNN only supports padding height/width attributes. Thus, "
          "padding_top/padding_left must equal padding_bottom/padding_right "
          "for the op to execute as expected.");
    }

    // Padding only supports 2 values in ttnn
    auto reducedPaddingAttr =
        rewriter.getDenseI32ArrayAttr({paddingArrayRef[0], paddingArrayRef[1]});

    auto dilationAttr =
        attrToDenseI32ArrayAttr(adaptor.getDilation(), rewriter);
    if (auto error = dilationAttr.takeError()) {
      return rewriter.notifyMatchFailure(op, llvm::toString(std::move(error)));
    }

    auto groupsAttr = rewriter.getI32IntegerAttr(adaptor.getGroups());

    Value flattenedInput = generateTTIRNHWFlatten(
        mlir::cast<mlir::TypedValue<RankedTensorType>>(adaptor.getInput()),
        rewriter);

    // Convolution in ttnn returns a tensor in a flattened shape
    // (1 x 1 x N * H * W x C)
    llvm::ArrayRef<std::int64_t> outputShape = outputTy.getShape();
    llvm::SmallVector<std::int64_t, 4> flattenedOutputShape = {
        1, 1, outputShape[0] * outputShape[1] * outputShape[2], outputShape[3]};
    outputTy = mlir::cast<RankedTensorType>(getTypeConverter()->convertType(
        outputTy.cloneWith(flattenedOutputShape, outputTy.getElementType())));

    outputTy = mlir::RankedTensorType::get(flattenedOutputShape,
                                           outputTy.getElementType(),
                                           outputTy.getEncoding());

    auto convDPS = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), outputTy.getShape(), outputTy.getElementType());
    ttir::OptimizedFlattenedConv2dOp newConv =
        rewriter.create<ttir::OptimizedFlattenedConv2dOp>(
            op.getLoc(), outputTy, flattenedInput, adaptor.getWeight(),
            adaptor.getBias(), convDPS, inChannelsAttr, outChannelsAttr,
            batchSizeAttr, inputHeightAttr, inputWidthAttr, kernelSizeAttr,
            *strideAttr, reducedPaddingAttr, *dilationAttr, groupsAttr);

    Value output = generateTTIRReshape(newConv, outputShape, rewriter);

    rewriter.replaceOp(op, output);
    return success();
  }

private:
  llvm::Expected<DenseI32ArrayAttr>
  attrToDenseI32ArrayAttr(mlir::Attribute attr,
                          ConversionPatternRewriter &rewriter,
                          uint32_t elementCount = 2) const {
    switch (elementCount) {
    case 2: {
      // Handles attributes requiring 2 spatial dimensions (e.g., stride,
      // dilation). Converts the attribute into a pair of integers.
      auto pair = ttmlir::utils::getPairOfInteger<int32_t>(attr);
      if (auto error = pair.takeError()) {
        return std::move(error);
      }
      return rewriter.getDenseI32ArrayAttr({pair->first, pair->second});
    }
    case 4: {
      // Handles attributes requiring 4 spatial dimensions (e.g., padding in
      // this case). Converts the attribute into a quadruple of integers.
      auto quadruple = ttmlir::utils::getQuadrupleOfInteger<int32_t>(attr);
      if (auto error = quadruple.takeError()) {
        return std::move(error);
      }
      return rewriter.getDenseI32ArrayAttr(
          {std::get<0>(*quadruple), std::get<1>(*quadruple),
           std::get<2>(*quadruple), std::get<3>(*quadruple)});
    }
    default: {
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Unsupported element count: %d",
                                     elementCount);
    }
    }
  }
};
} // namespace
} // namespace mlir::tt::ttir
using namespace mlir;
using namespace mlir::tt;
namespace {

class TTIREraseInverseOps
    : public ttir::impl::TTIREraseInverseOpsBase<TTIREraseInverseOps> {
public:
  using ttir::impl::TTIREraseInverseOpsBase<
      TTIREraseInverseOps>::TTIREraseInverseOpsBase;
  void runOnOperation() final {

    RewritePatternSet conversionPatterns(&getContext());
    TypeConverter typeConverter;
    // All types map 1:1.
    typeConverter.addConversion([](Type type) { return type; });
    conversionPatterns.add<ttir::ConvertToOptimizedFlattenedConv2dPattern>(
        typeConverter, &getContext());
    FrozenRewritePatternSet conversionPatternSet(std::move(conversionPatterns));

    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<ttir::TTIRDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    // target.addLegalDialect<BuiltinDialect>(); // This contains the "module"
    // op which is necessary

    target.addLegalOp<tensor::EmptyOp>(); // DPS operands are create with
    // tensor::EmptyOp
    target.addIllegalOp<ttir::Conv2dOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(conversionPatternSet)))) {
      signalPassFailure();
      return;
    }

    RewritePatternSet commutePatterns(&getContext());
    commutePatterns.add<ttir::TTIRCommuteTmsAboveElementwiseUnaryRewriter>(
        &getContext());
    commutePatterns.add<ttir::TTIRCommuteTmsAboveElementwiseBinaryRewriter>(
        &getContext());
    FrozenRewritePatternSet commutePatternSet(std::move(commutePatterns));

    RewritePatternSet erasePatterns(&getContext());
    erasePatterns.add<ttir::TTIREraseInverseTransposes>(&getContext());
    erasePatterns.add<ttir::TTIREraseInversePermutations>(&getContext());
    erasePatterns.add<ttir::TTIREraseInverseReshapes>(&getContext());
    FrozenRewritePatternSet erasePatternSet(std::move(erasePatterns));

    // We want to commute all TMs upwards as much as possible so they are are
    // placed back to back Then we can erase back to back inverses.
    //
    //
    // Because there are multiple TMs we wish to commute and erase, we must
    // continuously run the commute and erase patterns until the graph stops
    // changing. This is because erasing a pair of TMs may free up a path
    // for another pair of TMs to be erased.
    //
    // We do have some canonicalizatios for these ops will erase back to back
    // ops, however they are not run during this pass (yet). Maybe we can call
    // them instead.
    GreedyRewriteConfig rewriteConfig = GreedyRewriteConfig();
    bool changed = false;
    do {
      if (failed(applyPatternsGreedily(getOperation(), commutePatternSet,
                                       rewriteConfig, &changed))) {
        signalPassFailure();
        return;
      }
      if (failed(applyPatternsGreedily(getOperation(), erasePatternSet,
                                       rewriteConfig, &changed))) {
        signalPassFailure();
        return;
      }
    } while (changed);
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};

} // namespace
