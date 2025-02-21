// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <limits>
#include <vector>

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "ttmlir/Conversion/StableHLOToTTIR/ShardingUtils.h"
#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOToTTIR.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TT/Utils/Mesh.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

#include <llvm/ADT/APFloat.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>
#include <stablehlo/dialect/StablehloOps.h>

using namespace mlir;
using namespace mlir::tt;

namespace {
template <typename SrcOp, typename DestOp,
          typename Adaptor = typename SrcOp::Adaptor>
class StableHLOToTTIROpDefaultConversionPattern
    : public OpConversionPattern<SrcOp> {

  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(SrcOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));

    ttmlir::utils::replaceOpWithNewDPSOp<DestOp>(rewriter, srcOp, outputType,
                                                 adaptor.getOperands());

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRReduceOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ReduceOp> {

  using OpConversionPattern<mlir::stablehlo::ReduceOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReduceOp srcOp,
                  mlir::stablehlo::ReduceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LogicalResult legalityResult = checkBasicLegality(srcOp, adaptor, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    const mlir::Operation &innerOp = srcOp.getBody().front().front();

    if (mlir::isa<mlir::stablehlo::AddOp>(innerOp)) {
      return matchAndRewriteInternal<mlir::tt::ttir::SumOp>(srcOp, adaptor,
                                                            rewriter);
    }
    if (mlir::isa<mlir::stablehlo::MaxOp>(innerOp)) {
      return matchAndRewriteInternal<mlir::tt::ttir::MaxOp>(srcOp, adaptor,
                                                            rewriter);
    }
    if (mlir::isa<mlir::stablehlo::MinOp>(innerOp)) {
      return matchAndRewriteInternal<mlir::tt::ttir::MinOp>(srcOp, adaptor,
                                                            rewriter);
    }
    if (mlir::isa<mlir::stablehlo::MulOp>(innerOp)) {
      return matchAndRewriteInternal<mlir::tt::ttir::ProdOp>(srcOp, adaptor,
                                                             rewriter);
    }
    if (mlir::isa<mlir::stablehlo::AndOp>(innerOp)) {
      return matchAndRewriteInternal<mlir::tt::ttir::ReduceAndOp>(
          srcOp, adaptor, rewriter);
    }
    if (isArgMax(srcOp, adaptor, rewriter)) {
      return matchAndRewriteInternalArgMax(srcOp, adaptor, rewriter);
    }

    return failure();
  }

private:
  // Typical initialization values for reduction ops.
  enum TypicalInitReductionValue {
    NEG_INF, // It is also used for minimum integer value.
    ZERO,
  };

  LogicalResult checkBasicLegality(mlir::stablehlo::ReduceOp &srcOp,
                                   mlir::stablehlo::ReduceOp::Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
    if (!srcOp.getBody().hasOneBlock()) {
      return rewriter.notifyMatchFailure(
          srcOp,
          "Expecting StableHLO Reduce OP to have one block inside its body.");
    }

    if (srcOp.getBody().front().empty()) {
      return rewriter.notifyMatchFailure(
          srcOp,
          "Expecting StableHLO Reduce OP to have a body operation defined.");
    }

    mlir::Operation &innerOp = srcOp.getBody().front().front();
    if (mlir::isa<mlir::stablehlo::AndOp>(innerOp)) {
      bool allOperandsAreBoolean = std::all_of(
          srcOp->operand_begin(), srcOp->operand_end(), [](auto operand) {
            return mlir::cast<RankedTensorType>(operand.getType())
                       .getElementTypeBitWidth() == 1;
          });
      if (!allOperandsAreBoolean) {
        return rewriter.notifyMatchFailure(
            srcOp, "stablehlo.reduce for stablehlo.and operator is only "
                   "supported for logical and.");
      }
    }

    return success();
  }

  template <typename DestOp>
  LogicalResult
  matchAndRewriteInternal(mlir::stablehlo::ReduceOp &srcOp,
                          mlir::stablehlo::ReduceOp::Adaptor &adaptor,
                          ConversionPatternRewriter &rewriter) const {
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResultTypes().front()));

    // Can't reuse the original dimensions attribute because it uses i64 type.
    mlir::ArrayAttr dimArg = rewriter.getI32ArrayAttr(
        llvm::SmallVector<int32_t>(srcOp.getDimensions()));

    ttmlir::utils::replaceOpWithNewDPSOp<DestOp>(rewriter, srcOp, outputType,
                                                 adaptor.getInputs().front(),
                                                 /*keep_dim=*/false, dimArg);

    return success();
  }

  LogicalResult
  matchAndRewriteInternalArgMax(mlir::stablehlo::ReduceOp &srcOp,
                                mlir::stablehlo::ReduceOp::Adaptor &adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResultTypes()[1]));
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    // Can't reuse the original dimensions attribute because it uses i64 type.
    mlir::ArrayAttr dimArg = rewriter.getI32ArrayAttr(
        llvm::SmallVector<int32_t>(srcOp.getDimensions()));

    // stablehlo.reduce op generates two results; the first output is maximum
    // value which is not consumed in subsequent graph and the second out is the
    // index of maximum value which is consumend in subsequent graph. On other
    // hand ttir.argmax generates one result only.
    // So 'rewriter.replaceOpWithNewOp' will not work due to difference in
    // number of outputs.
    // We are creating the new ttir op explicitly here, then replace the
    // original op uses with the new op, and finally, erase the original op
    // explicitly.
    ttir::ArgMaxOp newOp = rewriter.create<tt::ttir::ArgMaxOp>(
        srcOp->getLoc(), outputType, adaptor.getInputs().front(), outputTensor,
        false /* keep_dim */, dimArg);

    srcOp->getResults().back().replaceAllUsesWith(newOp->getResults().front());
    rewriter.eraseOp(srcOp);

    return success();
  }

  // This function verify all the required conditions to convert stablehlo
  // reduce op to TTIR argmax op
  // 1. Two inputs values ot the op.
  //   a. first input is the input tensor.
  //   b. second input is defined with stablehlo.iota
  // 2. Two init values
  //   a. first init value is -inf (for float) or int_min (for integer input)
  //   b. second init value is 0
  // 3. Two results generated; the first result is maximum value which is not
  //    consumed in subsequent graph and the second result is index of maximum
  //    value which is consumend in subsequent graph.
  // 4. One block in reducer body.
  // 5. Pattern match the ops of reducer body; tt-torch and tt-xla generates
  //    different pattern. So pattern matching is performed separately.
  bool isArgMax(mlir::stablehlo::ReduceOp &srcOp,
                mlir::stablehlo::ReduceOp::Adaptor &adaptor,
                ConversionPatternRewriter &rewriter) const {
    if (!hasValidArgMaxInputs(srcOp.getInputs())) {
      return false;
    }

    if (!hasValidArgMaxOutputs(srcOp.getResults())) {
      return false;
    }

    if (!hasValidArgMaxInitValues(srcOp.getInitValues())) {
      return false;
    }

    if (!hasValidArgMaxReducerBody(srcOp.getBody())) {
      return false;
    }

    return true;
  }

  // Validate the inputs.
  bool hasValidArgMaxInputs(mlir::OperandRange inputs) const {
    if (inputs.size() != 2) {
      return false;
    }

    return isa<stablehlo::IotaOp>(inputs.back().getDefiningOp()) ? true : false;
  }

  // Validate the outputs.
  bool hasValidArgMaxOutputs(mlir::ResultRange results) const {
    if (results.size() != 2) {
      return false;
    }

    if (!results.front().getUsers().empty() ||
        results.back().getUsers().empty()) {
      return false;
    }
    return true;
  }

  // Validate initialization values.
  bool hasValidArgMaxInitValues(mlir::OperandRange initValues) const {
    if (initValues.size() != 2) {
      return false;
    }

    if (!verifyInitValue(initValues.front(),
                         TypicalInitReductionValue::NEG_INF)) {
      return false;
    }

    if (!verifyInitValue(initValues.back(), TypicalInitReductionValue::ZERO)) {
      return false;
    }
    return true;
  }

  // Validate reducer body.
  bool hasValidArgMaxReducerBody(mlir::Region &body) const {
    auto &blocks = body.getBlocks();
    if (blocks.size() != 1) {
      return false;
    }

    auto &operations = blocks.front().getOperations();
    if (operations.size() == 7) {
      if (verifyTorchOpArgMaxPattern(operations.front())) {
        return true;
      }
    } else if (operations.size() == 10) {
      if (verifyJaxOpArgMaxPattern(operations.front())) {
        return true;
      }
    }

    return false;
  }

  // Pattern match the ops for tt-torch ArgMax op.
  bool verifyTorchOpArgMaxPattern(mlir::Operation &operation) const {
    mlir::Operation *op = &operation;
    if (!isa<stablehlo::CompareOp>(op)) {
      return false;
    }
    stablehlo::CompareOp compareOp = mlir::cast<stablehlo::CompareOp>(op);
    if (compareOp.getComparisonDirection() !=
        mlir::stablehlo::ComparisonDirection::GE) {
      return false;
    }

    op = op->getNextNode();
    if (!isa<stablehlo::SelectOp>(op)) {
      return false;
    }

    op = op->getNextNode();
    if (!isa<stablehlo::CompareOp>(op)) {
      return false;
    }

    op = op->getNextNode();
    if (!isa<stablehlo::MinOp>(op)) {
      return false;
    }

    op = op->getNextNode();
    if (!isa<stablehlo::SelectOp>(op)) {
      return false;
    }

    op = op->getNextNode();
    if (!isa<stablehlo::SelectOp>(op)) {
      return false;
    }

    op = op->getNextNode();
    if (!isa<stablehlo::ReturnOp>(op)) {
      return false;
    }

    return true;
  }

  // Pattern match the ops for tt-xla ArgMax op.
  bool verifyJaxOpArgMaxPattern(mlir::Operation &operation) const {
    mlir::Operation *op = &operation;
    if (!isa<stablehlo::CompareOp>(op)) {
      return false;
    }
    stablehlo::CompareOp compareOp = mlir::cast<stablehlo::CompareOp>(op);
    if (compareOp.getComparisonDirection() !=
        mlir::stablehlo::ComparisonDirection::GT) {
      return false;
    }

    op = op->getNextNode();
    if (!isa<stablehlo::CompareOp>(op)) {
      return false;
    }

    op = op->getNextNode();
    if (!isa<stablehlo::OrOp>(op)) {
      return false;
    }

    op = op->getNextNode();
    if (!isa<stablehlo::CompareOp>(op)) {
      return false;
    }

    op = op->getNextNode();
    if (!isa<stablehlo::CompareOp>(op)) {
      return false;
    }

    op = op->getNextNode();
    if (!isa<stablehlo::AndOp>(op)) {
      return false;
    }

    op = op->getNextNode();
    if (!isa<stablehlo::OrOp>(op)) {
      return false;
    }

    op = op->getNextNode();
    if (!isa<stablehlo::SelectOp>(op)) {
      return false;
    }

    op = op->getNextNode();
    if (!isa<stablehlo::SelectOp>(op)) {
      return false;
    }

    op = op->getNextNode();
    if (!isa<stablehlo::ReturnOp>(op)) {
      return false;
    }

    return true;
  }

  // Verify that the init value is defined by a constant op and initialize with
  // desired value.
  bool verifyInitValue(mlir::Value val,
                       TypicalInitReductionValue desired) const {
    Operation *initValue = val.getDefiningOp();
    while (initValue->getOpOperands().size() == 1) {
      initValue = initValue->getOpOperand(0).get().getDefiningOp();
    }
    if (!isa<stablehlo::ConstantOp>(initValue)) {
      return false;
    }

    stablehlo::ConstantOp initValueOp =
        mlir::cast<stablehlo::ConstantOp>(initValue);

    if (!checkInitValue(initValueOp, desired)) {
      return false;
    }
    return true;
  }

  // Check if the constant op is initialized with the desired init value.
  bool checkInitValue(stablehlo::ConstantOp initValueOp,
                      TypicalInitReductionValue desired) const {
    if (initValueOp.getValueAttr().size() != 1) {
      return false;
    }

    float desiredF32;
    double desiredF64;
    uint16_t desiredBF16;
    int32_t desiredI32;
    int64_t desiredI64;
    if (desired == TypicalInitReductionValue::NEG_INF) {
      desiredF32 = -std::numeric_limits<float>::infinity();
      desiredF64 = -std::numeric_limits<double>::infinity();
      desiredBF16 = 0xff80; // This is -inf in bfloat16 raw bits
      desiredI32 = std::numeric_limits<int32_t>::min();
      desiredI64 = std::numeric_limits<int64_t>::min();
    } else if (desired == TypicalInitReductionValue::ZERO) {
      desiredF32 = 0.0;
      desiredF64 = 0.0;
      desiredBF16 = 0x0000; // This is 0 in bfloat16 raw bits
      desiredI32 = 0;
      desiredI64 = 0;
    } else {
      return false;
    }

    // Constant operand must be -inf if this is to be a max pool
    // since bfloat16 is not a type we actually have I must compare the raw
    // bits
    if (initValueOp.getResult().getType().getElementType().isBF16()) {
      // Collect the values into a vector
      std::vector<mlir::Attribute> values;
      for (int64_t i = 0; i < initValueOp.getValueAttr().size(); ++i) {
        values.push_back(
            initValueOp.getValueAttr().getValues<mlir::Attribute>()[i]);
      }

      auto denseValues = ::mlir::DenseElementsAttr::get(
          initValueOp.getValueAttr().getShapedType(), values);
      uint16_t bfloat_bits =
          static_cast<uint16_t>(*denseValues.getRawData().data());
      if (bfloat_bits != desiredBF16) { // This is -inf in bfloat16
        return false;
      }
    } else if (initValueOp.getResult().getType().getElementType().isF32()) {
      if (*initValueOp.getValue().value_begin<float>() != desiredF32) {
        return false;
      }
    } else if (initValueOp.getResult().getType().getElementType().isF64()) {
      if (*initValueOp.getValue().value_begin<double>() != desiredF64) {
        return false;
      }
    } else if (initValueOp.getResult().getType().getElementType().isInteger(
                   32)) {
      if (*initValueOp.getValue().value_begin<int32_t>() != desiredI32) {
        return false;
      }
    } else if (initValueOp.getResult().getType().getElementType().isInteger(
                   64)) {
      if (*initValueOp.getValue().value_begin<int64_t>() != desiredI64) {
        return false;
      }
    } else {
      return false;
    }

    return true;
  }
};
} // namespace

namespace {
class StableHLOToTTIRDotGeneralOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::DotGeneralOp> {
  using OpConversionPattern<mlir::stablehlo::DotGeneralOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::DotGeneralOp srcOp,
                  mlir::stablehlo::DotGeneralOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::DotGeneralOp>(
        srcOp, outputType, adaptor.getLhs(), adaptor.getRhs(),
        adaptor.getDotDimensionNumbers().getLhsBatchingDimensions(),
        adaptor.getDotDimensionNumbers().getLhsContractingDimensions(),
        adaptor.getDotDimensionNumbers().getRhsBatchingDimensions(),
        adaptor.getDotDimensionNumbers().getRhsContractingDimensions());

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRTransposeOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::TransposeOp> {
  using OpConversionPattern<mlir::stablehlo::TransposeOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::TransposeOp srcOp,
                  mlir::stablehlo::TransposeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ::mlir::RankedTensorType outputType = mlir::cast<mlir::RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));

    // The stablehlo.transpose and ttir.permute have the same semantics.
    ttmlir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::PermuteOp>(
        rewriter, srcOp, outputType, adaptor.getOperand(),
        adaptor.getPermutation());

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRReshapeOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ReshapeOp> {
  using OpConversionPattern<mlir::stablehlo::ReshapeOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReshapeOp srcOp,
                  mlir::stablehlo::ReshapeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    ArrayAttr newShapeAttr = rewriter.getI32ArrayAttr(
        llvm::SmallVector<int32_t>(outputType.getShape()));

    ttmlir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::ReshapeOp>(
        rewriter, srcOp, outputType, adaptor.getOperand(), newShapeAttr);

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRGetDimensionSizeOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::GetDimensionSizeOp> {

  using OpConversionPattern<
      mlir::stablehlo::GetDimensionSizeOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::GetDimensionSizeOp srcOp,
                  mlir::stablehlo::GetDimensionSizeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    IntegerType intType = IntegerType::get(getContext(), 32);
    RankedTensorType outputType = RankedTensorType::get({1}, intType);
    mlir::OpBuilder builder(getContext());
    IntegerAttr dimension_attr = builder.getIntegerAttr(
        intType, static_cast<int32_t>(srcOp.getDimension()));

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::GetDimensionSizeOp>(
        srcOp, outputType, adaptor.getOperand(), dimension_attr);

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRConstantOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ConstantOp> {

  using OpConversionPattern<mlir::stablehlo::ConstantOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ConstantOp srcOp,
                  mlir::stablehlo::ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LogicalResult legalityResult = checkBasicLegality(srcOp, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    mlir::ElementsAttr valueAttr = getValueAttr(srcOp.getValue());

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::ConstantOp>(srcOp, outputType,
                                                            valueAttr);
    return success();
  }

private:
  LogicalResult checkBasicLegality(mlir::stablehlo::ConstantOp &srcOp,
                                   ConversionPatternRewriter &rewriter) const {
    if (srcOp.getValue().getShapedType().getShape().empty() &&
        !srcOp.getValue().getElementType().isIntOrFloat()) {
      return rewriter.notifyMatchFailure(srcOp, "Unsupported element type.");
    }

    return success();
  }

  // Rebuilding value of constant op for following cases.
  // 1. Scalar values: TTNN does not support scalar types. So they are converted
  //    1-D tensors.
  // 2. Boolean tensor: TTNN does not support boolean data. So they are
  //    converted to bfloat16 tensors.
  // 3. Integer tensor: TTNN does not support 64 bit integer. So they are
  //    converted to 32 bit tensor.
  // 4. Float tensor: TTNN does not support 64 bit float. So they are converted
  //    to 32 bit tensor.
  mlir::ElementsAttr getValueAttr(mlir::ElementsAttr valueAttr) const {
    Type elementType = valueAttr.getElementType();
    size_t bitWidth = elementType.getIntOrFloatBitWidth();
    bool isTensor = !valueAttr.getShapedType().getShape().empty();
    bool isIntTensor = isTensor && isa<IntegerType>(elementType) &&
                       bitWidth != 1 && bitWidth != 64;
    bool isFloatTensor = isTensor && isa<FloatType>(elementType) &&
                         bitWidth != 1 && bitWidth != 64;

    if (isTensor && (isIntTensor || isFloatTensor)) {
      return valueAttr;
    }

    mlir::ShapedType valueType = mlir::cast<mlir::ShapedType>(
        getTypeConverter()->convertType(valueAttr.getShapedType()));
    if (isa<IntegerType>(elementType)) {
      switch (bitWidth) {
      case 1: {
        return rebuildValueAttr<bool>(valueAttr, 1);
      }
      case 8: {
        return elementType.isUnsignedInteger()
                   ? rebuildValueAttr<uint8_t>(valueAttr, 8)
                   : rebuildValueAttr<int8_t>(valueAttr, 8);
      }
      case 16: {
        return elementType.isUnsignedInteger()
                   ? rebuildValueAttr<uint16_t>(valueAttr, 16)
                   : rebuildValueAttr<int16_t>(valueAttr, 16);
      }
      case 32: {
        return elementType.isUnsignedInteger()
                   ? rebuildValueAttr<uint32_t>(valueAttr, 32)
                   : rebuildValueAttr<int32_t>(valueAttr, 32);
      }
      case 64: {
        return elementType.isUnsignedInteger()
                   ? rebuildValueAttr<uint64_t>(valueAttr, 32)
                   : rebuildValueAttr<int64_t>(valueAttr, 32);
      }
      default: {
        assert(false && "Unsupported integer type.");
      }
      }
    }
    if (isa<FloatType>(elementType)) {
      // Convert 64 bit floating point numbers to 32 bit floating point numbers.
      if (bitWidth == 64) {
        std::vector<mlir::APFloat> floatValues;
        for (mlir::APFloat value : valueAttr.getValues<mlir::APFloat>()) {
          float fl = static_cast<float>(value.convertToDouble());
          mlir::APFloat input = mlir::APFloat(fl);
          floatValues.emplace_back(input);
        }
        return mlir::DenseElementsAttr::get(valueType, floatValues);
      }
      // In case of float values llvm has a bug where not all float types are
      // supported for iterating in DenseElementsAttr, so we have to use a
      // different constructor.
      std::vector<mlir::APFloat> floatValues(
          valueAttr.getValues<mlir::APFloat>().begin(),
          valueAttr.getValues<mlir::APFloat>().end());
      return mlir::DenseElementsAttr::get(valueType, floatValues);
    }
    assert(false && "Unsupported data type.");
  }

  // Extract the values (using the given ElementType) and create new data
  // structure. This is used to convert scalars (of type boolean, int8, int16,
  // int32, int64, uint8, uint16, uint32, uint64) and tensors (of type boolean
  // and int64).
  template <typename ElementType>
  mlir::ElementsAttr rebuildValueAttr(mlir::ElementsAttr valueAttr,
                                      size_t bitWidth) const {
    mlir::ShapedType valueType = mlir::cast<mlir::ShapedType>(
        getTypeConverter()->convertType(valueAttr.getShapedType()));

    // Create data structure for boolean type with bfloat16.
    if (bitWidth == 1) {
      std::vector<mlir::APFloat> booleanValue = {};
      for (ElementType value : valueAttr.getValues<ElementType>()) {
        mlir::APFloat input(mlir::APFloat::BFloat(), value);
        booleanValue.emplace_back(input);
      }
      return mlir::DenseElementsAttr::get(valueType, booleanValue);
    }

    // Create data structure for other types.
    std::vector<mlir::APInt> IntegerValue = {};
    for (ElementType value : valueAttr.getValues<ElementType>()) {
      mlir::APInt input(bitWidth, value);
      IntegerValue.emplace_back(input);
    }
    return mlir::DenseElementsAttr::get(valueType, IntegerValue);
  }
};
} // namespace

namespace {
class StableHLOToTTIRConvolutionOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ConvolutionOp> {
  using OpConversionPattern<
      mlir::stablehlo::ConvolutionOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ConvolutionOp srcOp,
                  mlir::stablehlo::ConvolutionOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    RankedTensorType outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    auto dimNums = adaptor.getDimensionNumbers();
    uint64_t numSpatialDims = dimNums.getInputSpatialDimensions().size();

    // These are the defaults intended by stablehlo when the attrs are not
    // populated
    DenseI64ArrayAttr windowStridesAttr =
        adaptor.getWindowStridesAttr()
            ? adaptor.getWindowStridesAttr()
            : rewriter.getDenseI64ArrayAttr(
                  SmallVector<int64_t>(numSpatialDims, 1));
    DenseI64ArrayAttr paddingAttr =
        adaptor.getPaddingAttr()
            ? rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(
                  adaptor.getPaddingAttr().getValues<int64_t>()))
            : rewriter.getDenseI64ArrayAttr(
                  SmallVector<int64_t>(numSpatialDims * 2, 0));
    DenseI64ArrayAttr inputDilationAttr =
        adaptor.getLhsDilationAttr()
            ? adaptor.getLhsDilationAttr()
            : rewriter.getDenseI64ArrayAttr(
                  SmallVector<int64_t>(numSpatialDims, 1));
    DenseI64ArrayAttr kernelDilationAttr =
        adaptor.getRhsDilationAttr()
            ? adaptor.getRhsDilationAttr()
            : rewriter.getDenseI64ArrayAttr(
                  SmallVector<int64_t>(numSpatialDims, 1));
    DenseBoolArrayAttr windowReversalAttr =
        adaptor.getWindowReversalAttr()
            ? adaptor.getWindowReversalAttr()
            : rewriter.getDenseBoolArrayAttr(
                  SmallVector<bool>(numSpatialDims, false));

    ttmlir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::ConvolutionOp>(
        rewriter, srcOp, outputType, adaptor.getLhs(), adaptor.getRhs(),
        Value(), windowStridesAttr, paddingAttr, inputDilationAttr,
        kernelDilationAttr, windowReversalAttr,
        mlir::tt::ttir::ConvolutionLayoutAttr::get(
            getContext(), dimNums.getInputBatchDimension(),
            dimNums.getInputFeatureDimension(),
            dimNums.getInputSpatialDimensions(),
            dimNums.getKernelOutputFeatureDimension(),
            dimNums.getKernelInputFeatureDimension(),
            dimNums.getKernelSpatialDimensions(),
            dimNums.getOutputBatchDimension(),
            dimNums.getOutputFeatureDimension(),
            dimNums.getOutputSpatialDimensions()),
        adaptor.getFeatureGroupCountAttr(), adaptor.getBatchGroupCountAttr());

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRReduceWindowOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ReduceWindowOp> {
  using OpConversionPattern<
      mlir::stablehlo::ReduceWindowOp>::OpConversionPattern;

public:
  bool isMaxPool(mlir::stablehlo::ReduceWindowOp &srcOp) const {
    if (srcOp.getBody().getBlocks().size() != 1) {
      return false;
    }

    // Find constant input(s)
    Operation *initValue;
    for (uint64_t i = 0; i < srcOp.getInitValues().size(); i++) {
      initValue = srcOp.getInitValues()[i].getDefiningOp();
      while (initValue->getOpOperands().size() == 1) {
        initValue = initValue->getOpOperand(0).get().getDefiningOp();
      }
      if (!isa<stablehlo::ConstantOp>(initValue)) {
        return false;
      }

      stablehlo::ConstantOp initValueOp =
          mlir::cast<stablehlo::ConstantOp>(initValue);

      if (!checkInitValue(initValueOp, TypicalInitReductionValue::NEG_INF)) {
        return false;
      }
    }

    Block &block = *srcOp.getBody().getBlocks().begin();
    uint32_t opIdx = 0;
    for (Operation &op : block) {
      if (opIdx == 0 && !isa<mlir::stablehlo::MaxOp>(op)) {
        return false;
      }
      if (opIdx == 1 && !isa<mlir::stablehlo::ReturnOp>(op)) {
        return false;
      }
      if (opIdx >= 2) {
        return false; // More than two ops in the block
      }
      opIdx++;
    }

    return true;
  }

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReduceWindowOp srcOp,
                  mlir::stablehlo::ReduceWindowOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    RankedTensorType outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult(0).getType()));

    SmallVector<Value> outputsVec;
    for (uint32_t i = 0; i < srcOp.getResults().size(); i++) {
      tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
          srcOp.getLoc(), outputType.getShape(), outputType.getElementType());
      outputsVec.push_back(outputTensor);
    }
    ValueRange outputs = outputsVec;

    auto windowDimensions = adaptor.getWindowDimensionsAttr();
    auto windowStrides = adaptor.getWindowStridesAttr();
    auto baseDilations = adaptor.getBaseDilationsAttr();
    auto window_dilations = adaptor.getWindowDilationsAttr();
    auto padding_ = adaptor.getPaddingAttr();

    // Generate defaults if they dont exist (these defaults are what the
    // stablehlo dialect intends when they are not provided)
    windowStrides = windowStrides
                        ? windowStrides
                        : rewriter.getDenseI64ArrayAttr(
                              SmallVector<int64_t>(windowDimensions.size(), 1));
    baseDilations = baseDilations
                        ? baseDilations
                        : rewriter.getDenseI64ArrayAttr(
                              SmallVector<int64_t>(windowDimensions.size(), 1));
    window_dilations = window_dilations
                           ? window_dilations
                           : rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(
                                 windowDimensions.size(), 1));
    auto padding =
        padding_ ? rewriter.getDenseI64ArrayAttr(
                       SmallVector<int64_t>(padding_.getValues<int64_t>()))
                 : rewriter.getDenseI64ArrayAttr(
                       SmallVector<int64_t>(windowDimensions.size() * 2, 0));

    mlir::tt::ttir::PoolingMethod poolingMethod;
    int64_t dimension = -1;
    if (isMaxPool(srcOp)) {
      poolingMethod = mlir::tt::ttir::PoolingMethod::Max;
    } else if (isCumSum(srcOp, adaptor, dimension)) {
      rewriter.replaceOpWithNewOp<ttir::CumSumOp>(
          srcOp, outputType, adaptor.getInputs()[0],
          rewriter.getI64IntegerAttr(dimension), outputs[0]);
      return success();
    } else {
      return rewriter.notifyMatchFailure(srcOp, "Unsupported pooling method");
    }

    rewriter.replaceOpWithNewOp<ttir::PoolingOp>(
        srcOp, outputType, adaptor.getInputs(), outputs, poolingMethod,
        windowDimensions, windowStrides, baseDilations, window_dilations,
        padding);

    return success();
  }

private:
  // This function verify all the required conditions to convert stablehlo
  // reduce_window op to TTIR cumsum op and also determine the dimension
  // attribute along which the cumulative sum will be computed.
  // The reduce_window op must satisfy the following conditions.
  // 1. One input / one output, one block in body and two ops with in block.
  // 2. Ops in the block must be 'add' and 'return'.
  // 3. InitValue must be zero.
  // 4. There are no strides or dilations for window-related attributes.
  // 5. The size of padding attribute is equal to two times input tensor rank.
  // 6. Padding value must be zero in case of splat vector. Window dimension
  //    attribute must have all elements equal to one in this case.
  // 7. Padding attribute have one non-zero element in case of non-splat vector
  //    and this non-zero element must be equal to size of specified dimension
  //    minus one.
  // The dimension attribute is determined in following two ways.
  // 1. (If padding is splat vector): First dimension in the input tensor shape,
  //    whose size is 1, is the required dimension.
  // 2. (If padding is non-splat vector): Window dimension attribute must have
  //    all elements equal to 1 except one; whose location is the required
  //    dimension and value must be qual to size of the required dimension.
  bool isCumSum(mlir::stablehlo::ReduceWindowOp &srcOp,
                mlir::stablehlo::ReduceWindowOp::Adaptor adaptor,
                int64_t &dimension) const {

    // Check basic structure of the ReduceWindowOp
    if (!hasValidOpStructure(srcOp)) {
      return false;
    }

    // Verify operations in the block
    if (!hasValidOperationsInBlock(srcOp)) {
      return false;
    }

    // Check init values
    if (!hasValidInitValues(srcOp)) {
      return false;
    }

    // Verify window-related attributes (strides, dilations)
    if (!hasValidWindowAttributes(adaptor)) {
      return false;
    }

    // Check input tensor type and padding
    if (!hasValidInputAndPadding(srcOp, adaptor, dimension)) {
      return false;
    }

    return true;
  }

  // validate basic structure of the ReduceWindowOp.
  bool hasValidOpStructure(mlir::stablehlo::ReduceWindowOp &srcOp) const {
    if (srcOp.getBody().getBlocks().size() != 1 ||
        srcOp.getBody().getBlocks().begin()->getOperations().size() != 2) {
      return false;
    }
    if (srcOp.getInputs().size() != 1 || srcOp->getResults().size() != 1) {
      return false;
    }
    return true;
  }

  // Check init values (must be constant and zero).
  bool hasValidInitValues(mlir::stablehlo::ReduceWindowOp &srcOp) const {
    for (auto initValue : srcOp.getInitValues()) {
      auto *defOp = initValue.getDefiningOp();
      while (defOp->getOpOperands().size() == 1) {
        defOp = defOp->getOpOperand(0).get().getDefiningOp();
      }
      if (!isa<stablehlo::ConstantOp>(defOp)) {
        return false;
      }
      stablehlo::ConstantOp initValueOp =
          mlir::cast<stablehlo::ConstantOp>(defOp);
      if (!checkInitValue(initValueOp, TypicalInitReductionValue::ZERO)) {
        return false;
      }
    }
    return true;
  }

  // Verify operations inside the block (AddOp followed by ReturnOp).
  bool hasValidOperationsInBlock(mlir::stablehlo::ReduceWindowOp &srcOp) const {
    Block &block = *srcOp.getBody().getBlocks().begin();
    auto &operations = block.getOperations();
    if (!isa<mlir::stablehlo::AddOp>(operations.front())) {
      return false;
    }
    if (!isa<mlir::stablehlo::ReturnOp>(operations.back())) {
      return false;
    }
    return true;
  }

  // Verify that window attributes (strides, dilations) are all set to 1.
  bool hasValidWindowAttributes(
      mlir::stablehlo::ReduceWindowOp::Adaptor adaptor) const {
    auto verifyAttributes = [](mlir::DenseI64ArrayAttr arrAttr) -> bool {
      if (!arrAttr) {
        return true;
      }
      return std::all_of(arrAttr.asArrayRef().begin(),
                         arrAttr.asArrayRef().end(),
                         [](int value) { return value == 1; });
    };
    return verifyAttributes(adaptor.getWindowStridesAttr()) &&
           verifyAttributes(adaptor.getBaseDilationsAttr()) &&
           verifyAttributes(adaptor.getWindowDilationsAttr());
  }

  // Check input tensor type and validate padding.
  bool hasValidInputAndPadding(mlir::stablehlo::ReduceWindowOp &srcOp,
                               mlir::stablehlo::ReduceWindowOp::Adaptor adaptor,
                               int64_t &dimension) const {
    RankedTensorType inputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getInputs()[0].getType()));
    int64_t inputRank = inputType.getRank();
    llvm::ArrayRef<int64_t> windowDimensions =
        adaptor.getWindowDimensionsAttr().asArrayRef();
    mlir::DenseIntElementsAttr padding = adaptor.getPaddingAttr();

    // Validate padding size
    if (padding.size() != (inputRank * 2)) {
      return false;
    }

    // Check for splat padding (all zeroes expected).
    if (padding.isSplat()) {
      if (padding.getSplatValue<int64_t>() != 0) {
        return false;
      }
      if (!std::all_of(windowDimensions.begin(), windowDimensions.end(),
                       [](int value) { return value == 1; })) {
        return false;
      }
      // Determine the dimension using input tensor shape.
      return findDimensionWithShape(inputType, dimension);
    }

    // Check non-splat padding and ensure the window dimensions and padding are
    // consistent and determine the dimension attribute.
    return validateNonSplatPadding(windowDimensions, padding, inputType,
                                   dimension);
  }

  // Find the dimension using input tensor shape.
  bool findDimensionWithShape(RankedTensorType inputType,
                              int64_t &dimension) const {
    dimension = -1;
    for (int64_t size : inputType.getShape()) {
      ++dimension;
      if (size == 1) {
        return true;
      }
    }
    return false;
  }

  // Determine and validate dimension attribute for non-splat padding attribute.
  bool validateNonSplatPadding(llvm::ArrayRef<int64_t> windowDimensions,
                               mlir::DenseIntElementsAttr padding,
                               RankedTensorType inputType,
                               int64_t &dimension) const {
    int64_t dimArgValue = -1;
    int64_t idx = -1;
    auto padding_values = padding.getValues<int64_t>();

    // Determine dimension attribute.
    for (int64_t windowDim : windowDimensions) {
      ++idx;
      if (windowDim == 1) {
        continue;
      }
      if (dimArgValue != -1) {
        return false; // Ensure only one non-1 element.
      }
      dimArgValue = windowDim;
      dimension = idx;
    }

    // Validate dimension attribute.
    if (dimArgValue != inputType.getShape()[dimension] || dimArgValue <= 1) {
      return false;
    }

    for (int64_t i = 0; i < padding.size(); ++i) {
      if (i == (dimension * 2)) {
        if (padding_values[i] != (dimArgValue - 1)) {
          return false;
        }
      } else if (padding_values[i] != 0) {
        return false;
      }
    }

    return true;
  }

  enum TypicalInitReductionValue {
    NEG_INF, // used for max pooling
    ZERO,    // used for sum pooling
  };

  // Using the value enum rather than actual values because of different data
  // types the init value could be
  bool checkInitValue(stablehlo::ConstantOp initValueOp,
                      TypicalInitReductionValue desired) const {
    if (initValueOp.getValueAttr().size() != 1) {
      return false;
    }

    float desiredF32;
    double desiredF64;
    uint16_t desiredBF16;
    int32_t desiredI32;
    int64_t desiredI64;
    if (desired == TypicalInitReductionValue::NEG_INF) {
      desiredF32 = -std::numeric_limits<float>::infinity();
      desiredF64 = -std::numeric_limits<double>::infinity();
      desiredBF16 = 0xff80; // This is -inf in bfloat16 raw bits
      desiredI32 = std::numeric_limits<int32_t>::min();
      desiredI64 = std::numeric_limits<int64_t>::min();
    } else if (desired == TypicalInitReductionValue::ZERO) {
      desiredF32 = 0.0;
      desiredF64 = 0.0;
      desiredBF16 = 0x0000; // This is 0 in bfloat16 raw bits
      desiredI32 = 0;
      desiredI64 = 0;
    } else {
      return false;
    }

    // Constant operand must be -inf if this is to be a max pool
    // since bfloat16 is not a type we actually have I must compare the raw
    // bits
    if (initValueOp.getResult().getType().getElementType().isBF16()) {
      // Collect the values into a vector
      std::vector<mlir::Attribute> values;
      for (int64_t i = 0; i < initValueOp.getValueAttr().size(); ++i) {
        values.push_back(
            initValueOp.getValueAttr().getValues<mlir::Attribute>()[i]);
      }

      auto denseValues = ::mlir::DenseElementsAttr::get(
          initValueOp.getValueAttr().getShapedType(), values);
      uint16_t bfloat_bits =
          static_cast<uint16_t>(*denseValues.getRawData().data());
      if (bfloat_bits != desiredBF16) { // This is -inf in bfloat16
        return false;
      }
    } else if (initValueOp.getResult().getType().getElementType().isF32()) {
      if (*initValueOp.getValue().value_begin<float>() != desiredF32) {
        return false;
      }
    } else if (initValueOp.getResult().getType().getElementType().isF64()) {
      if (*initValueOp.getValue().value_begin<double>() != desiredF64) {
        return false;
      }
    } else if (initValueOp.getResult().getType().getElementType().isInteger(
                   32)) {
      if (*initValueOp.getValue().value_begin<int32_t>() != desiredI32) {
        return false;
      }
    } else if (initValueOp.getResult().getType().getElementType().isInteger(
                   64)) {
      if (*initValueOp.getValue().value_begin<int64_t>() != desiredI64) {
        return false;
      }
    } else {
      return false;
    }

    return true;
  }
};
} // namespace

namespace {
class StableHLOToTTIRBroadcastInDimOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::BroadcastInDimOp> {
  using OpConversionPattern<
      mlir::stablehlo::BroadcastInDimOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::BroadcastInDimOp srcOp,
                  mlir::stablehlo::BroadcastInDimOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LogicalResult legalityResult = checkBasicLegality(srcOp, adaptor, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    auto inputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getOperand().getType()));

    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    if (inputType.getRank() == outputType.getRank()) {
      // No unsqueeze is needed in this case and this broadcast can be
      // represented by broadcast op.
      ::llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
      ::llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

      SmallVector<int64_t> broadcastShape =
          ttmlir::utils::getBroadcastDimensions<int64_t>(inputShape,
                                                         outputShape);

      ttmlir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::BroadcastOp>(
          rewriter, srcOp, outputType, adaptor.getOperand(), broadcastShape);
    } else {
      // This stablehlo operation cannot be represented by a single TTIR
      // operation. It has to be split into ttir.reshape followed by a
      // ttir.broadcast op.
      SmallVector<int64_t> unsqueezeShape(outputType.getRank(), 1);
      ::llvm::ArrayRef<int64_t> broadcastInDim =
          adaptor.getBroadcastDimensions();

      // Since we convert scalars to 1D tensors as a special case,
      // so check input dimension is not empty.
      if (!broadcastInDim.empty()) {
        for (int64_t i = 0; i < inputType.getRank(); i++) {
          unsqueezeShape[broadcastInDim[i]] = inputType.getDimSize(i);
        }
      }

      SmallVector<int32_t> reshapeDim(unsqueezeShape.begin(),
                                      unsqueezeShape.end());
      auto reshapeDimAttr = rewriter.getI32ArrayAttr(reshapeDim);

      ttir::ReshapeOp reshapeOp = ttmlir::utils::createDPSOp<ttir::ReshapeOp>(
          rewriter, srcOp.getLoc(), unsqueezeShape, outputType.getElementType(),
          outputType.getEncoding(), adaptor.getOperand(), reshapeDimAttr);

      ::llvm::ArrayRef<int64_t> inputShape = unsqueezeShape;
      ::llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

      SmallVector<int64_t> broadcastShape =
          ttmlir::utils::getBroadcastDimensions<int64_t>(inputShape,
                                                         outputShape);

      ttmlir::utils::replaceOpWithNewDPSOp<ttir::BroadcastOp>(
          rewriter, srcOp, outputType, reshapeOp, broadcastShape);
    }

    return success();
  }

private:
  LogicalResult
  checkBasicLegality(mlir::stablehlo::BroadcastInDimOp &srcOp,
                     mlir::stablehlo::BroadcastInDimOp::Adaptor adaptor,
                     ConversionPatternRewriter &rewriter) const {

    llvm::SmallVector<int64_t, 4> broadcastedShape;
    auto srcType =
        getTypeConverter()->convertType(adaptor.getOperand().getType());
    auto inputShape = mlir::cast<mlir::RankedTensorType>(srcType).getShape();
    auto outputShape = mlir::cast<mlir::RankedTensorType>(srcType).getShape();

    if (!OpTrait::util::getBroadcastedShape(inputShape, outputShape,
                                            broadcastedShape)) {
      return rewriter.notifyMatchFailure(
          srcOp, "Input cannot be broadcasted to provided dimensions.");
    }

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRCompareOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompareOp> {
  using OpConversionPattern<mlir::stablehlo::CompareOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompareOp srcOp,
                  mlir::stablehlo::CompareOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // StableHLO has one 'compare' op to do all type of comparison (EQ, NE, GE,
    // GT, LE, and LT) and use direction to determine the type of comparison.
    mlir::stablehlo::ComparisonDirection direction =
        srcOp.getComparisonDirection();

    switch (direction) {
    case mlir::stablehlo::ComparisonDirection::EQ: {
      return matchAndRewriteInternal<mlir::tt::ttir::EqualOp>(srcOp, adaptor,
                                                              rewriter);
      break;
    }
    case mlir::stablehlo::ComparisonDirection::NE: {
      return matchAndRewriteInternal<mlir::tt::ttir::NotEqualOp>(srcOp, adaptor,
                                                                 rewriter);
      break;
    }
    case mlir::stablehlo::ComparisonDirection::GE: {
      return matchAndRewriteInternal<mlir::tt::ttir::GreaterEqualOp>(
          srcOp, adaptor, rewriter);
      break;
    }
    case mlir::stablehlo::ComparisonDirection::GT: {
      return matchAndRewriteInternal<mlir::tt::ttir::GreaterThanOp>(
          srcOp, adaptor, rewriter);
      break;
    }
    case mlir::stablehlo::ComparisonDirection::LE: {
      return matchAndRewriteInternal<mlir::tt::ttir::LessEqualOp>(
          srcOp, adaptor, rewriter);
      break;
    }
    case mlir::stablehlo::ComparisonDirection::LT: {
      return matchAndRewriteInternal<mlir::tt::ttir::LessThanOp>(srcOp, adaptor,
                                                                 rewriter);
      break;
    }
    }
    return success();
  }

private:
  template <typename DestOp>
  LogicalResult
  matchAndRewriteInternal(mlir::stablehlo::CompareOp srcOp,
                          mlir::stablehlo::CompareOp::Adaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    mlir::RankedTensorType outputType =
        mlir::cast<RankedTensorType>(this->getTypeConverter()->convertType(
            srcOp->getResults()[0].getType()));

    ttmlir::utils::replaceOpWithNewDPSOp<DestOp>(rewriter, srcOp, outputType,
                                                 adaptor.getOperands());

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRConcatOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ConcatenateOp> {

  using OpConversionPattern<
      mlir::stablehlo::ConcatenateOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ConcatenateOp srcOp,
                  mlir::stablehlo::ConcatenateOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Check legality of the operation
    LogicalResult err = checkBasicLegality(srcOp, adaptor, rewriter);
    if (failed(err)) {
      return err;
    }

    // Create the output tensor type based on inputs
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    ttmlir::utils::replaceOpWithNewDPSOp<ttir::ConcatOp>(
        rewriter, srcOp, outputType, adaptor.getInputs(),
        static_cast<int32_t>(adaptor.getDimension()));

    return success();
  }

private:
  LogicalResult
  checkBasicLegality(mlir::stablehlo::ConcatenateOp &srcOp,
                     mlir::stablehlo::ConcatenateOp::Adaptor adaptor,
                     ConversionPatternRewriter &rewriter) const {
    if (srcOp.getInputs().empty()) {
      return rewriter.notifyMatchFailure(
          srcOp, "ConcatOp must have at least one input.");
    }
    if (adaptor.getDimension() >=
        INT32_MAX) { // stablehlo.concatenate dimension is i64,
                     // ttir.concat dimension is si32
      return rewriter.notifyMatchFailure(srcOp,
                                         "ConcatOp dimension is too large.");
    }

    auto rankedTensorType = mlir::dyn_cast<mlir::RankedTensorType>(
        adaptor.getOperands()[0].getType());
    if (static_cast<int64_t>(adaptor.getDimension()) >=
        rankedTensorType.getRank()) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "Invalid concatenation dimension.");
    }

    return success();
  }
};
} // namespace

// Class implementing conversion from StableHLO to TTIR logical and bitwise ops.
// StableHLO has AND, OR, XOR and NOT ops defined in such a way that they do two
// different things based on type of inputs. In case of booleans, they perform
// logical version of the op, and in case of integers they perform bitwise
// version of the op. We made a decision to make those two cases completely
// distinct ops in TTIR. Thus, a StableHLO `SrcOp` is rewritten to one of
// `DestOp`s based on operand types.
namespace {
template <typename SrcOp, typename LogicalDestOp, typename BitwiseDestOp,
          typename Adaptor = typename SrcOp::Adaptor>
class StableHLOToTTIRLogicalAndBitwiseOpConversionPattern
    : public OpConversionPattern<SrcOp> {

  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(SrcOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));

    if (getStableHLOOpType(srcOp) == StableHLOOpType::kLogical) {
      ttmlir::utils::replaceOpWithNewDPSOp<LogicalDestOp>(
          rewriter, srcOp, outputType, adaptor.getOperands());
    } else {
      ttmlir::utils::replaceOpWithNewDPSOp<BitwiseDestOp>(
          rewriter, srcOp, outputType, adaptor.getOperands());
    }

    return success();
  }

private:
  enum StableHLOOpType { kLogical = 0, kBitwise = 1 };

  // Determines stablehlo op type based on its operand types (i.e. their
  // bit width). This assumes boolean operands are modeled as 1bit wide ints.
  static StableHLOOpType getStableHLOOpType(const SrcOp &srcOp) {
    // Checks if all operands are boolean (have bit width equal to 1).
    bool allOperandsAreBoolean = std::all_of(
        srcOp->operand_begin(), srcOp->operand_end(), [](auto operand) {
          return mlir::cast<RankedTensorType>(operand.getType())
                     .getElementTypeBitWidth() == 1;
        });

    return allOperandsAreBoolean ? StableHLOOpType::kLogical
                                 : StableHLOOpType::kBitwise;
  }
};
} // namespace

template <typename SrcOpTy>
LogicalResult getReduceType(SrcOpTy &srcOp, ReduceType &reduceType) {
  if constexpr (!std::is_same<SrcOpTy, mlir::stablehlo::AllReduceOp>::value) {
    return failure();
  }
  // Check operations in the first block and determine reduce type for now
  // TODO(wooseoklee): This pattern matching mechanism may need to be updated as
  // we see complicated patterns of reduce block in the future.
  auto &block = srcOp.getRegion().front();
  for (Operation &op : block) {
    if (isa<mlir::stablehlo::AddOp>(op)) {
      reduceType = ReduceType::Sum;
      return success();
    }
    if (isa<mlir::stablehlo::MaxOp>(op)) {
      reduceType = ReduceType::Max;
      return success();
    }
    if (isa<mlir::stablehlo::MinOp>(op)) {
      reduceType = ReduceType::Min;
      return success();
    }
  }
  // Other reduce types are currently not supported
  return failure();
}

// StalbeHLO spec.md defines following channel type for ccl ops
enum StableHLOChannelType {
  // CHANNEL_TYPE_INVALID = 0 : Invalid primitive type to serve as
  // default.
  kChannelTypeInvalid = 0,
  // DEVICE_TO_DEVICE = 1 : A channel for sending data between
  // devices.
  kChannelTypeDeviceToDevice = 1,
  // DEVICE_TO_HOST = 2 : A channel for sending data from the
  // device to the host. Can only be used with a Send operation.
  kChannelTypeDeviceToHost = 2,
  // HOST_TO_DEVICE = 3 : A channel for sending data from the host to
  // the device. Can only be used with a Recv operation.
  kChannelTypeHostToDevice = 3,
};

namespace {
class StableHLOToTTIRAllReduceOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::AllReduceOp> {

  using OpConversionPattern<mlir::stablehlo::AllReduceOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::AllReduceOp srcOp,
                  mlir::stablehlo::AllReduceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Check legality of the operation
    LogicalResult err = checkBasicLegality(srcOp, adaptor, rewriter);
    if (failed(err)) {
      return err;
    }

    IntegerAttr channelHandleAttr;
    if (auto srcChannelHandleAttr = adaptor.getChannelHandleAttr()) {
      // channelType is supposed to be DEVICE_TO_DEVICE or Invalid for CCL ops.
      // Currently, we ensure if it is DEVICE_TO_DEVICE commmuincaiton.
      // Consider preserving this information in the future if the attribute
      // is non-DEVICE_TO_DEVICE values.
      auto channelType = static_cast<int32_t>(srcChannelHandleAttr.getType());
      if (channelType != kChannelTypeDeviceToDevice &&
          channelType != kChannelTypeInvalid) {
        return failure();
      }

      channelHandleAttr = rewriter.getSI32IntegerAttr(
          static_cast<int32_t>(srcChannelHandleAttr.getHandle()));
    }
    mlir::DenseIntElementsAttr replicaGroupsAttr =
        adaptor.getReplicaGroupsAttr();
    bool useGlobalDeviceIds = adaptor.getUseGlobalDeviceIds();

    // Parse computation in region and add it to ttirAttrs
    ReduceType reduceType;
    if (failed(getReduceType(srcOp, reduceType))) {
      return rewriter.notifyMatchFailure(
          srcOp, "AllReduceOp cannot specify reduce type.");
    }

    // stablehlo all_reduce op has no dimension defined in the op. Thus, we
    // estimate possible all reduce dimension. Current algorithm is to search
    // for first non-one dimension of input tensor from back.
    auto estimateDim = [](mlir::RankedTensorType inputType) -> int32_t {
      if (inputType.getRank() == 1) {
        return 0;
      }
      auto inputShape = inputType.getShape();
      auto nonOneIt = std::find_if(inputShape.rbegin(), inputShape.rend(),
                                   [](int64_t s) { return s != 1; });
      int32_t dim = inputType.getRank() - 1 -
                    std::distance(inputShape.rbegin(), nonOneIt);
      // all one shape, then select the deepest dim
      if (dim < 0) {
        dim = inputType.getRank() - 1;
      }
      return dim;
    };

    // Handle variadic input/output pairs by creating mulitple AllReduceOps.
    llvm::SmallVector<mlir::Value> allReduceOpResults;
    for (auto [inputOperand, resultOperand] :
         llvm::zip_equal(adaptor.getOperands(), srcOp->getResults())) {
      auto inputType = mlir::cast<RankedTensorType>(inputOperand.getType());
      auto outputType = mlir::cast<RankedTensorType>(
          getTypeConverter()->convertType(resultOperand.getType()));

      auto allReduceOp =
          ttmlir::utils::createDPSOp<mlir::tt::ttir::AllReduceOp>(
              rewriter, srcOp.getLoc(), outputType, inputOperand,
              replicaGroupsAttr, estimateDim(inputType), channelHandleAttr,
              useGlobalDeviceIds, reduceType);

      allReduceOpResults.push_back(allReduceOp.getResult());
    }
    rewriter.replaceOp(srcOp, allReduceOpResults);

    return success();
  }

private:
  LogicalResult
  checkBasicLegality(mlir::stablehlo::AllReduceOp &srcOp,
                     mlir::stablehlo::AllReduceOp::Adaptor adaptor,
                     ConversionPatternRewriter &rewriter) const {
    if (srcOp.getOperands().empty()) {
      return rewriter.notifyMatchFailure(
          srcOp, "AllReduceOp must have at least one input/output.");
    }

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRAllGatherOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::AllGatherOp> {
  using OpConversionPattern<mlir::stablehlo::AllGatherOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::AllGatherOp srcOp,
                  mlir::stablehlo::AllGatherOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check legality of the operation
    LogicalResult err = checkBasicLegality(srcOp, adaptor, rewriter);
    if (failed(err)) {
      return err;
    }

    // Create the output tensor type based on inputs
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult(0).getType()));

    // Create an empty output tensor with the computed shape
    tensor::EmptyOp outputTensor = rewriter.create<tensor::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    SmallVector<Type> ttirTypes;
    if (failed(this->getTypeConverter()->convertTypes(srcOp->getResultTypes(),
                                                      ttirTypes))) {
      return failure();
    }

    auto ttirOperands = srcOp.getOperandsMutable();
    ttirOperands.append(ValueRange(outputTensor));

    /*
    We need to figure out what the cluster axis is based on replica_groups.
    Replica groups define which device axis we are performing all_gather on.
    It is a 2D vector. Each element in replica_groups contains a list of devices
    that will perform all_gather with each other. Currently we only support 2D
    meshes, but this algorithm can be expanded for ND.

    ex.
    mesh = [2, 4]
    replica_groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    0 1 2 3
    4 5 6 7

    all_gather happens on (0, 1, 2, 3) and (4, 5, 6, 7) so cluster_axis = 1
    (mesh[1])

    mesh = [2, 4]
    replica_groups = [[0, 4], [1, 5], [2, 6], [3, 7]]
    0 1 2 3
    4 5 6 7

    all_gather happens on (0, 4), (1, 5), (2, 6), (3, 7) so cluster_axis = 0
    (mesh[0])

    */

    uint32_t clusterAxis = 0;
    auto replicaGroups = adaptor.getReplicaGroups();
    auto replicaGroupsShape = adaptor.getReplicaGroups().getType().getShape();

    if (replicaGroupsShape.size() == 0) {
      // Cannot have replicas of size 0, this means we are not performing the
      // all_gather across any device.
      return failure();
    }

    // Case where we have single devices in each replica_group (ie perform
    // all_gather against itself which should be optimized away).
    // We also assume we are only using our constrained mesh types (ie 1x8, 1x32
    // etc) and cannot have (32x1, 8x1).
    if (replicaGroupsShape[1] != 1) {
      auto firstElementIt = replicaGroups.begin();
      auto secondElementIt = firstElementIt + 1;

      clusterAxis = (((*firstElementIt) + 1) == *secondElementIt);
    }

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::AllGatherOp>(
        srcOp, outputType, adaptor.getOperands()[0], outputTensor,
        adaptor.getAllGatherDim(), clusterAxis);

    return success();
  }

private:
  LogicalResult
  checkBasicLegality(mlir::stablehlo::AllGatherOp &srcOp,
                     mlir::stablehlo::AllGatherOp::Adaptor adaptor,
                     ConversionPatternRewriter &rewriter) const {
    if (srcOp.getOperands().empty() || srcOp.getOperands().size() > 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "AllGatherOp must have one input/output for now.");
    }

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRCustomCallOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CustomCallOp> {

  using OpConversionPattern<mlir::stablehlo::CustomCallOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp srcOp,
                  mlir::stablehlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Check legality of the operation
    LogicalResult err = checkBasicLegality(srcOp, adaptor, rewriter);
    if (failed(err)) {
      return err;
    }

    auto callTargetName = adaptor.getCallTargetNameAttr();

    // Currently stablehlo.custom_call with following functions from
    // jax/openxla are supported
    if (callTargetName !=
            mlir::tt::sharding_utils::kShardingCustomCallTargetName &&
        callTargetName !=
            mlir::tt::sharding_utils::kSPMDFullToShardShapeCallTargetName &&
        callTargetName !=
            mlir::tt::sharding_utils::kSPMDShardToFullShapeCallTargetName) {
      return failure();
    }

    auto shardingAttr =
        dyn_cast_if_present<StringAttr>(adaptor.getAttributes().get(
            mlir::tt::sharding_utils::kXlaShardingAttr));
    if (!shardingAttr) {
      return failure();
    }

    mlir::tt::sharding_utils::MeshSharding meshSharding;
    auto error = meshSharding.convertGSPMDShardingToMeshSharding(
        shardingAttr.getValue());
    if (auto e = error.takeError()) {
      return rewriter.notifyMatchFailure(srcOp, llvm::toString(std::move(e)));
    }

    // For GSPMD, meshShape is extracted by the parser. Then, add it as module
    // attribute such that the information is used by later pipeline stage.
    auto meshShape = meshSharding.getMeshShape();
    if (meshShape.size() > 1) {
      auto module = srcOp->getParentOfType<ModuleOp>();
      if (!module) {
        llvm_unreachable("Require module as one of parent ops.");
      }
      mlir::tt::utils::addMeshToModuleAttribute(
          rewriter, module, StringAttr::get(getContext(), "mesh_gspmd"),
          meshShape);
    }

    if (callTargetName ==
        mlir::tt::sharding_utils::kSPMDFullToShardShapeCallTargetName) {
      // @Sharding => @SPMDFullToShardShape pattern
      Operation *shardingOp = srcOp->getOperand(0).getDefiningOp();
      if (!shardingOp) {
        return rewriter.notifyMatchFailure(
            srcOp, "Requires operand to be defined by prior Sharding op.");
      }
      rewriter.replaceOp(srcOp, shardingOp->getResult(0));
    } else if (callTargetName ==
               mlir::tt::sharding_utils::kSPMDShardToFullShapeCallTargetName) {
      // @Sharding => @SPMDShardToFullShape pattern
      Operation *shardingOp = srcOp->getOperand(0).getDefiningOp();
      if (!shardingOp) {
        return rewriter.notifyMatchFailure(
            srcOp, "Requires operand to be defined by prior Sharding op.");
      }

      auto outputType = mlir::cast<RankedTensorType>(
          getTypeConverter()->convertType(srcOp->getResult(0).getType()));

      ttmlir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::MeshShardOp>(
          rewriter, srcOp, outputType, adaptor.getInputs().front(),
          meshSharding.getShardType(),
          mlir::tt::MeshShardDirection::ShardToFull,
          meshSharding.getShardShape(), meshSharding.getShardDims());

    } else if (callTargetName ==
               mlir::tt::sharding_utils::kShardingCustomCallTargetName) {
      if (meshSharding.getShardType() == mlir::tt::MeshShardType::Manual) {
        // @Sharding => @SPMDShardToFullShape pattern
        // "manual" sharding indicates no sharding is required.
        rewriter.replaceOp(srcOp, srcOp->getOperand(0));
      } else {
        // @Sharding => @SPMDFullToShardShape pattern
        auto fullToShardCustomCall =
            mlir::dyn_cast_if_present<mlir::stablehlo::CustomCallOp>(
                *srcOp->user_begin());
        if (!fullToShardCustomCall || !fullToShardCustomCall->hasOneUse()) {
          return failure();
        }

        // JAX automatic sharding pre-shards input tensors and provides multiple
        // buffers. Thus, mesh sharding operations should not shard the tensors
        // twice if they are function arguments and pre-sharded by frontend.
        // Runtime ignores mesh sharding operation if it is set as manual
        // sharding.
        auto inputOperand = adaptor.getInputs().front();
        auto funcOp = srcOp->getParentOfType<mlir::func::FuncOp>();
        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(inputOperand)) {
          auto argNum = blockArg.getArgNumber();
          if (auto argShardingAttr = funcOp.getArgAttrOfType<mlir::StringAttr>(
                  argNum, mlir::tt::sharding_utils::kXlaShardingAttr)) {
            if (argShardingAttr == shardingAttr) {
              meshSharding.setDummyShardingOp();
              rewriter.modifyOpInPlace(funcOp, [&]() {
                funcOp.removeArgAttr(
                    argNum, mlir::tt::sharding_utils::kXlaShardingAttr);
              });
            } else {
              llvm_unreachable("GSPMD customCallOp and function argument "
                               "shardings are different.");
            }
          }
        }

        auto outputType =
            mlir::cast<RankedTensorType>(getTypeConverter()->convertType(
                fullToShardCustomCall->getResult(0).getType()));

        ttmlir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::MeshShardOp>(
            rewriter, srcOp, outputType, inputOperand,
            meshSharding.getShardType(),
            mlir::tt::MeshShardDirection::FullToShard,
            meshSharding.getShardShape(), meshSharding.getShardDims());
      }
    }
    return success();
  }

private:
  LogicalResult
  checkBasicLegality(mlir::stablehlo::CustomCallOp &srcOp,
                     mlir::stablehlo::CustomCallOp::Adaptor adaptor,
                     ConversionPatternRewriter &rewriter) const {
    // Expect single input/output and at least one use of result.
    if (srcOp->getNumOperands() != 1 || srcOp->getNumResults() != 1 ||
        srcOp->getResult(0).use_empty()) {
      return failure();
    }

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRSliceOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::SliceOp> {

  using OpConversionPattern<mlir::stablehlo::SliceOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::SliceOp srcOp,
                  mlir::stablehlo::SliceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Create the output tensor type based on inputs
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    llvm::SmallVector<int32_t> startIndices(adaptor.getStartIndices());
    llvm::SmallVector<int32_t> endIndices(adaptor.getLimitIndices());
    llvm::SmallVector<int32_t> step(adaptor.getStrides());

    ttmlir::utils::replaceOpWithNewDPSOp<ttir::SliceOp>(
        rewriter, srcOp, outputType, adaptor.getOperand(),
        rewriter.getI32ArrayAttr(startIndices),
        rewriter.getI32ArrayAttr(endIndices), rewriter.getI32ArrayAttr(step));

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIROpClampOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ClampOp> {

  using OpConversionPattern<mlir::stablehlo::ClampOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ClampOp srcOp,
                  mlir::stablehlo::ClampOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));

    if (std::optional<float> minValue = getConstantValue(adaptor.getMin()),
        maxValue = getConstantValue(adaptor.getMax());
        minValue && maxValue) {
      ttmlir::utils::replaceOpWithNewDPSOp<ttir::ClampOp>(
          rewriter, srcOp, outputType, adaptor.getOperand(),
          mlir::APFloat(*minValue), mlir::APFloat(*maxValue));

      return success();
    }

    ttir::MaximumOp maximumOp = ttmlir::utils::createDPSOp<ttir::MaximumOp>(
        rewriter, srcOp->getLoc(), outputType, adaptor.getMin(),
        adaptor.getOperand());
    ttmlir::utils::replaceOpWithNewDPSOp<ttir::MinimumOp>(
        rewriter, srcOp, outputType, maximumOp.getResult(0), adaptor.getMax());

    return success();
  }

private:
  std::optional<float> getConstantValue(Value value) const {
    if (auto constantOp = value.getDefiningOp<ttir::ConstantOp>()) {
      auto attr = constantOp.getValueAttr();
      if (!attr.isSplat()) {
        return {};
      }
      return attr.getElementType().isInteger()
                 ? static_cast<float>(attr.getSplatValue<int>())
                 : attr.getSplatValue<float>();
    }
    return {};
  }
};
} // namespace

namespace {
class StableHLOToTTIRGatherOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::GatherOp> {
  using OpConversionPattern<mlir::stablehlo::GatherOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::GatherOp srcOp,
                  mlir::stablehlo::GatherOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    auto dimensionNumbers = srcOp.getDimensionNumbers();

    ttmlir::utils::replaceOpWithNewDPSOp<ttir::GatherOp>(
        rewriter, srcOp, outputType, adaptor.getOperands()[0],
        adaptor.getOperands()[1], dimensionNumbers.getOffsetDims(),
        dimensionNumbers.getCollapsedSliceDims(),
        dimensionNumbers.getOperandBatchingDims(),
        dimensionNumbers.getStartIndicesBatchingDims(),
        dimensionNumbers.getStartIndexMap(),
        dimensionNumbers.getIndexVectorDim(), srcOp.getSliceSizes(),
        /*indices_are_sorted=*/false);

    return success();
  }
};
} // namespace

namespace {
template <typename SrcIotaOp, typename Adaptor = typename SrcIotaOp::Adaptor>
class StableHLOToTTIROpIotaOpConversionPattern
    : public OpConversionPattern<SrcIotaOp> {

  using OpConversionPattern<SrcIotaOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(SrcIotaOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    RankedTensorType outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResult().getType()));
    rewriter.replaceOpWithNewOp<ttir::ArangeOp>(
        srcOp, outputType, 0, outputType.getDimSize(adaptor.getIotaDimension()),
        1, adaptor.getIotaDimension());

    // Dynamic Iota has an output_shape attribute but the output shape is
    // already known by the result type This is to remove the operand that
    // will become dead code
    for (auto operand : adaptor.getOperands()) {
      if (operand.getDefiningOp()) {
        rewriter.eraseOp(operand.getDefiningOp());
      }
    }

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRScatterOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ScatterOp> {

  using OpConversionPattern<mlir::stablehlo::ScatterOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ScatterOp srcOp,
                  mlir::stablehlo::ScatterOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResults()[0].getType()));

    Value operand = srcOp.getInputs()[0];
    Value scatterIndices = adaptor.getScatterIndices();
    Value update = srcOp.getUpdates()[0];
    auto updateWindowsDims =
        adaptor.getScatterDimensionNumbers().getUpdateWindowDims();
    auto insertedWindowDims =
        adaptor.getScatterDimensionNumbers().getInsertedWindowDims();
    auto inputBatchingDims =
        adaptor.getScatterDimensionNumbers().getInputBatchingDims();
    auto scatterIndicesBatchingDims =
        adaptor.getScatterDimensionNumbers().getScatterIndicesBatchingDims();
    auto scatterDimsToOperandDims =
        adaptor.getScatterDimensionNumbers().getScatterDimsToOperandDims();
    auto indexVectorDim =
        adaptor.getScatterDimensionNumbers().getIndexVectorDim();
    auto indicesAreSorted = adaptor.getIndicesAreSorted();
    auto uniqueIndices = adaptor.getUniqueIndices();

    auto newScatterOp = ttmlir::utils::createDPSOp<ttir::ScatterOp>(
        rewriter, srcOp->getLoc(), outputType, operand, scatterIndices, update,
        llvm::SmallVector<int32_t>(updateWindowsDims),
        llvm::SmallVector<int32_t>(insertedWindowDims),
        llvm::SmallVector<int32_t>(inputBatchingDims),
        llvm::SmallVector<int32_t>(scatterIndicesBatchingDims),
        llvm::SmallVector<int32_t>(scatterDimsToOperandDims), indexVectorDim,
        indicesAreSorted, uniqueIndices);

    newScatterOp->getRegion(0).takeBody(adaptor.getUpdateComputation());
    changeRegionTypes(newScatterOp->getRegion(0), *getTypeConverter(),
                      rewriter);

    rewriter.replaceOp(srcOp, newScatterOp);

    return success();
  }

private:
  void changeRegionTypes(mlir::Region &region,
                         const mlir::TypeConverter &typeConverter,
                         mlir::PatternRewriter &rewriter) const {
    Block &block = *region.getBlocks().begin();
    llvm::SmallVector<mlir::BlockArgument, 4> oldArguments(
        block.getArguments().begin(), block.getArguments().end());
    llvm::SmallVector<mlir::Value, 4> newArguments;

    // Add new arguments with updated types to the block.
    for (auto arg : oldArguments) {
      if (auto newType = typeConverter.convertType(arg.getType())) {
        mlir::BlockArgument newArg = block.addArgument(newType, arg.getLoc());
        newArguments.push_back(newArg);
      } else {
        newArguments.push_back(arg); // Type didn't change
      }
    }

    for (auto it : llvm::zip(oldArguments, newArguments)) {
      mlir::BlockArgument oldArg = std::get<0>(it);
      mlir::Value newArg = std::get<1>(it);
      if (oldArg != newArg) {
        oldArg.replaceAllUsesWith(newArg);
      }
    }

    for (auto arg : oldArguments) {
      if (!llvm::is_contained(newArguments, arg)) {
        block.eraseArgument(arg.getArgNumber());
      }
    }
  }
};
} // namespace

namespace {
class StableHLOToTTIRReturnOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ReturnOp> {

  using OpConversionPattern<mlir::stablehlo::ReturnOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReturnOp srcOp,
                  mlir::stablehlo::ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::YieldOp>(srcOp,
                                                         srcOp.getResults());

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIROpReverseOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ReverseOp> {

  using OpConversionPattern<mlir::stablehlo::ReverseOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReverseOp srcOp,
                  mlir::stablehlo::ReverseOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    ttmlir::utils::replaceOpWithNewDPSOp<ttir::ReverseOp>(
        rewriter, srcOp, outputType, adaptor.getOperand(),
        adaptor.getDimensions());

    return success();
  }
};
} // namespace
namespace {
class StableHLOToTTIROpPadOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::PadOp> {

  using OpConversionPattern<mlir::stablehlo::PadOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::PadOp srcOp,
                  mlir::stablehlo::PadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LogicalResult legalityResult = checkBasicLegality(srcOp, adaptor, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    SmallVector<int32_t> padDim;
    for (uint32_t i = 0; i < adaptor.getEdgePaddingLow().size(); i++) {
      padDim.push_back(adaptor.getEdgePaddingLow()[i]);
      padDim.push_back(adaptor.getEdgePaddingHigh()[i]);
    }

    ttir::ConstantOp valueDef =
        getConstantValueDefiningOp(adaptor.getPaddingValue());

    mlir::ElementsAttr paddingValueAttr = valueDef.getValueAttr();

    float value =
        paddingValueAttr.getElementType().isInteger()
            ? static_cast<float>(paddingValueAttr.getSplatValue<int>())
            : paddingValueAttr.getSplatValue<float>();

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::PadOp>(
        srcOp,
        outputType,                            // result type
        adaptor.getOperand(),                  // input
        rewriter.getDenseI32ArrayAttr(padDim), // padding dimensions
        rewriter.getF32FloatAttr(value)        // padding value
    );

    return success();
  }

private:
  ttir::ConstantOp getConstantValueDefiningOp(Value value) const {
    Operation *valueDef = value.getDefiningOp();

    // equivalent to valueDef != nullptr && isa<...>(valueDef)
    while (
        isa_and_nonnull<ttir::ReshapeOp, ttir::BroadcastOp, ttir::TypecastOp>(
            valueDef)) {
      valueDef = valueDef->getOperand(0).getDefiningOp();
    }
    return mlir::dyn_cast_or_null<ttir::ConstantOp>(valueDef);
  }

  LogicalResult checkBasicLegality(mlir::stablehlo::PadOp &srcOp,
                                   mlir::stablehlo::PadOp::Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {

    // Due to lack of support by device, we do not support interior padding,
    // so verify if interior padding is requested and exit early with error.
    for (int64_t eachVal : srcOp.getInteriorPadding()) {
      if (eachVal != 0) {
        return rewriter.notifyMatchFailure(srcOp,
                                           "Unsupported Interior Padding, only "
                                           "0 interior padding is supported.");
      }
    }

    if (srcOp.getEdgePaddingLow().size() != srcOp.getEdgePaddingHigh().size()) {
      return rewriter.notifyMatchFailure(
          srcOp, "Low and High padding dimensions should match.");
    }

    if (!getConstantValueDefiningOp(adaptor.getPaddingValue())) {
      return rewriter.notifyMatchFailure(
          srcOp, "Padding value cannot be traced back to a constant value "
                 "which has not been altered mathematically.");
    }

    return success();
  }
};
} // namespace

static void
addElementwiseUnaryOpsConversionPatterns(MLIRContext *ctx,
                                         RewritePatternSet &patterns,
                                         TypeConverter &typeConverter) {

  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::AbsOp, mlir::tt::ttir::AbsOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::CbrtOp, mlir::tt::ttir::CbrtOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::ConvertOp, mlir::tt::ttir::TypecastOp>>(typeConverter,
                                                               ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::CeilOp, mlir::tt::ttir::CeilOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::CosineOp, mlir::tt::ttir::CosOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::ExpOp, mlir::tt::ttir::ExpOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::FloorOp, mlir::tt::ttir::FloorOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::IsFiniteOp, mlir::tt::ttir::IsFiniteOp>>(typeConverter,
                                                                ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::NegOp, mlir::tt::ttir::NegOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::RsqrtOp, mlir::tt::ttir::RsqrtOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::SineOp, mlir::tt::ttir::SinOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::SqrtOp, mlir::tt::ttir::SqrtOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::Log1pOp, mlir::tt::ttir::Log1pOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::Expm1Op, mlir::tt::ttir::Expm1Op>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::SignOp, mlir::tt::ttir::SignOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::LogisticOp, mlir::tt::ttir::SigmoidOp>>(typeConverter,
                                                               ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::TanOp, mlir::tt::ttir::TanOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::TanhOp, mlir::tt::ttir::TanhOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::LogOp, mlir::tt::ttir::LogOp>>(typeConverter, ctx);
}

static void
addElementwiseBinaryOpsConversionPatterns(MLIRContext *ctx,
                                          RewritePatternSet &patterns,
                                          TypeConverter &typeConverter) {

  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::AddOp, mlir::tt::ttir::AddOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::DivOp, mlir::tt::ttir::DivOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::MaxOp, mlir::tt::ttir::MaximumOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::MinOp, mlir::tt::ttir::MinimumOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::MulOp, mlir::tt::ttir::MultiplyOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::SubtractOp, mlir::tt::ttir::SubtractOp>>(typeConverter,
                                                                ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::RemOp, mlir::tt::ttir::RemainderOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::SelectOp, mlir::tt::ttir::WhereOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::PowOp, mlir::tt::ttir::PowerOp>>(typeConverter, ctx);
}

static void addReduceOpsConversionPatterns(MLIRContext *ctx,
                                           RewritePatternSet &patterns,
                                           TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRReduceOpConversionPattern>(typeConverter, ctx);
}

static void addDotGeneralOpConversionPatterns(MLIRContext *ctx,
                                              RewritePatternSet &patterns,
                                              TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRDotGeneralOpConversionPattern>(typeConverter,
                                                             ctx);
}

static void
addGetDimensionSizeOpsConversionPatterns(MLIRContext *ctx,
                                         RewritePatternSet &patterns,
                                         TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRGetDimensionSizeOpConversionPattern>(
      typeConverter, ctx);
}

static void
addTensorCreationOpsConversionPatterns(MLIRContext *ctx,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRConstantOpConversionPattern>(typeConverter, ctx);
}

static void addBroadcastOpConversionPattern(MLIRContext *ctx,
                                            RewritePatternSet &patterns,
                                            TypeConverter &typeConverter) {

  patterns.add<StableHLOToTTIRBroadcastInDimOpConversionPattern>(typeConverter,
                                                                 ctx);
}

static void addConv2dOpConversionPattern(MLIRContext *ctx,
                                         RewritePatternSet &patterns,
                                         TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRConvolutionOpConversionPattern>(typeConverter,
                                                              ctx);
}

static void addReduceWindowOpConversionPattern(MLIRContext *ctx,
                                               RewritePatternSet &patterns,
                                               TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRReduceWindowOpConversionPattern>(typeConverter,
                                                               ctx);
}

static void addCompareOpsConversionPatterns(MLIRContext *ctx,
                                            RewritePatternSet &patterns,
                                            TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRCompareOpConversionPattern>(typeConverter, ctx);
}

static void addConcatOpsConversionPatterns(MLIRContext *ctx,
                                           RewritePatternSet &patterns,
                                           TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRConcatOpConversionPattern>(typeConverter, ctx);
}

static void addTransposeOpConversionPattern(MLIRContext *ctx,
                                            RewritePatternSet &patterns,
                                            TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRTransposeOpConversionPattern>(typeConverter, ctx);
}

static void addReshapeOpConversionPattern(MLIRContext *ctx,
                                          RewritePatternSet &patterns,
                                          TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRReshapeOpConversionPattern>(typeConverter, ctx);
}

static void addCCLOpsConversionPattern(MLIRContext *ctx,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRAllReduceOpConversionPattern>(typeConverter, ctx);
  patterns.add<StableHLOToTTIRAllGatherOpConversionPattern>(typeConverter, ctx);
  patterns.add<StableHLOToTTIRCustomCallOpConversionPattern>(typeConverter,
                                                             ctx);
}

static void
addLogicalAndBitwiseOpsConversionPatterns(MLIRContext *ctx,
                                          RewritePatternSet &patterns,
                                          TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRLogicalAndBitwiseOpConversionPattern<
      mlir::stablehlo::AndOp, mlir::tt::ttir::LogicalAndOp,
      mlir::tt::ttir::BitwiseAndOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIRLogicalAndBitwiseOpConversionPattern<
      mlir::stablehlo::OrOp, mlir::tt::ttir::LogicalOrOp,
      mlir::tt::ttir::BitwiseOrOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIRLogicalAndBitwiseOpConversionPattern<
      mlir::stablehlo::XorOp, mlir::tt::ttir::LogicalXorOp,
      mlir::tt::ttir::BitwiseXorOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIRLogicalAndBitwiseOpConversionPattern<
      mlir::stablehlo::NotOp, mlir::tt::ttir::LogicalNotOp,
      mlir::tt::ttir::BitwiseNotOp>>(typeConverter, ctx);
}

static void addSliceOpConversionPattern(MLIRContext *ctx,
                                        RewritePatternSet &patterns,
                                        TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRSliceOpConversionPattern>(typeConverter, ctx);
}

static void addClampOpConversionPattern(MLIRContext *ctx,
                                        RewritePatternSet &patterns,
                                        TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIROpClampOpConversionPattern>(typeConverter, ctx);
}

static void addGatherOpConversionPattern(MLIRContext *ctx,
                                         RewritePatternSet &patterns,
                                         TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRGatherOpConversionPattern>(typeConverter, ctx);
}

static void addIotaOpConversionPattern(MLIRContext *ctx,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIROpIotaOpConversionPattern<stablehlo::IotaOp>>(
      typeConverter, ctx);
  patterns
      .add<StableHLOToTTIROpIotaOpConversionPattern<stablehlo::DynamicIotaOp>>(
          typeConverter, ctx);
}

static void addScatterOpConversionPatterns(MLIRContext *ctx,
                                           RewritePatternSet &patterns,
                                           TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRScatterOpConversionPattern>(typeConverter, ctx);
}

static void addReturnOpConversionPatterns(MLIRContext *ctx,
                                          RewritePatternSet &patterns,
                                          TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRReturnOpConversionPattern>(typeConverter, ctx);
}

static void addReverseOpConversionPattern(MLIRContext *ctx,
                                          RewritePatternSet &patterns,
                                          TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIROpReverseOpConversionPattern>(typeConverter, ctx);
}

static void addPadOpConversionPattern(MLIRContext *ctx,
                                      RewritePatternSet &patterns,
                                      TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIROpPadOpConversionPattern>(typeConverter, ctx);
}

namespace mlir::tt {

void populateStableHLOToTTIRPatterns(MLIRContext *ctx,
                                     RewritePatternSet &patterns,
                                     TypeConverter &typeConverter) {
  addElementwiseUnaryOpsConversionPatterns(ctx, patterns, typeConverter);
  addElementwiseBinaryOpsConversionPatterns(ctx, patterns, typeConverter);
  addReduceOpsConversionPatterns(ctx, patterns, typeConverter);
  addDotGeneralOpConversionPatterns(ctx, patterns, typeConverter);
  addGetDimensionSizeOpsConversionPatterns(ctx, patterns, typeConverter);
  addTensorCreationOpsConversionPatterns(ctx, patterns, typeConverter);
  addBroadcastOpConversionPattern(ctx, patterns, typeConverter);
  addConv2dOpConversionPattern(ctx, patterns, typeConverter);
  addReduceWindowOpConversionPattern(ctx, patterns, typeConverter);
  addCompareOpsConversionPatterns(ctx, patterns, typeConverter);
  addConcatOpsConversionPatterns(ctx, patterns, typeConverter);
  addTransposeOpConversionPattern(ctx, patterns, typeConverter);
  addReshapeOpConversionPattern(ctx, patterns, typeConverter);
  addCCLOpsConversionPattern(ctx, patterns, typeConverter);
  addLogicalAndBitwiseOpsConversionPatterns(ctx, patterns, typeConverter);
  addSliceOpConversionPattern(ctx, patterns, typeConverter);
  addClampOpConversionPattern(ctx, patterns, typeConverter);
  addGatherOpConversionPattern(ctx, patterns, typeConverter);
  addIotaOpConversionPattern(ctx, patterns, typeConverter);
  addScatterOpConversionPatterns(ctx, patterns, typeConverter);
  addReturnOpConversionPatterns(ctx, patterns, typeConverter);
  addReverseOpConversionPattern(ctx, patterns, typeConverter);
  addPadOpConversionPattern(ctx, patterns, typeConverter);
}

} // namespace mlir::tt
