// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOToTTIR.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/StableHLO/Utils/GSPMDUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/Utils/Mesh.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace mlir;
using namespace mlir::tt;

// Helper to extract values from optional StableHLO attributes with defaults.
// StableHLO convolution attributes like window_strides, lhs_dilation, etc.
// are optional and default to 1 (or 0 for padding) when not specified.
static llvm::SmallVector<int64_t>
getI64ArrayOrDefault(mlir::DenseI64ArrayAttr attr, size_t size,
                     int64_t defaultValue) {
  if (attr) {
    return llvm::SmallVector<int64_t>(attr.asArrayRef());
  }
  return llvm::SmallVector<int64_t>(size, defaultValue);
}

// Helper for padding which uses DenseIntElementsAttr in StableHLO.
static llvm::SmallVector<int64_t>
getPaddingOrDefault(mlir::DenseIntElementsAttr attr, size_t numSpatialDims) {
  if (attr) {
    return llvm::SmallVector<int64_t>(attr.getValues<int64_t>());
  }
  return llvm::SmallVector<int64_t>(numSpatialDims * 2, 0);
}

// Typical initialization values for reduction ops.
enum TypicalInitReductionValue {
  NEG_INF, // It is also used for minimum integer value.
  ZERO,
};

// Check if the constant op is initialized with the desired init value.
static bool checkInitValue(mlir::stablehlo::ConstantOp initValueOp,
                           TypicalInitReductionValue desired) {
  if (initValueOp.getValueAttr().size() != 1) {
    return false;
  }

  float desiredF32;
  double desiredF64;
  uint16_t desiredBF16;
  int32_t desiredI32;
  int64_t desiredI64;
  int8_t desiredI8;
  bool desiredI1;
  if (desired == TypicalInitReductionValue::NEG_INF) {
    desiredF32 = -std::numeric_limits<float>::infinity();
    desiredF64 = -std::numeric_limits<double>::infinity();
    desiredBF16 = 0xff80; // This is -inf in bfloat16 raw bits
    desiredI32 = std::numeric_limits<int32_t>::min();
    desiredI64 = std::numeric_limits<int64_t>::min();
    desiredI8 = std::numeric_limits<int8_t>::min();
    desiredI1 = false;
  } else if (desired == TypicalInitReductionValue::ZERO) {
    desiredF32 = 0.0;
    desiredF64 = 0.0;
    desiredBF16 = 0x0000; // This is 0 in bfloat16 raw bits
    desiredI32 = 0;
    desiredI64 = 0;
    desiredI8 = 0;
    desiredI1 = false;
  } else {
    return false;
  }

  // Comparing actual bits in case of bfloat16.
  if (initValueOp.getResult().getType().getElementType().isBF16()) {
    // Collect the values into a vector
    std::vector<mlir::Attribute> values;
    for (int64_t i = 0; i < initValueOp.getValueAttr().size(); ++i) {
      values.push_back(
          initValueOp.getValueAttr().getValues<mlir::Attribute>()[i]);
    }

    auto denseValues = ::mlir::DenseElementsAttr::get(
        initValueOp.getValueAttr().getShapedType(), values);
    uint16_t bfloatBits =
        static_cast<uint16_t>(*denseValues.getRawData().data());
    return bfloatBits == desiredBF16;
  }
  if (initValueOp.getResult().getType().getElementType().isF32()) {
    return *initValueOp.getValue().value_begin<float>() == desiredF32;
  }
  if (initValueOp.getResult().getType().getElementType().isF64()) {
    return *initValueOp.getValue().value_begin<double>() == desiredF64;
  }
  if (initValueOp.getResult().getType().getElementType().isInteger(32)) {
    return *initValueOp.getValue().value_begin<int32_t>() == desiredI32;
  }
  if (initValueOp.getResult().getType().getElementType().isInteger(64)) {
    return *initValueOp.getValue().value_begin<int64_t>() == desiredI64;
  }
  if (initValueOp.getResult().getType().getElementType().isInteger(8)) {
    return *initValueOp.getValue().value_begin<uint8_t>() == desiredI8;
  }
  if (initValueOp.getResult().getType().getElementType().isInteger(1)) {
    return *initValueOp.getValue().value_begin<bool>() == desiredI1;
  }

  return false;
}

// Helper function to find the TTIR constant defining op by traversing through
// operations that preserve constant semantics (ReshapeOp, BroadcastOp, and
// TypecastOp).
static ttir::ConstantOp getConstantValueDefiningOp(Value value) {
  Operation *valueDef = value.getDefiningOp();

  // equivalent to valueDef != nullptr && isa<...>(valueDef)
  while (isa_and_nonnull<ttir::ReshapeOp, ttir::BroadcastOp, ttir::TypecastOp>(
      valueDef)) {
    valueDef = valueDef->getOperand(0).getDefiningOp();
  }
  return mlir::dyn_cast_if_present<ttir::ConstantOp>(valueDef);
}

static LogicalResult parseBoolFromStringAttr(const mlir::StringAttr &stringAttr,
                                             bool &result) {
  if (stringAttr.getValue().lower() == "true") {
    result = true;
  } else if (stringAttr.getValue().lower() == "false") {
    result = false;
  } else {
    return failure();
  }
  return success();
}

static LogicalResult
parseFloatFromStringAttr(const mlir::StringAttr &stringAttr, float &result) {
  if (!llvm::to_float(stringAttr.getValue(), result)) {
    return failure();
  }
  return success();
}

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

    rewriter.replaceOpWithNewOp<DestOp>(srcOp, outputType,
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
    LogicalResult legalityResult =
        checkConversionLegality(srcOp, adaptor, rewriter);
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
    if (mlir::isa<mlir::stablehlo::OrOp>(innerOp)) {
      return matchAndRewriteInternal<mlir::tt::ttir::ReduceOrOp>(srcOp, adaptor,
                                                                 rewriter);
    }
    if (isArgMax(srcOp, adaptor, rewriter)) {
      return matchAndRewriteInternalArgMax(srcOp, adaptor, rewriter);
    }

    return failure();
  }

private:
  LogicalResult
  checkConversionLegality(mlir::stablehlo::ReduceOp &srcOp,
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
    if (mlir::isa<mlir::stablehlo::AndOp>(innerOp) ||
        mlir::isa<mlir::stablehlo::OrOp>(innerOp)) {
      bool allOperandsAreBoolean = std::all_of(
          srcOp->operand_begin(), srcOp->operand_end(), [](auto operand) {
            return mlir::cast<RankedTensorType>(operand.getType())
                       .getElementTypeBitWidth() == 1;
          });
      // Stablehlo (unlike other dialects) has single op for both logical and
      // bitwise operation. Data type is used to distinguish between logical and
      // bitwise operation. If the datatype is boolean then it is a logical
      // operation; otherwise it is bitwise operation. This check ensure that
      // the inputs are boolean as tt-metal only supports logical operations.
      if (!allOperandsAreBoolean) {
        return rewriter.notifyMatchFailure(
            srcOp,
            "stablehlo.reduce for stablehlo.and/stablehlo.or operator is only "
            "supported for logical operator.");
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

    rewriter.replaceOpWithNewOp<DestOp>(srcOp, outputType,
                                        adaptor.getInputs().front(),
                                        /*keep_dim=*/false, dimArg);

    return success();
  }

  LogicalResult
  matchAndRewriteInternalArgMax(mlir::stablehlo::ReduceOp &srcOp,
                                mlir::stablehlo::ReduceOp::Adaptor &adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult(1).getType()));

    // Can't reuse the original dimensions attribute because it uses i64 type.
    mlir::ArrayAttr dimArg = rewriter.getI32ArrayAttr(
        llvm::SmallVector<int32_t>(srcOp.getDimensions()));

    // stablehlo.reduce op generates two results; the first output is maximum
    // value which is not consumed in subsequent graph and the second out is the
    // index of maximum value which is consumed in subsequent graph. On other
    // hand ttir.argmax generates one result only.
    // So 'rewriter.replaceOpWithNewOp' will not work due to difference in
    // number of outputs.
    // We are creating the new ttir op explicitly here, then replace the
    // original op uses with the new op, and finally, erase the original op
    // explicitly.
    ttir::ArgMaxOp newOp = rewriter.create<tt::ttir::ArgMaxOp>(
        srcOp->getLoc(), outputType, adaptor.getInputs().front(),
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
  //    value which is consumed in subsequent graph.
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

    mlir::Value val = inputs.back();
    auto *op = val.getDefiningOp();

    // IotaOp can be preceded by either a BroadcastInDim or a Reshape.
    while (op) {
      if (isa<mlir::stablehlo::IotaOp>(op)) {
        return true;
      }

      if (isa<mlir::stablehlo::BroadcastInDimOp, mlir::stablehlo::ReshapeOp>(
              op)) {
        val = op->getOperand(0);
        op = val.getDefiningOp();
        continue;
      }

      return false;
    }

    return false;
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
    if (operations.empty()) {
      return false;
    }

    mlir::Operation &firstOp = operations.front();

    // Torch and Jax ArgMax patterns are different.
    if (verifyTorchOpArgMaxPattern(firstOp)) {
      return true;
    }
    if (verifyJaxOpArgMaxPattern(firstOp)) {
      return true;
    }

    return false;
  }

  // Pattern match the ops for tt-torch ArgMax op.
  // The pattern is as follows:
  //  stablehlo.compare (GE)
  //  stablehlo.select / stablehlo.maximum
  //  stablehlo.compare
  //  stablehlo.min
  //  stablehlo.select
  //  stablehlo.select
  //  stablehlo.return
  bool verifyTorchOpArgMaxPattern(mlir::Operation &operation) const {
    mlir::Operation *op = &operation;
    if (!isa<mlir::stablehlo::CompareOp>(op)) {
      return false;
    }
    mlir::stablehlo::CompareOp compareOp =
        mlir::cast<mlir::stablehlo::CompareOp>(op);
    if (compareOp.getComparisonDirection() !=
        mlir::stablehlo::ComparisonDirection::GE) {
      return false;
    }

    op = op->getNextNode();
    if (!isa_and_nonnull<mlir::stablehlo::SelectOp, mlir::stablehlo::MaxOp>(
            op)) {
      return false;
    }

    op = op->getNextNode();
    if (!isa_and_nonnull<mlir::stablehlo::CompareOp>(op)) {
      return false;
    }

    op = op->getNextNode();
    if (!isa_and_nonnull<mlir::stablehlo::MinOp>(op)) {
      return false;
    }

    op = op->getNextNode();
    if (!isa_and_nonnull<mlir::stablehlo::SelectOp>(op)) {
      return false;
    }

    op = op->getNextNode();
    if (!isa_and_nonnull<mlir::stablehlo::SelectOp>(op)) {
      return false;
    }

    op = op->getNextNode();
    if (!isa_and_nonnull<mlir::stablehlo::ReturnOp>(op)) {
      return false;
    }

    return true;
  }

  // Pattern match the ops for tt-xla ArgMax op.
  // There are two potential patterns, one is as follows:
  // stablehlo.compare (GT)
  // stablehlo.compare
  // stablehlo.or
  // stablehlo.compare
  // stablehlo.compare
  // stablehlo.and
  // stablehlo.or
  // stablehlo.select
  // stablehlo.select
  // stablehlo.return
  //
  // The other pattern is as follows:
  // stablehlo.compare (GT)
  // stablehlo.compare (EQ)
  // stablehlo.compare (LT)
  // stablehlo.and
  // stablehlo.or
  // stablehlo.select / stablehlo.maximum
  // stablehlo.select
  // stablehlo.return
  bool verifyJaxOpArgMaxPattern(mlir::Operation &operation) const {
    mlir::Operation *op = &operation;
    if (!isa<mlir::stablehlo::CompareOp>(op)) {
      return false;
    }
    mlir::stablehlo::CompareOp compareOp0 =
        mlir::cast<mlir::stablehlo::CompareOp>(op);
    if (compareOp0.getComparisonDirection() !=
        mlir::stablehlo::ComparisonDirection::GT) {
      return false;
    }

    op = op->getNextNode();
    if (!isa<mlir::stablehlo::CompareOp>(op)) {
      return false;
    }
    mlir::stablehlo::CompareOp compareOp1 =
        mlir::cast<mlir::stablehlo::CompareOp>(op);
    op = op->getNextNode();
    if (!op) {
      return false;
    }
    if (isa<mlir::stablehlo::OrOp>(op)) {
      op = op->getNextNode();
      if (!isa_and_nonnull<mlir::stablehlo::CompareOp>(op)) {
        return false;
      }

      op = op->getNextNode();
      if (!isa_and_nonnull<mlir::stablehlo::CompareOp>(op)) {
        return false;
      }

      op = op->getNextNode();
      if (!isa_and_nonnull<mlir::stablehlo::AndOp>(op)) {
        return false;
      }

      op = op->getNextNode();
      if (!isa_and_nonnull<mlir::stablehlo::OrOp>(op)) {
        return false;
      }

      op = op->getNextNode();
      if (!isa_and_nonnull<mlir::stablehlo::SelectOp>(op)) {
        return false;
      }

      op = op->getNextNode();
      if (!isa_and_nonnull<mlir::stablehlo::SelectOp>(op)) {
        return false;
      }

      op = op->getNextNode();
      if (!isa_and_nonnull<mlir::stablehlo::ReturnOp>(op)) {
        return false;
      }

      return true;
    }
    if (isa<mlir::stablehlo::CompareOp>(op)) {
      if (compareOp1.getComparisonDirection() !=
          mlir::stablehlo::ComparisonDirection::EQ) {
        return false;
      }
      mlir::stablehlo::CompareOp compareOp2 =
          mlir::cast<mlir::stablehlo::CompareOp>(op);
      if (compareOp2.getComparisonDirection() !=
          mlir::stablehlo::ComparisonDirection::LT) {
        return false;
      }

      op = op->getNextNode();
      if (!isa_and_nonnull<mlir::stablehlo::AndOp>(op)) {
        return false;
      }

      op = op->getNextNode();
      if (!isa_and_nonnull<mlir::stablehlo::OrOp>(op)) {
        return false;
      }

      op = op->getNextNode();
      if (!isa_and_nonnull<mlir::stablehlo::SelectOp>(op) &&
          !isa_and_nonnull<mlir::stablehlo::MaxOp>(op)) {
        return false;
      }

      op = op->getNextNode();
      if (!isa_and_nonnull<mlir::stablehlo::SelectOp>(op)) {
        return false;
      }

      op = op->getNextNode();
      if (!isa_and_nonnull<mlir::stablehlo::ReturnOp>(op)) {
        return false;
      }

      return true;
    }

    return false;
  }

  // Verify that the init value is defined by a constant op and initialize with
  // desired value.
  bool verifyInitValue(mlir::Value val,
                       TypicalInitReductionValue desired) const {
    Operation *initValue = val.getDefiningOp();
    while (initValue->getOpOperands().size() == 1) {
      initValue = initValue->getOpOperand(0).get().getDefiningOp();
    }
    if (!isa<mlir::stablehlo::ConstantOp>(initValue)) {
      return false;
    }

    mlir::stablehlo::ConstantOp initValueOp =
        mlir::cast<mlir::stablehlo::ConstantOp>(initValue);

    if (!checkInitValue(initValueOp, desired)) {
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
    rewriter.replaceOpWithNewOp<mlir::tt::ttir::PermuteOp>(
        srcOp, outputType, adaptor.getOperand(), adaptor.getPermutation());

    return success();
  }
};
} // namespace

namespace {

// Used by both BatchNormInferenceOp and BatchNormTrainingOp.
template <typename OpType, typename OpAdaptor>
static LogicalResult
checkBatchNormConversionLegality(OpType &srcOp, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) {
  auto inputType = mlir::cast<RankedTensorType>(adaptor.getOperand().getType());
  int64_t rank = inputType.getRank();
  uint64_t featureIndex = srcOp.getFeatureIndex();

  // BatchNorm requires at least 2 dimensions (batch and feature)
  if (rank < 2) {
    return rewriter.notifyMatchFailure(
        srcOp,
        srcOp.getOperationName() + " input must have at least 2 dimensions.");
  }

  // BatchNorm supports up to 5 dimensions
  if (rank > 5) {
    return rewriter.notifyMatchFailure(
        srcOp,
        srcOp.getOperationName() + " input must have at most 5 dimensions.");
  }

  // Feature index must be valid
  if (featureIndex >= static_cast<uint64_t>(rank)) {
    return rewriter.notifyMatchFailure(
        srcOp, srcOp.getOperationName() + " feature_index is out of bounds.");
  }

  return success();
}

class StableHLOToBatchNormInferenceOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::BatchNormInferenceOp> {

  using OpConversionPattern<
      mlir::stablehlo::BatchNormInferenceOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::BatchNormInferenceOp srcOp,
                  mlir::stablehlo::BatchNormInferenceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check legality of the conversion.
    LogicalResult legalityResult =
        checkBatchNormConversionLegality(srcOp, adaptor, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    // Convert feature_index to dimension attribute
    mlir::Type integerType = mlir::IntegerType::get(getContext(), 32);
    IntegerAttr dimensionAttr =
        mlir::IntegerAttr::get(integerType, srcOp.getFeatureIndex());

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::BatchNormInferenceOp>(
        srcOp, outputType, adaptor.getOperand(), adaptor.getScale(),
        adaptor.getOffset(), adaptor.getMean(), adaptor.getVariance(),
        adaptor.getEpsilonAttr(), dimensionAttr);

    return success();
  }
};
} // namespace

namespace {
class StableHLOToBatchNormTrainingOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::BatchNormTrainingOp> {

  using OpConversionPattern<
      mlir::stablehlo::BatchNormTrainingOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::BatchNormTrainingOp srcOp,
                  mlir::stablehlo::BatchNormTrainingOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check legality of the conversion.
    LogicalResult legalityResult =
        checkBatchNormConversionLegality(srcOp, adaptor, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    auto loc = srcOp.getLoc();

    // Get output types
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult(0).getType()));
    auto meanType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult(1).getType()));
    auto varianceType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult(2).getType()));

    // Convert feature_index to dimension attribute
    mlir::Type integerType = mlir::IntegerType::get(getContext(), 32);
    IntegerAttr dimensionAttr =
        mlir::IntegerAttr::get(integerType, srcOp.getFeatureIndex());

    // Default momentum for batch norm training
    FloatAttr momentumAttr = rewriter.getF32FloatAttr(1.0f);

    auto runningMean = rewriter.create<ttir::ZerosOp>(
        loc, meanType, llvm::to_vector_of<int32_t>(meanType.getShape()));
    auto runningVariance = rewriter.create<ttir::OnesOp>(
        loc, varianceType,
        llvm::to_vector_of<int32_t>(varianceType.getShape()));

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::BatchNormTrainingOp>(
        srcOp, TypeRange{outputType, meanType, varianceType},
        adaptor.getOperand(), adaptor.getScale(), adaptor.getOffset(),
        runningMean, runningVariance, adaptor.getEpsilonAttr(), dimensionAttr,
        momentumAttr);

    return success();
  }
};
} // namespace

namespace {

class StableHLOToBatchNormGradOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::BatchNormGradOp> {

  using OpConversionPattern<
      mlir::stablehlo::BatchNormGradOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::BatchNormGradOp srcOp,
                  mlir::stablehlo::BatchNormGradOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check legality of the general batch norm conversion.
    LogicalResult legalityResult =
        checkBatchNormConversionLegality(srcOp, adaptor, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    // Check that parameters have the expected shape.
    legalityResult =
        checkBatchNormGradParametersLegality(srcOp, adaptor, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    auto loc = srcOp.getLoc();

    // Get the input types and shapes.
    auto operandType =
        mlir::cast<RankedTensorType>(adaptor.getOperand().getType());
    auto scaleType = mlir::cast<RankedTensorType>(adaptor.getScale().getType());
    auto gradOutputType =
        mlir::cast<RankedTensorType>(adaptor.getGradOutput().getType());

    // Get output types.
    auto gradOperandType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getGradOperand().getType()));
    auto gradScaleType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getGradScale().getType()));
    auto gradOffsetType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getGradOffset().getType()));

    // Get feature index and epsilon.
    int64_t featureIndex = srcOp.getFeatureIndex();
    FloatAttr epsilonAttr = srcOp.getEpsilonAttr();
    float epsilon = epsilonAttr.getValueAsDouble();

    // Create reduction dimensions (all dims except feature_index).
    int64_t rank = operandType.getRank();
    auto reductionDims = llvm::to_vector(llvm::seq<int32_t>(rank));
    reductionDims.erase(reductionDims.begin() + featureIndex);
    ArrayAttr reductionDimsAttr = rewriter.getI32ArrayAttr(reductionDims);

    // Broadcast inputs to operand shape.
    Value scaleBcast = broadcastFeatureToShape(
        rewriter, loc, adaptor.getScale(), operandType, featureIndex);
    Value meanBcast = broadcastFeatureToShape(rewriter, loc, adaptor.getMean(),
                                              operandType, featureIndex);
    Value varianceBcast = broadcastFeatureToShape(
        rewriter, loc, adaptor.getVariance(), operandType, featureIndex);

    // Create epsilon broadcast.
    auto scalarType = RankedTensorType::get({}, operandType.getElementType());
    auto epsilonDenseAttr = DenseElementsAttr::get(
        scalarType,
        rewriter.getFloatAttr(operandType.getElementType(), epsilon));
    auto epsilonConstant =
        rewriter.create<ttir::ConstantOp>(loc, scalarType, epsilonDenseAttr);
    Value epsilonBcast = broadcastFeatureToShape(rewriter, loc, epsilonConstant,
                                                 operandType, featureIndex);

    // centered_operand = operand - mean
    auto centeredOperand = rewriter.create<ttir::SubtractOp>(
        loc, operandType, adaptor.getOperand(), meanBcast);

    // stddev = sqrt(variance + epsilon)
    auto variancePlusEpsilon = rewriter.create<ttir::AddOp>(
        loc, operandType, varianceBcast, epsilonBcast);

    auto stddev =
        rewriter.create<ttir::SqrtOp>(loc, operandType, variancePlusEpsilon);

    // normalized_operand = centered_operand / stddev
    auto normalizedOperand =
        rewriter.create<ttir::DivOp>(loc, operandType, centeredOperand, stddev);

    // elements_per_feature = total_elements / feature_dim_size
    int64_t totalElements = operandType.getNumElements();
    float elementsPerFeature =
        static_cast<float>(totalElements) /
        static_cast<float>(operandType.getShape()[featureIndex]);
    auto elementsPerFeatureAttr = DenseElementsAttr::get(
        scalarType, rewriter.getFloatAttr(operandType.getElementType(),
                                          elementsPerFeature));
    auto elementsPerFeatureConst = rewriter.create<ttir::ConstantOp>(
        loc, scalarType, elementsPerFeatureAttr);
    auto elementsPerFeatureBcast = broadcastFeatureToShape(
        rewriter, loc, elementsPerFeatureConst, operandType, featureIndex);

    // i1 = grad_output * elements_per_feature
    auto i1 = rewriter.create<ttir::MultiplyOp>(
        loc, gradOutputType, adaptor.getGradOutput(), elementsPerFeatureBcast);

    // i2 = broadcast(sum(grad_output, reduction_dims))
    auto sumGradOutput = rewriter.create<ttir::SumOp>(
        loc, scaleType, adaptor.getGradOutput(), rewriter.getBoolAttr(false),
        reductionDimsAttr);
    auto i2 = broadcastFeatureToShape(rewriter, loc, sumGradOutput, operandType,
                                      featureIndex);

    // grad_output * centered_operand
    auto gradTimesCentered = rewriter.create<ttir::MultiplyOp>(
        loc, operandType, adaptor.getGradOutput(), centeredOperand);

    // i3 = broadcast(sum(grad_output * centered_operand))
    auto sumGradTimesCentered = rewriter.create<ttir::SumOp>(
        loc, scaleType, gradTimesCentered, rewriter.getBoolAttr(false),
        reductionDimsAttr);
    auto i3 = broadcastFeatureToShape(rewriter, loc, sumGradTimesCentered,
                                      operandType, featureIndex);

    // i4 = i3 * centered_operand
    auto i4 = rewriter.create<ttir::MultiplyOp>(loc, operandType, i3,
                                                centeredOperand);

    // i5 = i4 / (variance + epsilon)
    auto i5 =
        rewriter.create<ttir::DivOp>(loc, operandType, i4, variancePlusEpsilon);

    // i6 = i1 - i2 - i5
    auto i1MinusI2 =
        rewriter.create<ttir::SubtractOp>(loc, operandType, i1, i2);

    auto i6 =
        rewriter.create<ttir::SubtractOp>(loc, operandType, i1MinusI2, i5);

    // grad_operand = (scale / stddev / elements_per_feature) * i6
    auto scaleOverStddev =
        rewriter.create<ttir::DivOp>(loc, operandType, scaleBcast, stddev);

    auto scaleOverStddevOverElem = rewriter.create<ttir::DivOp>(
        loc, operandType, scaleOverStddev, elementsPerFeatureBcast);

    auto gradOperand = rewriter.create<ttir::MultiplyOp>(
        loc, gradOperandType, scaleOverStddevOverElem, i6);

    // grad_scale = sum(grad_output * normalized_operand)
    auto gradTimesNorm = rewriter.create<ttir::MultiplyOp>(
        loc, operandType, adaptor.getGradOutput(), normalizedOperand);

    auto gradScale = rewriter.create<ttir::SumOp>(
        loc, gradScaleType, gradTimesNorm, rewriter.getBoolAttr(false),
        reductionDimsAttr);

    // grad_offset = sum(grad_output)
    auto gradOffset = rewriter.create<ttir::SumOp>(
        loc, gradOffsetType, adaptor.getGradOutput(),
        rewriter.getBoolAttr(false), reductionDimsAttr);

    // Replace the operation with the three results.
    rewriter.replaceOp(srcOp, {gradOperand, gradScale, gradOffset});

    return success();
  }

private:
  // Verify that scale, mean, variance parameters are 1D tensors with correct
  // size.
  LogicalResult checkBatchNormGradParametersLegality(
      mlir::stablehlo::BatchNormGradOp srcOp,
      mlir::stablehlo::BatchNormGradOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const {
    auto operandType =
        mlir::cast<RankedTensorType>(adaptor.getOperand().getType());
    auto scaleType = mlir::cast<RankedTensorType>(adaptor.getScale().getType());
    auto meanType = mlir::cast<RankedTensorType>(adaptor.getMean().getType());
    auto varianceType =
        mlir::cast<RankedTensorType>(adaptor.getVariance().getType());

    int64_t featureIndex = srcOp.getFeatureIndex();
    int64_t expectedSize = operandType.getShape()[featureIndex];

    // Check that parameters are 1D tensors.
    if (scaleType.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "batch_norm_grad scale must be a 1D tensor");
    }
    if (meanType.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "batch_norm_grad mean must be a 1D tensor");
    }
    if (varianceType.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "batch_norm_grad variance must be a 1D tensor");
    }

    // Check that sizes match feature dimension.
    if (scaleType.getDimSize(0) != expectedSize) {
      return rewriter.notifyMatchFailure(
          srcOp, "batch_norm_grad scale size must match feature dimension");
    }
    if (meanType.getDimSize(0) != expectedSize) {
      return rewriter.notifyMatchFailure(
          srcOp, "batch_norm_grad mean size must match feature dimension");
    }
    if (varianceType.getDimSize(0) != expectedSize) {
      return rewriter.notifyMatchFailure(
          srcOp, "batch_norm_grad variance size must match feature dimension");
    }

    return success();
  }

  // Helper to create broadcast shape from feature dimension to full shape.
  Value broadcastFeatureToShape(ConversionPatternRewriter &rewriter,
                                Location loc, Value input,
                                RankedTensorType targetType,
                                int64_t featureIndex) const {
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    const int64_t rank = targetType.getRank();

    // First reshape to match target rank if needed.
    if (inputType.getRank() < rank) {
      input =
          reshapeFeatureToShape(rewriter, loc, input, targetType, featureIndex);
    }

    // Broadcast the input to the target type.
    Value output;
    LogicalResult result =
        ttir::utils::broadcastValue(rewriter, input, targetType, output, loc,
                                    /*frontUnsqueeze=*/false);
    assert(result.succeeded() &&
           "Broadcast should succeed after legality checks");
    return output;
  }

  Value reshapeFeatureToShape(ConversionPatternRewriter &rewriter, Location loc,
                              Value input, RankedTensorType targetType,
                              int64_t featureIndex) const {
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    const int64_t rank = targetType.getRank();

    SmallVector<int64_t> unsqueezeShape(rank, 1);

    if (inputType.getRank() > 0) {
      unsqueezeShape[featureIndex] = inputType.getDimSize(0);
    }

    auto unsqueezeType =
        RankedTensorType::get(unsqueezeShape, targetType.getElementType());

    return rewriter.create<ttir::ReshapeOp>(
        loc, unsqueezeType, input,
        rewriter.getI32ArrayAttr(llvm::to_vector_of<int32_t>(unsqueezeShape)));
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

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::ReshapeOp>(
        srcOp, outputType, adaptor.getOperand(), newShapeAttr);

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
    RankedTensorType outputType = RankedTensorType::get({}, intType);
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
    // Check legality of the conversion.
    if (failed(checkConversionLegality(srcOp, rewriter))) {
      return failure();
    }

    auto convertedType =
        getTypeConverter()->convertType(srcOp.getResult().getType());
    auto outputType = cast<mlir::RankedTensorType>(convertedType);

    // In case value attr is scalar we need to convert it to 1D tensor.
    ElementsAttr newValueAttr = getValueAttr(srcOp.getValue());
    if (!newValueAttr) {
      return rewriter.notifyMatchFailure(
          srcOp, "Expected DenseElementsAttr or DenseResourceElementsAttr");
    }

    // Replace with ttir.constant.
    rewriter.replaceOpWithNewOp<mlir::tt::ttir::ConstantOp>(srcOp, outputType,
                                                            newValueAttr);

    return success();
  }

private:
  ElementsAttr getValueAttr(ElementsAttr valueAttr) const {
    // Shape is not empty, so we can return the value as is.
    if (!valueAttr.getShapedType().getShape().empty()) {
      return valueAttr;
    }

    auto valueShapedType = cast<mlir::ShapedType>(
        getTypeConverter()->convertType(valueAttr.getType()));
    if (auto denseAttr = dyn_cast<mlir::DenseElementsAttr>(valueAttr)) {
      llvm::SmallVector<mlir::Attribute> values(
          denseAttr.getValues<mlir::Attribute>());
      return mlir::DenseElementsAttr::get(valueShapedType, values);
    }

    if (auto resourceAttr =
            dyn_cast<mlir::DenseResourceElementsAttr>(valueAttr)) {
      // Rebuild with new type + same resource handle.
      return mlir::DenseResourceElementsAttr::get(valueShapedType,
                                                  resourceAttr.getRawHandle());
    }

    return nullptr;
  }

  LogicalResult
  checkConversionLegality(mlir::stablehlo::ConstantOp &srcOp,
                          ConversionPatternRewriter &rewriter) const {
    if (isa<DenseElementsAttr, DenseResourceElementsAttr>(srcOp.getValue()) &&
        !srcOp.getValue().getElementType().isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          srcOp, "ttir.constant only supports DenseElementsAttr or "
                 "DenseResourceElementsAttr with int or float types.");
    }

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRQuantizeOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::UniformQuantizeOp> {
  using OpConversionPattern<
      mlir::stablehlo::UniformQuantizeOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::UniformQuantizeOp srcOp,
                  mlir::stablehlo::UniformQuantizeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    LogicalResult legalityResult = checkConversionLegality(srcOp, rewriter);

    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    RankedTensorType inputType =
        mlir::cast<RankedTensorType>(adaptor.getOperand().getType());

    // Call the Quantize op if the input type is float.
    if (mlir::isa<FloatType>(inputType.getElementType())) {
      return matchAndRewriteInternal<mlir::tt::ttir::QuantizeOp>(srcOp, adaptor,
                                                                 rewriter);
    }

    // Call the Requantize op if the input type is quantized
    // per-tensor/per-axis.
    if (mlir::isa<mlir::quant::UniformQuantizedType,
                  mlir::quant::UniformQuantizedPerAxisType>(
            inputType.getElementType())) {
      return matchAndRewriteInternal<mlir::tt::ttir::RequantizeOp>(
          srcOp, adaptor, rewriter);
    }

    return failure();
  }

private:
  LogicalResult
  checkConversionLegality(mlir::stablehlo::UniformQuantizeOp &srcOp,
                          ConversionPatternRewriter &rewriter) const {
    // Expect a single input/output.
    if (srcOp->getNumOperands() != 1 || srcOp->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "Expected a single input/output.");
    }
    RankedTensorType outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    // Get the element type of the output tensor.
    if (!isa<mlir::quant::QuantizedType>(outputType.getElementType())) {
      return rewriter.notifyMatchFailure(
          srcOp, "Expected output element type to be quant.uniform");
    }

    return success();
  }

  template <typename DestOp>
  LogicalResult
  matchAndRewriteInternal(mlir::stablehlo::UniformQuantizeOp srcOp,
                          mlir::stablehlo::UniformQuantizeOp::Adaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    // Replace the StableHLO UniformQuantizeOp with the TTIR QuantizeOp or
    // RequantizeOp.
    RankedTensorType outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    rewriter.replaceOpWithNewOp<DestOp>(srcOp, outputType,
                                        adaptor.getOperand());

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRDequantizeOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::UniformDequantizeOp> {
  using OpConversionPattern<
      mlir::stablehlo::UniformDequantizeOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::UniformDequantizeOp srcOp,
                  mlir::stablehlo::UniformDequantizeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    LogicalResult legalityResult = checkConversionLegality(srcOp, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    RankedTensorType outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::DequantizeOp>(
        srcOp, outputType, adaptor.getOperand());
    return success();
  }

private:
  LogicalResult
  checkConversionLegality(mlir::stablehlo::UniformDequantizeOp &srcOp,
                          ConversionPatternRewriter &rewriter) const {
    // Expect a single input/output.
    if (srcOp->getNumOperands() != 1 || srcOp->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "Expected a single input/output.");
    }
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convolution passes
//===----------------------------------------------------------------------===//

template <uint32_t NDims>
using PaddingMatrix = std::array<std::array<int64_t, 2>, NDims>;

template <uint32_t NDims>
static PaddingMatrix<NDims> getPaddingMatrix(ArrayRef<int64_t> padding) {
  assert(padding.size() >= 2 * NDims &&
         "padding must be at least 2 * NDims sized array");

  PaddingMatrix<NDims> paddingMatrix;

  for (std::size_t i = 0; i < NDims; ++i) {
    paddingMatrix[i] = {padding[i * 2], padding[i * 2 + 1]};
  }
  return paddingMatrix;
}

namespace {
struct ConvolutionDecompositionPattern
    : public OpConversionPattern<mlir::stablehlo::ConvolutionOp> {
public:
  using OpConversionPattern<
      mlir::stablehlo::ConvolutionOp>::OpConversionPattern;

  enum ConvolutionDimension { BATCH = -1, FEATURE = -2, INVALID_DIM = -3 };
  enum ConvolutionKernelDimension {
    INPUT_FEATURES = -1,
    OUTPUT_FEATURES = -2,
    INVALID_KERNEL_DIM = -3
  };

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ConvolutionOp op,
                  mlir::stablehlo::ConvolutionOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override = 0;

protected:
  static bool isNDimensional(mlir::stablehlo::ConvolutionOp op,
                             uint32_t numSpatialDims) {
    return op.getDimensionNumbers().getInputSpatialDimensions().size() ==
           numSpatialDims;
  }

  static bool isSupportedConv(mlir::stablehlo::ConvolutionOp op) {

    if (op.getWindowReversal() &&
        llvm::any_of(*op.getWindowReversal(), ttmlir::utils::identity<bool>)) {
      return false;
    }

    return true;
  }

  // This function will generate the transpose indices needed to convert a
  // convolution input to a desired layout. The reason for the separate
  // function is to encapsulate the logic for constructing the inputLayout.
  static llvm::SmallVector<int64_t>
  generateConvPermutation(mlir::stablehlo::ConvolutionOp op,
                          llvm::ArrayRef<int64_t> ttnnConvolutionLayout) {

    llvm::SmallVector<int64_t> inputLayout(ttnnConvolutionLayout.size(),
                                           ConvolutionDimension::INVALID_DIM);
    inputLayout[op.getDimensionNumbers().getInputBatchDimension()] =
        ConvolutionDimension::BATCH;
    inputLayout[op.getDimensionNumbers().getInputFeatureDimension()] =
        ConvolutionDimension::FEATURE;

    for (const auto [spatialCount, spatialDim] : llvm::enumerate(
             op.getDimensionNumbers().getInputSpatialDimensions())) {
      inputLayout[spatialDim] = spatialCount;
    }

    return ttmlir::utils::generatePermutation(llvm::ArrayRef(inputLayout),
                                              ttnnConvolutionLayout);
  }

  // This function will generate the transpose indices needed to convert a
  // convolution input to a desired layout. The reason for the separate
  // function is to encapsulate the logic for constructing the kernelLayout.
  static llvm::SmallVector<int64_t> generateConvKernelPermutation(
      mlir::stablehlo::ConvolutionOp op,
      llvm::ArrayRef<int64_t> ttnnConvolutionKernelLayout) {

    llvm::SmallVector<int64_t> kernelLayout(
        ttnnConvolutionKernelLayout.size(),
        ConvolutionKernelDimension::INVALID_KERNEL_DIM);
    kernelLayout[op.getDimensionNumbers().getKernelOutputFeatureDimension()] =
        ConvolutionKernelDimension::OUTPUT_FEATURES;
    kernelLayout[op.getDimensionNumbers().getKernelInputFeatureDimension()] =
        ConvolutionKernelDimension::INPUT_FEATURES;

    for (const auto [spatialCount, spatialDim] : llvm::enumerate(
             op.getDimensionNumbers().getKernelSpatialDimensions())) {
      kernelLayout[spatialDim] = spatialCount;
    }

    return ttmlir::utils::generatePermutation(llvm::ArrayRef(kernelLayout),
                                              ttnnConvolutionKernelLayout);
  }
};
} // namespace

// Helper structure to hold sliced convolution inputs for batch group
// decomposition.
struct ConvolutionSlices {
  llvm::SmallVector<Value> inputs;
  llvm::SmallVector<Value> weights;
  llvm::SmallVector<llvm::SmallVector<int64_t>> outputShapes;
};

// Helper function to slice inputs and weights for batch group decomposition.
static ConvolutionSlices
sliceForBatchGroups(ConversionPatternRewriter &rewriter, Location loc,
                    Value input, Value weight, int64_t kernelOutputFeatureDim,
                    uint64_t groupCount, int64_t groupDimensionIndex) {
  ConvolutionSlices slices;

  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  auto weightType = mlir::cast<RankedTensorType>(weight.getType());

  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  llvm::ArrayRef<int64_t> weightShape = weightType.getShape();

  int64_t inputSliceSize = inputShape[groupDimensionIndex] / groupCount;
  int64_t weightSliceSize = weightShape[kernelOutputFeatureDim] / groupCount;

  for (uint64_t i = 0; i < groupCount; ++i) {
    // Slice input.
    llvm::SmallVector<int32_t> inputBegins(inputShape.size(), 0);
    llvm::SmallVector<int32_t> inputEnds(inputShape.begin(), inputShape.end());
    llvm::SmallVector<int32_t> inputSteps(inputShape.size(), 1);
    inputBegins[groupDimensionIndex] = i * inputSliceSize;
    inputEnds[groupDimensionIndex] = (i + 1) * inputSliceSize;

    llvm::SmallVector<int64_t> inputSliceShape(inputShape.begin(),
                                               inputShape.end());
    inputSliceShape[groupDimensionIndex] = inputSliceSize;

    auto inputSlice = rewriter.create<ttir::SliceStaticOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_inputSlice"),
        RankedTensorType::get(inputSliceShape, inputType.getElementType(),
                              inputType.getEncoding()),
        input, rewriter.getI32ArrayAttr(inputBegins),
        rewriter.getI32ArrayAttr(inputEnds),
        rewriter.getI32ArrayAttr(inputSteps));
    slices.inputs.push_back(inputSlice);

    // Slice weight.
    llvm::SmallVector<int32_t> weightBegins(weightShape.size(), 0);
    llvm::SmallVector<int32_t> weightEnds(weightShape.begin(),
                                          weightShape.end());
    llvm::SmallVector<int32_t> weightSteps(weightShape.size(), 1);
    weightBegins[kernelOutputFeatureDim] = i * weightSliceSize;
    weightEnds[kernelOutputFeatureDim] = (i + 1) * weightSliceSize;

    llvm::SmallVector<int64_t> weightSliceShape(weightShape.begin(),
                                                weightShape.end());
    weightSliceShape[kernelOutputFeatureDim] = weightSliceSize;

    auto weightSlice = rewriter.create<ttir::SliceStaticOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_weightSlice"),
        RankedTensorType::get(weightSliceShape, weightType.getElementType(),
                              weightType.getEncoding()),
        weight, rewriter.getI32ArrayAttr(weightBegins),
        rewriter.getI32ArrayAttr(weightEnds),
        rewriter.getI32ArrayAttr(weightSteps));
    slices.weights.push_back(weightSlice);
  }

  return slices;
}

// A decomposition pattern that matches to a stablehlo.convolution op that does
// 1D convolution. Since that is not supported in ttnn, we reshape the inputs
// and the output to match a 2D stablehlo.convolution op. The expectation is
// that the new stablehlo.convolution op will be picked up by the
// ConvolutionToConv2dPattern and translated into ttir.conv2d op.
namespace {
struct Legalize1DConvolutionPattern : public ConvolutionDecompositionPattern {
public:
  using ConvolutionDecompositionPattern::ConvolutionDecompositionPattern;
  constexpr static uint32_t NUM_SPATIAL_DIMS = 1;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ConvolutionOp op,
                  mlir::stablehlo::ConvolutionOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!(isSupportedConv(op) && isNDimensional(op, NUM_SPATIAL_DIMS))) {
      return failure();
    }

    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));

    llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

    uint64_t batchGroupCount = adaptor.getBatchGroupCount();
    uint64_t featureGroupCount = adaptor.getFeatureGroupCount();

    // Prepare inputs/weights (slice if batchGroupCount > 1).
    llvm::SmallVector<Value> inputSlices;
    llvm::SmallVector<Value> weightSlices;
    llvm::SmallVector<llvm::SmallVector<int64_t>> outputSliceShapes;
    assert(featureGroupCount == 1 ||
           batchGroupCount == 1 &&
               "At least one of the group counts must be 1.");

    // Split (X, Y, Z) is defined as splitting X into Y groups along the Z
    // dimension. If batch_group_count > 1: lhses = split(lhs,
    // batch_group_count, input_batch_dimension). rhses = split(rhs,
    // batch_group_count, kernel_output_feature_dimension). results... =
    // convolution(lhses..., rhses..., ..., batch_group_count=1, ...). result =
    // concatenate(results, output_feature_dimension).
    if (batchGroupCount > 1) {
      auto slices = sliceForBatchGroups(
          rewriter, op.getLoc(), adaptor.getLhs(), adaptor.getRhs(),
          adaptor.getDimensionNumbers().getKernelOutputFeatureDimension(),
          batchGroupCount,
          adaptor.getDimensionNumbers().getInputBatchDimension());
      inputSlices = std::move(slices.inputs);
      weightSlices = std::move(slices.weights);

      int64_t outputFeatureDim =
          adaptor.getDimensionNumbers().getOutputFeatureDimension();
      int64_t outputSliceSize = outputShape[outputFeatureDim] / batchGroupCount;
      for (uint64_t i = 0; i < batchGroupCount; ++i) {
        llvm::SmallVector<int64_t> outputSliceShape(outputShape.begin(),
                                                    outputShape.end());
        outputSliceShape[outputFeatureDim] = outputSliceSize;
        outputSliceShapes.push_back(outputSliceShape);
      }
    } else {
      inputSlices.push_back(adaptor.getLhs());
      weightSlices.push_back(adaptor.getRhs());
      outputSliceShapes.push_back(
          llvm::SmallVector<int64_t>(outputShape.begin(), outputShape.end()));
    }

    llvm::SmallVector<Value> results;
    for (size_t i = 0; i < inputSlices.size(); ++i) {
      Value result = convert1DConvTo2D(rewriter, op.getLoc(), inputSlices[i],
                                       weightSlices[i], outputSliceShapes[i],
                                       outputType.getElementType(),
                                       outputType.getEncoding(), adaptor);
      results.push_back(result);
    }

    if (batchGroupCount > 1) {
      int64_t outputFeatureDim =
          adaptor.getDimensionNumbers().getOutputFeatureDimension();
      auto concatOp = rewriter.create<ttir::ConcatOp>(
          op.getLoc(), outputType, results, outputFeatureDim);
      rewriter.replaceOp(op, concatOp);
    } else {
      rewriter.replaceOp(op, results[0]);
    }

    return success();
  }

private:
  // Convert a 1D convolution to a 2D convolution by adding a dimension.
  Value convert1DConvTo2D(ConversionPatternRewriter &rewriter, Location loc,
                          Value input, Value weight,
                          llvm::ArrayRef<int64_t> expectedOutputShape,
                          Type outputElementType, Attribute outputEncoding,
                          OpAdaptor adaptor) const {
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto weightType = mlir::cast<RankedTensorType>(weight.getType());

    llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
    llvm::ArrayRef<int64_t> weightShape = weightType.getShape();

    // Add dimension to shapes for 2D conversion.
    llvm::SmallVector<int64_t> conv2dInputShape(inputShape.begin(),
                                                inputShape.end());
    conv2dInputShape.push_back(1);
    llvm::SmallVector<int64_t> conv2dWeightShape(weightShape.begin(),
                                                 weightShape.end());
    conv2dWeightShape.push_back(1);
    llvm::SmallVector<int64_t> conv2dOutputShape(expectedOutputShape.begin(),
                                                 expectedOutputShape.end());
    conv2dOutputShape.push_back(1);

    // Reshape input and weight to 2D.
    ttir::ReshapeOp reshapeInput = createReshapeOp(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_reshapeInput"),
        input, conv2dInputShape);
    ttir::ReshapeOp reshapeWeight = createReshapeOp(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_reshapeWeight"),
        weight, conv2dWeightShape);

    // Get 1D attrs with defaults, then extend to 2D by adding default for new
    // dim.
    auto windowStrides1D =
        getI64ArrayOrDefault(adaptor.getWindowStridesAttr(), 1, 1);
    auto lhsDilation1D =
        getI64ArrayOrDefault(adaptor.getLhsDilationAttr(), 1, 1);
    auto rhsDilation1D =
        getI64ArrayOrDefault(adaptor.getRhsDilationAttr(), 1, 1);

    windowStrides1D.push_back(1);
    lhsDilation1D.push_back(1);
    rhsDilation1D.push_back(1);

    mlir::DenseI64ArrayAttr conv2dOpWindowsStridesAttr =
        rewriter.getDenseI64ArrayAttr(windowStrides1D);
    mlir::DenseI64ArrayAttr conv2dOpInputDilationAttr =
        rewriter.getDenseI64ArrayAttr(lhsDilation1D);
    mlir::DenseI64ArrayAttr conv2dOpWeightDilationAttr =
        rewriter.getDenseI64ArrayAttr(rhsDilation1D);

    // Window reversal: default to false for 1D, add false for new dimension.
    llvm::SmallVector<bool> windowReversal1D(1, false);
    if (auto windowReversalAttr = adaptor.getWindowReversalAttr()) {
      windowReversal1D =
          llvm::SmallVector<bool>(windowReversalAttr.asArrayRef());
    }
    windowReversal1D.push_back(false);
    mlir::DenseBoolArrayAttr conv2dOpWindowReversalAttr =
        rewriter.getDenseBoolArrayAttr(windowReversal1D);

    // Prepare padding: get 1D padding (or default [0,0]) and add [0,0] for 2D.
    auto conv2dOpPaddingVec = getPaddingOrDefault(adaptor.getPaddingAttr(), 1);
    conv2dOpPaddingVec.push_back(0); // Low padding for new dimension
    conv2dOpPaddingVec.push_back(0); // High padding for new dimension
    mlir::DenseIntElementsAttr conv2dOpPaddingAttr =
        mlir::DenseIntElementsAttr::get(
            mlir::RankedTensorType::get({2, 2}, rewriter.getI64Type()),
            conv2dOpPaddingVec);

    // The additional spatial dimension is added at the end (3rd in 0 indexed
    // array).
    llvm::SmallVector<int64_t> conv2dInputSpatialDimensions(
        adaptor.getDimensionNumbers().getInputSpatialDimensions());
    conv2dInputSpatialDimensions.push_back(3);

    llvm::SmallVector<int64_t> conv2dKernelSpatialDimensions(
        adaptor.getDimensionNumbers().getKernelSpatialDimensions());
    conv2dKernelSpatialDimensions.push_back(3);

    llvm::SmallVector<int64_t> conv2dOutputSpatialDimensions(
        adaptor.getDimensionNumbers().getOutputSpatialDimensions());
    conv2dOutputSpatialDimensions.push_back(3);

    // Create StableHLO ConvDimensionNumbersAttr.
    mlir::stablehlo::ConvDimensionNumbersAttr conv2dDimensionNumbers =
        mlir::stablehlo::ConvDimensionNumbersAttr::get(
            rewriter.getContext(),
            adaptor.getDimensionNumbers().getInputBatchDimension(),
            adaptor.getDimensionNumbers().getInputFeatureDimension(),
            conv2dInputSpatialDimensions,
            adaptor.getDimensionNumbers().getKernelInputFeatureDimension(),
            adaptor.getDimensionNumbers().getKernelOutputFeatureDimension(),
            conv2dKernelSpatialDimensions,
            adaptor.getDimensionNumbers().getOutputBatchDimension(),
            adaptor.getDimensionNumbers().getOutputFeatureDimension(),
            conv2dOutputSpatialDimensions);

    // Create the 2D convolution using StableHLO ConvolutionOp.
    // batch_group_count is always 1 here because slicing is handled by the
    // caller (matchAndRewrite) when batch_group_count > 1.
    auto new2dConvolutionOp = mlir::stablehlo::ConvolutionOp::create(
        rewriter, loc,
        RankedTensorType::get(conv2dOutputShape, outputElementType,
                              outputEncoding),
        reshapeInput.getResult(), reshapeWeight.getResult(),
        conv2dOpWindowsStridesAttr, conv2dOpPaddingAttr,
        conv2dOpInputDilationAttr, conv2dOpWeightDilationAttr,
        conv2dOpWindowReversalAttr, conv2dDimensionNumbers,
        adaptor.getFeatureGroupCount(), /*batchGroupCount=*/1,
        /*precisionConfig=*/nullptr);

    ttir::ReshapeOp reshapeOutput = createReshapeOp(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_reshapeOutput"),
        new2dConvolutionOp.getResult(), expectedOutputShape);

    return reshapeOutput;
  }

  ttir::ReshapeOp createReshapeOp(PatternRewriter &rewriter, Location loc,
                                  Value input,
                                  ::llvm::ArrayRef<int64_t> targetShape) const {
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shapeAttr =
        rewriter.getI32ArrayAttr(llvm::SmallVector<int32_t>(targetShape));

    return rewriter.create<ttir::ReshapeOp>(
        loc,
        RankedTensorType::get(targetShape, inputType.getElementType(),
                              inputType.getEncoding()),
        input, shapeAttr);
  }
};
} // namespace

namespace {
struct ConvolutionToConv2dPattern : public ConvolutionDecompositionPattern {
public:
  using ConvolutionDecompositionPattern::ConvolutionDecompositionPattern;

  constexpr static uint32_t NUM_SPATIAL_DIMS = 2;
  constexpr static uint32_t SPATIAL_DIM_HEIGHT = 0;
  constexpr static uint32_t SPATIAL_DIM_WIDTH = 1;

  // NHWC
  static inline const std::vector<int64_t> conv2dLayout = {
      ConvolutionDimension::BATCH,
      SPATIAL_DIM_HEIGHT,
      SPATIAL_DIM_WIDTH,
      ConvolutionDimension::FEATURE,
  };
  // OIHW
  static inline const std::vector<int64_t> conv2dKernelLayout = {
      ConvolutionKernelDimension::OUTPUT_FEATURES,
      ConvolutionKernelDimension::INPUT_FEATURES,
      SPATIAL_DIM_HEIGHT,
      SPATIAL_DIM_WIDTH,
  };
  // IOHW; for conv_transpose2d
  static inline const std::vector<int64_t> conv2dTransposeKernelLayout = {
      ConvolutionKernelDimension::INPUT_FEATURES,
      ConvolutionKernelDimension::OUTPUT_FEATURES,
      SPATIAL_DIM_HEIGHT,
      SPATIAL_DIM_WIDTH,
  };

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ConvolutionOp op,
                  mlir::stablehlo::ConvolutionOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!(isSupportedConv(op) && isNDimensional(op, NUM_SPATIAL_DIMS))) {
      return failure();
    }

    uint64_t batchGroupCount = adaptor.getBatchGroupCount();
    uint64_t featureGroupCount = adaptor.getFeatureGroupCount();

    assert(batchGroupCount == 1 ||
           featureGroupCount == 1 &&
               "At least one of the group counts must be 1.");
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));

    // Prepare inputs/weights (slice if batchGroupCount > 1).
    llvm::SmallVector<Value> inputSlices;
    llvm::SmallVector<Value> weightSlices;
    llvm::SmallVector<llvm::SmallVector<int64_t>> outputSliceShapes;

    if (batchGroupCount > 1) {

      auto slices = sliceForBatchGroups(
          rewriter, op.getLoc(), adaptor.getLhs(), adaptor.getRhs(),
          adaptor.getDimensionNumbers().getKernelOutputFeatureDimension(),
          batchGroupCount,
          adaptor.getDimensionNumbers().getInputBatchDimension());
      inputSlices = std::move(slices.inputs);
      weightSlices = std::move(slices.weights);

      llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
      int64_t outputBatchDim =
          adaptor.getDimensionNumbers().getOutputBatchDimension();
      int64_t outputFeatureDim =
          adaptor.getDimensionNumbers().getOutputFeatureDimension();
      int64_t outputBatchSliceSize = outputShape[outputBatchDim];
      int64_t outputFeatureSliceSize =
          outputShape[outputFeatureDim] / batchGroupCount;
      for (uint64_t i = 0; i < batchGroupCount; ++i) {
        llvm::SmallVector<int64_t> outputSliceShape(outputShape.begin(),
                                                    outputShape.end());
        outputSliceShape[outputBatchDim] = outputBatchSliceSize;
        outputSliceShape[outputFeatureDim] = outputFeatureSliceSize;
        outputSliceShapes.push_back(outputSliceShape);
      }
    } else {
      inputSlices.push_back(adaptor.getLhs());
      weightSlices.push_back(adaptor.getRhs());
      llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
      outputSliceShapes.push_back(
          llvm::SmallVector<int64_t>(outputShape.begin(), outputShape.end()));
    }

    llvm::SmallVector<Value> results;
    for (size_t i = 0; i < inputSlices.size(); ++i) {
      Value result =
          createConv2dForSlice(rewriter, op, adaptor, inputSlices[i],
                               weightSlices[i], outputSliceShapes[i]);
      results.push_back(result);
    }

    Value finalResult;
    if (batchGroupCount > 1) {
      // Concat on feature dimension in the output layout.
      // Find which dimension is the feature dimension.
      int64_t featureDim =
          adaptor.getDimensionNumbers().getOutputFeatureDimension();
      finalResult = rewriter.create<ttir::ConcatOp>(op.getLoc(), outputType,
                                                    results, featureDim);
    } else {
      finalResult = results[0];
    }

    rewriter.replaceOp(op, finalResult);

    return success();
  }

private:
  // Create a Conv2d or ConvTranspose2d operation for a single slice.
  Value createConv2dForSlice(ConversionPatternRewriter &rewriter,
                             mlir::stablehlo::ConvolutionOp op,
                             OpAdaptor adaptor, Value input, Value weight,
                             llvm::ArrayRef<int64_t> outputShape) const {

    bool isTransposed = tt::stablehlo::utils::isTransposedConv(op);

    auto windowStrides = getI64ArrayOrDefault(adaptor.getWindowStridesAttr(),
                                              NUM_SPATIAL_DIMS, 1);
    auto rhsDilation =
        getI64ArrayOrDefault(adaptor.getRhsDilationAttr(), NUM_SPATIAL_DIMS, 1);
    auto lhsDilation =
        getI64ArrayOrDefault(adaptor.getLhsDilationAttr(), NUM_SPATIAL_DIMS, 1);
    auto padding =
        getPaddingOrDefault(adaptor.getPaddingAttr(), NUM_SPATIAL_DIMS);

    auto strideAttr = rewriter.getDenseI32ArrayAttr({
        static_cast<int32_t>(windowStrides[SPATIAL_DIM_HEIGHT]),
        static_cast<int32_t>(windowStrides[SPATIAL_DIM_WIDTH]),
    });
    auto dilationAttr = rewriter.getDenseI32ArrayAttr({
        static_cast<int32_t>(rhsDilation[SPATIAL_DIM_HEIGHT]),
        static_cast<int32_t>(rhsDilation[SPATIAL_DIM_WIDTH]),
    });

    // Padding is a list of 2-tuples, the order of the 2-tuples is in
    // most-significant spatial dimension first order For Conv2d the most
    // significant spatial dimension is the height, followed by the width.
    auto paddingMatrix = getPaddingMatrix<NUM_SPATIAL_DIMS>(padding);
    auto paddingAttr = rewriter.getDenseI32ArrayAttr({
        static_cast<int32_t>(paddingMatrix[SPATIAL_DIM_HEIGHT][0]),
        static_cast<int32_t>(paddingMatrix[SPATIAL_DIM_WIDTH][0]),
        static_cast<int32_t>(paddingMatrix[SPATIAL_DIM_HEIGHT][1]),
        static_cast<int32_t>(paddingMatrix[SPATIAL_DIM_WIDTH][1]),
    });

    uint32_t groups = adaptor.getFeatureGroupCount();

    RankedTensorType inputType = mlir::cast<RankedTensorType>(input.getType());

    // Extract input dimension indices from StableHLO dimension numbers.
    int64_t inputBatchDim =
        adaptor.getDimensionNumbers().getInputBatchDimension();
    int64_t inputChannelDim =
        adaptor.getDimensionNumbers().getInputFeatureDimension();
    auto inputSpatialDims =
        adaptor.getDimensionNumbers().getInputSpatialDimensions();
    int64_t inputHeightDim = inputSpatialDims[SPATIAL_DIM_HEIGHT];
    int64_t inputWidthDim = inputSpatialDims[SPATIAL_DIM_WIDTH];

    // Extract output dimension indices from StableHLO dimension numbers.
    int64_t outputBatchDim =
        adaptor.getDimensionNumbers().getOutputBatchDimension();
    int64_t outputChannelDim =
        adaptor.getDimensionNumbers().getOutputFeatureDimension();
    auto outputSpatialDims =
        adaptor.getDimensionNumbers().getOutputSpatialDimensions();
    int64_t outputHeightDim = outputSpatialDims[SPATIAL_DIM_HEIGHT];
    int64_t outputWidthDim = outputSpatialDims[SPATIAL_DIM_WIDTH];

    // Build input layout array: inputLayout[i] tells us what semantic
    // dimension is at position i.
    llvm::SmallVector<int64_t> inputLayout(4);
    inputLayout[inputBatchDim] = ConvolutionDimension::BATCH;
    inputLayout[inputChannelDim] = ConvolutionDimension::FEATURE;
    inputLayout[inputHeightDim] = SPATIAL_DIM_HEIGHT;
    inputLayout[inputWidthDim] = SPATIAL_DIM_WIDTH;

    // Build output layout array.
    llvm::SmallVector<int64_t> outputLayout(4);
    outputLayout[outputBatchDim] = ConvolutionDimension::BATCH;
    outputLayout[outputChannelDim] = ConvolutionDimension::FEATURE;
    outputLayout[outputHeightDim] = SPATIAL_DIM_HEIGHT;
    outputLayout[outputWidthDim] = SPATIAL_DIM_WIDTH;

    // If input and output have different layouts, permute input to match
    // output.
    Value convInput = input;
    if (inputLayout != outputLayout) {
      // Generate permutation from input layout to output layout.
      auto permutation = ttmlir::utils::generatePermutation(
          llvm::ArrayRef(inputLayout), llvm::ArrayRef(outputLayout));
      auto permutedShape =
          ttmlir::utils::applyPermutation(inputType.getShape(), permutation);
      convInput = rewriter.create<ttir::PermuteOp>(
          ttmlir::utils::appendLocationSuffix(op.getLoc(), "_input"),
          RankedTensorType::get(permutedShape, inputType.getElementType(),
                                inputType.getEncoding()),
          input, permutation);
    }

    // Use output layout dimensions for the conv op since input is now permuted
    // to match.
    int64_t batchDim = outputBatchDim;
    int64_t heightDim = outputHeightDim;
    int64_t widthDim = outputWidthDim;
    int64_t channelDim = outputChannelDim;

    RankedTensorType outputType = inputType.clone(outputShape);

    Value permutedWeight = weight;
    // TTNN api handles reversing weights internally for transposed convolution.
    // So stablehlo.reverse op is ignored and its operand is used as weight.
    if (auto reverseOp = permutedWeight.getDefiningOp<ttir::ReverseOp>();
        isTransposed && reverseOp) {
      permutedWeight = reverseOp.getInput();
    }
    auto weightType = mlir::cast<RankedTensorType>(permutedWeight.getType());
    auto kernelPermutation = generateConvKernelPermutation(
        op, isTransposed ? conv2dTransposeKernelLayout : conv2dKernelLayout);
    auto weightOutputShape = ::ttmlir::utils::applyPermutation(
        weightType.getShape(), kernelPermutation);
    permutedWeight = rewriter.create<ttir::PermuteOp>(
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_weight"),
        RankedTensorType::get(weightOutputShape, weightType.getElementType(),
                              weightType.getEncoding()),
        permutedWeight, kernelPermutation);

    mlir::Value newConv;
    if (isTransposed) {
      // [TODO](mmanzoor) Verify the implementation of transposed convolution
      // for tt-xla. https://github.com/tenstorrent/tt-mlir/issues/3293
      // stablehlo.convolution op doesn't have output_padding
      // attribute. So Torch-MLIR adds output_padding with padding attribute for
      // transposed convolution during lowering.
      // https://github.com/llvm/torch-mlir/blob/main/lib/Conversion/TorchToStablehlo/Linear.cpp
      auto outputPaddingAttr = rewriter.getDenseI32ArrayAttr(
          {static_cast<int32_t>(paddingMatrix[SPATIAL_DIM_HEIGHT][1] -
                                paddingMatrix[SPATIAL_DIM_HEIGHT][0]),
           static_cast<int32_t>(paddingMatrix[SPATIAL_DIM_WIDTH][1] -
                                paddingMatrix[SPATIAL_DIM_WIDTH][0])});
      // Recomputing padding attribute based on Torch-MLIR lowering of
      // conv_transposed2d op: [top, left, bottom, right].
      // https://github.com/llvm/torch-mlir/blob/main/lib/Conversion/TorchToStablehlo/Linear.cpp
      // Get the kernel spatial dimension indices from the dimension numbers.
      auto kernelSpatialDims =
          adaptor.getDimensionNumbers().getKernelSpatialDimensions();
      int64_t kernelHeightDim = kernelSpatialDims[SPATIAL_DIM_HEIGHT];
      int64_t kernelWidthDim = kernelSpatialDims[SPATIAL_DIM_WIDTH];
      paddingAttr = rewriter.getDenseI32ArrayAttr({
          static_cast<int32_t>((weightType.getShape()[kernelHeightDim] - 1) *
                                   rhsDilation[SPATIAL_DIM_HEIGHT] -
                               paddingMatrix[SPATIAL_DIM_HEIGHT][0]),
          static_cast<int32_t>((weightType.getShape()[kernelWidthDim] - 1) *
                                   rhsDilation[SPATIAL_DIM_WIDTH] -
                               paddingMatrix[SPATIAL_DIM_WIDTH][0]),
          static_cast<int32_t>((weightType.getShape()[kernelHeightDim] - 1) *
                                   rhsDilation[SPATIAL_DIM_HEIGHT] -
                               paddingMatrix[SPATIAL_DIM_HEIGHT][0]),
          static_cast<int32_t>((weightType.getShape()[kernelWidthDim] - 1) *
                                   rhsDilation[SPATIAL_DIM_WIDTH] -
                               paddingMatrix[SPATIAL_DIM_WIDTH][0]),
      });
      // Input dilation (lhs dilation) is used for stride for transposed
      // convolution.
      auto inputDilationAttr = rewriter.getDenseI32ArrayAttr({
          static_cast<int32_t>(lhsDilation[SPATIAL_DIM_HEIGHT]),
          static_cast<int32_t>(lhsDilation[SPATIAL_DIM_WIDTH]),
      });

      if (int64_t groups = adaptor.getFeatureGroupCount(); groups > 1) {
        // Stablehlo.convolution/ttir.convolution op weights are in the format:
        // (C/G, O, K_H, K_W). Torch and TTNN expect the weights to be in the
        // format (C, O/G, K_H, K_W). Therefore it is necessary to transform the
        // weights.
        // (C/G, O, K_H, K_W) -> (C/G, G, O/G, K_H, K_W) -> (G, C/G,O/G, K_H,
        // K_W) -> (C, O/G, K_H, K_W)
        auto permutedWeightType =
            mlir::cast<RankedTensorType>(permutedWeight.getType());
        auto permutedWeightShape = permutedWeightType.getShape();

        int64_t inChannelsPerGroup = permutedWeightShape[0];
        int64_t totalOutChannels = permutedWeightShape[1];
        int64_t kH = permutedWeightShape[2];
        int64_t kW = permutedWeightShape[3];
        int64_t outChannelsPerGroup = totalOutChannels / groups;
        int64_t totalInChannels = inChannelsPerGroup * groups;

        // Reshape (C/G, O, K_H, K_W) -> (C/G, G, O/G, K_H, K_W)
        llvm::SmallVector<int64_t> extractedGroupsShape = {
            inChannelsPerGroup, groups, outChannelsPerGroup, kH, kW};
        auto extractedGroups = ttir::utils::createReshapeOp(
            rewriter,
            ttmlir::utils::appendLocationSuffix(op.getLoc(),
                                                "_weight_extracted_groups"),
            permutedWeight, extractedGroupsShape);

        // Permute (C/G, G, O/G, K_H, K_W) -> (G, C/G, O/G, K_H, K_W)
        llvm::SmallVector<int64_t> permuteOrder = {1, 0, 2, 3, 4};
        llvm::SmallVector<int64_t> permutedGroupsShape =
            ttmlir::utils::applyPermutation(
                llvm::ArrayRef(extractedGroupsShape),
                llvm::ArrayRef(permuteOrder));
        auto permutedGroups = rewriter.create<ttir::PermuteOp>(
            ttmlir::utils::appendLocationSuffix(op.getLoc(),
                                                "_weight_permuted_groups"),
            permutedWeightType.cloneWith(permutedGroupsShape,
                                         permutedWeightType.getElementType()),
            extractedGroups, permuteOrder);

        // Reshape (G, C/G, O/G, K_H, K_W) -> (C, O/G, K_H, K_W)
        llvm::SmallVector<int64_t> mergedGroupsShape = {
            totalInChannels, outChannelsPerGroup, kH, kW};
        permutedWeight = ttir::utils::createReshapeOp(
            rewriter,
            ttmlir::utils::appendLocationSuffix(op.getLoc(),
                                                "_weight_merged_groups"),
            permutedGroups, mergedGroupsShape);
      }

      // Use full builder with explicit dimension attributes.
      newConv = rewriter.create<ttir::ConvTranspose2dOp>(
          op.getLoc(), outputType, convInput, Value(permutedWeight), Value(),
          inputDilationAttr, paddingAttr, outputPaddingAttr, dilationAttr,
          rewriter.getI32IntegerAttr(groups),
          rewriter.getI64IntegerAttr(batchDim),
          rewriter.getI64IntegerAttr(heightDim),
          rewriter.getI64IntegerAttr(widthDim),
          rewriter.getI64IntegerAttr(channelDim));

    } else {

      // Use full builder with explicit dimension attributes.
      newConv = rewriter.create<ttir::Conv2dOp>(
          op.getLoc(), outputType, convInput, Value(permutedWeight), Value(),
          strideAttr, paddingAttr, dilationAttr,
          rewriter.getI32IntegerAttr(groups),
          rewriter.getI64IntegerAttr(batchDim),
          rewriter.getI64IntegerAttr(heightDim),
          rewriter.getI64IntegerAttr(widthDim),
          rewriter.getI64IntegerAttr(channelDim),
          /*flattenedCompatInfo=*/nullptr);
    }

    return newConv;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Conv3d Pattern Matching
//===----------------------------------------------------------------------===//

namespace {
struct ConvolutionToConv3dPattern : public ConvolutionDecompositionPattern {
public:
  using ConvolutionDecompositionPattern::ConvolutionDecompositionPattern;

  constexpr static uint32_t NUM_SPATIAL_DIMS = 3;
  constexpr static uint32_t SPATIAL_DIM_DEPTH = 0;
  constexpr static uint32_t SPATIAL_DIM_HEIGHT = 1;
  constexpr static uint32_t SPATIAL_DIM_WIDTH = 2;

  // OIDHW
  static inline const std::vector<int64_t> conv3dKernelLayout = {
      ConvolutionKernelDimension::OUTPUT_FEATURES,
      ConvolutionKernelDimension::INPUT_FEATURES,
      SPATIAL_DIM_DEPTH,
      SPATIAL_DIM_HEIGHT,
      SPATIAL_DIM_WIDTH,
  };

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!(isSupportedConv(op) && isNDimensional(op, NUM_SPATIAL_DIMS))) {
      return failure();
    }

    uint64_t batchGroupCount = adaptor.getBatchGroupCount();

    // For now, only support simple case without batch groups
    if (batchGroupCount > 1) {
      return rewriter.notifyMatchFailure(
          op, "Conv3d does not support batch_group_count > 1 yet");
    }

    // Check that dilation is 1 for all dimensions since Conv3d doesn't support
    // dilation yet
    auto rhsDilation =
        getI64ArrayOrDefault(adaptor.getRhsDilationAttr(), NUM_SPATIAL_DIMS, 1);
    for (int64_t dilation : rhsDilation) {
      if (dilation != 1) {
        return rewriter.notifyMatchFailure(
            op, "Conv3d does not support dilation != 1");
      }
    }

    // Padding must be symmetric for all dimensions since Conv3d only
    // supports symmetric padding
    auto padding =
        getPaddingOrDefault(adaptor.getPaddingAttr(), NUM_SPATIAL_DIMS);
    auto paddingMatrix = getPaddingMatrix<NUM_SPATIAL_DIMS>(padding);
    for (uint32_t i = 0; i < NUM_SPATIAL_DIMS; i++) {
      if (paddingMatrix[i][0] != paddingMatrix[i][1]) {
        return rewriter.notifyMatchFailure(
            op, "Conv3d only supports symmetric padding");
      }
    }

    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getType()));

    Value result = createConv3d(rewriter, op, adaptor, adaptor.getLhs(),
                                adaptor.getRhs(), outputType);

    rewriter.replaceOp(op, result);

    return success();
  }

private:
  // Create a Conv3d operation with explicit dimension attributes.
  Value createConv3d(ConversionPatternRewriter &rewriter,
                     mlir::stablehlo::ConvolutionOp op, OpAdaptor adaptor,
                     Value input, Value weight,
                     RankedTensorType outputType) const {

    auto windowStrides = getI64ArrayOrDefault(adaptor.getWindowStridesAttr(),
                                              NUM_SPATIAL_DIMS, 1);
    auto padding =
        getPaddingOrDefault(adaptor.getPaddingAttr(), NUM_SPATIAL_DIMS);

    auto strideAttr = rewriter.getDenseI32ArrayAttr({
        static_cast<int32_t>(windowStrides[SPATIAL_DIM_DEPTH]),
        static_cast<int32_t>(windowStrides[SPATIAL_DIM_HEIGHT]),
        static_cast<int32_t>(windowStrides[SPATIAL_DIM_WIDTH]),
    });

    // Padding is a list of 2-tuples, the order of the 2-tuples is in
    // most-significant spatial dimension first order. For Conv3d the most
    // significant spatial dimension is the depth, followed by height, then
    // width.
    // Note: Conv3d only supports symmetric padding (validated in
    // matchAndRewrite), so we only use the "before" padding values.
    auto paddingMatrix = getPaddingMatrix<NUM_SPATIAL_DIMS>(padding);
    auto paddingAttr = rewriter.getDenseI32ArrayAttr({
        static_cast<int32_t>(paddingMatrix[SPATIAL_DIM_DEPTH][0]),
        static_cast<int32_t>(paddingMatrix[SPATIAL_DIM_HEIGHT][0]),
        static_cast<int32_t>(paddingMatrix[SPATIAL_DIM_WIDTH][0]),
    });

    auto groupsAttr =
        rewriter.getI32IntegerAttr(adaptor.getFeatureGroupCount());

    auto outputSpatialDims =
        adaptor.getDimensionNumbers().getOutputSpatialDimensions();
    int64_t batchDim = adaptor.getDimensionNumbers().getOutputBatchDimension();
    int64_t depthDim = outputSpatialDims[SPATIAL_DIM_DEPTH];
    int64_t heightDim = outputSpatialDims[SPATIAL_DIM_HEIGHT];
    int64_t widthDim = outputSpatialDims[SPATIAL_DIM_WIDTH];
    int64_t channelDim =
        adaptor.getDimensionNumbers().getOutputFeatureDimension();

    // Permute weight to OIDHW layout
    Value permutedWeight = weight;
    auto weightType = mlir::cast<RankedTensorType>(permutedWeight.getType());
    auto kernelPermutation =
        generateConvKernelPermutation(op, conv3dKernelLayout);
    auto weightOutputShape = ::ttmlir::utils::applyPermutation(
        weightType.getShape(), kernelPermutation);
    permutedWeight = rewriter.create<ttir::PermuteOp>(
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_weight"),
        RankedTensorType::get(weightOutputShape, weightType.getElementType(),
                              weightType.getEncoding()),
        permutedWeight, kernelPermutation);

    mlir::Value newConv = rewriter.create<ttir::Conv3dOp>(
        op.getLoc(), outputType, Value(input), Value(permutedWeight), Value(),
        strideAttr, paddingAttr, groupsAttr,
        rewriter.getI64IntegerAttr(batchDim),
        rewriter.getI64IntegerAttr(depthDim),
        rewriter.getI64IntegerAttr(heightDim),
        rewriter.getI64IntegerAttr(widthDim),
        rewriter.getI64IntegerAttr(channelDim),
        /*padding_mode=*/nullptr);

    return newConv;
  }
};
} // namespace
namespace {
class StableHLOToTTIRConvolutionOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ConvolutionOp> {
public:
  StableHLOToTTIRConvolutionOpConversionPattern(TypeConverter &typeConverter,
                                                MLIRContext *ctx,
                                                PatternBenefit benefit = 2)
      : OpConversionPattern<mlir::stablehlo::ConvolutionOp>(typeConverter, ctx,
                                                            benefit) {}

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

    // Negative padding handling strategy:
    // 1. Convolution ops don't support negative padding directly, so we split
    // the operation into two steps when negative padding is detected
    // 2. First, run convolution with padding clamped to zero (resulting in a
    // larger intermediate output than the final desired shape)
    // 3. Then, slice the intermediate output to achieve the effect of negative
    // padding (cropping edges based on the magnitude of negative padding
    // values)
    //
    // Example: padding=[-1, -1] means "crop 1 pixel from each edge of the
    // output"
    //   - Run conv with padding=[0, 0] to get larger intermediate result
    //   - Slice [1:-1, 1:-1] to remove the edges and get final output
    // This pattern only handles negative padding.
    ArrayRef<int64_t> paddingArray = paddingAttr.asArrayRef();
    bool hasNegativePadding =
        llvm::any_of(paddingArray, [](int64_t p) { return p < 0; });
    if (!hasNegativePadding) {
      return failure();
    }

    SmallVector<int64_t> adjustedPadding = llvm::to_vector(llvm::map_range(
        paddingArray, [](int64_t p) { return std::max<int64_t>(p, 0); }));

    paddingAttr = rewriter.getDenseI64ArrayAttr(adjustedPadding);

    // Calculate intermediate output shape for convolution when negative padding
    // exists.
    SmallVector<int64_t> intermediateOutputShape(outputType.getShape().begin(),
                                                 outputType.getShape().end());
    auto spatialDims = dimNums.getOutputSpatialDimensions();
    for (size_t i = 0; i < numSpatialDims; i++) {
      int64_t padLow = paddingArray[2 * i];
      int64_t padHigh = paddingArray[2 * i + 1];

      if (padLow < 0) {
        intermediateOutputShape[spatialDims[i]] -= padLow;
      }
      if (padHigh < 0) {
        intermediateOutputShape[spatialDims[i]] -= padHigh;
      }
    }

    RankedTensorType intermediateOutputType = RankedTensorType::get(
        intermediateOutputShape, outputType.getElementType());

    // Convert padding to DenseIntElementsAttr with shape [numSpatialDims, 2].
    mlir::DenseIntElementsAttr stablehloPaddingAttr =
        mlir::DenseIntElementsAttr::get(
            mlir::RankedTensorType::get(
                {static_cast<int64_t>(numSpatialDims), 2},
                rewriter.getI64Type()),
            adjustedPadding);

    // Create StableHLO ConvDimensionNumbersAttr.
    mlir::stablehlo::ConvDimensionNumbersAttr stablehloDimNums =
        mlir::stablehlo::ConvDimensionNumbersAttr::get(
            rewriter.getContext(), dimNums.getInputBatchDimension(),
            dimNums.getInputFeatureDimension(),
            dimNums.getInputSpatialDimensions(),
            dimNums.getKernelInputFeatureDimension(),
            dimNums.getKernelOutputFeatureDimension(),
            dimNums.getKernelSpatialDimensions(),
            dimNums.getOutputBatchDimension(),
            dimNums.getOutputFeatureDimension(),
            dimNums.getOutputSpatialDimensions());

    mlir::stablehlo::ConvolutionOp convOp =
        mlir::stablehlo::ConvolutionOp::create(
            rewriter, srcOp.getLoc(), intermediateOutputType, adaptor.getLhs(),
            adaptor.getRhs(), windowStridesAttr, stablehloPaddingAttr,
            inputDilationAttr, kernelDilationAttr, windowReversalAttr,
            stablehloDimNums, adaptor.getFeatureGroupCount(),
            adaptor.getBatchGroupCount(),
            /*precisionConfig=*/nullptr);

    auto convOutputType = cast<RankedTensorType>(convOp.getResult().getType());
    auto convOutputShape = convOutputType.getShape();

    SmallVector<int32_t> sliceBegins(convOutputShape.size(), 0);
    SmallVector<int32_t> sliceEnds(convOutputShape.begin(),
                                   convOutputShape.end());
    SmallVector<int32_t> sliceSteps(convOutputShape.size(), 1);

    // Adjust slice parameters for spatial dimensions with negative padding.
    for (size_t i = 0; i < numSpatialDims; i++) {
      int64_t padLow = paddingArray[2 * i];
      int64_t padHigh = paddingArray[2 * i + 1];

      if (padLow < 0 || padHigh < 0) {
        int64_t spatialDim = spatialDims[i];
        sliceBegins[spatialDim] =
            std::abs(std::min(static_cast<int64_t>(0), padLow));
        sliceEnds[spatialDim] =
            convOutputShape[spatialDim] -
            std::abs(std::min(static_cast<int64_t>(0), padHigh));
      }
    }

    // Create slice operation to crop the output.
    auto sliceOp = rewriter.create<ttir::SliceStaticOp>(
        srcOp.getLoc(), outputType, convOp.getResult(),
        rewriter.getI32ArrayAttr(sliceBegins),
        rewriter.getI32ArrayAttr(sliceEnds),
        rewriter.getI32ArrayAttr(sliceSteps));

    rewriter.replaceOp(srcOp, sliceOp.getResult());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// StableHLOToTTIRReduceWindowOpConversionPattern
// The lowering is specialized for a few well-structured cases and **does not**
// handle all valid StableHLO patterns. Current assumptions:
//  - The body block must contain only `stablehlo.{add,max}` ops followed by a
//    `stablehlo.return`. Other reductions (e.g., min, multiply) are
//    unsupported.
//  - The number of body reduction ops must match the number of inputs.
//  - The initial values (`init_values`) must be stablehlo.constant ops that are
//    either zero or negative infinity (NEG_INF). Function arguments or more
//    complex expressions are not currently supported.
//  - Mixed dtypes across inputs are supported, but reduction op must match
//    the input type.
//  - `CumSum` lowering only works for single-input/single-output cases and
//    must satisfy specific window/padding rules (see isCumSum()).
// This conversion is tailored toward cases like max_pool2d, avg_pool2d (via
// sum+div), and cumulative sum.
// TODO(anusingh):
//  - Support initialization via function arguments
//  - Generalize to other reduction ops
//  - Extract and match nested operations in reduction blocks
//===----------------------------------------------------------------------===//

namespace {

class StableHLOToTTIRReduceWindowOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ReduceWindowOp> {
  using OpConversionPattern<
      mlir::stablehlo::ReduceWindowOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReduceWindowOp srcOp,
                  mlir::stablehlo::ReduceWindowOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    using TypicalInitReductionValue::NEG_INF;
    using TypicalInitReductionValue::ZERO;

    // Validate basic op structure.
    if (!hasValidOpStructure(srcOp)) {
      return rewriter.notifyMatchFailure(
          srcOp, "Invalid structure of reduce window block.");
    }

    // Validate input shapes and ranks.
    RankedTensorType firstInputType =
        mlir::cast<RankedTensorType>(srcOp.getInputs()[0].getType());
    for (Value input : srcOp.getInputs()) {
      if (mlir::cast<RankedTensorType>(input.getType()).getShape() !=
          firstInputType.getShape()) {
        return rewriter.notifyMatchFailure(
            srcOp, "All inputs must have the same shape.");
      }
    }
    int64_t inputRank = firstInputType.getRank();
    if (inputRank > 4) {
      return rewriter.notifyMatchFailure(srcOp, "Invalid input tensor rank.");
    }

    // Validate init values.
    std::optional<llvm::SmallVector<TypicalInitReductionValue>> initValues =
        extractInitValues(srcOp);
    if (!initValues) {
      return rewriter.notifyMatchFailure(
          srcOp, "Failed to extract constant init values.");
    }
    if (initValues->size() != srcOp.getInputs().size()) {
      return rewriter.notifyMatchFailure(
          srcOp, "Mismatch between inputs and init values.");
    }

    // Validate block body.
    Block &block = srcOp.getBody().getBlocks().front();
    auto &operations = block.getOperations();
    SmallVector<mlir::Operation *> reductionOps;
    for (Operation &op : llvm::drop_end(operations, 1)) {
      if (!isa<mlir::stablehlo::AddOp, mlir::stablehlo::MaxOp>(&op)) {
        return rewriter.notifyMatchFailure(srcOp, "Unsupported reduction op.");
      }
      reductionOps.push_back(&op);
    }
    if (!isa<mlir::stablehlo::ReturnOp>(operations.back())) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "Invalid last op in the block.");
    }
    if (reductionOps.size() != srcOp.getInputs().size()) {
      return rewriter.notifyMatchFailure(
          srcOp, "Mismatch between inputs and body ops.");
    }

    // Validate op attributes or assign default values if not provided.

    DenseI64ArrayAttr windowDimensions = adaptor.getWindowDimensionsAttr();
    if (windowDimensions.size() != inputRank) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "Invalid pooling window dimensions.");
    }

    DenseI64ArrayAttr windowStrides = adaptor.getWindowStridesAttr();
    if (!windowStrides) {
      windowStrides = rewriter.getDenseI64ArrayAttr(
          SmallVector<int64_t>(windowDimensions.size(), 1));
    }
    if (windowStrides.size() != inputRank) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "Invalid pooling window strides.");
    }

    DenseI64ArrayAttr windowDilations = adaptor.getWindowDilationsAttr();
    if (!windowDilations) {
      windowDilations = rewriter.getDenseI64ArrayAttr(
          SmallVector<int64_t>(windowDimensions.size(), 1));
    }
    if (windowDilations.size() != inputRank) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "Invalid pooling window dilations.");
    }

    DenseI64ArrayAttr baseDilations = adaptor.getBaseDilationsAttr();
    if (!baseDilations) {
      baseDilations = rewriter.getDenseI64ArrayAttr(
          SmallVector<int64_t>(windowDimensions.size(), 1));
    }
    if (baseDilations.size() != inputRank) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "Invalid base dilations in pooling.");
    }

    DenseI64ArrayAttr padding =
        adaptor.getPaddingAttr()
            ? rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(
                  adaptor.getPaddingAttr().getValues<int64_t>()))
            : rewriter.getDenseI64ArrayAttr(
                  SmallVector<int64_t>(windowDimensions.size() * 2, 0));

    BoolAttr ceilMode = rewriter.getBoolAttr(false);

    BoolAttr countIncludesPad = rewriter.getBoolAttr(true);

    // Handle the special case of lowering to CumSumOp.
    if (srcOp.getInputs().size() == 1) {
      std::optional<int64_t> dimension =
          isCumSum(srcOp, adaptor, (*initValues)[0], reductionOps[0], padding);
      if (dimension) {
        mlir::RankedTensorType resultType = cast<RankedTensorType>(
            getTypeConverter()->convertType(srcOp.getResult(0).getType()));
        rewriter.replaceOpWithNewOp<ttir::CumSumOp>(
            srcOp, resultType, adaptor.getInputs()[0],
            rewriter.getI64IntegerAttr(*dimension));
        return success();
      }
    }

    // Not a special case of CumSumOp - lowering to TTIR pooling ops is
    // supported only for 2D and 4D input tensors.
    if (!(inputRank == 2 || inputRank == 4)) {
      return rewriter.notifyMatchFailure(srcOp, "Invalid input tensor rank.");
    }

    // Deduce whether a reshape (2D->4D) or a permute (4D->4D) operation is
    // needed to prepare input for TTIR pooling ops.
    bool needsReshape = false;
    bool needsPermute = false;
    SmallVector<int64_t> permutation;        // populated only if needsPermute
    SmallVector<int64_t> inversePermutation; // populated only if needsPermute
    SmallVector<size_t, 2> spatialDimIndices = {1, 2}; // default for NHWC
    if (inputRank == 4) {
      // 4D input that maybe needs to be permuted to NHWC format which is
      // expected by TTIR pooling ops. TTIR ops operate on spatial dimensions
      // H and W. To deduce whether a permute operation is needed, non-1 values
      // in windowDimensions attribute signalize a spatial dimension.
      SmallVector<size_t> indicesGreaterThanOne =
          indicesOfValuesGreaterThanOne(windowDimensions.asArrayRef());
      if (indicesGreaterThanOne.size() > 2) {
        return rewriter.notifyMatchFailure(
            srcOp, "Conversion is not supported when more than 2 spatials "
                   "dimensions are specified.");
      }
      if (indicesGreaterThanOne.size() < 2) {
        // The default behavior is to assume a channel-first tensor (NCHW),
        // if spatial dimensions cannot be determined from windowDimensions.
        spatialDimIndices = {2, 3};
      } else { // exactly two found
        spatialDimIndices = {indicesGreaterThanOne[0],
                             indicesGreaterThanOne[1]};
      }

      // PermuteOp is needed if spatial dimensions (HW) are not 1 and 2 (NHWC).
      if (spatialDimIndices[0] != 1 || spatialDimIndices[1] != 2) {
        needsPermute = true;

        // Build desired layout: spatial H at index 1, spatial W at index 2.
        const int64_t SPATIAL_H = -3; // -3 is a placeholder to indicate H dim
        const int64_t SPATIAL_W = -2; // -2 is a placeholder to indicate W dim

        std::vector<int64_t> desiredLayout(inputRank, -1);
        desiredLayout[1] = SPATIAL_H;
        desiredLayout[2] = SPATIAL_W;
        int64_t nonSpatialCount = 0;
        for (size_t i = 0; i < desiredLayout.size(); ++i) {
          if (desiredLayout[i] == -1) {
            desiredLayout[i] = nonSpatialCount++;
          }
        }

        std::vector<int64_t> currentLayout(inputRank, -1);
        currentLayout[spatialDimIndices[0]] = SPATIAL_H;
        currentLayout[spatialDimIndices[1]] = SPATIAL_W;
        nonSpatialCount = 0;
        for (size_t i = 0; i < currentLayout.size(); ++i) {
          if (currentLayout[i] == -1) {
            currentLayout[i] = nonSpatialCount++;
          }
        }

        permutation = ttmlir::utils::generatePermutation(
            llvm::ArrayRef(currentLayout), llvm::ArrayRef(desiredLayout));

        inversePermutation = ttmlir::utils::inversePermutation(permutation);
      }
    } else {
      // 2D input needs reshape to 4D.
      needsReshape = true;
      spatialDimIndices = {0, 1};
    }

    // Construct attributes for TTIR pooling ops.

    DenseI32ArrayAttr kernelForTTIROps = extract2xI32For2DPoolOpAttr(
        windowDimensions, spatialDimIndices, rewriter);

    DenseI32ArrayAttr strideForTTIROps =
        extract2xI32For2DPoolOpAttr(windowStrides, spatialDimIndices, rewriter);

    DenseI32ArrayAttr dilationForTTIROps = extract2xI32For2DPoolOpAttr(
        windowDilations, spatialDimIndices, rewriter);

    // Padding is constructed later, and per-input.

    // Build per-input pooling ops.
    SmallVector<Value> resultVals;
    for (size_t i = 0; i < srcOp.getInputs().size(); ++i) {
      Value input = adaptor.getInputs()[i];
      Value result;
      TypicalInitReductionValue initVal = (*initValues)[i];
      mlir::Operation *reductionOp = reductionOps[i];

      // Check if this input comes from a fusable PadOp and compute per-input
      // effective padding.
      SmallVector<int64_t> effectivePadding(padding.asArrayRef());
      if (mlir::stablehlo::PadOp padOp =
              getFusablePadOp(srcOp.getInputs()[i], spatialDimIndices)) {
        effectivePadding = combinePaddingFromPadOp(padOp, padding.asArrayRef(),
                                                   spatialDimIndices);

        // Use PadOp's input instead of PadOp's output as input for pooling.
        input = rewriter.getRemappedValue(padOp.getOperand());
      }

      DenseI32ArrayAttr paddingForTTIROps = extract4xI32PaddingAttr(
          rewriter.getDenseI64ArrayAttr(effectivePadding), spatialDimIndices,
          rewriter);

      RankedTensorType inputType =
          mlir::cast<RankedTensorType>(input.getType());
      RankedTensorType resultType = cast<RankedTensorType>(
          getTypeConverter()->convertType(srcOp.getResult(i).getType()));

      if (needsReshape) {
        ArrayRef<int64_t> shape2D = inputType.getShape();
        SmallVector<int64_t> shape4DI64 = {1, shape2D[0], shape2D[1], 1};
        SmallVector<int32_t> shape4D(shape4DI64.begin(), shape4DI64.end());
        RankedTensorType inputType4D = RankedTensorType::get(
            shape4DI64, inputType.getElementType(), inputType.getEncoding());

        input =
            rewriter.create<ttir::ReshapeOp>(srcOp.getLoc(), inputType4D, input,
                                             rewriter.getI32ArrayAttr(shape4D));
        resultType = RankedTensorType::get(
            /*shape*/ {1, resultType.getShape()[0], resultType.getShape()[1],
                       1},
            resultType.getElementType(), resultType.getEncoding());
      }

      if (needsPermute) {
        SmallVector<int64_t> permutedInputShape =
            ttmlir::utils::applyPermutation(inputType.getShape(), permutation);
        input = rewriter.create<ttir::PermuteOp>(
            srcOp.getLoc(),
            RankedTensorType::get(permutedInputShape,
                                  inputType.getElementType(),
                                  inputType.getEncoding()),
            input, permutation);

        // Apply output permutation.
        SmallVector<int64_t> permutedResultShape =
            ttmlir::utils::applyPermutation(resultType.getShape(), permutation);
        resultType = RankedTensorType::get(permutedResultShape,
                                           resultType.getElementType(),
                                           resultType.getEncoding());
      }

      auto restoreOriginalLayout = [&](Value result) -> Value {
        RankedTensorType originalResultType = cast<RankedTensorType>(
            getTypeConverter()->convertType(srcOp.getResult(i).getType()));
        if (needsReshape) {
          result = rewriter.create<ttir::ReshapeOp>(
              srcOp.getLoc(), originalResultType, result,
              rewriter.getI32ArrayAttr(
                  SmallVector<int32_t>(originalResultType.getShape().begin(),
                                       originalResultType.getShape().end())));
        }
        if (needsPermute) {
          result = rewriter.create<ttir::PermuteOp>(
              srcOp.getLoc(), originalResultType, result, inversePermutation);
        }
        return result;
      };

      if (isa<mlir::stablehlo::MaxOp>(reductionOp) && initVal == NEG_INF) {
        result = rewriter
                     .create<ttir::MaxPool2dOp>(
                         srcOp.getLoc(), resultType, input, kernelForTTIROps,
                         strideForTTIROps, dilationForTTIROps,
                         paddingForTTIROps, ceilMode)
                     .getResult();
      } else if (isa<mlir::stablehlo::AddOp>(reductionOp) && initVal == ZERO) {
        // Special case of sum pooling followed by a convenient div op.
        // TODO(acicovic): Check why was i == 0 originally added as a condition.
        std::optional<mlir::Operation *> divOp = extractDivisor(srcOp);
        if (divOp && i == 0) {
          // Average pooling: sum pooling followed by division.
          // Create AvgPool2dOp directly.
          ttir::AvgPool2dOp avgPool2dOp = rewriter.create<ttir::AvgPool2dOp>(
              srcOp.getLoc(), resultType, input, kernelForTTIROps,
              strideForTTIROps, dilationForTTIROps, paddingForTTIROps, ceilMode,
              countIncludesPad);
          result = restoreOriginalLayout(avgPool2dOp.getResult());
          resultVals.push_back(result);
          (*divOp)->getResult(0).replaceAllUsesWith(result);
          rewriter.eraseOp(*divOp);
          continue;
        }

        // Sum pooling imitated as average pooling followed by multiplication.
        ttir::AvgPool2dOp avgPool2dOp = rewriter.create<ttir::AvgPool2dOp>(
            srcOp.getLoc(), resultType, input, kernelForTTIROps,
            strideForTTIROps, dilationForTTIROps, paddingForTTIROps, ceilMode,
            countIncludesPad);
        int32_t kernelSize = kernelForTTIROps[0] * kernelForTTIROps[1];
        DenseElementsAttr splatAttr = DenseElementsAttr::get(
            resultType, rewriter.getFloatAttr(resultType.getElementType(),
                                              static_cast<double>(kernelSize)));
        ttir::ConstantOp kernelSizeConst = rewriter.create<ttir::ConstantOp>(
            srcOp.getLoc(), resultType, splatAttr);
        ttir::MultiplyOp mulOp = rewriter.create<ttir::MultiplyOp>(
            srcOp.getLoc(), resultType, avgPool2dOp.getResult(),
            kernelSizeConst.getResult());
        result = mulOp.getResult();
      } else {
        return rewriter.notifyMatchFailure(
            srcOp, "Invalid combination of reduction function and init value.");
      }

      result = restoreOriginalLayout(result);
      resultVals.push_back(result);
    }

    rewriter.replaceOp(srcOp, resultVals);
    return success();
  }

private:
  // This function verifies all the required conditions to convert stablehlo
  // reduce_window op to TTIR cumsum op and also determine the dimension
  // attribute along which the cumulative sum will be computed.
  // The reduce_window op must satisfy the following conditions.
  // 1. Front op in the block must be 'add'.
  // 2. InitValue must be zero.
  // 3. There are no strides or dilations for window-related attributes.
  // 4. The size of padding attribute is equal to two times input tensor rank.
  // 5. Padding value must be zero in case of splat vector. Window dimension
  //    attribute must have all elements equal to one in this case.
  // 6. Padding attribute have one non-zero element in case of non-splat vector
  //    and this non-zero element must be equal to size of specified dimension
  //    minus one.
  // The dimension attribute is determined in following two ways.
  // 1. (If padding is splat vector): First dimension in the input tensor shape,
  //    whose size is 1, is the required dimension.
  // 2. (If padding is non-splat vector): Window dimension attribute must have
  //    all elements equal to 1 except one; whose location is the required
  //    dimension and value must be equal to size of the required dimension.
  std::optional<int64_t>
  isCumSum(mlir::stablehlo::ReduceWindowOp &srcOp,
           mlir::stablehlo::ReduceWindowOp::Adaptor adaptor,
           TypicalInitReductionValue initValue, mlir::Operation *frontOp,
           DenseI64ArrayAttr padding) const {
    if (!isa<mlir::stablehlo::AddOp>(frontOp)) {
      return std::nullopt;
    }

    if (initValue != TypicalInitReductionValue::ZERO) {
      return std::nullopt;
    }

    // Verify window-related attributes (strides, dilations)
    if (!hasValidWindowAttributes(adaptor)) {
      return std::nullopt;
    }

    int64_t dimension;
    // Check input tensor type and padding
    if (!hasValidInputAndPadding(srcOp, adaptor, dimension, padding)) {
      return std::nullopt;
    }

    return dimension;
  }

  // Helper function to find the StableHLO constant defining op by traversing
  // through operations that preserve constant semantics (similar to
  // getConstantValueDefiningOp).
  mlir::stablehlo::ConstantOp
  getStableHLOConstantDefiningOp(Value value) const {
    Operation *valueDef = value.getDefiningOp();

    // Only traverse through operations that preserve constant semantics
    while (isa_and_nonnull<mlir::stablehlo::ReshapeOp,
                           mlir::stablehlo::BroadcastInDimOp,
                           mlir::stablehlo::ConvertOp>(valueDef)) {
      valueDef = valueDef->getOperand(0).getDefiningOp();
    }
    return mlir::dyn_cast_if_present<mlir::stablehlo::ConstantOp>(valueDef);
  }

  // Extract constant initialization values.
  std::optional<llvm::SmallVector<TypicalInitReductionValue>>
  extractInitValues(mlir::stablehlo::ReduceWindowOp &srcOp) const {
    llvm::SmallVector<TypicalInitReductionValue> initValues;
    for (auto initValue : srcOp.getInitValues()) {
      auto constantOp = getStableHLOConstantDefiningOp(initValue);
      if (!constantOp) {
        return std::nullopt;
      }
      if (checkInitValue(constantOp, TypicalInitReductionValue::NEG_INF)) {
        initValues.push_back(TypicalInitReductionValue::NEG_INF);
      } else if (checkInitValue(constantOp, TypicalInitReductionValue::ZERO)) {
        initValues.push_back(TypicalInitReductionValue::ZERO);
      } else {
        return std::nullopt;
      }
    }
    return initValues;
  }

  // Validate structure of the ReduceWindowOp.
  // - Body must have exactly one block.
  // - Block must contain at least one reduction op.
  // - The number of inputs must equal the number of outputs.
  bool hasValidOpStructure(mlir::stablehlo::ReduceWindowOp &srcOp) const {
    auto &blocks = srcOp.getBody().getBlocks();
    if (blocks.size() != 1) {
      return false;
    }
    const auto &ops = blocks.front().getOperations();
    if (ops.size() < 2) {
      return false;
    }
    if (srcOp.getInputs().size() != srcOp.getResults().size()) {
      return false;
    }
    return true;
  }

  // Verify that all window-related attributes (strides and dilations) are
  // either absent or explicitly set to 1 for every dimension. This ensures the
  // op represents a simple sliding window without dilation or subsampling.
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

  // Validate input rank, padding shape, and window dimensions to determine
  // whether this reduce_window can be interpreted as a cumsum along one
  // dimension.
  bool hasValidInputAndPadding(mlir::stablehlo::ReduceWindowOp &srcOp,
                               mlir::stablehlo::ReduceWindowOp::Adaptor adaptor,
                               int64_t &dimension,
                               DenseI64ArrayAttr padding) const {
    RankedTensorType inputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getInputs()[0].getType()));
    int64_t inputRank = inputType.getRank();
    llvm::ArrayRef<int64_t> windowDimensions =
        adaptor.getWindowDimensionsAttr().asArrayRef();

    // Validate padding size.
    if (padding.size() != (inputRank * 2)) {
      return false;
    }

    // Check for splat padding (all zeroes expected).
    if (llvm::all_equal(padding.asArrayRef())) {
      if (padding[0] != 0) {
        return false;
      }
      if (!llvm::all_of(windowDimensions,
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

  // Finds the first dimension of the input with size == 1.
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
                               DenseI64ArrayAttr padding,
                               RankedTensorType inputType,
                               int64_t &dimension) const {
    int64_t dimArgValue = -1;
    int64_t idx = -1;

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
        if (padding[i] != (dimArgValue - 1)) {
          return false;
        }
      } else if (padding[i] != 0) {
        return false;
      }
    }

    return true;
  }

  // Verify that the output of reduce window op is consumed by division op and
  // the divisor is initialized with a constant op which is equal to number of
  // elements in the kernel.
  std::optional<mlir::Operation *>
  extractDivisor(mlir::stablehlo::ReduceWindowOp &srcOp) const {
    mlir::Operation *op = *srcOp->getUsers().begin();
    if (isa_and_nonnull<mlir::stablehlo::BroadcastInDimOp>(op)) {
      op = *op->getUsers().begin();
    }
    if (!isa_and_nonnull<mlir::stablehlo::DivOp>(op)) {
      return std::nullopt;
    }
    mlir::Operation *divOp = op;

    op = op->getOperand(1).getDefiningOp();
    while (mlir::isa_and_present<mlir::stablehlo::BroadcastInDimOp,
                                 mlir::stablehlo::ReshapeOp,
                                 mlir::stablehlo::ConvertOp>(op)) {
      op = op->getOperand(0).getDefiningOp();
    }

    auto constantOp =
        mlir::dyn_cast_if_present<mlir::stablehlo::ConstantOp>(op);
    if (!constantOp) {
      return std::nullopt;
    }
    if (constantOp.getType().getShape().size()) {
      return std::nullopt;
    }
    auto kernel = srcOp.getWindowDimensions();
    int64_t kernelSize = 1;
    for (int64_t element : kernel) {
      kernelSize *= element;
    }
    int64_t divisor =
        constantOp.getValueAttr().getSplatValue<llvm::APInt>().getZExtValue();
    if (divisor == kernelSize) {
      return divOp;
    }
    return std::nullopt;
  }

  // Extracts indices of all elements in the given array that have the value
  // greter than 1.
  static SmallVector<size_t>
  indicesOfValuesGreaterThanOne(const ArrayRef<int64_t> input) {
    SmallVector<size_t> results;
    for (size_t i = 0; i < input.size(); ++i) {
      if (input[i] > 1) {
        results.push_back(i);
      }
    }
    return results;
  }

  // Extract attribute values for two spatial dimensions from an attribute of
  // a 4D tensor input.
  static DenseI32ArrayAttr
  extract2xI32For2DPoolOpAttr(DenseI64ArrayAttr attr,
                              SmallVector<size_t, 2> spatialDimIndices,
                              PatternRewriter &rewriter) {
    return rewriter.getDenseI32ArrayAttr(
        {static_cast<int32_t>(attr[spatialDimIndices[0]]),   // H
         static_cast<int32_t>(attr[spatialDimIndices[1]])}); // W
  }

  // Check if the given Value comes from a stablehlo::PadOp that can be fused
  // into the pooling operation. Returns the PadOp if fusable, nullptr
  // otherwise.
  mlir::stablehlo::PadOp
  getFusablePadOp(Value input, ArrayRef<size_t> spatialDimIndices) const {
    auto padOp = input.getDefiningOp<mlir::stablehlo::PadOp>();
    if (!padOp) {
      return nullptr;
    }

    // Interior padding is not supported for this fusion.
    for (int64_t interiorPad : padOp.getInteriorPadding()) {
      if (interiorPad != 0) {
        return nullptr;
      }
    }

    // Padding must only be on spatial dimensions (H, W).
    // For non-spatial dimensions, both low and high padding must be 0.
    ArrayRef<int64_t> edgePaddingLow = padOp.getEdgePaddingLow();
    ArrayRef<int64_t> edgePaddingHigh = padOp.getEdgePaddingHigh();

    for (size_t i = 0; i < edgePaddingLow.size(); ++i) {
      bool isSpatialDim =
          (i == spatialDimIndices[0] || i == spatialDimIndices[1]);
      if (!isSpatialDim &&
          (edgePaddingLow[i] != 0 || edgePaddingHigh[i] != 0)) {
        return nullptr;
      }
    }

    return padOp;
  }

  // Combine padding from PadOp with the base padding array.
  // Returns updated padding array with PadOp's spatial padding added.
  static SmallVector<int64_t>
  combinePaddingFromPadOp(mlir::stablehlo::PadOp padOp,
                          ArrayRef<int64_t> basePadding,
                          ArrayRef<size_t> spatialDimIndices) {

    SmallVector<int64_t> combinedPadding(basePadding);
    for (size_t i : spatialDimIndices) {
      combinedPadding[i * 2] += padOp.getEdgePaddingLow()[i];
      combinedPadding[i * 2 + 1] += padOp.getEdgePaddingHigh()[i];
    }
    return combinedPadding;
  }

  // Extract attribute values for padding from an attribute of
  // a 4D tensor input (8 padding values).
  static DenseI32ArrayAttr
  extract4xI32PaddingAttr(DenseI64ArrayAttr padding8,
                          SmallVector<size_t, 2> spatialDimIndices,
                          PatternRewriter &rewriter) {
    return rewriter.getDenseI32ArrayAttr({
        static_cast<int32_t>(padding8[spatialDimIndices[0] * 2]),     // H low
        static_cast<int32_t>(padding8[spatialDimIndices[1] * 2]),     // W low
        static_cast<int32_t>(padding8[spatialDimIndices[0] * 2 + 1]), // H high
        static_cast<int32_t>(padding8[spatialDimIndices[1] * 2 + 1]), // W high
    });
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
    LogicalResult legalityResult =
        checkConversionLegality(srcOp, adaptor, rewriter);
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

      rewriter.replaceOpWithNewOp<mlir::tt::ttir::BroadcastOp>(
          srcOp, outputType, adaptor.getOperand(), broadcastShape);
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

      RankedTensorType unsqueezedType =
          RankedTensorType::get(unsqueezeShape, inputType.getElementType());
      ttir::ReshapeOp reshapeOp = rewriter.create<ttir::ReshapeOp>(
          srcOp.getLoc(), unsqueezedType, adaptor.getOperand(), reshapeDimAttr);

      ::llvm::ArrayRef<int64_t> inputShape = unsqueezeShape;
      ::llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

      SmallVector<int64_t> broadcastShape =
          ttmlir::utils::getBroadcastDimensions<int64_t>(inputShape,
                                                         outputShape);

      rewriter.replaceOpWithNewOp<ttir::BroadcastOp>(srcOp, outputType,
                                                     reshapeOp, broadcastShape);
    }

    return success();
  }

private:
  LogicalResult
  checkConversionLegality(mlir::stablehlo::BroadcastInDimOp &srcOp,
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

    rewriter.replaceOpWithNewOp<DestOp>(srcOp, outputType, adaptor.getLhs(),
                                        adaptor.getRhs());

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

    // Check legality of the conversion.
    LogicalResult err = checkConversionLegality(srcOp, adaptor, rewriter);
    if (failed(err)) {
      return err;
    }

    // Create the output tensor type based on inputs
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    rewriter.replaceOpWithNewOp<ttir::ConcatOp>(
        srcOp, outputType, adaptor.getInputs(),
        static_cast<int32_t>(adaptor.getDimension()));

    return success();
  }

private:
  LogicalResult
  checkConversionLegality(mlir::stablehlo::ConcatenateOp &srcOp,
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

    // Bitwise ops for boolean operands:
    // XLA lowering converts boolean to int8 and then bitwise ops is applied.
    // Here we detect such cases and convert them to logical ops.
    // This solution is required as TTNN doesn't support boolean types natively
    // and tt-mlir converts boolean to bfloat16 and then to uint8 generating
    // incorrect results.
    // XLA lowering adds a NotEqual comparison op following the bitwise op to
    // restore the original boolean semantics; which is removed here after
    // replacing the bitwise op.
    if (areOperandsBoolean(srcOp)) {
      auto logicalOpType =
          RankedTensorType::get(outputType.getShape(), rewriter.getI1Type(),
                                outputType.getEncoding());
      auto logicalOp = rewriter.create<LogicalDestOp>(
          srcOp.getLoc(), logicalOpType,
          ValueRange{
              adaptor.getOperands()[0].getDefiningOp()->getOperands()[0],
              adaptor.getOperands()[1].getDefiningOp()->getOperands()[0]});

      auto numUsers = llvm::range_size(srcOp.getResult().getUsers());
      if (numUsers == 1) {
        // Erase NotEqual comparison op following the bitwise op and update its
        // uses to use the logical op result directly.
        auto userOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(
            *srcOp.getResult().getUsers().begin());

        if (userOp && userOp.getComparisonDirection() ==
                          mlir::stablehlo::ComparisonDirection::NE) {
          userOp.getResult().replaceAllUsesWith(logicalOp.getResult());
          rewriter.eraseOp(userOp);
          return success();
        }
      }

      // Add a typecast to convert i1 result to original output type (if
      // required).
      rewriter.replaceOpWithNewOp<mlir::tt::ttir::TypecastOp>(
          srcOp, outputType, logicalOp.getResult());
      return success();
    }

    if (getStableHLOOpType(srcOp) == StableHLOOpType::kLogical) {
      rewriter.replaceOpWithNewOp<LogicalDestOp>(srcOp, outputType,
                                                 adaptor.getOperands());
    } else {
      rewriter.replaceOpWithNewOp<BitwiseDestOp>(srcOp, outputType,
                                                 adaptor.getOperands());
    }

    return success();
  }

private:
  enum StableHLOOpType { kLogical = 0, kBitwise = 1 };

  // Determines stablehlo op type based on its operand types (i.e. their
  // bit width). This assumes boolean operands are modeled as 1bit wide ints.
  static StableHLOOpType getStableHLOOpType(const SrcOp &srcOp) {
    // Checks if all operands are boolean (have bit width equal to 1).
    bool allOperandsAreBoolean =
        llvm::all_of(srcOp->getOperands(), [](auto operand) {
          return mlir::cast<RankedTensorType>(operand.getType())
                     .getElementTypeBitWidth() == 1;
        });

    return allOperandsAreBoolean ? StableHLOOpType::kLogical
                                 : StableHLOOpType::kBitwise;
  }

  // This function iterates over each operand of the provided source operation
  // and determines if each operand represents a boolean value through a
  // stablehlo::ConvertOp from a boolean tensor.
  // return true if all operands of a given StableHLO operation are effectively
  // boolean.
  bool areOperandsBoolean(const SrcOp &srcOp) const {
    bool allOperandsBoolean =
        llvm::all_of(srcOp->getOperands(), [](auto operand) {
          // Check if operand is a ranked tensor of bit width 1.
          if (mlir::cast<RankedTensorType>(operand.getType())
                  .getElementTypeBitWidth() == 1) {
            return false; // Already boolean, so not counted as converted.
          }

          // Check if the defining op is a stablehlo::ConvertOp.
          auto definingOp = operand.getDefiningOp();
          if (!isa_and_nonnull<mlir::stablehlo::ConvertOp>(definingOp)) {
            return false; // Not a conversion from boolean
          }

          // Check if the input to the conversion is boolean.
          return mlir::cast<RankedTensorType>(
                     definingOp->getOperand(0).getType())
                     .getElementTypeBitWidth() == 1;
        });

    return allOperandsBoolean;
  }
};
} // namespace

static llvm::ErrorOr<ttcore::ReduceType>
getReduceTypeFromRegion(Region &region) {
  // TODO(wooseoklee): This pattern matching mechanism may need to be updated as
  // we see complicated patterns of reduce block in the future.

  assert(region.getBlocks().size() == 1 &&
         "Region should have exactly one block");

  // Add, Prod, Max, Min are the only supported reduce types for now.
  // Invalid is default reduction type in TT-Metal, used when we do not do
  // reduction, only copy source to output.
  auto &block = region.front();
  for (Operation &op : block) {
    if (isa<mlir::stablehlo::AddOp>(op)) {
      return ttcore::ReduceType::Sum;
    }
    if (isa<mlir::stablehlo::MulOp>(op)) {
      return ttcore::ReduceType::Prod;
    }
    if (isa<mlir::stablehlo::MaxOp>(op)) {
      return ttcore::ReduceType::Max;
    }
    if (isa<mlir::stablehlo::MinOp>(op)) {
      return ttcore::ReduceType::Min;
    }
    if (isa<mlir::stablehlo::ReturnOp>(op)) {
      return ttcore::ReduceType::Invalid;
    }
  }
  // Reduction type is not supported
  return llvm::ErrorOr<ttcore::ReduceType>(
      std::make_error_code(std::errc::operation_not_supported));
}

static llvm::ErrorOr<ttcore::ReduceType> getReduceType(Operation *op) {
  if (!llvm::isa<mlir::stablehlo::AllReduceOp, mlir::stablehlo::ReduceScatterOp,
                 mlir::stablehlo::ScatterOp>(op)) {
    return llvm::ErrorOr<ttcore::ReduceType>(
        std::make_error_code(std::errc::operation_not_supported));
  }
  return getReduceTypeFromRegion(op->getRegion(0));
}

static LogicalResult
determineClusterAxis(::mlir::DenseIntElementsAttr replicaGroups,
                     uint32_t &clusterAxis) {
  /*
  We need to figure out what the cluster axis is based on replica_groups.
  Replica groups define which device axis we are performing the collective
  communication operation on. It is a 2D vector. Each element in replica_groups
  contains a list of devices that will perform the collective communication
  operation with each other. Currently we only support 2D meshes, but this
  algorithm can be expanded for ND.

  ex.
  mesh = [2, 4]
  replica_groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
  0 1 2 3
  4 5 6 7

  collective communication operation happens on (0, 1, 2, 3) and (4, 5, 6, 7) so
  cluster_axis = 1 (mesh[1])

  mesh = [2, 4]
  replica_groups = [[0, 4], [1, 5], [2, 6], [3, 7]]
  0 1 2 3
  4 5 6 7

  collective communication operation happens on (0, 4), (1, 5), (2, 6), (3, 7)
  so cluster_axis = 0 (mesh[0])

  */
  auto replicaGroupsShape = replicaGroups.getType().getShape();

  if (replicaGroupsShape.size() != 2) {
    // Can only have replica groups of size 2. Otherwise, this is an ill formed
    // graph and needs to be asserted.
    return failure();
  }

  // Case where we have single devices in each replica_group (ie perform
  // collective communication operation against itself which should be optimized
  // away). We also assume we are only using our constrained mesh types (ie 1x8,
  // 1x32 etc) and cannot have (32x1, 8x1).
  if (replicaGroupsShape[1] != 1) {
    auto firstElementIt = replicaGroups.begin();
    auto secondElementIt = firstElementIt + 1;

    clusterAxis = (((*firstElementIt) + 1) == *secondElementIt);
    return success();
  }

  // If replicaGroupsShape[1] == 1, then the cluster axis should be set to 0.
  clusterAxis = 0;
  return success();
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

// Decompose SelectAndScatter into MaxPool2dWithIndices + Scatter:
// 1. MaxPool2dWithIndices finds the maximum values and their flattened indices
// within each pooling window.
// 2. Scatter scatters the corresponding source values back into those
// positions.
//
// This decomposition currently supports only SelectAndScatter operations where
// the select function uses MAX, which corresponds to the case appearing in
// MaxPool2d backward. Other types in Select block are not used in our
// workloads.
//
// If multiple windows overlap (e.g., stride < window size), several source
// values may map to the same index. In that case, Scatter reduces them
// using the reduction function specified in the scatter operation (e.g., add,
// multiply, etc.).
//
// Example:
// --------
// Input tensor (4x4):
//   [[ 1,  5,  2,  4],
//    [ 7,  3,  8,  6],
//    [ 0,  9, 11, 10],
//    [12, 13, 14, 15]]
//
// Window size: 2x2, stride: 2
//
// Source tensor (same shape as pooled output):
//   [[10, 20],
//    [30, 40]]
//
// Step 1: MaxPool2dWithIndices
//   - For each 2x2 window, find the maximum value and record its **flattened
//   index** within the input:
//       Window (0,0): max = 7  → index = 4
//       Window (0,1): max = 8  → index = 6
//       Window (1,0): max = 13 → index = 13
//       Window (1,1): max = 15 → index = 15
//
//   - Max values: [[ 7,  8],
//                  [13, 15]]
//   - Indices:    [[ 4,  6],
//                  [13, 15]]
//
// Step 2: Scatter
//   - Scatter the source values [[10,20],[30,40]] into the flattened positions
//   above.
//   - Result (reshaped back to 4x4):
//       [[ 0,  0,  0,  0],
//        [10,  0, 20,  0],
//        [ 0,  0,  0,  0],
//        [ 0, 30,  0, 40]]
namespace {
class StableHLOToTTIRSelectAndScatterOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::SelectAndScatterOp> {
  using OpConversionPattern<
      mlir::stablehlo::SelectAndScatterOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::SelectAndScatterOp srcOp,
                  mlir::stablehlo::SelectAndScatterOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Verify that the select block contains only a compare op
    if (failed(verifySelectBlock(srcOp, rewriter))) {
      return failure();
    }

    Location loc = srcOp.getLoc();
    auto operand = srcOp.getOperand();
    auto source = srcOp.getSource();
    auto initValue_ = srcOp.getInitValue();

    // Get window attributes
    auto windowDims = srcOp.getWindowDimensionsAttr();
    auto windowStrides_ = srcOp.getWindowStridesAttr();
    auto padding_ = srcOp.getPaddingAttr();

    // Initial value in tensor which we scatter into
    // If not present, defaults to zero
    auto operandType = mlir::cast<RankedTensorType>(operand.getType());
    float fillValue = 0.0f;

    if (initValue_) {
      if (auto constOp =
              initValue_.getDefiningOp<mlir::stablehlo::ConstantOp>()) {
        auto valAttr = mlir::cast<DenseFPElementsAttr>(constOp.getValue());
        fillValue = static_cast<float>(
            valAttr.getValues<APFloat>()[0].convertToDouble());
      } else {
        llvm::report_fatal_error("initValue_ must be a stablehlo.constant");
      }
    }

    auto fullTensorOp = rewriter.create<ttir::FullOp>(
        loc, operandType, rewriter.getF32FloatAttr(fillValue));

    // Tensor which we scatter into
    auto fullTensor = fullTensorOp.getResult();

    // Generate defaults if they dont exist
    // Default window strides is all ones
    auto windowStrides = windowStrides_
                             ? windowStrides_
                             : rewriter.getDenseI64ArrayAttr(
                                   SmallVector<int64_t>(windowDims.size(), 1));

    // Default padding is all zeros
    auto padding =
        padding_ ? rewriter.getDenseI64ArrayAttr(
                       SmallVector<int64_t>(padding_.getValues<int64_t>()))
                 : rewriter.getDenseI64ArrayAttr(
                       SmallVector<int64_t>(windowDims.size() * 2, 0));

    // Adjust tensor layouts and args for MaxPool2dWithIndices
    // Get indices of elements larger than one which correspond to spatial dims
    // (H, W)
    auto spatialDims = getIndicesofElementsLargerThanOne(windowDims);
    if (spatialDims.empty()) {
      return rewriter.notifyMatchFailure(srcOp, "No elements larger than one");
    }

    // Generate desired and current layouts (desired = N,H,W,C)
    // Generate permutation for current->desired layout

    const int64_t SPATIAL_H = -3;
    const int64_t SPATIAL_W = -2;
    const int64_t NON_SPATIAL = -1;

    // Desired layout (N,H,W,C)
    std::vector<int64_t> desiredLayout(operand.getType().getRank(),
                                       NON_SPATIAL);
    desiredLayout[operand.getType().getRank() - 3] = SPATIAL_H;
    desiredLayout[operand.getType().getRank() - 2] = SPATIAL_W;

    int64_t nonSpatialCount = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(desiredLayout.size()); ++i) {
      if (desiredLayout[i] == NON_SPATIAL) {
        desiredLayout[i] = nonSpatialCount++;
      }
    }

    // Current layout
    std::vector<int64_t> currentLayout(operand.getType().getRank(),
                                       NON_SPATIAL);
    currentLayout[spatialDims[0]] = SPATIAL_H;
    currentLayout[spatialDims[1]] = SPATIAL_W;

    nonSpatialCount = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(currentLayout.size()); ++i) {
      if (currentLayout[i] == NON_SPATIAL) {
        currentLayout[i] = nonSpatialCount++;
      }
    }

    // Permutation for current->desired layout and it's inverse
    auto permutation = ttmlir::utils::generatePermutation(
        llvm::ArrayRef(currentLayout), llvm::ArrayRef(desiredLayout));
    auto inverseOfPermutation = ttmlir::utils::inversePermutation(permutation);

    // Adjust args to match definition in MaxPool2dWithIndices
    auto kernel = rewriter.getDenseI32ArrayAttr(
        {static_cast<int32_t>(windowDims[spatialDims[0]]),
         static_cast<int32_t>(windowDims[spatialDims[1]])});

    auto stride = rewriter.getDenseI32ArrayAttr(
        {static_cast<int32_t>(windowStrides[spatialDims[0]]),
         static_cast<int32_t>(windowStrides[spatialDims[1]])});

    auto dilations = rewriter.getDenseI32ArrayAttr({1, 1});

    auto paddingAttr = rewriter.getDenseI32ArrayAttr({
        static_cast<int32_t>(padding[2 * spatialDims[0]]),     // top
        static_cast<int32_t>(padding[2 * spatialDims[1]]),     // left
        static_cast<int32_t>(padding[2 * spatialDims[0] + 1]), // bottom
        static_cast<int32_t>(padding[2 * spatialDims[1] + 1]), // right
    });

    auto ceilMode = rewriter.getBoolAttr(false);

    // Apply input permutation to operand, source and tensor which we scatter
    // into (to N,H,W,C)
    auto operandPermShape = ::ttmlir::utils::applyPermutation(
        operand.getType().getShape(), permutation);
    auto sourcePermShape = ::ttmlir::utils::applyPermutation(
        source.getType().getShape(), permutation);
    auto fullTensorPermShape = ::ttmlir::utils::applyPermutation(
        fullTensor.getType().getShape(), permutation);

    operand = applyPermutationToValue(rewriter, loc, operand, operandPermShape,
                                      operand.getType(), permutation,
                                      "_permuteInput");
    source = applyPermutationToValue(rewriter, loc, source, sourcePermShape,
                                     source.getType(), permutation,
                                     "_permuteSource");
    fullTensor = applyPermutationToValue(
        rewriter, loc, fullTensor, fullTensorPermShape, fullTensor.getType(),
        permutation, "_permuteFullTensor");

    // Calling MaxPool2dWithIndices op on operand
    // and obtaining indices which will be used for Scatter
    auto pooledType = RankedTensorType::get(sourcePermShape,
                                            source.getType().getElementType());
    auto indicesType =
        RankedTensorType::get(sourcePermShape,
                              rewriter.getIntegerType(32)); // i32 for indices

    Value pooledEmpty = rewriter.create<ttir::EmptyOp>(loc, pooledType);
    Value indicesEmpty = rewriter.create<ttir::EmptyOp>(loc, indicesType);

    auto maxPoolOp = rewriter.create<ttir::MaxPool2dWithIndicesOp>(
        loc, TypeRange{pooledType, indicesType}, operand,
        ValueRange{pooledEmpty, indicesEmpty}, kernel, stride, dilations,
        paddingAttr, ceilMode);

    auto indices = maxPoolOp.getResultIndices();

    // Reshape for Scatter (N,H*W,1,C)
    auto reshapedIndicesType =
        getNHWFlattenedType(mlir::cast<RankedTensorType>(indices.getType()));
    auto reshapedIndices = generateReshape(indices, reshapedIndicesType,
                                           rewriter, "_reshapeIndices");

    auto reshapedSourceType =
        getNHWFlattenedType(mlir::cast<RankedTensorType>(source.getType()));
    auto reshapedSource =
        generateReshape(source, reshapedSourceType, rewriter, "_reshapeSource");

    auto reshapedFullTensorType =
        getNHWFlattenedType(mlir::cast<RankedTensorType>(fullTensor.getType()));
    auto reshapedFullTensor = generateReshape(
        fullTensor, reshapedFullTensorType, rewriter, "_reshapeFullTensor");

    // Calling Scatter to scatter source values back to positions
    // in the full tensor as indicated by previously obtained indices
    auto scatterOutputType =
        mlir::cast<RankedTensorType>(reshapedFullTensorType);

    auto dimAttr = rewriter.getI32IntegerAttr(1);

    // Convert reduceType stablehlo attribute into ttir attribute
    // We are using getReduceTypeFromRegion instead of getReduceType here
    // because SelectAndScatterOp has a select region and a scatter region.
    llvm::ErrorOr<ttcore::ReduceType> scatterReduceType =
        getReduceTypeFromRegion(srcOp.getScatter());
    if (!scatterReduceType) {
      return rewriter.notifyMatchFailure(
          srcOp, "SelectAndScatterOp cannot specify reduce type.");
    }
    auto reduceTypeAttr =
        ttcore::ReduceTypeAttr::get(rewriter.getContext(), *scatterReduceType);
    auto scatterResult = rewriter.create<ttir::ScatterOp>(
        loc, scatterOutputType,
        reshapedFullTensor, // input tensor
        reshapedIndices,    // index tensor
        reshapedSource,     // source tensor
        dimAttr,            // dim = 1 (the H*W flattened dimension)
        reduceTypeAttr);    // reduction type

    // Reshape back to N,H,W,C
    auto finalOutputType = RankedTensorType::get(
        fullTensorPermShape, scatterOutputType.getElementType());
    auto finalResult = generateReshape(scatterResult, finalOutputType, rewriter,
                                       "_reshapeToNHWC");

    // Apply inverse permutation to get back to original layout
    auto originalShape = ::ttmlir::utils::applyPermutation(
        finalResult.getType().getShape(), inverseOfPermutation);
    auto finalPermutedResult = applyPermutationToValue(
        rewriter, loc, finalResult, originalShape, finalResult.getType(),
        inverseOfPermutation, "_permuteBackToOriginal");

    rewriter.replaceOp(srcOp, finalPermutedResult);
    return success();
  }

private:
  llvm::SmallVector<int64_t>
  getIndicesofElementsLargerThanOne(llvm::ArrayRef<int64_t> array) const {
    llvm::SmallVector<int64_t> indices;
    for (int32_t i = 0; i < static_cast<int64_t>(array.size()); ++i) {
      if (array[i] > 1) {
        indices.push_back(i);
      }
    }
    return indices;
  }

  RankedTensorType getNHWFlattenedType(RankedTensorType unflattenedType) const {
    llvm::ArrayRef<int64_t> shape = unflattenedType.getShape();
    assert(shape.size() == 4 && "Expected 4D NHWC tensor");
    llvm::SmallVector<int64_t, 4> flattenedShape = {
        shape[0], shape[1] * shape[2], 1, shape[3]};
    return RankedTensorType::get(flattenedShape,
                                 unflattenedType.getElementType());
  }

  ttir::ReshapeOp
  generateReshape(mlir::TypedValue<mlir::RankedTensorType> input,
                  RankedTensorType outputType, PatternRewriter &rewriter,
                  StringRef suffix) const {
    return rewriter.create<ttir::ReshapeOp>(
        ttmlir::utils::appendLocationSuffix(input.getLoc(), suffix), outputType,
        input,
        rewriter.getI32ArrayAttr(SmallVector<int32_t>(
            outputType.getShape().begin(), outputType.getShape().end())));
  }

  mlir::TypedValue<mlir::RankedTensorType> applyPermutationToValue(
      OpBuilder &rewriter, Location loc, mlir::Value input,
      ArrayRef<int64_t> permutedShape, RankedTensorType inputType,
      ArrayRef<int64_t> permutation, StringRef suffix) const {
    RankedTensorType permuteType = RankedTensorType::get(
        permutedShape, inputType.getElementType(), inputType.getEncoding());
    return rewriter.create<ttir::PermuteOp>(
        ttmlir::utils::appendLocationSuffix(loc, suffix), permuteType, input,
        permutation);
  }

  LogicalResult verifySelectBlock(mlir::stablehlo::SelectAndScatterOp srcOp,
                                  PatternRewriter &rewriter) const {
    auto &selectBlock = srcOp.getSelect().front();
    for (Operation &op : selectBlock) {
      // Skip the return operation
      if (mlir::isa<mlir::stablehlo::ReturnOp>(op)) {
        continue;
      }
      if (!mlir::isa<mlir::stablehlo::CompareOp>(op)) {
        return rewriter.notifyMatchFailure(
            srcOp,
            "SelectAndScatter select block must contain only a compare op.");
      }
    }
    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRAllReduceOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::AllReduceOp> {

  using OpConversionPattern<mlir::stablehlo::AllReduceOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::AllReduceOp srcOp,
                  mlir::stablehlo::AllReduceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Check legality of the conversion.
    LogicalResult err = checkConversionLegality(srcOp, adaptor, rewriter);
    if (failed(err)) {
      return err;
    }

    if (auto srcChannelHandleAttr = adaptor.getChannelHandleAttr()) {
      // channelType is supposed to be DEVICE_TO_DEVICE or Invalid for CCL ops.
      // Currently, we ensure if it is DEVICE_TO_DEVICE communication.
      // Consider preserving this information in the future if the attribute
      // is non-DEVICE_TO_DEVICE values.
      auto channelType =
          static_cast<StableHLOChannelType>(srcChannelHandleAttr.getType());
      if (channelType != StableHLOChannelType::kChannelTypeDeviceToDevice &&
          channelType != StableHLOChannelType::kChannelTypeInvalid) {
        return failure();
      }
    }

    // Determine cluster axis based on replica groups
    uint32_t clusterAxis;
    if (failed(determineClusterAxis(adaptor.getReplicaGroups(), clusterAxis))) {
      return rewriter.notifyMatchFailure(
          srcOp, "AllReduceOp cannot specify cluster axis.");
    }

    // Convert reduceType stablehlo attribute into ttir attribute
    llvm::ErrorOr<ttcore::ReduceType> reduceType = getReduceType(srcOp);
    if (!reduceType) {
      return rewriter.notifyMatchFailure(
          srcOp, "AllReduceOp cannot specify reduce type.");
    }

    // Handle variadic input/output pairs by creating mulitple AllReduceOps.
    llvm::SmallVector<mlir::Value> allReduceOpResults;
    for (auto [inputOperand, resultOperand] :
         llvm::zip_equal(adaptor.getOperands(), srcOp->getResults())) {
      auto outputType = mlir::cast<RankedTensorType>(
          getTypeConverter()->convertType(resultOperand.getType()));

      auto allReduceOp = rewriter.create<mlir::tt::ttir::AllReduceOp>(
          srcOp.getLoc(), outputType, inputOperand, *reduceType, clusterAxis);

      allReduceOpResults.push_back(allReduceOp.getResult());
    }
    rewriter.replaceOp(srcOp, allReduceOpResults);

    return success();
  }

private:
  LogicalResult
  checkConversionLegality(mlir::stablehlo::AllReduceOp &srcOp,
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
class StableHLOToTTIRReduceScatterOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ReduceScatterOp> {
  using OpConversionPattern<
      mlir::stablehlo::ReduceScatterOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReduceScatterOp srcOp,
                  mlir::stablehlo::ReduceScatterOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Create the output tensor type based on inputs
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    if (auto srcChannelHandleAttr = adaptor.getChannelHandleAttr()) {
      // channelType is supposed to be DEVICE_TO_DEVICE or Invalid for CCL ops.
      // Currently, we ensure if it is DEVICE_TO_DEVICE communication.
      // Consider preserving this information in the future if the attribute
      // is non-DEVICE_TO_DEVICE values.
      auto channelType =
          static_cast<StableHLOChannelType>(srcChannelHandleAttr.getType());
      if (channelType != StableHLOChannelType::kChannelTypeDeviceToDevice &&
          channelType != StableHLOChannelType::kChannelTypeInvalid) {
        return failure();
      }
    }

    // Determine cluster axis based on replica groups
    uint32_t clusterAxis;
    if (failed(determineClusterAxis(adaptor.getReplicaGroups(), clusterAxis))) {
      return rewriter.notifyMatchFailure(
          srcOp, "ReduceScatterOp cannot specify cluster axis.");
    }

    // Convert reduceType stablehlo attribute into ttir attribute
    llvm::ErrorOr<ttcore::ReduceType> reduceType = getReduceType(srcOp);
    if (!reduceType) {
      return rewriter.notifyMatchFailure(
          srcOp, "ReduceScatterOp cannot specify reduce type.");
    }

    rewriter.replaceOpWithNewOp<ttir::ReduceScatterOp>(
        srcOp, outputType, adaptor.getOperands()[0], *reduceType,
        adaptor.getScatterDimension(), clusterAxis);

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
    // Check legality of the conversion.
    LogicalResult err = checkConversionLegality(srcOp, adaptor, rewriter);
    if (failed(err)) {
      return err;
    }

    // Create the output tensor type based on inputs
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult(0).getType()));

    // Determine cluster axis based on replica groups
    uint32_t clusterAxis;
    if (failed(determineClusterAxis(adaptor.getReplicaGroups(), clusterAxis))) {
      return rewriter.notifyMatchFailure(
          srcOp, "AllGather cannot specify cluster axis.");
    }

    rewriter.replaceOpWithNewOp<ttir::AllGatherOp>(
        srcOp, outputType, adaptor.getOperands()[0], adaptor.getAllGatherDim(),
        clusterAxis);

    return success();
  }

private:
  LogicalResult
  checkConversionLegality(mlir::stablehlo::AllGatherOp &srcOp,
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
class StableHLOToTTIRCollectivePermuteOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CollectivePermuteOp> {
  using OpConversionPattern<
      mlir::stablehlo::CollectivePermuteOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::CollectivePermuteOp srcOp,
                  mlir::stablehlo::CollectivePermuteOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Create the output tensor type based on inputs
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    if (auto srcChannelHandleAttr = adaptor.getChannelHandleAttr()) {
      // channelType is supposed to be DEVICE_TO_DEVICE or Invalid for CCL ops.
      // Currently, we ensure if it is DEVICE_TO_DEVICE communication.
      // Consider preserving this information in the future if the attribute
      // is non-DEVICE_TO_DEVICE values.
      auto channelType =
          static_cast<StableHLOChannelType>(srcChannelHandleAttr.getType());
      if (channelType != StableHLOChannelType::kChannelTypeDeviceToDevice &&
          channelType != StableHLOChannelType::kChannelTypeInvalid) {
        return failure();
      }
    }

    rewriter.replaceOpWithNewOp<ttir::CollectivePermuteOp>(
        srcOp, outputType, adaptor.getOperand(),
        adaptor.getSourceTargetPairs());

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
    // Check legality of the conversion.
    LogicalResult err = checkConversionLegality(srcOp, adaptor, rewriter);
    if (failed(err)) {
      return err;
    }

    auto callTargetName = adaptor.getCallTargetNameAttr();

    // There are three call target names that we handle:
    // 1. `@Sharding` - This is the custom call for sharding
    // 2. `@SPMDFullToShardShape` - This is the custom call for converting full
    // shape to shard shape.
    // 3. `@SPMDShardToFullShape` - This is the custom call for converting
    // shard shape to full shape.
    // All @SPMD* calls have @Sharding as their first operand. Therefore, we
    // skip @Sharding and handle it in the other two cases together.
    if (callTargetName !=
            mlir::tt::gspmd_utils::kSPMDFullToShardShapeCallTargetName &&
        callTargetName !=
            mlir::tt::gspmd_utils::kSPMDShardToFullShapeCallTargetName) {
      return success();
    }

    // Set the shard direction.
    mlir::tt::ttcore::MeshShardDirection shardDirection =
        mlir::tt::ttcore::MeshShardDirection::ShardToFull;
    if (callTargetName ==
        mlir::tt::gspmd_utils::kSPMDFullToShardShapeCallTargetName) {
      shardDirection = mlir::tt::ttcore::MeshShardDirection::FullToShard;
    }

    // We want to extract the mhlo.sharding attribute from the
    // CustomCallOp.
    auto opShardingAttr = dyn_cast_if_present<StringAttr>(
        adaptor.getAttributes().get(mlir::tt::gspmd_utils::kXlaShardingAttr));
    if (!opShardingAttr) {
      return rewriter.notifyMatchFailure(
          srcOp, "@SPMD* custom call is missing mhlo.sharding attribute.");
    }

    // We also want to extract the mhlo.sharding attribute from this op's
    // @Sharding operand.
    auto shardingOperand = srcOp->getOperand(0);
    auto definingOp =
        shardingOperand.getDefiningOp<mlir::stablehlo::CustomCallOp>();
    auto operandShardingAttr = definingOp->getAttrOfType<mlir::StringAttr>(
        mlir::tt::gspmd_utils::kXlaShardingAttr);

    if (!operandShardingAttr) {
      return rewriter.notifyMatchFailure(
          srcOp, "@Sharding custom call is missing mhlo.sharding attribute.");
    }

    // We also extract the shard status from the @Sharding op.
    auto runtimeTensorShardingAttr =
        definingOp->getAttrOfType<mlir::tt::ttcore::RuntimeTensorShardingAttr>(
            mlir::tt::ttcore::RuntimeTensorShardingAttr::name);
    auto shardStatusAttr = runtimeTensorShardingAttr.getShardStatus();

    // Insert default sharding status if not present.
    if (!shardStatusAttr) {
      shardStatusAttr = mlir::tt::ttcore::ShardStatusAttr::get(
          getContext(), mlir::tt::ttcore::ShardStatus::Unsharded);
    }

    // Once extracted, we can generate the GSPMDMeshSharding object.
    llvm::Expected<mlir::tt::gspmd_utils::GSPMDMeshSharding> gspmdMeshSharding =
        mlir::tt::gspmd_utils::GSPMDMeshSharding::generate(
            opShardingAttr.getValue(), operandShardingAttr.getValue(),
            shardStatusAttr.getValue(), shardDirection);
    if (auto err = gspmdMeshSharding.takeError()) {
      return rewriter.notifyMatchFailure(
          srcOp, "Error trying to parse GSPMD annotation.");
    }

    // Insert the new MeshShardOp with the generated GSPMDMeshSharding.
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp->getResult(0).getType()));
    rewriter.replaceOpWithNewOp<mlir::tt::ttir::MeshShardOp>(
        srcOp, outputType, definingOp.getInputs().front(),
        gspmdMeshSharding->getShardType(),
        gspmdMeshSharding->getShardDirection(),
        gspmdMeshSharding->getShardShape(), gspmdMeshSharding->getShardDims());

    // Erase the @Sharding op as well.
    rewriter.eraseOp(definingOp);
    return success();
  }

private:
  LogicalResult
  checkConversionLegality(mlir::stablehlo::CustomCallOp &srcOp,
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

    rewriter.replaceOpWithNewOp<ttir::SliceStaticOp>(
        srcOp, outputType, adaptor.getOperand(),
        rewriter.getI32ArrayAttr(startIndices),
        rewriter.getI32ArrayAttr(endIndices), rewriter.getI32ArrayAttr(step));

    return success();
  }
};
} // namespace

// NOTE: StableHLO Dynamic Slice Op clamps start indices to ensure validity,
// but we don't. Would add multiple ops (worse perf) and doesn't appear in test
// models.
namespace {
class StableHLOToTTIRDynamicSliceOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::DynamicSliceOp> {
  using OpConversionPattern<
      mlir::stablehlo::DynamicSliceOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::DynamicSliceOp srcOp,
                  mlir::stablehlo::DynamicSliceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    // Reshape start indices to 1D tensors for concat op.
    ValueRange startIndicesRange = adaptor.getStartIndices();
    SmallVector<Value> startIndicesValues1D;
    auto startIndexElementType =
        mlir::cast<RankedTensorType>(startIndicesRange[0].getType())
            .getElementType();
    auto singleElementTensorType =
        RankedTensorType::get({1}, startIndexElementType);

    for (Value startIndex : startIndicesRange) {
      auto reshapedIndex = rewriter.create<ttir::ReshapeOp>(
          srcOp.getLoc(),
          RankedTensorType::get(singleElementTensorType.getShape(),
                                startIndexElementType,
                                singleElementTensorType.getEncoding()),
          startIndex, rewriter.getI32ArrayAttr({1}));
      startIndicesValues1D.push_back(reshapedIndex);
    }
    // Create a single 1D tensor from start indices values using concat op.
    auto startIndicesTensorType = RankedTensorType::get(
        {static_cast<int64_t>(startIndicesValues1D.size())},
        startIndexElementType);
    auto startIndicesTensor = rewriter.create<mlir::tt::ttir::ConcatOp>(
        srcOp.getLoc(),
        RankedTensorType::get(startIndicesTensorType.getShape(),
                              startIndexElementType,
                              startIndicesTensorType.getEncoding()),
        startIndicesValues1D, /*dim=*/0);

    // Create a 1D constant tensor with slice_sizes values.
    auto sliceSizes = srcOp.getSliceSizes();
    SmallVector<int32_t> sliceSizesInt32(sliceSizes.begin(), sliceSizes.end());
    auto sliceSizesTensorType = RankedTensorType::get(
        {static_cast<int64_t>(sliceSizesInt32.size())}, rewriter.getI32Type());
    auto sliceSizesAttr = mlir::DenseElementsAttr::get(
        sliceSizesTensorType, llvm::ArrayRef<int32_t>(sliceSizesInt32));
    auto sliceSizesConstant = rewriter.create<mlir::tt::ttir::ConstantOp>(
        srcOp.getLoc(), sliceSizesTensorType, sliceSizesAttr);

    // Create an add op that adds the slice sizes to start indices to get end
    // indices.
    auto endIndicesTensor = rewriter.create<mlir::tt::ttir::AddOp>(
        srcOp.getLoc(),
        RankedTensorType::get(startIndicesTensorType.getShape(),
                              startIndexElementType,
                              startIndicesTensorType.getEncoding()),
        startIndicesTensor, sliceSizesConstant);

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::SliceDynamicOp>(
        srcOp, outputType, adaptor.getOperand(), startIndicesTensor,
        endIndicesTensor, ArrayAttr());

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

    rewriter.replaceOpWithNewOp<ttir::ClampTensorOp>(
        srcOp, outputType, adaptor.getOperand(), adaptor.getMin(),
        adaptor.getMax());
    return success();
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

    rewriter.replaceOpWithNewOp<ttir::GatherOp>(
        srcOp, outputType, adaptor.getOperands()[0], adaptor.getOperands()[1],
        dimensionNumbers.getOffsetDims(),
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
class CacheFillUpdatePattern
    : public OpConversionPattern<mlir::stablehlo::ScatterOp> {

  using OpConversionPattern<mlir::stablehlo::ScatterOp>::OpConversionPattern;

public:
  // Use higher benefit to ensure this pattern is tried before generic scatter.
  CacheFillUpdatePattern(TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::ScatterOp>(typeConverter, context,
                                                        /*benefit=*/2) {}
  /// Pattern: scatter(input, indices, updates)
  ///
  /// This pattern detects when a ScatterOp is used as a fill/update for a
  /// cache. We check for its input, indices, and update tensors to ensure they
  /// match the expected cache fill/update pattern.
  ///
  /// Input pattern:
  ///   %result = scatter(%cache, %indices, %updates)
  ///   - Given a cache with shape (B, N, M, H) and a updates tensor with shape
  ///   (B, N, S, H), the indices tensor represents the index where each element
  ///   in %updates should placed in the %cache.
  ///   - %indices can be tracked back to the function's cachePositions input
  ///   that represents the indices of the cache to fill/update.
  /// Output pattern:
  ///   %result = fillCacheOp(%cache, %updates)
  ///   or (if S == 1)
  ///   %result = updateCacheOp(%cache, %updates, %update_index)
  mlir::LogicalResult
  matchAndRewrite(mlir::stablehlo::ScatterOp scatterOp,
                  mlir::stablehlo::ScatterOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto CachePositions = getCacheUpdatePositions(scatterOp);
    if (!CachePositions) {
      return mlir::failure();
    }

    auto cacheUpdateInputType =
        mlir::cast<RankedTensorType>((*CachePositions).getType());
    auto cacheUpdateInputShape = cacheUpdateInputType.getShape();
    if (cacheUpdateInputShape.size() != 1) {
      return mlir::failure();
    }

    Value cache = scatterOp.getInputs()[0];
    Value updates = scatterOp.getUpdates()[0];
    int32_t batchSize =
        mlir::cast<RankedTensorType>(cache.getType()).getShape()[0];

    RankedTensorType updatesType =
        mlir::cast<RankedTensorType>(updates.getType());

    // If the cachePositions tensor has more than one element we assume it
    // represents a set of arranged indices (0, cachePositions.size), so we
    // replace it with FillCacheOp. If the tensor has only one element, we
    // assume it represents the update index for UpateCacheOp.
    if (cacheUpdateInputShape[0] != 1) {
      // Fill cache requires that each batch is filled separately. So, we will
      // insert a FillCacheOp for each batch. This requires slicing out each
      // batch.
      if (batchSize > 1) {

        for (int32_t batchOffset = 0; batchOffset < batchSize; batchOffset++) {
          auto batchOffsetAttr = rewriter.getI32IntegerAttr(batchOffset);

          // Slice starts at the batch offset for the batch dim, and starts at 0
          // for all other dims.
          SmallVector<int32_t> sliceStarts = {batchOffset, 0, 0, 0};

          // Slice ends at the dim size for every dim, except the batch dim
          // where the slice ends at batch offset + 1.
          SmallVector<int32_t> sliceEnds = SmallVector<int32_t>(
              updatesType.getShape().begin(), updatesType.getShape().end());
          sliceEnds[0] = batchOffset + 1;

          // Slice steps is 1 for every dim as we do not wish to skip any
          SmallVector<int32_t> sliceSteps = {1, 1, 1, 1};

          // Slice output shape is the same as the fill value shape, except the
          // batch dim is 1 since we sliced out a single batch.
          SmallVector<int64_t> sliceOutputShape(updatesType.getShape());
          sliceOutputShape[0] = 1;

          // Encoding should not be set when this pass is run. Guard against it.
          assert(!updatesType.getEncoding() &&
                 "Encoding should not be set when this pass is run");

          RankedTensorType slicedUpdatesType = RankedTensorType::get(
              sliceOutputShape, updatesType.getElementType(), nullptr);

          // Create slice op.
          auto slicedUpdates = rewriter.create<ttir::SliceStaticOp>(
              scatterOp.getLoc(), slicedUpdatesType, updates,
              rewriter.getI32ArrayAttr(sliceStarts),
              rewriter.getI32ArrayAttr(sliceEnds),
              rewriter.getI32ArrayAttr(sliceSteps));
          // create fill cache op for this batch.
          cache = rewriter.create<mlir::tt::ttir::FillCacheOp>(
              scatterOp.getLoc(),
              scatterOp.getResult(0).getType(), // Result type
              cache,                            // Cache tensor
              slicedUpdates,                    // Updates tensor
              batchOffsetAttr                   // Batch offset
          );
        }
      } else {
        cache = rewriter.create<mlir::tt::ttir::FillCacheOp>(
            scatterOp.getLoc(), scatterOp.getResult(0).getType(), // Result type
            cache,   // Cache tensor
            updates, // Updates tensor
            0        // Batch offset
        );
      }
    } else {
      // Unlike ttnn.fill_cache, we can perform ttnn.update_cache on the entire
      // batch at once. However this requires that the fill value is in the form
      // [1, num_heads, B, head_size]. So, we must permute the updates tensor to
      // this shape.
      if (batchSize > 1) {
        SmallVector<int64_t> permutedShape = ttmlir::utils::applyPermutation(
            updatesType.getShape(), {2, 1, 0, 3});

        // Encoding should not be set when this pass is run. Guard against it.
        assert(!updatesType.getEncoding() &&
               "Encoding should not be set when this pass is run");
        RankedTensorType permutedUpdatesType = RankedTensorType::get(
            permutedShape, updatesType.getElementType(), nullptr);
        updates = rewriter.create<ttir::PermuteOp>(
            scatterOp.getLoc(), permutedUpdatesType, updates,
            rewriter.getDenseI64ArrayAttr({2, 1, 0, 3}));
      }
      cache = rewriter.create<mlir::tt::ttir::UpdateCacheOp>(
          scatterOp.getLoc(),
          scatterOp.getResult(0).getType(), // Result type
          cache,                            // Cache tensor
          updates,                          // Updates tensor
          *CachePositions,                  // Cache Idx
          0                                 // Batch offset
      );
    }

    rewriter.replaceOp(scatterOp, cache);

    return mlir::success();
  }

private:
  // Check if the scatter op is a cache fill/update, and track the
  // cachePositions input tensor if it is.
  //
  // We are looking for:
  // %result = "stablehlo.scatter"(%cache, %indices, %updates)
  // Where:
  //    1. %cache and %updates are 4D tensors who's shape match except on the
  //    3rd dimension,
  //       (B, N, M, H) and (B, N, S, H) respectively, M being the max cache
  //       length and S being the sequence length of the update.
  //    2. %indices comes from a block argument representing the cachePositions
  //    tensor.
  static std::optional<mlir::Value>
  getCacheUpdatePositions(mlir::stablehlo::ScatterOp scatterOp) {
    // Check that the scatter op inputs represent a cache fill/update:
    //    1. The input is a 4D (B, N, M, H)
    //    2. The update tensor is a 4D tensor (B, N, S, H)
    //    3. The scatter indices is either a 1D equivalent tensor or 5D index
    //       grid tensor (B, N, S, H, 4). Both can be tracked to a block
    //       argument representing the cachePositions input.
    auto scatterIndices = scatterOp.getScatterIndices();
    ArrayRef<int64_t> inputShape =
        mlir::cast<RankedTensorType>(scatterOp.getInputs()[0].getType())
            .getShape();
    ArrayRef<int64_t> scatterIdxShape =
        mlir::cast<RankedTensorType>(scatterIndices.getType()).getShape();
    ArrayRef<int64_t> updateShape =
        mlir::cast<RankedTensorType>(scatterOp.getUpdates()[0].getType())
            .getShape();
    if (inputShape.size() != 4 || updateShape.size() != 4) {
      return std::nullopt;
    }

    if (!(inputShape[0] == updateShape[0] && inputShape[1] == updateShape[1] &&
          inputShape[3] == updateShape[3])) {
      return std::nullopt;
    }

    int cacheUpdateSize = updateShape[2];

    bool effectively1D = isEffectively1D(scatterIdxShape);
    if (effectively1D &&
        ttmlir::utils::volume(scatterIdxShape) != cacheUpdateSize) {
      return std::nullopt;
    }

    bool isIndexGrid =
        (scatterIdxShape.size() == 5 && scatterIdxShape[0] == inputShape[0] &&
         scatterIdxShape[1] == inputShape[1] &&
         scatterIdxShape[2] == cacheUpdateSize &&
         scatterIdxShape[3] == inputShape[3] && scatterIdxShape[4] == 4);

    // Check that scatter indices is either a 1D cache positions tensor or a 5D
    // index grid.
    if (!effectively1D && !isIndexGrid) {
      return std::nullopt;
    }

    // The cachePositions tensor is expected to be a 1D blockargument tensor
    // with the same size as the cache update size.
    auto useDefChain = ttmlir::utils::getUseDefChain(scatterIndices);
    auto blockArgs =
        ttmlir::utils::filterBlockArguments(useDefChain.getArrayRef());
    for (auto blockArg : blockArgs) {
      // Check if the block argument is a cachePositions input.
      auto argTensorShape =
          mlir::cast<RankedTensorType>(blockArg.getType()).getShape();
      effectively1D = isEffectively1D(argTensorShape);
      if (!effectively1D) {
        continue;
      }
      if (ttmlir::utils::volume(argTensorShape) == cacheUpdateSize) {
        // We found the cachePositions input tensor.
        return blockArg;
      }
    }

    return std::nullopt;
  }

  static bool isEffectively1D(ArrayRef<int64_t> shape) {
    return llvm::count_if(shape, [](int64_t dim) { return dim != 1; }) <= 1;
  }
};
} // namespace

namespace {
class StableHLOToTTIREmbeddingBackwardOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ScatterOp> {

  using OpConversionPattern<mlir::stablehlo::ScatterOp>::OpConversionPattern;

public:
  StableHLOToTTIREmbeddingBackwardOpConversionPattern(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<mlir::stablehlo::ScatterOp>(typeConverter, ctx,
                                                        /*benefit=*/2) {}

  // Benefit is 2 to ensure this pattern is tried before generic scatter.
  // This is because embedding_backward is a more specific pattern than scatter.
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ScatterOp srcOp,
                  mlir::stablehlo::ScatterOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check if this scatter can be converted to embedding_backward.
    // embedding_backward requires:
    // 1. Add reduction (for gradient accumulation)
    // 2. Specific dimension configuration matching embedding semantics

    llvm::ErrorOr<ttcore::ReduceType> reduceType =
        getReduceTypeFromRegion(srcOp.getRegion());
    if (!reduceType || *reduceType != ttcore::ReduceType::Sum) {
      return rewriter.notifyMatchFailure(
          srcOp,
          "EmbeddingBackward requires sum reduction in update computation");
    }

    assert(adaptor.getInputs().size() == 1 &&
           "EmbeddingBackward requires 1 inputs");
    assert(adaptor.getUpdates().size() == 1 &&
           "EmbeddingBackward requires 1 update");

    Value operand = adaptor.getInputs()[0];
    Value scatterIndices = adaptor.getScatterIndices();
    Value update = adaptor.getUpdates()[0];

    auto operandType = mlir::cast<RankedTensorType>(operand.getType());
    auto updateType = mlir::cast<RankedTensorType>(update.getType());

    if (operandType.getRank() != 2) {
      return rewriter.notifyMatchFailure(
          srcOp, "EmbeddingBackward requires 2D weight table (operand)");
    }

    auto scatterDimsToOperandDims =
        adaptor.getScatterDimensionNumbers().getScatterDimsToOperandDims();
    auto insertedWindowDims =
        adaptor.getScatterDimensionNumbers().getInsertedWindowDims();

    // embedding_backward requires:
    // 1. scatterDimsToOperandDims = [0] (scatter into first dim)
    // 2. insertedWindowDims = [0] (first dim is scatter dim)
    // 3. updateWindowDims should cover the embedding dimension
    if (scatterDimsToOperandDims.size() != 1 ||
        scatterDimsToOperandDims[0] != 0) {
      return rewriter.notifyMatchFailure(
          srcOp,
          "EmbeddingBackward requires scattering into first dimension only");
    }

    if (insertedWindowDims.size() != 1 || insertedWindowDims[0] != 0) {
      return rewriter.notifyMatchFailure(
          srcOp, "EmbeddingBackward requires first dimension as scatter dim");
    }

    int64_t embeddingDim = operandType.getDimSize(1);
    int64_t updateLastDim = updateType.getDimSize(updateType.getRank() - 1);
    if (embeddingDim != updateLastDim) {
      return rewriter.notifyMatchFailure(
          srcOp, "EmbeddingBackward update tensor last dimension must match "
                 "embedding dimension");
    }

    assert(srcOp.getResults().size() == 1 &&
           "EmbeddingBackward requires 1 result");
    auto outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResults()[0].getType()));

    auto scatterIndicesType =
        mlir::cast<RankedTensorType>(scatterIndices.getType());

    int64_t indexVectorDim =
        adaptor.getScatterDimensionNumbers().getIndexVectorDim();

    // StableHLO uses vectorized indices for scatter.
    // TT-metal expects indices to be [B, N] for embedding_backward. If the
    // indices are 2D and the index vector dim is 1, reshape the indices to 3D.
    if (scatterIndicesType.getRank() == 2 && indexVectorDim == 1) {
      llvm::SmallVector<int64_t> newShape{1};
      llvm::append_range(newShape, scatterIndicesType.getShape());
      scatterIndices = ttir::utils::createReshapeOp(rewriter, srcOp.getLoc(),
                                                    scatterIndices, newShape);
    }

    rewriter.replaceOpWithNewOp<ttir::EmbeddingBackwardOp>(
        srcOp, outputType, scatterIndices, operand, update);

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

    if (LogicalResult result = checkBasicLegality(srcOp, adaptor, rewriter);
        !result.succeeded()) {
      return result;
    }

    // Convert reduceType stablehlo attribute into ttir attribute.
    llvm::ErrorOr<ttcore::ReduceType> scatterReduceType = getReduceType(srcOp);
    if (!scatterReduceType) {
      return rewriter.notifyMatchFailure(
          srcOp, "ScatterOp cannot specify reduce type.");
    }

    Value inputTensor = srcOp.getInputs()[0];
    Value updateTensor = srcOp.getUpdates()[0];
    auto scatterDimsToOperandDims =
        adaptor.getScatterDimensionNumbers().getScatterDimsToOperandDims();
    RankedTensorType inputType =
        mlir::cast<RankedTensorType>(inputTensor.getType());
    ArrayRef<int64_t> inputShape = inputType.getShape();
    RankedTensorType outputType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(srcOp.getResults()[0].getType()));
    RankedTensorType updateType =
        mlir::cast<RankedTensorType>(updateTensor.getType());
    ArrayRef<int64_t> updateShape = updateType.getShape();

    // Single-dimensional scatter.
    if (scatterDimsToOperandDims.size() == 1) {
      // Process indices to match update tensor shape.
      int32_t dim = scatterDimsToOperandDims[0];
      Value finalIndexTensor =
          extractElementWiseScatterIndices(srcOp, rewriter);

      // Create ScatterOp.
      rewriter.replaceOpWithNewOp<ttir::ScatterOp>(
          srcOp, outputType, inputTensor, finalIndexTensor, updateTensor,
          rewriter.getI32IntegerAttr(dim),
          ttcore::ReduceTypeAttr::get(rewriter.getContext(),
                                      *scatterReduceType));
      return success();
    }

    // Multi-dimensional scatter.
    if (scatterDimsToOperandDims.size() > 1) {
      // Always scatter along dimension 0 for flattened tensors.
      constexpr int32_t SCATTER_DIMENSION = 0;

      // Scatter indices, input, and update tensors flattened to 1D.
      Value flattenedIndices = flattenMultiDimScatterIndices(
          srcOp, inputShape, updateShape, rewriter);
      Value flattenedInput = ttir::utils::flattenTensor(
          rewriter, srcOp.getLoc(), inputTensor, "_input_flatten");
      Value flattenedUpdate = ttir::utils::flattenTensor(
          rewriter, srcOp.getLoc(), updateTensor, "_update_flatten");

      // Scatter scalars on flattened tensors.
      Value scatterResult = rewriter.create<ttir::ScatterOp>(
          srcOp.getLoc(),
          mlir::cast<RankedTensorType>(flattenedInput.getType()),
          flattenedInput, flattenedIndices, flattenedUpdate,
          rewriter.getI32IntegerAttr(SCATTER_DIMENSION),
          ttcore::ReduceTypeAttr::get(rewriter.getContext(),
                                      *scatterReduceType));

      // Reshape result back to original input shape.
      Value reshapedResult =
          ttir::utils::createReshapeOp(rewriter,
                                       ttmlir::utils::appendLocationSuffix(
                                           srcOp.getLoc(), "_result_reshape"),
                                       scatterResult, inputShape);

      rewriter.replaceOp(srcOp, reshapedResult);
      return success();
    }

    return failure();
  }

private:
  LogicalResult checkBasicLegality(mlir::stablehlo::ScatterOp &op,
                                   mlir::stablehlo::ScatterOp::Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
    auto inputBatchingDims =
        adaptor.getScatterDimensionNumbers().getInputBatchingDims();
    auto scatterIndicesBatchingDims =
        adaptor.getScatterDimensionNumbers().getScatterIndicesBatchingDims();
    if (!inputBatchingDims.empty() || !scatterIndicesBatchingDims.empty()) {
      return rewriter.notifyMatchFailure(
          op, "Scatter doesn't currently support scatter with batching "
              "dimensions");
    }

    ArrayRef<int64_t> insertedWindowDims =
        adaptor.getScatterDimensionNumbers().getInsertedWindowDims();
    RankedTensorType updateType =
        mlir::cast<RankedTensorType>(op.getUpdates()[0].getType());
    ArrayRef<int64_t> updateShape = updateType.getShape();

    // Get index tensor shape.
    RankedTensorType indexType = op.getScatterIndices().getType();
    ArrayRef<int64_t> indexShape = indexType.getShape();

    // Check that scatter_dims_to_operand_dims is in order.
    ArrayRef<int64_t> scatterDimsToOperandDims =
        adaptor.getScatterDimensionNumbers().getScatterDimsToOperandDims();
    if (!llvm::is_sorted(scatterDimsToOperandDims)) {
      return rewriter.notifyMatchFailure(
          op,
          "scatter_dims_to_operand_dims must be in strictly increasing order.");
    }

    bool multiDimensionalScatter = scatterDimsToOperandDims.size() > 1;
    uint32_t indexVectorDim =
        adaptor.getScatterDimensionNumbers().getIndexVectorDim();

    // Checks that apply to multi dimensional scatter.

    if (multiDimensionalScatter &&
        indexVectorDim != static_cast<uint32_t>(indexShape.size() - 1)) {
      return rewriter.notifyMatchFailure(
          op, "TTIR multi-dimensional scatter currently only supports "
              "index_vector_dim being the last dimension");
    }

    if (multiDimensionalScatter &&
        llvm::DenseSet<int64_t>(scatterDimsToOperandDims.begin(),
                                scatterDimsToOperandDims.end()) !=
            llvm::DenseSet<int64_t>(insertedWindowDims.begin(),
                                    insertedWindowDims.end())) {
      return rewriter.notifyMatchFailure(
          op, "TTIR multi-dimensional scatter requires "
              "scatter_dims_to_operand_dims and inserted_window_dims to "
              "contain the same elements");
    }

    // Checks that apply to single dimensional scatter.

    if (!multiDimensionalScatter && indexShape.size() > updateShape.size()) {
      return rewriter.notifyMatchFailure(
          op, "TTIR scatter requires indices.rank <= updates.rank. Please add "
              "support for rank promotion if needed.");
    }

    if (!multiDimensionalScatter && indexVectorDim != 1u) {
      return rewriter.notifyMatchFailure(
          op,
          "TTIR single dimensional scatter requires index_vector_dim to be 1");
    }

    if (!multiDimensionalScatter && scatterDimsToOperandDims[0] != 0) {
      return rewriter.notifyMatchFailure(
          op, "TTIR single dimensional scatter currently only supports "
              "scattering along dimension 0");
    }
    return success();
  }

  /// Computes flat 1D indices from per-dimension index slices using strides.
  ///
  /// Takes a vector of index tensors (one per operand dimension) and computes:
  ///   flat_index = sum(indices[d] * stride[d])
  /// where stride[d] = product of operandShape[d+1..operandRank-1].
  Value
  computeFlatIndicesFromSlices(Location loc, PatternRewriter &rewriter,
                               const llvm::SmallVector<Value> &indexSlices,
                               ArrayRef<int64_t> operandShape,
                               Type indexElementType) const {
    int64_t operandRank = static_cast<int64_t>(operandShape.size());

    // Calculate strides for each dimension.
    // stride[d] = product of operandShape[d+1..operandRank-1]
    llvm::SmallVector<int64_t> strides(operandRank);
    int64_t stride = 1;
    for (int64_t d = operandRank - 1; d >= 0; --d) {
      strides[d] = stride;
      stride *= operandShape[d];
    }

    Value flatIndices = nullptr;

    for (int64_t d = 0; d < operandRank; ++d) {
      Value dimIndices = indexSlices[d];
      RankedTensorType dimIndicesType =
          mlir::cast<RankedTensorType>(dimIndices.getType());
      ArrayRef<int64_t> dimIndicesShape = dimIndicesType.getShape();

      // Multiply by stride if stride > 1.
      if (strides[d] > 1) {
        auto scalarAttr =
            rewriter.getI32IntegerAttr(static_cast<int32_t>(strides[d]));

        RankedTensorType strideType = RankedTensorType::get(
            dimIndicesShape, indexElementType, dimIndicesType.getEncoding());

        Value strideTensor = rewriter.create<ttir::FullOp>(
            ttmlir::utils::appendLocationSuffix(loc,
                                                "_stride_" + std::to_string(d)),
            strideType, scalarAttr);

        dimIndices = rewriter.create<ttir::MultiplyOp>(
            ttmlir::utils::appendLocationSuffix(
                loc, "_dim_" + std::to_string(d) + "_stride_mul"),
            strideType, dimIndices, strideTensor);
      }

      // Accumulate into flatIndices.
      if (flatIndices == nullptr) {
        flatIndices = dimIndices;
      } else {
        RankedTensorType addType = RankedTensorType::get(
            dimIndicesShape, indexElementType, dimIndicesType.getEncoding());

        flatIndices = rewriter.create<ttir::AddOp>(
            ttmlir::utils::appendLocationSuffix(loc, "_add_dim_" +
                                                         std::to_string(d)),
            addType, flatIndices, dimIndices);
      }
    }

    return flatIndices;
  }

  /// Computes flattened 1D indices for multi-dimensional scatter.
  ///
  /// Handles both cases:
  /// - Non-empty updateWindowDims: expands indices for window positions,
  ///   converting each scatter of a window into multiple scalar scatters.
  /// - Empty updateWindowDims: degenerates to simple case (windowSize=1).
  ///
  /// Before: 1 index scatters a window of N values
  /// After:  N indices scatter N scalar values
  ///
  /// Example with window expansion:
  /// Before:
  ///   operand:  [[[a, b], [c, d], [e, f], [g, h], [i, j]]] shape (1, 5, 2)
  ///   indices:  [[0, 3]]                                   shape (1, 2)
  ///   updates:  [[x, y]]                                   shape (1, 2)
  ///   update_window_dims = [1]
  //    inserted_window_dims = [0, 1],
  ///   index_vector_dim = 1
  /// After:
  ///   flattened indices: [6, 7], shape (2) - positions in flattened operand
  ///
  /// Returns flattened 1D indices tensor.
  Value flattenMultiDimScatterIndices(mlir::stablehlo::ScatterOp op,
                                      ArrayRef<int64_t> operandShape,
                                      ArrayRef<int64_t> updateShape,
                                      PatternRewriter &rewriter) const {
    Value indices = op.getScatterIndices();
    RankedTensorType indicesType =
        mlir::cast<RankedTensorType>(indices.getType());
    ArrayRef<int64_t> indicesShape = indicesType.getShape();
    auto scatterDimNumbers = op.getScatterDimensionNumbers();
    ArrayRef<int64_t> insertedWindowDims =
        scatterDimNumbers.getInsertedWindowDims();
    ArrayRef<int64_t> updateWindowDims =
        scatterDimNumbers.getUpdateWindowDims();
    int64_t indexVectorDim = scatterDimNumbers.getIndexVectorDim();
    int64_t operandRank = operandShape.size();
    Location loc = op.getLoc();

    // Build set of inserted window dims for fast lookup.
    llvm::DenseSet<int64_t> insertedDimsSet(insertedWindowDims.begin(),
                                            insertedWindowDims.end());

    // Calculate window shape from update dimensions at update_window_dims.
    // The window size is determined by the update tensor.
    llvm::SmallVector<int64_t> windowShape;
    for (int64_t dim : updateWindowDims) {
      windowShape.push_back(updateShape[dim]);
    }

    // Calculate total window size (product of window dimensions).
    int64_t windowSize = 1;
    for (int64_t size : windowShape) {
      windowSize *= size;
    }

    // Calculate number of scatter positions from indices tensor.
    // All dimensions except index_vector_dim contribute to scatter positions.
    int64_t numScatterPositions = 1;
    for (int64_t d = 0; d < static_cast<int64_t>(indicesShape.size()); ++d) {
      if (d != indexVectorDim) {
        numScatterPositions *= indicesShape[d];
      }
    }

    // Total number of expanded indices (one per one scalar update).
    int64_t expandedNumIndices = numScatterPositions * windowSize;

    // First, reshape the original indices to shape
    // [numScatterPositions, originalIndexSize].
    int64_t originalIndexSize = indicesShape[indexVectorDim];
    Value reshapedOrigIndices = ttir::utils::createReshapeOp(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_reshape_indices"),
        indices, {numScatterPositions, originalIndexSize});

    // Reshape to [numScatterPositions, 1, originalIndexSize] for repeat.
    Value reshapedForRepeat = ttir::utils::createReshapeOp(
        rewriter,
        ttmlir::utils::appendLocationSuffix(loc, "_reshape_for_repeat"),
        reshapedOrigIndices, {numScatterPositions, 1, originalIndexSize});

    // Repeat along dim 1 to get [numScatterPositions, windowSize,
    // originalIndexSize].
    llvm::SmallVector<int64_t> afterRepeatShape = {
        numScatterPositions, windowSize, originalIndexSize};
    RankedTensorType afterRepeatType =
        RankedTensorType::get(afterRepeatShape, indicesType.getElementType());
    Value repeatedIndices = rewriter.create<ttir::RepeatOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_repeat_indices"),
        afterRepeatType, reshapedForRepeat,
        rewriter.getDenseI64ArrayAttr({1, windowSize, 1}));

    // Flatten to [numScatterPositions * windowSize, originalIndexSize].
    Value flatRepeatedIndices = ttir::utils::createReshapeOp(
        rewriter,
        ttmlir::utils::appendLocationSuffix(loc, "_flatten_repeated_indices"),
        repeatedIndices, {expandedNumIndices, originalIndexSize});

    // Generate window offset coordinates for each window dimension, and create
    // a single ConstantOp per dimension.
    //
    // For window shape [W1, W2, ...], we compute the coordinate for each
    // position in the flattened window. For position i in the window:
    //   coord[d] = (i / product(windowShape[d+1:])) % windowShape[d]
    //
    // These coordinates are later repeated for each scatter position.
    llvm::SmallVector<Value> windowOffsetSlices;

    // Compute window offsets for each dimension in memory.
    // windowOffsets[d][i] = coordinate of window position i in dimension d.
    llvm::SmallVector<llvm::SmallVector<int64_t>> windowOffsets(
        windowShape.size());
    for (size_t d = 0; d < windowShape.size(); ++d) {
      windowOffsets[d].reserve(windowSize);
    }

    // For each position in the flattened window, compute multi-dimensional
    // coordinates. For dimension d: coord = (i / stride) % windowShape[d],
    // where stride = product of windowShape[d+1:].
    for (int64_t i = 0; i < windowSize; ++i) {
      int64_t stride = 1;
      for (size_t d = windowShape.size(); d-- > 0;) {
        windowOffsets[d].push_back((i / stride) % windowShape[d]);
        stride *= windowShape[d]; // repeats the same seq during each outer loop
      }
    }

    // Create constant tensors for each window dimension's offsets.
    Type indexElementType = indicesType.getElementType();
    for (size_t dim = 0; dim < windowShape.size(); ++dim) {
      // Build the final offset values: repeat window offsets for each scatter
      // position. Pattern: [w0, w1, ..., wN-1, w0, w1, ..., wN-1, ...]
      //                     |---- window ---|  |--- window ----|
      //                    |--------- numScatterPositions times -----|
      llvm::SmallVector<int64_t> finalOffsetValues;
      finalOffsetValues.reserve(expandedNumIndices);
      for (int64_t i = 0; i < numScatterPositions; ++i) {
        for (int64_t w = 0; w < windowSize; ++w) {
          finalOffsetValues.push_back(windowOffsets[dim][w]);
        }
      }

      // Create constant tensor with shape [expandedNumIndices, 1].
      // (to match the shape of individual slices generated in the next
      // step of the algorithm, since they are all pushed to indexSlices)
      RankedTensorType finalOffsetType =
          RankedTensorType::get({expandedNumIndices, 1}, indexElementType);
      auto finalOffsetAttr =
          DenseIntElementsAttr::get(finalOffsetType, finalOffsetValues);
      Value finalOffset = rewriter.create<ttir::ConstantOp>(
          ttmlir::utils::appendLocationSuffix(loc, "_window_offset_" +
                                                       std::to_string(dim)),
          finalOffsetType, finalOffsetAttr);

      windowOffsetSlices.push_back(finalOffset);
    }

    // Build per-dimension index slices by interleaving original indices
    // and window offsets according to operand dimension order.
    llvm::SmallVector<Value> indexSlices;
    size_t origIdxPos = 0;
    size_t windowIdxPos = 0;
    for (int64_t operandDim = 0; operandDim < operandRank; ++operandDim) {
      if (insertedDimsSet.contains(operandDim)) {
        // This dimension is indexed by scatter - slice from original indices.
        llvm::SmallVector<int32_t> begins = {0,
                                             static_cast<int32_t>(origIdxPos)};
        llvm::SmallVector<int32_t> ends = {
            static_cast<int32_t>(expandedNumIndices),
            static_cast<int32_t>(origIdxPos + 1)};
        llvm::SmallVector<int32_t> steps = {1, 1};

        llvm::SmallVector<int64_t> sliceShape = {expandedNumIndices, 1};
        RankedTensorType sliceType =
            RankedTensorType::get(sliceShape, indicesType.getElementType());

        Value sliced = rewriter.create<ttir::SliceStaticOp>(
            ttmlir::utils::appendLocationSuffix(
                loc, "_slice_orig_idx_" + std::to_string(operandDim)),
            sliceType, flatRepeatedIndices, rewriter.getI32ArrayAttr(begins),
            rewriter.getI32ArrayAttr(ends), rewriter.getI32ArrayAttr(steps));

        indexSlices.push_back(sliced);
        ++origIdxPos;
      } else {
        // This dimension is a window dimension - use window offset.
        indexSlices.push_back(windowOffsetSlices[windowIdxPos]);
        ++windowIdxPos;
      }
    }

    Value flatIndices = computeFlatIndicesFromSlices(
        loc, rewriter, indexSlices, operandShape, indicesType.getElementType());

    // Flatten the computed indices to 1D.
    return ttir::utils::flattenTensor(rewriter, loc, flatIndices,
                                      "_flatten_expanded_indices");
  }

  Value extractElementWiseScatterIndices(mlir::stablehlo::ScatterOp op,
                                         PatternRewriter &rewriter) const {
    // Indices need to match updates tensor.
    TypedValue<RankedTensorType> indexTensor = op.getScatterIndices();
    RankedTensorType updateType =
        mlir::cast<RankedTensorType>(op.getUpdates()[0].getType());
    RankedTensorType indexType = indexTensor.getType();
    llvm::SmallVector<int64_t> indexShape(indexType.getShape());
    ArrayRef<int64_t> updateShape = updateType.getShape();

    if (indexShape.size() < updateShape.size()) {
      // Need to reshape indices by appending 1s to the shape.
      llvm::SmallVector<int64_t> newShape(indexShape.begin(), indexShape.end());
      newShape.resize(updateShape.size(), 1);

      indexTensor = ttir::utils::createReshapeOp(rewriter, op.getLoc(),
                                                 indexTensor, newShape);
      indexType = mlir::cast<RankedTensorType>(indexTensor.getType());
      indexShape = newShape;
    }

    // Repeat along update_window_dims to match update tensor shape.
    ArrayRef<int64_t> updateWindowDims =
        op.getScatterDimensionNumbers().getUpdateWindowDims();
    llvm::SmallVector<int64_t> repeatDims(indexShape.size(), 1);
    bool needsRepeat = false;

    // For each update_window_dim, set repeat factor to match update tensor
    // size.
    for (auto dimAttr : updateWindowDims) {
      int64_t dim = dimAttr;
      if (indexShape[dim] != updateShape[dim]) {
        repeatDims[dim] = updateShape[dim];
        needsRepeat = true;
      }
    }

    if (needsRepeat) {
      llvm::SmallVector<int64_t> targetIndexShape(updateShape.begin(),
                                                  updateShape.end());
      RankedTensorType targetIndexType =
          RankedTensorType::get(targetIndexShape, indexType.getElementType(),
                                indexType.getEncoding());
      auto repeatDimsAttr = rewriter.getDenseI64ArrayAttr(repeatDims);

      indexTensor = rewriter.create<ttir::RepeatOp>(
          op.getLoc(), targetIndexType, indexTensor, repeatDimsAttr);
    }

    return indexTensor;
  }
};
} // namespace

namespace {
// Conversion: stablehlo::SortOp → ttir::SortOp + optional ttir::GatherOp(s)
//
// StableHLO's SortOp supports sorting tuples of tensors with an arbitrary
// comparator function. This pattern lowers such SortOps into TTIR by
// decomposing them into one or more of the following:
//   - ttir::SortOp: handles sorting a single tensor and producing sorted tensor
//                   along with sort indices.
//   - ttir::GatherOp: reorders other tensors based on the computed indices.
//
// This conversion supports three types of SortOps:
//
// [1] ValueOnly Sort:
//     - Only one input (e.g., SortOp(values))
//     - Lowered to ttir::SortOp producing sorted values
//     - indices output ignored.
//     - No gather needed.
//
// [2] ValueIndex Sort:
//     - Two inputs: a value tensor and an index tensor.
//     - If the second input is a recognized as stablehlo::iota (possibly after
//       reshape/broadcast), the pattern assumes it's requesting both sorted
//       values and their original indices.
//     - Lowered to ttir::SortOp producing both values and indices
//     - No gather needed.
//
// [3] KeyValue Sort:
//     - Two inputs where the second input is not recognized as iota or
//       more than two inputs.
//     - Only the first input is directly sorted.
//     - The resulting indices are used to reorder all other inputs via
//       ttir::GatherOp
//     - This emulates tuple sorting (e.g., SortOp(keys, values, ...)) by
//       aligning all value tensors with the sorted indices of the key.
//
// Key implementation steps:
// - Determine SortType (ValueOnly, ValueIndex, or KeyValue) based on input
//   count and type.
// - Emit ttir::SortOp using only the first input tensor.
// - If needed, emit one or more ttir::GatherOps to reorder the rest of the
//   inputs.
// - Replace the original stablehlo::SortOp with the results of the new
//   operations.
// - TTIR GatherOp is based on StableHLO GatherOp which requires full
//   multi-dimensional index tuples for each index into the input. This is
//   generated using iota (ArangeOp) tensors for static dimensions and using
//   ConcatOp to combine them with the index tensor.
class StableHLOToTTIRSortOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::SortOp> {
  using OpConversionPattern::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::SortOp srcOp,
                  mlir::stablehlo::SortOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = srcOp.getLoc();
    // Get sorting metadata.
    int64_t sortDim = srcOp.getDimension();
    bool isStable = srcOp.getIsStable();
    mlir::FailureOr<bool> isDescending = getSortDirection(srcOp);
    if (mlir::failed(isDescending)) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "Cannot determine sort direction");
    }

    // Step 1: Type conversion and output tensor preparation for 'values'.

    auto valueType = cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResultTypes().front()));
    SmallVector<Type> outputTypes{valueType};

    // Step 2: Determine Sort Type and output tensor preparation for 'indices'.

    SortType sortType;
    if (isValueIndexSort(srcOp.getInputs())) {
      sortType = SortType::kValueIndex;

      auto indicesType = cast<RankedTensorType>(
          getTypeConverter()->convertType(srcOp.getResultTypes()[1]));

      outputTypes.push_back(indicesType);
    } else {
      sortType = (srcOp.getInputs().size() == 1) ? SortType::kValueOnly
                                                 : SortType::kKeyValue;

      IntegerType indexType = rewriter.getI32Type();
      RankedTensorType indicesType = RankedTensorType::get(
          valueType.getShape(), indexType, valueType.getEncoding());

      outputTypes.push_back(indicesType);
    }

    // Step 3: Emit SortOp.

    auto sortOp = rewriter.create<ttir::SortOp>(
        loc, outputTypes, adaptor.getInputs().front(),
        rewriter.getSI32IntegerAttr(sortDim),
        rewriter.getBoolAttr(*isDescending), rewriter.getBoolAttr(isStable));

    // Step 4: SortType-specific lowering.

    // SortType::kValueOnly - Replace values output and ignore indices output.
    if (sortType == SortType::kValueOnly) {
      rewriter.replaceOp(srcOp, sortOp.getValues());
      return success();
    }

    // SortType::kValueIndex - Replace the source op with values/indices output.
    if (sortType == SortType::kValueIndex) {
      rewriter.replaceOp(srcOp, sortOp->getResults());
      return success();
    }

    // SortType::kKeyValue — sort additional inputs using indices and GatherOp.
    Value indices = sortOp.getIndices();
    auto indicesType = cast<RankedTensorType>(indices.getType());
    int64_t rank = indicesType.getRank();

    // TTIR GatherOp is based on StableHLO GatherOp which requires full
    // multi-dimensional index tuples for each index into the input. This is
    // generated using iota (ArangeOp) tensors for static dimensions and using
    // ConcatOp to combine them with the index tensor.
    SmallVector<int64_t> shape(indicesType.getShape());
    shape.push_back(1);
    auto expandedType = RankedTensorType::get(
        shape, indicesType.getElementType(), indicesType.getEncoding());

    // Reshape indices to [*shape, 1]
    SmallVector<int32_t> reshapeDim(shape.begin(), shape.end());
    auto reshape = rewriter.create<ttir::ReshapeOp>(
        loc, expandedType, indices, rewriter.getI32ArrayAttr(reshapeDim));

    // Generate iota-based index components (for all dims except sorting dim).
    // Sorted indices is used for sorting dim.
    SmallVector<Value> toConcat;
    for (int64_t idx = 0; idx < rank; ++idx) {
      if (idx == sortDim) {
        toConcat.push_back(reshape);
        continue;
      }

      Value arangeOp = rewriter.create<ttir::ArangeOp>(
          loc, expandedType, /*start=*/0,
          /*end=*/shape[idx], /*step=*/1, /*arange_dimension=*/idx);
      toConcat.push_back(arangeOp);
    }

    // Concat along new trailing dimension to get [*, rank].
    shape.back() = rank;
    auto concatType = RankedTensorType::get(shape, indicesType.getElementType(),
                                            indicesType.getEncoding());
    // Concatenate iota(s) with the original indices tensor; this will act as
    // index tensor for GatherOp.
    Value concatIndices =
        rewriter.create<ttir::ConcatOp>(loc, concatType, toConcat, rank);

    // Prepare Gather attributes
    // collapsedDims specifies which dimensions of the gathered slice should be
    // "collapsed" (i.e., dropped from the result shape).
    // Since we're gathering scalar elements (sliceSize = 1 in every dimension),
    // we collapse all dimensions to get a scalar output at each gather point.
    // collapsedDims = [0, 1, ..., rank-1]
    SmallVector<int64_t> collapsedDims(/*size=*/rank);
    std::iota(collapsedDims.begin(), collapsedDims.end(), /*start_value=*/0);

    // startIndexMap defines how to map each element of a start index vector to
    // a dimension in the input.
    // A value of `i` at position `i` means index[i] maps to dimension i in the
    // input. This is a one-to-one mapping from index vector components to input
    // dimensions.
    // startIndexMap = [0, 1, ..., rank-1]
    SmallVector<int64_t> startIndexMap(/*size=*/rank);
    std::iota(startIndexMap.begin(), startIndexMap.end(), /*start_value=*/0);

    // These are left empty because we are not using batching in this gather.
    // - operand_batch_dims: for batching in the input tensor
    // - start_index_batch_dims: for batching in the start_indices tensor
    // Since we're gathering without batching semantics, these remain empty.
    llvm::ArrayRef<int64_t> empty;

    // sliceSizes determines the size of the slice to extract at each index.
    // Since we are gathering scalars (individual elements), the slice size is 1
    // in every dimension. e.g., for rank=4 → [1, 1, 1, 1]
    SmallVector<int64_t> sliceSizes(rank, 1);

    // Collect output values: sorted keys + gathered values
    SmallVector<Value> results{sortOp.getValues()};

    for (size_t i = 1; i < srcOp.getInputs().size(); ++i) {
      auto valType = cast<RankedTensorType>(
          getTypeConverter()->convertType(srcOp.getResultTypes()[i]));

      auto gathered = rewriter.create<ttir::GatherOp>(
          loc, valType, srcOp.getInputs()[i], concatIndices,
          /*offsetDims=*/empty,
          /*collapsedSliceDims=*/collapsedDims,
          /*operandBatchDims=*/empty,
          /*startIndexBatchDims=*/empty,
          /*startIndexMap=*/startIndexMap,
          /*indexVectorDim=*/rank,
          /*sliceSizes=*/sliceSizes,
          /*indicesAreSorted=*/false);

      results.push_back(gathered);
    }

    rewriter.replaceOp(srcOp, results);
    return success();
  }

private:
  enum SortType { kValueOnly = 0, kValueIndex = 1, kKeyValue = 2 };

  bool isValueIndexSort(mlir::OperandRange inputs) const {
    if (inputs.size() != 2) {
      return false;
    }

    Operation *op = inputs.back().getDefiningOp();
    while (
        isa_and_nonnull<mlir::stablehlo::BroadcastInDimOp,
                        mlir::stablehlo::ReshapeOp, mlir::stablehlo::ConvertOp>(
            op)) {
      op = op->getOperand(0).getDefiningOp();
    }
    return isa_and_nonnull<mlir::stablehlo::IotaOp>(op);
  }

  mlir::FailureOr<bool> getSortDirection(mlir::stablehlo::SortOp &srcOp) const {
    Block &block = srcOp.getComparator().front();
    for (auto &op : block.getOperations()) {
      auto cmpOp = dyn_cast<mlir::stablehlo::CompareOp>(op);
      if (!cmpOp || !(cmpOp.getComparisonDirection() ==
                          mlir::stablehlo::ComparisonDirection::LT ||
                      cmpOp.getComparisonDirection() ==
                          mlir::stablehlo::ComparisonDirection::GT)) {
        continue;
      }
      return cmpOp.getComparisonDirection() ==
             mlir::stablehlo::ComparisonDirection::GT;
    }
    return mlir::failure();
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

    rewriter.replaceOpWithNewOp<ttir::ReverseOp>(
        srcOp, outputType, adaptor.getOperand(), adaptor.getDimensions());

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

    LogicalResult legalityResult =
        checkConversionLegality(srcOp, adaptor, rewriter);
    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    ttir::ConstantOp valueDef =
        getConstantValueDefiningOp(adaptor.getPaddingValue());

    mlir::ElementsAttr paddingValueAttr = valueDef.getValueAttr();
    SmallVector<int64_t> steps;
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    float value;
    if (paddingValueAttr.getElementType().isInteger()) {
      value = paddingValueAttr.getSplatValue<APInt>().signedRoundToDouble();
    } else {
      value = paddingValueAttr.getSplatValue<APFloat>().convertToDouble();
    }

    int64_t sum = 0;
    for (int64_t eachVal : srcOp.getInteriorPadding()) {
      steps.push_back(eachVal + 1);
      sum += eachVal;
    }
    if (sum != 0) {
      auto fullOp = rewriter.create<ttir::FullOp>(
          srcOp.getLoc(), outputType, rewriter.getF32FloatAttr(value));
      llvm::SmallVector<int64_t> upperbounds;
      llvm::copy(outputType.getShape(), std::back_inserter(upperbounds));
      int64_t index = 0;
      for (int64_t pad : srcOp.getEdgePaddingHigh()) {
        upperbounds[index] -= pad;
        index++;
      }

      llvm::SmallVector<int64_t> counters;
      llvm::SmallVector<int64_t> lowerbounds;
      for (int64_t counter : srcOp.getEdgePaddingLow()) {
        counters.push_back(counter);
        lowerbounds.push_back(counter);
      }
      llvm::SmallVector<int64_t> flatIndices;
      flatIndices.append(counters.begin(), counters.end());
      int64_t numIndices = 1;

      size_t current_index = 0;
      while (current_index < counters.size() &&
             counters[counters.size() - 1] <
                 upperbounds[upperbounds.size() - 1]) {
        counters[current_index] += steps[current_index];
        bool reset = false;
        while (current_index < counters.size() &&
               counters[current_index] >= upperbounds[current_index]) {
          counters[current_index] = lowerbounds[current_index];
          current_index++;
          if (current_index < counters.size()) {
            counters[current_index] += steps[current_index];
          }
          reset = true;
        }
        if (current_index >= counters.size()) {
          break;
        }
        if (reset) {
          current_index = 0;
        }
        flatIndices.append(counters.begin(), counters.end());
        numIndices++;
      }
      auto inputType =
          mlir::cast<RankedTensorType>(adaptor.getOperand().getType());

      int64_t rank = counters.size();

      // Calculate strides for converting multi-dimensional indices to 1D.
      ArrayRef<int64_t> outputShape = outputType.getShape();
      llvm::SmallVector<int64_t> strides(rank);
      for (int64_t i = 0; i < rank; ++i) {
        int64_t stride = 1;
        for (int64_t j = i + 1; j < rank; ++j) {
          stride *= outputShape[j];
        }
        strides[i] = stride;
      }

      // Convert multi-dimensional indices to 1D flat indices.
      llvm::SmallVector<int64_t> flatIndices1D;
      flatIndices1D.reserve(numIndices);
      for (int64_t i = 0; i < numIndices; ++i) {
        int64_t flatIdx = 0;
        for (int64_t d = 0; d < rank; ++d) {
          flatIdx += flatIndices[i * rank + d] * strides[d];
        }
        flatIndices1D.push_back(flatIdx);
      }

      auto flatIndicesType = RankedTensorType::get(
          {numIndices}, rewriter.getI64Type(), inputType.getEncoding());
      auto flatIndicesAttr =
          DenseIntElementsAttr::get(flatIndicesType, flatIndices1D);
      Value flatIndicesTensor = rewriter.create<ttir::ConstantOp>(
          srcOp.getLoc(), flatIndicesType, flatIndicesAttr);

      // Flatten input and update tensors to 1D.
      Value flattenedInput = ttir::utils::flattenTensor(
          rewriter, srcOp.getLoc(), fullOp.getResult(), "_input_flatten");
      Value flattenedUpdate = ttir::utils::flattenTensor(
          rewriter, srcOp.getLoc(), adaptor.getOperand(), "_update_flatten");

      RankedTensorType flattenedInputType =
          mlir::cast<RankedTensorType>(flattenedInput.getType());

      auto dimAttr = rewriter.getI32IntegerAttr(0);
      auto reduceTypeAttr = ttcore::ReduceTypeAttr::get(
          rewriter.getContext(), ttcore::ReduceType::Invalid);

      Value scatterResult = rewriter.create<ttir::ScatterOp>(
          srcOp.getLoc(), flattenedInputType, flattenedInput, flatIndicesTensor,
          flattenedUpdate, dimAttr, reduceTypeAttr);

      // Reshape result back to original output shape.
      rewriter.replaceOpWithNewOp<ttir::ReshapeOp>(
          srcOp, outputType, scatterResult,
          rewriter.getI32ArrayAttr(llvm::SmallVector<int32_t>(
              outputShape.begin(), outputShape.end())));

      return success();
    }

    SmallVector<int32_t> padDim;
    for (uint32_t i = 0; i < adaptor.getEdgePaddingLow().size(); i++) {
      padDim.push_back(adaptor.getEdgePaddingLow()[i]);
      padDim.push_back(adaptor.getEdgePaddingHigh()[i]);
    }

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::PadOp>(
        srcOp, outputType, adaptor.getOperand(),
        rewriter.getDenseI32ArrayAttr(padDim), rewriter.getF32FloatAttr(value));

    return success();
  }

private:
  LogicalResult
  checkConversionLegality(mlir::stablehlo::PadOp &srcOp,
                          mlir::stablehlo::PadOp::Adaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {

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

namespace {
class StableHLOToTTIRAllToAllOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::AllToAllOp> {
  using OpConversionPattern<mlir::stablehlo::AllToAllOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::AllToAllOp srcOp,
                  mlir::stablehlo::AllToAllOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult(0).getType()));

    rewriter.replaceOpWithNewOp<ttir::AllToAllOp>(
        srcOp, outputType, adaptor.getOperands()[0],
        adaptor.getSplitDimension(), adaptor.getConcatDimension(),
        adaptor.getSplitCount(), adaptor.getReplicaGroups());

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRCollectiveBroadcastOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CollectiveBroadcastOp> {
  using OpConversionPattern<
      mlir::stablehlo::CollectiveBroadcastOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::CollectiveBroadcastOp srcOp,
                  mlir::stablehlo::CollectiveBroadcastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Create the output tensor type based on inputs.
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));

    rewriter.replaceOpWithNewOp<ttir::CollectiveBroadcastOp>(
        srcOp, outputType, adaptor.getOperand(), adaptor.getReplicaGroups());

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRRngOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::RngOp> {
  using OpConversionPattern<mlir::stablehlo::RngOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::RngOp srcOp,
                  mlir::stablehlo::RngOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LogicalResult legalityResult = checkConversionLegality(srcOp, rewriter);

    if (!legalityResult.succeeded()) {
      return legalityResult;
    }

    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult().getType()));
    auto low = extractConstantScalarAsFloat(adaptor.getA());
    auto high = extractConstantScalarAsFloat(adaptor.getB());

    if (!low || !high) {
      return rewriter.notifyMatchFailure(
          srcOp, "Unable to extract low and high scalars for random number "
                 "generator interval.");
    }

    if (*low >= *high) {
      return rewriter.notifyMatchFailure(
          srcOp, "Expects 'low' value to be < 'high' value.");
    }

    llvm::SmallVector<int32_t> size(outputType.getShape());

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::RandOp>(
        srcOp, outputType, rewriter.getI32ArrayAttr(size),
        mlir::TypeAttr::get(outputType.getElementType()),
        rewriter.getF32FloatAttr(*low), rewriter.getF32FloatAttr(*high));

    return success();
  }

private:
  LogicalResult
  checkConversionLegality(mlir::stablehlo::RngOp &srcOp,
                          ConversionPatternRewriter &rewriter) const {
    // [TODO] Remove this legality check when tt-metal supports other kind of
    // distribution for random number generation.
    if (srcOp.getRngDistribution() !=
        mlir::stablehlo::RngDistribution::UNIFORM) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "Unsupported RNG distribution type.");
    }

    return success();
  }

  std::optional<float> extractConstantScalarAsFloat(mlir::Value val) const {
    auto constantOp = getConstantValueDefiningOp(val);
    if (!constantOp) {
      return std::nullopt;
    }

    if (!constantOp.getValueAttr().isSplat()) {
      return std::nullopt;
    }

    auto elementType = constantOp.getValueAttr().getElementType();
    if (!elementType.isIntOrFloat()) {
      return std::nullopt;
    }

    float scalarValue = 0.0;
    if (isa<IntegerType>(elementType)) {
      scalarValue = static_cast<float>(constantOp.getValueAttr()
                                           .getSplatValue<llvm::APInt>()
                                           .getSExtValue());
    } else {
      scalarValue = static_cast<float>(constantOp.getValueAttr()
                                           .getSplatValue<llvm::APFloat>()
                                           .convertToDouble());
    }

    return scalarValue;
  }
};
} // namespace

namespace {
class StableHLOToTTIRRngBitGeneratorOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::RngBitGeneratorOp> {
  using OpConversionPattern<
      mlir::stablehlo::RngBitGeneratorOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::RngBitGeneratorOp srcOp,
                  mlir::stablehlo::RngBitGeneratorOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getOutput().getType()));

    llvm::SmallVector<int32_t> size(outputType.getShape());

    auto floatElementType = rewriter.getF32Type();
    auto floatOutputType = mlir::RankedTensorType::get(
        outputType.getShape(), floatElementType, outputType.getEncoding());

    // Use min and max unsigned int value to cover most of the range of uint32.
    auto fromFloat = rewriter.getF32FloatAttr(
        static_cast<float>(std::numeric_limits<unsigned int>::min()));
    auto toFloat = rewriter.getF32FloatAttr(
        static_cast<float>(std::numeric_limits<unsigned int>::max()));

    // Seed set to 0 is a special case that is equivalent to no seed.
    // Using any other value would make every flatbuffer run deterministic.
    auto seed = rewriter.getUI32IntegerAttr(0);

    auto randOp = rewriter.create<mlir::tt::ttir::RandOp>(
        srcOp.getLoc(), floatOutputType, rewriter.getI32ArrayAttr(size),
        mlir::TypeAttr::get(floatElementType), fromFloat, toFloat, seed);

    // TODO (pglusac): Change to bit cast once we support it or remove if
    // rand starts supporting uint32.
    // See https://github.com/tenstorrent/tt-mlir/issues/5078
    auto typecastOp = rewriter.create<mlir::tt::ttir::TypecastOp>(
        srcOp.getLoc(), outputType, randOp.getResult());

    // HACK (pglusac): Output state is discarded, initial state is returned as
    // a result. https://github.com/tenstorrent/tt-mlir/issues/5101
    rewriter.replaceOp(srcOp,
                       {adaptor.getInitialState(), typecastOp.getResult()});

    return success();
  }
};
} // namespace

namespace {
// This pattern recognizes and converts stablehlo.custom_call @tt.fill_cache
// to ttir.fill_cache.
class StableHLOFillCacheConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CustomCallOp> {
  using OpConversionPattern<mlir::stablehlo::CustomCallOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp srcOp,
                  mlir::stablehlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr funcName = adaptor.getCallTargetNameAttr();
    if (funcName != "tt.fill_cache") {
      return failure();
    }

    if (adaptor.getOperands().size() != 2 || srcOp.getResults().size() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp,
          "FillCache op must have exactly two operands and one result. Got " +
              std::to_string(adaptor.getOperands().size()) +
              " operands "
              "and " +
              std::to_string(srcOp.getResults().size()) + " results.");
    }

    mlir::DictionaryAttr frontendAttributes =
        mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
            srcOp->getDiscardableAttr("mhlo.frontend_attributes"));
    if (!frontendAttributes) {
      return rewriter.notifyMatchFailure(
          srcOp, "FillCache op must have mhlo.frontend_attributes attribute.");
    }

    // Frontend attributes can only be populated via torch-xla as
    // string-to-string dicts. Thus, our batch_offset will be presented to
    // tt-mlir as a string.
    auto batchOffset =
        frontendAttributes.getAs<mlir::StringAttr>("batch_offset");
    if (!batchOffset) {
      return rewriter.notifyMatchFailure(
          srcOp, "FillCache op must have batch_offset attribute.");
    }

    int32_t batchOffsetInt = 0;
    if (!llvm::to_integer(batchOffset.getValue(), batchOffsetInt)) {
      return rewriter.notifyMatchFailure(
          srcOp, "Failed to convert batch_offset string to integer.");
    }

    if (batchOffsetInt < 0) {
      return rewriter.notifyMatchFailure(
          srcOp, "Batch offset must be non-negative. Received " +
                     std::to_string(batchOffsetInt) + ".");
    }

    auto cache = adaptor.getOperands()[0];
    auto input = adaptor.getOperands()[1];

    rewriter.replaceOpWithNewOp<ttir::FillCacheOp>(
        srcOp, cache.getType(), cache, input, batchOffsetInt);

    return success();
  }
};
} // namespace

namespace {
// This pattern recognizes and converts stablehlo.custom_call @tt.update_cache
// to ttir.update_cache.
class StableHLOUpdateCacheConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CustomCallOp> {
  using OpConversionPattern<mlir::stablehlo::CustomCallOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp srcOp,
                  mlir::stablehlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr funcName = adaptor.getCallTargetNameAttr();
    if (funcName != "tt.update_cache") {
      return failure();
    }

    if (adaptor.getOperands().size() != 3 || srcOp.getResults().size() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "UpdateCache op must have exactly three operands and one "
                 "result. Got " +
                     std::to_string(adaptor.getOperands().size()) +
                     " operands "
                     "and " +
                     std::to_string(srcOp.getResults().size()) + " results.");
    }

    mlir::DictionaryAttr frontendAttributes =
        mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
            srcOp->getDiscardableAttr("mhlo.frontend_attributes"));
    if (!frontendAttributes) {
      return rewriter.notifyMatchFailure(
          srcOp,
          "UpdateCache op must have mhlo.frontend_attributes attribute.");
    }

    // Frontend attributes can only be populated via torch-xla as
    // string-to-string dicts. Thus, our batch_offset will be presented to
    // tt-mlir as a string.
    auto batchOffset =
        frontendAttributes.getAs<mlir::StringAttr>("batch_offset");
    if (!batchOffset) {
      return rewriter.notifyMatchFailure(
          srcOp, "UpdateCache op must have batch_offset attribute.");
    }

    int32_t batchOffsetInt = 0;
    if (!llvm::to_integer(batchOffset.getValue(), batchOffsetInt)) {
      return rewriter.notifyMatchFailure(
          srcOp, "Failed to convert batch_offset string to integer.");
    }

    if (batchOffsetInt < 0) {
      return rewriter.notifyMatchFailure(
          srcOp, "Batch offset must be non-negative. Received " +
                     std::to_string(batchOffsetInt) + ".");
    }

    auto cache = adaptor.getOperands()[0];
    auto input = adaptor.getOperands()[1];
    auto updateIndex = adaptor.getOperands()[2];

    rewriter.replaceOpWithNewOp<ttir::UpdateCacheOp>(
        srcOp, cache.getType(), cache, input, updateIndex, batchOffsetInt);

    return success();
  }
};
} // namespace

namespace {
class StableHLOPagedUpdateCacheConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CustomCallOp> {
  using OpConversionPattern<mlir::stablehlo::CustomCallOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp srcOp,
                  mlir::stablehlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr funcName = adaptor.getCallTargetNameAttr();
    if (funcName != "tt.paged_update_cache") {
      return failure();
    }

    if (adaptor.getOperands().size() != 4 || srcOp.getResults().size() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "PagedUpdateCache op must have exactly four operands and one "
                 "result. Got " +
                     std::to_string(adaptor.getOperands().size()) +
                     " operands "
                     "and " +
                     std::to_string(srcOp.getResults().size()) + " results.");
    }

    Value cache = adaptor.getOperands()[0];
    Value input = adaptor.getOperands()[1];
    Value updateIndex = adaptor.getOperands()[2];
    Value pageTable = adaptor.getOperands()[3];

    bool shareCache = false;
    mlir::DictionaryAttr frontendAttributes =
        mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
            srcOp->getDiscardableAttr("mhlo.frontend_attributes"));
    if (frontendAttributes) {
      auto shareCacheStringAttr =
          frontendAttributes.getAs<mlir::StringAttr>("share_cache");
      if (shareCacheStringAttr) {
        if (failed(parseBoolFromStringAttr(shareCacheStringAttr, shareCache))) {
          return rewriter.notifyMatchFailure(
              srcOp, "Failed to parse share_cache attribute.");
        }
      }
    }

    rewriter.replaceOpWithNewOp<ttir::PagedUpdateCacheOp>(
        srcOp, cache.getType(), cache, input, updateIndex, shareCache,
        pageTable);

    return success();
  }
};
} // namespace

namespace {
class StableHLOPagedFillCacheConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CustomCallOp> {
  using OpConversionPattern<mlir::stablehlo::CustomCallOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp srcOp,
                  mlir::stablehlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr funcName = adaptor.getCallTargetNameAttr();
    if (funcName != "tt.paged_fill_cache") {
      return failure();
    }

    if ((adaptor.getOperands().size() != 4 &&
         adaptor.getOperands().size() != 3) ||
        srcOp.getResults().size() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "PagedFillCache op must have three or four operands and one "
                 "result. Got " +
                     std::to_string(adaptor.getOperands().size()) +
                     " operands "
                     "and " +
                     std::to_string(srcOp.getResults().size()) + " results.");
    }

    Value cache = adaptor.getOperands()[0];
    Value input = adaptor.getOperands()[1];
    Value pageTable = adaptor.getOperands()[2];
    Value batchIdxTensor =
        adaptor.getOperands().size() == 4 ? adaptor.getOperands()[3] : nullptr;

    rewriter.replaceOpWithNewOp<ttir::PagedFillCacheOp>(
        srcOp, cache.getType(), cache, input, pageTable, batchIdxTensor);

    return success();
  }
};
} // namespace

namespace {
class StableHLOErfOpMHLOConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CustomCallOp> {
  using OpConversionPattern<mlir::stablehlo::CustomCallOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp srcOp,
                  mlir::stablehlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr funcName = adaptor.getCallTargetNameAttr();
    if (funcName != "mhlo.erf") {
      return failure();
    }

    if (adaptor.getOperands().size() != 1 || srcOp.getResults().size() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "Erf op must have exactly one operand and one result. Got " +
                     std::to_string(adaptor.getOperands().size()) +
                     " operands "
                     "and " +
                     std::to_string(srcOp.getResults().size()) + " results.");
    }

    rewriter.replaceOpWithNewOp<ttir::ErfOp>(
        srcOp,
        cast<RankedTensorType>(
            getTypeConverter()->convertType(srcOp.getResult(0).getType())),
        adaptor.getOperands()[0]);

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRScaledDotProductAttentionDecodeOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CustomCallOp> {
  using OpConversionPattern<mlir::stablehlo::CustomCallOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp srcOp,
                  mlir::stablehlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr funcName = adaptor.getCallTargetNameAttr();
    if (funcName != "tt.scaled_dot_product_attention_decode") {
      return failure();
    }

    mlir::DictionaryAttr frontendAttributes =
        mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
            srcOp->getDiscardableAttr("mhlo.frontend_attributes"));
    if (!frontendAttributes) {
      return rewriter.notifyMatchFailure(
          srcOp, "ScaledDotProductAttentionDecode op must have "
                 "mhlo.frontend_attributes attribute.");
    }

    auto isCausalStringAttr =
        frontendAttributes.getAs<mlir::StringAttr>("is_causal");
    bool isCausal = true;
    if (isCausalStringAttr) {
      if (failed(parseBoolFromStringAttr(isCausalStringAttr, isCausal))) {
        return rewriter.notifyMatchFailure(
            srcOp, "Failed to parse is_causal attribute.");
      }
    }

    BoolAttr isCausalAttr = rewriter.getBoolAttr(isCausal);

    auto scaleStringAttr = frontendAttributes.getAs<mlir::StringAttr>("scale");
    FloatAttr scaleAttr = nullptr;
    if (scaleStringAttr) {
      float scale;
      if (failed(parseFloatFromStringAttr(scaleStringAttr, scale))) {
        return rewriter.notifyMatchFailure(srcOp,
                                           "Failed to parse scale attribute.");
      }
      scaleAttr = rewriter.getF32FloatAttr(scale);
    }

    auto hasAttentionMaskStringAttr =
        frontendAttributes.getAs<mlir::StringAttr>("has_attention_mask");
    bool hasAttentionMask = false;
    if (!hasAttentionMaskStringAttr) {
      return rewriter.notifyMatchFailure(
          srcOp, "has_attention_mask attribute must be present.");
    }
    if (failed(parseBoolFromStringAttr(hasAttentionMaskStringAttr,
                                       hasAttentionMask))) {
      return rewriter.notifyMatchFailure(
          srcOp, "Failed to parse has_attention_mask attribute.");
    }

    auto hasAttentionSinkStringAttr =
        frontendAttributes.getAs<mlir::StringAttr>("has_attention_sink");
    bool hasAttentionSink = false;
    if (!hasAttentionSinkStringAttr) {
      return rewriter.notifyMatchFailure(
          srcOp, "has_attention_sink attribute must be present.");
    }

    if (failed(parseBoolFromStringAttr(hasAttentionSinkStringAttr,
                                       hasAttentionSink))) {
      return rewriter.notifyMatchFailure(
          srcOp, "Failed to parse has_attention_sink attribute.");
    }

    Value query = adaptor.getOperands()[0];
    Value key = adaptor.getOperands()[1];
    Value value = adaptor.getOperands()[2];
    Value curPosTensor = adaptor.getOperands()[3];

    if (hasAttentionMask && hasAttentionSink) {
      rewriter.replaceOpWithNewOp<
          mlir::tt::ttir::ScaledDotProductAttentionDecodeOp>(
          srcOp,
          cast<RankedTensorType>(
              getTypeConverter()->convertType(srcOp.getResult(0).getType())),
          query, key, value, isCausalAttr, adaptor.getOperands()[4],
          curPosTensor, adaptor.getOperands()[5], scaleAttr);
    } else if (hasAttentionMask) {
      rewriter.replaceOpWithNewOp<
          mlir::tt::ttir::ScaledDotProductAttentionDecodeOp>(
          srcOp,
          cast<RankedTensorType>(
              getTypeConverter()->convertType(srcOp.getResult(0).getType())),
          query, key, value, isCausalAttr, adaptor.getOperands()[4],
          curPosTensor, nullptr, scaleAttr);
    } else if (hasAttentionSink) {
      rewriter.replaceOpWithNewOp<
          mlir::tt::ttir::ScaledDotProductAttentionDecodeOp>(
          srcOp,
          cast<RankedTensorType>(
              getTypeConverter()->convertType(srcOp.getResult(0).getType())),
          query, key, value, isCausalAttr, nullptr, curPosTensor,
          adaptor.getOperands()[4], scaleAttr);
    } else if (!hasAttentionMask && !hasAttentionSink) {
      rewriter.replaceOpWithNewOp<
          mlir::tt::ttir::ScaledDotProductAttentionDecodeOp>(
          srcOp,
          cast<RankedTensorType>(
              getTypeConverter()->convertType(srcOp.getResult(0).getType())),
          query, key, value, isCausalAttr, nullptr, curPosTensor, nullptr,
          scaleAttr);
    } else {
      if (hasAttentionMask || hasAttentionSink) {
        llvm_unreachable("All combinations of attention mask "
                         "and attention sink should have been handled");
      }
    }

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRPagedScaledDotProductAttentionDecodeOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CustomCallOp> {
  using OpConversionPattern<mlir::stablehlo::CustomCallOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp srcOp,
                  mlir::stablehlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr funcName = adaptor.getCallTargetNameAttr();
    if (funcName != "tt.paged_scaled_dot_product_attention_decode") {
      return failure();
    }

    mlir::DictionaryAttr frontendAttributes =
        mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
            srcOp->getDiscardableAttr("mhlo.frontend_attributes"));
    if (!frontendAttributes) {
      return rewriter.notifyMatchFailure(
          srcOp, "PagedScaledDotProductAttentionDecode op must have "
                 "mhlo.frontend_attributes attribute.");
    }

    auto isCausalStringAttr =
        frontendAttributes.getAs<mlir::StringAttr>("is_causal");
    bool isCausal = true;
    if (isCausalStringAttr) {

      if (isCausalStringAttr.getValue().lower() == "true") {
        isCausal = true;
      } else if (isCausalStringAttr.getValue().lower() == "false") {
        isCausal = false;
      } else {
        return rewriter.notifyMatchFailure(
            srcOp, "is_causal attribute must be true or false. Received \"" +
                       isCausalStringAttr.getValue() + "\".");
      }
    }

    BoolAttr isCausalAttr = rewriter.getBoolAttr(isCausal);

    auto scaleStringAttr = frontendAttributes.getAs<mlir::StringAttr>("scale");
    FloatAttr scaleAttr = nullptr;
    if (scaleStringAttr) {
      float scale;
      if (failed(parseFloatFromStringAttr(scaleStringAttr, scale))) {
        return rewriter.notifyMatchFailure(srcOp,
                                           "Failed to parse scale attribute.");
      }
      scaleAttr = rewriter.getF32FloatAttr(scale);
    }

    auto hasAttentionMaskStringAttr =
        frontendAttributes.getAs<mlir::StringAttr>("has_attention_mask");
    bool hasAttentionMask = false;
    if (!hasAttentionMaskStringAttr) {
      return rewriter.notifyMatchFailure(
          srcOp, "has_attention_mask attribute must be present.");
    }

    if (failed(parseBoolFromStringAttr(hasAttentionMaskStringAttr,
                                       hasAttentionMask))) {
      return rewriter.notifyMatchFailure(
          srcOp, "Failed to parse has_attention_mask attribute.");
    }

    auto hasCurPosTensorStringAttr =
        frontendAttributes.getAs<mlir::StringAttr>("has_cur_pos_tensor");
    bool hasCurPosTensor = false;
    if (!hasCurPosTensorStringAttr) {
      return rewriter.notifyMatchFailure(
          srcOp, "has_cur_pos_tensor attribute must be present.");
    }

    if (failed(parseBoolFromStringAttr(hasCurPosTensorStringAttr,
                                       hasCurPosTensor))) {
      return rewriter.notifyMatchFailure(
          srcOp, "Failed to parse has_cur_pos_tensor attribute.");
    }

    auto hasAttentionSinkStringAttr =
        frontendAttributes.getAs<mlir::StringAttr>("has_attention_sink");
    bool hasAttentionSink = false;
    if (!hasAttentionSinkStringAttr) {
      return rewriter.notifyMatchFailure(
          srcOp, "has_attention_sink attribute must be present.");
    }

    if (failed(parseBoolFromStringAttr(hasAttentionSinkStringAttr,
                                       hasAttentionSink))) {
      return rewriter.notifyMatchFailure(
          srcOp, "Failed to parse has_attention_sink attribute.");
    }

    Value query = adaptor.getOperands()[0];
    Value key = adaptor.getOperands()[1];
    Value value = adaptor.getOperands()[2];
    Value pageTable = adaptor.getOperands()[3];
    Value attentionMask = nullptr;
    Value curPosTensor = nullptr;
    Value attentionSink = nullptr;
    int64_t operandIndex = 4;
    if (hasAttentionMask) {
      attentionMask = adaptor.getOperands()[operandIndex];
      operandIndex++;
    }
    if (hasCurPosTensor) {
      curPosTensor = adaptor.getOperands()[operandIndex];
      operandIndex++;
    }
    if (hasAttentionSink) {
      attentionSink = adaptor.getOperands()[operandIndex];
      operandIndex++;
    }

    RankedTensorType outputType = cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult(0).getType()));
    ttir::EmptyOp outputTensor = rewriter.create<ttir::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    rewriter.replaceOpWithNewOp<
        mlir::tt::ttir::PagedScaledDotProductAttentionDecodeOp>(
        srcOp,
        cast<RankedTensorType>(
            getTypeConverter()->convertType(srcOp.getResult(0).getType())),
        query, key, value, pageTable, outputTensor, isCausalAttr, attentionMask,
        curPosTensor, attentionSink, scaleAttr);

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIROpOptimizationBarrierOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::OptimizationBarrierOp> {
  using OpConversionPattern<
      mlir::stablehlo::OptimizationBarrierOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::OptimizationBarrierOp srcOp,
                  mlir::stablehlo::OptimizationBarrierOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> convertedResultTypes;
    for (auto resultType : srcOp->getResultTypes()) {
      convertedResultTypes.push_back(
          this->getTypeConverter()->convertType(resultType));
    }

    rewriter.replaceOpWithNewOp<ttcore::OptimizationBarrierOp>(
        srcOp, convertedResultTypes, adaptor.getOperands());

    return success();
  }
};
} // namespace

namespace {
class StableHLOToTTIRScaledDotProductAttentionOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CustomCallOp> {
  using OpConversionPattern<mlir::stablehlo::CustomCallOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp srcOp,
                  mlir::stablehlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr funcName = adaptor.getCallTargetNameAttr();
    if (funcName != "tt.scaled_dot_product_attention") {
      return failure();
    }

    mlir::DictionaryAttr frontendAttributes =
        mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
            srcOp->getDiscardableAttr("mhlo.frontend_attributes"));
    if (!frontendAttributes) {
      return rewriter.notifyMatchFailure(
          srcOp, "ScaledDotProductAttention op must have "
                 "mhlo.frontend_attributes attribute.");
    }

    auto isCausalSringAttr =
        frontendAttributes.getAs<mlir::StringAttr>("is_causal");
    bool isCausal = true;
    if (isCausalSringAttr) {
      if (isCausalSringAttr.getValue().lower() == "true") {
        isCausal = true;
      } else if (isCausalSringAttr.getValue().lower() == "false") {
        isCausal = false;
      } else {
        return rewriter.notifyMatchFailure(
            srcOp, "is_causal attribute must be true or false. Received \"" +
                       isCausalSringAttr.getValue() + "\".");
      }
    }

    BoolAttr isCausalAttr = rewriter.getBoolAttr(isCausal);

    auto scaleStringAttr = frontendAttributes.getAs<mlir::StringAttr>("scale");
    std::optional<float> scale = std::nullopt;
    if (scaleStringAttr) {
      float _scale;
      if (!llvm::to_float(scaleStringAttr.getValue(), _scale)) {
        return rewriter.notifyMatchFailure(
            srcOp,
            "scale attribute string must be convertible to float. Received \"" +
                scaleStringAttr.getValue() + "\".");
      }
      scale = _scale;
    }

    FloatAttr scaleAttr =
        scale ? rewriter.getF32FloatAttr(scale.value()) : nullptr;

    Value query = adaptor.getOperands()[0];
    Value key = adaptor.getOperands()[1];
    Value value = adaptor.getOperands()[2];

    RankedTensorType outputType = cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult(0).getType()));

    Value attentionMask = nullptr;
    if (adaptor.getOperands().size() == 4) {
      attentionMask = adaptor.getOperands()[3];
    }

    rewriter.replaceOpWithNewOp<ttir::ScaledDotProductAttentionOp>(
        srcOp, outputType, query, key, value, attentionMask, isCausalAttr,
        scaleAttr, /*slidingWindowSize=*/nullptr);

    return success();
  }
};
} // namespace

namespace {
// Pattern to convert mhlo.topk to ttir.topk
class StableHLOTopKOpMHLOConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CustomCallOp> {
  using OpConversionPattern<mlir::stablehlo::CustomCallOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp srcOp,
                  mlir::stablehlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr funcName = adaptor.getCallTargetNameAttr();
    if (funcName != "mhlo.topk") {
      return failure();
    }

    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(srcOp->getResultTypes(),
                                                      resultTypes))) {
      return failure();
    }
    auto input = adaptor.getOperands()[0];
    IntegerAttr dimAttr = IntegerAttr::get(rewriter.getIntegerType(32), -1);
    auto sortedAttr = BoolAttr::get(rewriter.getContext(), false);

    auto dictAttr = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
        srcOp->getDiscardableAttr("mhlo.attributes"));
    if (!dictAttr) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "Missing mhlo.attributes dictionary");
    }

    auto kAttr = dictAttr.getAs<IntegerAttr>("k");
    if (!kAttr) {
      return rewriter.notifyMatchFailure(
          srcOp, "Missing k attribute in mhlo.attributes");
    }
    APInt kValueI64 = kAttr.getValue();
    // Check if value fits in i32
    if (!kValueI64.isIntN(32)) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "k value is too large for i32: " +
                                             Twine(kValueI64.getSExtValue()));
    }

    // Convert i64 to i32
    int32_t kValueI32 = static_cast<int32_t>(kValueI64.getSExtValue());
    kAttr = IntegerAttr::get(rewriter.getIntegerType(32), kValueI32);

    auto largestAttr = dictAttr.getAs<BoolAttr>("largest");
    if (!largestAttr) {
      return rewriter.notifyMatchFailure(
          srcOp, "Missing largest attribute in mhlo.attributes");
    }

    rewriter.replaceOpWithNewOp<ttir::TopKOp>(srcOp, resultTypes, input, kAttr,
                                              dimAttr, largestAttr, sortedAttr);
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
      mlir::stablehlo::PowOp, mlir::tt::ttir::PowOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::Atan2Op, mlir::tt::ttir::Atan2Op>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::ShiftRightLogicalOp,
      mlir::tt::ttir::LogicalRightShiftOp>>(typeConverter, ctx);
  patterns.add<StableHLOToTTIROpDefaultConversionPattern<
      mlir::stablehlo::ShiftLeftOp, mlir::tt::ttir::LogicalLeftShiftOp>>(
      typeConverter, ctx);
}

static void addReduceOpsConversionPatterns(MLIRContext *ctx,
                                           RewritePatternSet &patterns,
                                           TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRReduceOpConversionPattern>(typeConverter, ctx);
  patterns.add<StableHLOTopKOpMHLOConversionPattern>(typeConverter, ctx);
}

static void
addSelectAndScatterOpConversionPatterns(MLIRContext *ctx,
                                        RewritePatternSet &patterns,
                                        TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRSelectAndScatterOpConversionPattern>(
      typeConverter, ctx);
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
  patterns.add<Legalize1DConvolutionPattern>(typeConverter, ctx);
  patterns.add<ConvolutionToConv2dPattern>(typeConverter, ctx);
  patterns.add<ConvolutionToConv3dPattern>(typeConverter, ctx);
}

static void addQuantizeOpsConversionPattern(MLIRContext *ctx,
                                            RewritePatternSet &patterns,
                                            TypeConverter &typeConverter) {
  // Add the Quant and Requant ops.
  patterns.add<StableHLOToTTIRQuantizeOpConversionPattern>(typeConverter, ctx);
  // Add the Dequant op.
  patterns.add<StableHLOToTTIRDequantizeOpConversionPattern>(typeConverter,
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
  patterns.add<CacheFillUpdatePattern>(typeConverter, ctx);
  patterns.add<StableHLOToTTIRReduceScatterOpConversionPattern>(typeConverter,
                                                                ctx);
  patterns.add<StableHLOToTTIRCollectivePermuteOpConversionPattern>(
      typeConverter, ctx);
  patterns.add<StableHLOToTTIRCustomCallOpConversionPattern>(typeConverter,
                                                             ctx);
  patterns.add<StableHLOToTTIRAllToAllOpConversionPattern>(typeConverter, ctx);
  patterns.add<StableHLOToTTIRCollectiveBroadcastOpConversionPattern>(
      typeConverter, ctx);
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

static void addDynamicSliceOpConversionPattern(MLIRContext *ctx,
                                               RewritePatternSet &patterns,
                                               TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRDynamicSliceOpConversionPattern>(typeConverter,
                                                               ctx);
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
  patterns
      .add<StableHLOToTTIROpIotaOpConversionPattern<mlir::stablehlo::IotaOp>>(
          typeConverter, ctx);
  patterns.add<
      StableHLOToTTIROpIotaOpConversionPattern<mlir::stablehlo::DynamicIotaOp>>(
      typeConverter, ctx);
}

static void addScatterOpConversionPatterns(MLIRContext *ctx,
                                           RewritePatternSet &patterns,
                                           TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIREmbeddingBackwardOpConversionPattern,
               StableHLOToTTIRScatterOpConversionPattern>(typeConverter, ctx);
  ;
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

static void addBatchNormOpConversionPattern(MLIRContext *ctx,
                                            RewritePatternSet &patterns,
                                            TypeConverter &typeConverter) {
  patterns.add<StableHLOToBatchNormInferenceOpConversionPattern>(typeConverter,
                                                                 ctx);
  patterns.add<StableHLOToBatchNormTrainingOpConversionPattern>(typeConverter,
                                                                ctx);
  patterns.add<StableHLOToBatchNormGradOpConversionPattern>(typeConverter, ctx);
}

static void addRngOpConversionPattern(MLIRContext *ctx,
                                      RewritePatternSet &patterns,
                                      TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRRngOpConversionPattern>(typeConverter, ctx);
}

static void
addRngBitGeneratorOpConversionPattern(MLIRContext *ctx,
                                      RewritePatternSet &patterns,
                                      TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRRngBitGeneratorOpConversionPattern>(typeConverter,
                                                                  ctx);
}

static void addErfOpConversionPattern(MLIRContext *ctx,
                                      RewritePatternSet &patterns,
                                      TypeConverter &typeConverter) {
  patterns.add<StableHLOErfOpMHLOConversionPattern>(typeConverter, ctx);
}

static void addSortOpConversionPattern(MLIRContext *ctx,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIRSortOpConversionPattern>(typeConverter, ctx);
}

static void addCacheOpsConversionPattern(MLIRContext *ctx,
                                         RewritePatternSet &patterns,
                                         TypeConverter &typeConverter) {
  patterns.add<StableHLOFillCacheConversionPattern>(typeConverter, ctx);
  patterns.add<StableHLOUpdateCacheConversionPattern>(typeConverter, ctx);
  patterns.add<StableHLOPagedUpdateCacheConversionPattern>(typeConverter, ctx);
  patterns.add<StableHLOPagedFillCacheConversionPattern>(typeConverter, ctx);
}

static void
addOptimizationBarrierOpConversionPattern(MLIRContext *ctx,
                                          RewritePatternSet &patterns,
                                          TypeConverter &typeConverter) {
  patterns.add<StableHLOToTTIROpOptimizationBarrierOpConversionPattern>(
      typeConverter, ctx);
}

static void addScaledDotProductAttentionDecodeOpConversionPattern(
    MLIRContext *ctx, RewritePatternSet &patterns,
    TypeConverter &typeConverter) {
  patterns.add<
      StableHLOToTTIRScaledDotProductAttentionDecodeOpConversionPattern,
      StableHLOToTTIRScaledDotProductAttentionOpConversionPattern,
      StableHLOToTTIRPagedScaledDotProductAttentionDecodeOpConversionPattern>(
      typeConverter, ctx);
}

namespace mlir::tt {

void populateStableHLOToTTIRPatterns(MLIRContext *ctx,
                                     RewritePatternSet &patterns,
                                     TypeConverter &typeConverter) {
  addElementwiseUnaryOpsConversionPatterns(ctx, patterns, typeConverter);
  addElementwiseBinaryOpsConversionPatterns(ctx, patterns, typeConverter);
  addQuantizeOpsConversionPattern(ctx, patterns, typeConverter);
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
  addDynamicSliceOpConversionPattern(ctx, patterns, typeConverter);
  addClampOpConversionPattern(ctx, patterns, typeConverter);
  addGatherOpConversionPattern(ctx, patterns, typeConverter);
  addIotaOpConversionPattern(ctx, patterns, typeConverter);
  addScatterOpConversionPatterns(ctx, patterns, typeConverter);
  addReverseOpConversionPattern(ctx, patterns, typeConverter);
  addPadOpConversionPattern(ctx, patterns, typeConverter);
  addSelectAndScatterOpConversionPatterns(ctx, patterns, typeConverter);
  addBatchNormOpConversionPattern(ctx, patterns, typeConverter);
  addRngOpConversionPattern(ctx, patterns, typeConverter);
  addRngBitGeneratorOpConversionPattern(ctx, patterns, typeConverter);
  addErfOpConversionPattern(ctx, patterns, typeConverter);
  addSortOpConversionPattern(ctx, patterns, typeConverter);
  addCacheOpsConversionPattern(ctx, patterns, typeConverter);
  addOptimizationBarrierOpConversionPattern(ctx, patterns, typeConverter);
  addScaledDotProductAttentionDecodeOpConversionPattern(ctx, patterns,
                                                        typeConverter);
}

} // namespace mlir::tt
