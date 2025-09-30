// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOToTTIR.h"

#include "ttmlir/Dialect/StableHLO/Utils/GSPMDUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
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

#include "llvm/ADT/StringExtras.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace mlir;
using namespace mlir::tt;

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
  bool desiredI1;
  if (desired == TypicalInitReductionValue::NEG_INF) {
    desiredF32 = -std::numeric_limits<float>::infinity();
    desiredF64 = -std::numeric_limits<double>::infinity();
    desiredBF16 = 0xff80; // This is -inf in bfloat16 raw bits
    desiredI32 = std::numeric_limits<int32_t>::min();
    desiredI64 = std::numeric_limits<int64_t>::min();
    desiredI1 = false;
  } else if (desired == TypicalInitReductionValue::ZERO) {
    desiredF32 = 0.0;
    desiredF64 = 0.0;
    desiredBF16 = 0x0000; // This is 0 in bfloat16 raw bits
    desiredI32 = 0;
    desiredI64 = 0;
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

    ttir::utils::replaceOpWithNewDPSOp<DestOp>(rewriter, srcOp, outputType,
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

    ttir::utils::replaceOpWithNewDPSOp<DestOp>(rewriter, srcOp, outputType,
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
    ttir::EmptyOp outputTensor = rewriter.create<ttir::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType(),
        outputType.getEncoding());

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
    ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::PermuteOp>(
        rewriter, srcOp, outputType, adaptor.getOperand(),
        adaptor.getPermutation());

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

    ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::BatchNormInferenceOp>(
        rewriter, srcOp, outputType, adaptor.getOperand(), adaptor.getScale(),
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

    // Create empty output tensors
    auto outputEmpty = rewriter.create<ttir::EmptyOp>(loc, outputType);
    auto batchMeanEmpty = rewriter.create<ttir::EmptyOp>(loc, meanType);
    auto batchVarianceEmpty = rewriter.create<ttir::EmptyOp>(loc, varianceType);

    rewriter.replaceOpWithNewOp<mlir::tt::ttir::BatchNormTrainingOp>(
        srcOp, TypeRange{outputType, meanType, varianceType},
        adaptor.getOperand(), adaptor.getScale(), adaptor.getOffset(),
        runningMean, runningVariance,
        ValueRange{outputEmpty, batchMeanEmpty, batchVarianceEmpty},
        adaptor.getEpsilonAttr(), dimensionAttr, momentumAttr);

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
    auto centeredOperand = ttir::utils::createDPSOp<ttir::SubtractOp>(
        rewriter, loc, operandType, adaptor.getOperand(), meanBcast);

    // stddev = sqrt(variance + epsilon)
    auto variancePlusEpsilon = ttir::utils::createDPSOp<ttir::AddOp>(
        rewriter, loc, operandType, varianceBcast, epsilonBcast);

    auto stddev = ttir::utils::createDPSOp<ttir::SqrtOp>(
        rewriter, loc, operandType, variancePlusEpsilon);

    // normalized_operand = centered_operand / stddev
    auto normalizedOperand = ttir::utils::createDPSOp<ttir::DivOp>(
        rewriter, loc, operandType, centeredOperand, stddev);

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
    auto i1 = ttir::utils::createDPSOp<ttir::MultiplyOp>(
        rewriter, loc, gradOutputType, adaptor.getGradOutput(),
        elementsPerFeatureBcast);

    // i2 = broadcast(sum(grad_output, reduction_dims))
    auto sumGradOutput = ttir::utils::createDPSOp<ttir::SumOp>(
        rewriter, loc, scaleType, adaptor.getGradOutput(),
        rewriter.getBoolAttr(false), reductionDimsAttr);
    auto i2 = broadcastFeatureToShape(rewriter, loc, sumGradOutput, operandType,
                                      featureIndex);

    // grad_output * centered_operand
    auto gradTimesCentered = ttir::utils::createDPSOp<ttir::MultiplyOp>(
        rewriter, loc, operandType, adaptor.getGradOutput(), centeredOperand);

    // i3 = broadcast(sum(grad_output * centered_operand))
    auto sumGradTimesCentered = ttir::utils::createDPSOp<ttir::SumOp>(
        rewriter, loc, scaleType, gradTimesCentered,
        rewriter.getBoolAttr(false), reductionDimsAttr);
    auto i3 = broadcastFeatureToShape(rewriter, loc, sumGradTimesCentered,
                                      operandType, featureIndex);

    // i4 = i3 * centered_operand
    auto i4 = ttir::utils::createDPSOp<ttir::MultiplyOp>(
        rewriter, loc, operandType, i3, centeredOperand);

    // i5 = i4 / (variance + epsilon)
    auto i5 = ttir::utils::createDPSOp<ttir::DivOp>(rewriter, loc, operandType,
                                                    i4, variancePlusEpsilon);

    // i6 = i1 - i2 - i5
    auto i1MinusI2 = ttir::utils::createDPSOp<ttir::SubtractOp>(
        rewriter, loc, operandType, i1, i2);

    auto i6 = ttir::utils::createDPSOp<ttir::SubtractOp>(
        rewriter, loc, operandType, i1MinusI2, i5);

    // grad_operand = (scale / stddev / elements_per_feature) * i6
    auto scaleOverStddev = ttir::utils::createDPSOp<ttir::DivOp>(
        rewriter, loc, operandType, scaleBcast, stddev);

    auto scaleOverStddevOverElem = ttir::utils::createDPSOp<ttir::DivOp>(
        rewriter, loc, operandType, scaleOverStddev, elementsPerFeatureBcast);

    auto gradOperand = ttir::utils::createDPSOp<ttir::MultiplyOp>(
        rewriter, loc, gradOperandType, scaleOverStddevOverElem, i6);

    // grad_scale = sum(grad_output * normalized_operand)
    auto gradTimesNorm = ttir::utils::createDPSOp<ttir::MultiplyOp>(
        rewriter, loc, operandType, adaptor.getGradOutput(), normalizedOperand);

    auto gradScale = ttir::utils::createDPSOp<ttir::SumOp>(
        rewriter, loc, gradScaleType, gradTimesNorm,
        rewriter.getBoolAttr(false), reductionDimsAttr);

    // grad_offset = sum(grad_output)
    auto gradOffset = ttir::utils::createDPSOp<ttir::SumOp>(
        rewriter, loc, gradOffsetType, adaptor.getGradOutput(),
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

    return ttir::utils::createDPSOp<ttir::ReshapeOp>(
        rewriter, loc, unsqueezeType, input,
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

    ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::ReshapeOp>(
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

    ttir::utils::replaceOpWithNewDPSOp<DestOp>(rewriter, srcOp, outputType,
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

    ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::DequantizeOp>(
        rewriter, srcOp, outputType, adaptor.getOperand());
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

    ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::ConvolutionOp>(
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
//  type.
//  - `CumSum` lowering only works for single-input/single-output cases and
//    must satisfy specific window/padding rules (see isCumSum()).
// This conversion is tailored toward cases like maxpool2d, avgpool2d (via
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
    // Check basic structure of the ReduceWindowOp
    if (!hasValidOpStructure(srcOp)) {
      return rewriter.notifyMatchFailure(
          srcOp, "Invalid structure of reduce window block.");
    }

    // Extract initialization constant value per input.
    std::optional<llvm::SmallVector<TypicalInitReductionValue>> initValues =
        extractInitValues(srcOp);
    if (!initValues) {
      return rewriter.notifyMatchFailure(
          srcOp, "Unable to extract constant initialization value.");
      ;
    }
    if (initValues->size() != srcOp.getInputs().size()) {
      return rewriter.notifyMatchFailure(
          srcOp, "Mismatch between inputs and init values.");
    }

    // Validate block body.
    Block &block = *srcOp.getBody().getBlocks().begin();
    auto &operations = block.getOperations();
    // Collect reduction ops.
    SmallVector<mlir::Operation *> reductionOps;
    for (Operation &op : llvm::drop_end(operations, 1)) {
      if (!isa<mlir::stablehlo::AddOp, mlir::stablehlo::MaxOp>(&op)) {
        return rewriter.notifyMatchFailure(srcOp,
                                           "Unsupported reduction body op.");
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
    // Handle cumsum case.
    if (srcOp.getInputs().size() == 1) {
      std::optional<int64_t> dimension =
          isCumSum(srcOp, adaptor, (*initValues)[0], reductionOps[0], padding);
      if (dimension) {
        mlir::RankedTensorType resultType = cast<RankedTensorType>(
            getTypeConverter()->convertType(srcOp.getResult(0).getType()));
        ttir::utils::replaceOpWithNewDPSOp<ttir::CumSumOp>(
            rewriter, srcOp, resultType, adaptor.getInputs()[0],
            rewriter.getI64IntegerAttr(*dimension));
        return success();
      }
    }
    // Build per-input pooling ops.
    SmallVector<Value> resultVals;
    for (size_t i = 0; i < srcOp.getInputs().size(); ++i) {
      Value input = adaptor.getInputs()[i];
      mlir::RankedTensorType resultType = cast<RankedTensorType>(
          getTypeConverter()->convertType(srcOp.getResult(i).getType()));
      TypicalInitReductionValue initVal = (*initValues)[i];
      mlir::Operation *frontOp = reductionOps[i];
      ttir::PoolingMethod method;
      if (isMaxPool(srcOp, initVal, frontOp)) {
        method = ttir::PoolingMethod::Max;
      } else if (isSumPool(srcOp, initVal, frontOp)) {
        std::optional<mlir::Operation *> divOp = extractDivisor(srcOp);
        if (divOp && i == 0) {
          method = ttir::PoolingMethod::Average;
          ttir::PoolingOp poolingOp = ttir::utils::createDPSOp<ttir::PoolingOp>(
              rewriter, srcOp.getLoc(), resultType, ValueRange{input}, method,
              windowDimensions, windowStrides, baseDilations, window_dilations,
              padding);
          resultVals.push_back(poolingOp->getResult(0));
          (*divOp)->getResult(0).replaceAllUsesWith(poolingOp->getResult(0));
          rewriter.eraseOp(*divOp);
          continue;
        }
        method = ttir::PoolingMethod::Sum;
      } else {
        return rewriter.notifyMatchFailure(srcOp, "Unsupported pooling method");
      }
      ttir::PoolingOp poolingOp = ttir::utils::createDPSOp<ttir::PoolingOp>(
          rewriter, srcOp.getLoc(), resultType, ValueRange{input}, method,
          windowDimensions, windowStrides, baseDilations, window_dilations,
          padding);
      llvm::append_range(resultVals, poolingOp->getResults());
    }
    rewriter.replaceOp(srcOp, resultVals);
    return success();
  }

private:
  // Requirements for max pooling.
  // 1. Front op in the block must be 'max'.
  // 2. InitValue must be negative infinity.
  bool isMaxPool(mlir::stablehlo::ReduceWindowOp &srcOp,
                 TypicalInitReductionValue initValue,
                 mlir::Operation *frontOp) const {
    if (!isa<mlir::stablehlo::MaxOp>(frontOp)) {
      return false;
    }
    if (initValue != TypicalInitReductionValue::NEG_INF) {
      return false;
    }
    return true;
  }

  // Requirements for sum pooling.
  // 1. Front op in the block must be 'add'.
  // 2. InitValue must be zero.
  bool isSumPool(mlir::stablehlo::ReduceWindowOp &srcOp,
                 TypicalInitReductionValue initValue,
                 mlir::Operation *frontOp) const {
    if (!isa<mlir::stablehlo::AddOp>(frontOp)) {
      return false;
    }
    if (initValue != TypicalInitReductionValue::ZERO) {
      return false;
    }

    return true;
  }

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
  //    dimension and value must be qual to size of the required dimension.
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

  // Extract the constant initialization value.
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
    if (initValues.size() != srcOp.getInitValues().size()) {
      return std::nullopt;
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

      ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::BroadcastOp>(
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

      ttir::ReshapeOp reshapeOp = ttir::utils::createDPSOp<ttir::ReshapeOp>(
          rewriter, srcOp.getLoc(), unsqueezeShape, outputType.getElementType(),
          outputType.getEncoding(), adaptor.getOperand(), reshapeDimAttr);

      ::llvm::ArrayRef<int64_t> inputShape = unsqueezeShape;
      ::llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

      SmallVector<int64_t> broadcastShape =
          ttmlir::utils::getBroadcastDimensions<int64_t>(inputShape,
                                                         outputShape);

      ttir::utils::replaceOpWithNewDPSOp<ttir::BroadcastOp>(
          rewriter, srcOp, outputType, reshapeOp, broadcastShape);
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

    ttir::utils::replaceOpWithNewDPSOp<DestOp>(
        rewriter, srcOp, outputType, adaptor.getLhs(), adaptor.getRhs());

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

    ttir::utils::replaceOpWithNewDPSOp<ttir::ConcatOp>(
        rewriter, srcOp, outputType, adaptor.getInputs(),
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

    if (getStableHLOOpType(srcOp) == StableHLOOpType::kLogical) {
      ttir::utils::replaceOpWithNewDPSOp<LogicalDestOp>(
          rewriter, srcOp, outputType, adaptor.getOperands());
    } else {
      ttir::utils::replaceOpWithNewDPSOp<BitwiseDestOp>(
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

template <typename SrcOpT>
static llvm::ErrorOr<ttcore::ReduceType> getReduceType(SrcOpT srcOp) {
  if constexpr (!std::is_same<SrcOpT, mlir::stablehlo::AllReduceOp>::value &&
                !std::is_same<SrcOpT,
                              mlir::stablehlo::ReduceScatterOp>::value) {
    return llvm::ErrorOr<ttcore::ReduceType>(
        std::make_error_code(std::errc::operation_not_supported));
  }
  // Check operations in the first block and determine reduce type for now
  // TODO(wooseoklee): This pattern matching mechanism may need to be updated as
  // we see complicated patterns of reduce block in the future.
  auto &block = srcOp.getRegion().front();
  for (Operation &op : block) {
    if (isa<mlir::stablehlo::AddOp>(op)) {
      return ttcore::ReduceType::Sum;
    }
    if (isa<mlir::stablehlo::MaxOp>(op)) {
      return ttcore::ReduceType::Max;
    }
    if (isa<mlir::stablehlo::MinOp>(op)) {
      return ttcore::ReduceType::Min;
    }
  }
  // Other reduce types are currently not supported
  return llvm::ErrorOr<ttcore::ReduceType>(
      std::make_error_code(std::errc::operation_not_supported));
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
      // Currently, we ensure if it is DEVICE_TO_DEVICE commmuincaiton.
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

      auto allReduceOp = ttir::utils::createDPSOp<mlir::tt::ttir::AllReduceOp>(
          rewriter, srcOp.getLoc(), outputType, inputOperand, *reduceType,
          clusterAxis);

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
      // Currently, we ensure if it is DEVICE_TO_DEVICE commmuincaiton.
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

    ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::ReduceScatterOp>(
        rewriter, srcOp, outputType, adaptor.getOperands()[0], *reduceType,
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

    ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::AllGatherOp>(
        rewriter, srcOp, outputType, adaptor.getOperands()[0],
        adaptor.getAllGatherDim(), clusterAxis);

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
      // Currently, we ensure if it is DEVICE_TO_DEVICE commmuincaiton.
      // Consider preserving this information in the future if the attribute
      // is non-DEVICE_TO_DEVICE values.
      auto channelType =
          static_cast<StableHLOChannelType>(srcChannelHandleAttr.getType());
      if (channelType != StableHLOChannelType::kChannelTypeDeviceToDevice &&
          channelType != StableHLOChannelType::kChannelTypeInvalid) {
        return failure();
      }
    }

    ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::CollectivePermuteOp>(
        rewriter, srcOp, outputType, adaptor.getOperand(),
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
    auto shardStatusAttr =
        definingOp->getAttrOfType<mlir::tt::ttcore::ShardStatusAttr>(
            mlir::tt::ttcore::ShardStatusAttr::name);

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

    ttir::utils::replaceOpWithNewDPSOp<ttir::SliceStaticOp>(
        rewriter, srcOp, outputType, adaptor.getOperand(),
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
      auto reshapedIndex = ttir::utils::createDPSOp<ttir::ReshapeOp>(
          rewriter, srcOp.getLoc(), singleElementTensorType.getShape(),
          startIndexElementType, singleElementTensorType.getEncoding(),
          startIndex, rewriter.getI32ArrayAttr({1}));
      startIndicesValues1D.push_back(reshapedIndex);
    }
    // Create a single 1D tensor from start indices values using concat op.
    auto startIndicesTensorType = RankedTensorType::get(
        {static_cast<int64_t>(startIndicesValues1D.size())},
        startIndexElementType);
    auto startIndicesTensor =
        ttir::utils::createDPSOp<mlir::tt::ttir::ConcatOp>(
            rewriter, srcOp.getLoc(), startIndicesTensorType.getShape(),
            startIndexElementType, startIndicesTensorType.getEncoding(),
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
    auto endIndicesTensor = ttir::utils::createDPSOp<mlir::tt::ttir::AddOp>(
        rewriter, srcOp.getLoc(), startIndicesTensorType.getShape(),
        startIndexElementType, startIndicesTensorType.getEncoding(),
        startIndicesTensor, sliceSizesConstant);

    ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::SliceDynamicOp>(
        rewriter, srcOp, outputType, adaptor.getOperand(), startIndicesTensor,
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

    ttir::utils::replaceOpWithNewDPSOp<ttir::ClampTensorOp>(
        rewriter, srcOp, outputType, adaptor.getOperand(), adaptor.getMin(),
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

    ttir::utils::replaceOpWithNewDPSOp<ttir::GatherOp>(
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

    ttir::utils::replaceOpWithNewDPSOp<ttir::ScatterOp>(
        rewriter, srcOp, outputType, operand, scatterIndices, update,
        llvm::SmallVector<int32_t>(updateWindowsDims),
        llvm::SmallVector<int32_t>(insertedWindowDims),
        llvm::SmallVector<int32_t>(inputBatchingDims),
        llvm::SmallVector<int32_t>(scatterIndicesBatchingDims),
        llvm::SmallVector<int32_t>(scatterDimsToOperandDims), indexVectorDim,
        indicesAreSorted, uniqueIndices);

    return success();
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

    SmallVector<Value> outputTensors{rewriter.create<ttir::EmptyOp>(
        loc, valueType.getShape(), valueType.getElementType(),
        valueType.getEncoding())};

    // Step 2: Determine Sort Type and output tensor preparation for 'indices'.

    SortType sortType;
    if (isValueIndexSort(srcOp.getInputs())) {
      sortType = SortType::kValueIndex;

      auto indicesType = cast<RankedTensorType>(
          getTypeConverter()->convertType(srcOp.getResultTypes()[1]));

      outputTypes.push_back(indicesType);
      outputTensors.push_back(rewriter.create<ttir::EmptyOp>(
          loc, indicesType.getShape(), indicesType.getElementType(),
          indicesType.getEncoding()));
    } else {
      sortType = (srcOp.getInputs().size() == 1) ? SortType::kValueOnly
                                                 : SortType::kKeyValue;

      IntegerType indexType = rewriter.getI32Type();
      RankedTensorType indicesType = RankedTensorType::get(
          valueType.getShape(), indexType, valueType.getEncoding());

      outputTypes.push_back(indicesType);
      outputTensors.push_back(rewriter.create<ttir::EmptyOp>(
          loc, indicesType.getShape(), indicesType.getElementType(),
          indicesType.getEncoding()));
    }

    // Step 3: Emit SortOp.

    auto sortOp = rewriter.create<ttir::SortOp>(
        loc, outputTypes, adaptor.getInputs().front(), outputTensors,
        rewriter.getSI32IntegerAttr(sortDim),
        rewriter.getBoolAttr(*isDescending), rewriter.getBoolAttr(isStable));

    // Step 4: SortType-specific lowering.

    // SortType::kValueOnly - Replace values output and ingnore indices output.
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
    auto reshape = ttir::utils::createDPSOp<ttir::ReshapeOp>(
        rewriter, loc, expandedType, indices,
        rewriter.getI32ArrayAttr(reshapeDim));

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
    Value concatIndices = ttir::utils::createDPSOp<ttir::ConcatOp>(
        rewriter, loc, concatType, toConcat, rank);

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

      auto gathered = ttir::utils::createDPSOp<ttir::GatherOp>(
          rewriter, loc, valType, srcOp.getInputs()[i], concatIndices,
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

    ttir::utils::replaceOpWithNewDPSOp<ttir::ReverseOp>(
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
      auto indicesType = RankedTensorType::get(
          {numIndices, rank}, rewriter.getI64Type(), inputType.getEncoding());

      auto indicesAttr = DenseIntElementsAttr::get(indicesType, flatIndices);
      Value indicesTensor = rewriter.create<ttir::ConstantOp>(
          srcOp.getLoc(), indicesType, indicesAttr);
      SmallVector<int32_t> insertedWindowDims =
          llvm::to_vector(llvm::seq<int32_t>(0, rank));

      ttir::utils::replaceOpWithNewDPSOp<ttir::ScatterOp>(
          rewriter, srcOp, outputType,
          /*input=*/fullOp.getResult(),
          /*scatter_indices=*/indicesTensor,
          /*update=*/adaptor.getOperand(),
          /*update_window_dims=*/SmallVector<int32_t>{},
          /*inserted_window_dims=*/insertedWindowDims,
          /*input_batching_dims=*/SmallVector<int32_t>{},
          /*scatter_indices_batching_dims=*/SmallVector<int32_t>{},
          /*scatter_dims_to_operand_dims=*/insertedWindowDims,
          /*index_vector_dim=*/1,
          /*indices_are_sorted=*/false,
          /*unique_indices=*/true);

      return success();
    }

    SmallVector<int32_t> padDim;
    for (uint32_t i = 0; i < adaptor.getEdgePaddingLow().size(); i++) {
      padDim.push_back(adaptor.getEdgePaddingLow()[i]);
      padDim.push_back(adaptor.getEdgePaddingHigh()[i]);
    }

    ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::PadOp>(
        rewriter, srcOp, outputType, adaptor.getOperand(),
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

    ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::AllToAllOp>(
        rewriter, srcOp, outputType, adaptor.getOperands()[0],
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

    ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::CollectiveBroadcastOp>(
        rewriter, srcOp, outputType, adaptor.getOperand(),
        adaptor.getReplicaGroups());

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
    auto typecastOp =
        mlir::tt::ttir::utils::createDPSOp<mlir::tt::ttir::TypecastOp>(
            rewriter, srcOp.getLoc(), outputType, randOp.getResult());

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

    Value cache = adaptor.getOperands()[0];
    Value input = adaptor.getOperands()[1];
    Value updateIndex = adaptor.getOperands()[2];
    Value pageTable;
    if (adaptor.getOperands().size() == 4) {
      pageTable = adaptor.getOperands()[3];
    }

    bool shareCache = false;
    uint32_t batchOffset = 0;
    mlir::DictionaryAttr frontendAttributes =
        mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
            srcOp->getDiscardableAttr("mhlo.frontend_attributes"));
    if (!frontendAttributes) {
      return rewriter.notifyMatchFailure(
          srcOp,
          "PagedUpdateCache op must have mhlo.frontend_attributes attribute.");
    }

    auto shareCacheStringAttr =
        frontendAttributes.getAs<mlir::StringAttr>("share_cache");
    if (shareCacheStringAttr) {
      if (shareCacheStringAttr.getValue().lower() == "true") {
        shareCache = true;
      } else if (shareCacheStringAttr.getValue().lower() == "false") {
        shareCache = false;
      } else {
        return rewriter.notifyMatchFailure(
            srcOp, "share_cache attribute must be true or false. Received \"" +
                       shareCacheStringAttr.getValue() + "\".");
      }
    }

    auto batchOffsetStringAttr =
        frontendAttributes.getAs<mlir::StringAttr>("batch_offset");
    if (!batchOffsetStringAttr) {
      return rewriter.notifyMatchFailure(
          srcOp, "PagedUpdateCache op must have batch_offset attribute.");
    }

    if (!llvm::to_integer(batchOffsetStringAttr.getValue(), batchOffset)) {
      return rewriter.notifyMatchFailure(
          srcOp, "Failed to convert batch_offset string to integer.");
    }

    rewriter.replaceOpWithNewOp<ttir::PagedUpdateCacheOp>(
        srcOp, cache.getType(), cache, input, updateIndex, shareCache,
        pageTable, batchOffset);

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

    ttir::utils::replaceOpWithNewDPSOp<mlir::tt::ttir::ErfOp>(
        rewriter, srcOp,
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

    auto hasAttentionMaskStringAttr =
        frontendAttributes.getAs<mlir::StringAttr>("has_attention_mask");
    bool hasAttentionMask = false;
    if (!hasAttentionMaskStringAttr) {
      return rewriter.notifyMatchFailure(
          srcOp, "has_attention_mask attribute must be present.");
    }

    if (hasAttentionMaskStringAttr.getValue().lower() == "true") {
      hasAttentionMask = true;
    } else if (hasAttentionMaskStringAttr.getValue().lower() == "false") {
      hasAttentionMask = false;
    } else {
      return rewriter.notifyMatchFailure(
          srcOp,
          "has_attention_mask attribute must be true or false. Received \"" +
              hasAttentionMaskStringAttr.getValue() + "\".");
    }

    auto hasAttentionSinkStringAttr =
        frontendAttributes.getAs<mlir::StringAttr>("has_attention_sink");
    bool hasAttentionSink = false;
    if (!hasAttentionSinkStringAttr) {
      return rewriter.notifyMatchFailure(
          srcOp, "has_attention_sink attribute must be present.");
    }

    if (hasAttentionSinkStringAttr.getValue().lower() == "true") {
      hasAttentionSink = true;
    } else if (hasAttentionSinkStringAttr.getValue().lower() == "false") {
      hasAttentionSink = false;
    } else {
      return rewriter.notifyMatchFailure(
          srcOp,
          "has_attention_sink attribute must be true or false. Received \"" +
              hasAttentionSinkStringAttr.getValue() + "\".");
    }

    Value query = adaptor.getOperands()[0];
    Value key = adaptor.getOperands()[1];
    Value value = adaptor.getOperands()[2];
    Value curPosTensor = adaptor.getOperands()[3];

    RankedTensorType outputType = cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getResult(0).getType()));
    ttir::EmptyOp outputTensor = rewriter.create<ttir::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    if (hasAttentionMask && hasAttentionSink) {
      rewriter.replaceOpWithNewOp<
          mlir::tt::ttir::ScaledDotProductAttentionDecodeOp>(
          srcOp,
          cast<RankedTensorType>(
              getTypeConverter()->convertType(srcOp.getResult(0).getType())),
          query, key, value, isCausalAttr, adaptor.getOperands()[4],
          curPosTensor, adaptor.getOperands()[5], outputTensor, scaleAttr);
    } else if (hasAttentionMask) {
      rewriter.replaceOpWithNewOp<
          mlir::tt::ttir::ScaledDotProductAttentionDecodeOp>(
          srcOp,
          cast<RankedTensorType>(
              getTypeConverter()->convertType(srcOp.getResult(0).getType())),
          query, key, value, isCausalAttr, adaptor.getOperands()[4],
          curPosTensor, nullptr, outputTensor, scaleAttr);
    } else if (hasAttentionSink) {
      rewriter.replaceOpWithNewOp<
          mlir::tt::ttir::ScaledDotProductAttentionDecodeOp>(
          srcOp,
          cast<RankedTensorType>(
              getTypeConverter()->convertType(srcOp.getResult(0).getType())),
          query, key, value, isCausalAttr, nullptr, curPosTensor,
          adaptor.getOperands()[4], outputTensor, scaleAttr);
    } else if (!hasAttentionMask && !hasAttentionSink) {
      rewriter.replaceOpWithNewOp<
          mlir::tt::ttir::ScaledDotProductAttentionDecodeOp>(
          srcOp,
          cast<RankedTensorType>(
              getTypeConverter()->convertType(srcOp.getResult(0).getType())),
          query, key, value, isCausalAttr, nullptr, curPosTensor, nullptr,
          outputTensor, scaleAttr);
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
            srcOp, "is_causal attribute must be true or false. Recived \"" +
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
            "scale attribute string must be convertible to float. Recived \"" +
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

    ttir::EmptyOp outputTensor = rewriter.create<ttir::EmptyOp>(
        srcOp.getLoc(), outputType.getShape(), outputType.getElementType());

    Value attentionMask = nullptr;
    if (adaptor.getOperands().size() == 4) {
      attentionMask = adaptor.getOperands()[3];
    }

    rewriter.replaceOpWithNewOp<ttir::ScaledDotProductAttentionOp>(
        srcOp, outputType, query, key, value, attentionMask, outputTensor,
        isCausalAttr, scaleAttr, /*slidingWindowSize=*/nullptr);

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
  patterns.add<StableHLOToTTIRScatterOpConversionPattern>(typeConverter, ctx);
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
  patterns
      .add<StableHLOToTTIRScaledDotProductAttentionDecodeOpConversionPattern,
           StableHLOToTTIRScaledDotProductAttentionOpConversionPattern>(
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
