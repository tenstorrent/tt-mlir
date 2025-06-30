// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRTraits.h"
#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/QuantUtils.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRQUANTDEQUANTCONVERSION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Rewrites Q(op(x)) into op(Q(x)) if the op supports quantized execution.
// If the op does not support quantized execution, inserts DQ → op → Q sandwich
// instead.
class CommuteQuantizeAboveQuantizableOpRewriter
    : public TTIRCommuteOpInterfaceRewritePattern<ttir::QuantizeOp,
                                                  QuantizableOpInterface> {
public:
  using TTIRCommuteOpInterfaceRewritePattern<
      ttir::QuantizeOp,
      QuantizableOpInterface>::TTIRCommuteOpInterfaceRewritePattern;

private:
  bool isCommuteViable(QuantizableOpInterface op,
                       ttir::QuantizeOp quantOp) const override {
    // For now, always assume QuantizableOp is commutable with Quantize.
    return true;
  }

  bool isCommuteFavorable(QuantizableOpInterface op,
                          ttir::QuantizeOp quantOp) const override {
    // Skip commuting if the QuantizeOp has been explicitly marked to opt-out.
    if (quantOp->hasAttr("ttir.skip_qdq_commute")) {
      return false;
    }
    return true;
  }

  void performCommuteRewrite(QuantizableOpInterface op,
                             ttir::QuantizeOp quantOp,
                             PatternRewriter &rewriter) const override {
    llvm::SmallVector<Operation *> users(op->getUsers());
    auto oldOpType = cast<RankedTensorType>(op->getResult(0).getType());
    auto oldQuantizeResultType = quantOp.getResult().getType();
    auto quantType = mlir::dyn_cast<quant::QuantizedType>(
        oldQuantizeResultType.getElementType());
    // Construct the expected quantized result type by applying the quantized
    // element type to the original op's shape and encoding.
    auto newOpType = RankedTensorType::get(oldOpType.getShape(), quantType,
                                           oldOpType.getEncoding());
    SmallVector<Value> newQuantOperands;
    // Quantize all input operands to prepare for pushing Quantize above the op.
    for (auto [i, operand] : llvm::enumerate(op->getOperands().drop_back())) {
      auto oldOperandType = cast<RankedTensorType>(operand.getType());
      auto newOperandType = RankedTensorType::get(
          oldOperandType.getShape(), quantType, oldOperandType.getEncoding());
      auto q = ttir::utils::createDPSOp<ttir::QuantizeOp>(
          rewriter, op->getLoc(), newOperandType, operand, quantOp->getAttrs());
      newQuantOperands.push_back(q);
    }
    newQuantOperands.push_back(rewriter.create<ttir::EmptyOp>(
        op->getLoc(), newOpType.getShape(), newOpType.getElementType(),
        newOpType.getEncoding()));

    // Call rewriteWithQuantizedInputs which returns the new operation.
    Operation *newOp =
        op.rewriteWithQuantizedInputs(rewriter, newQuantOperands, newOpType);

    if (newOp && !newOp->getResults().empty()) {
      // Op successfully rewritten in quantized form — eliminate original
      // QuantizeOp.
      rewriter.replaceOp(quantOp, newOp->getResult(0));
    } else {
      // Op could not be quantized directly — fall back to inserting:
      //   Quantize(op(Dequantize(...)))
      llvm::SmallVector<Value> dequantizedOperands;
      // Iterate over all the indices in newQuantOperands except the last one
      // and create a DequantizeOp
      for (size_t i = 0; i < newQuantOperands.size() - 1; ++i) {
        // the output type of the DQ is always the Q's input type.
        auto floatType =
            cast<RankedTensorType>(newQuantOperands[i]
                                       .getDefiningOp<ttir::QuantizeOp>()
                                       .getInput()
                                       .getType());
        auto dq = ttir::utils::createDPSOp<ttir::DequantizeOp>(
            rewriter, op->getLoc(), floatType, newQuantOperands[i]);
        dequantizedOperands.push_back(dq);
      }
      dequantizedOperands.push_back(op->getOperands().back());
      // Recreate the original op with dequantized inputs and float output type.
      OperationState state(op->getLoc(), op->getName());
      state.addOperands(ValueRange(dequantizedOperands));
      state.addTypes(TypeRange(op->getResult(0).getType()));
      state.addAttributes(op->getAttrs());
      Operation *newOp = rewriter.create(state);

      // Create a new quantize op whose input is the original op.
      auto newQuantOp = ttir::utils::createDPSOp<ttir::QuantizeOp>(
          rewriter, op->getLoc(), newOpType, newOp->getResult(0));
      // Mark this QuantizeOp to prevent future rewrites from attempting to
      // commute it again.
      newQuantOp->setAttr("ttir.skip_qdq_commute", rewriter.getUnitAttr());
      rewriter.replaceOp(quantOp, newQuantOp.getResult());
    }
  }
};

struct QuantizeConvolutionRewriter
    : public OpRewritePattern<ttir::ConvolutionOp> {
  using OpRewritePattern<ttir::ConvolutionOp>::OpRewritePattern;

  // helper function that takes an arbitrary set of mlir::Operations, checks
  // their inputs are both dequantize ops, then checks the inputs of those are
  // quantize ops.
  LogicalResult checkQuantizeDequantizePair(SmallVector<Value> ops) const {
    for (auto op : ops) {
      if (op.getDefiningOp() == nullptr) {
        return failure();
      }
      ttir::DequantizeOp dequantize =
          mlir::dyn_cast<ttir::DequantizeOp>(op.getDefiningOp());
      if (!dequantize) {
        return failure();
      }
      if (dequantize.getInput().getDefiningOp() == nullptr) {
        return failure();
      }
      ttir::QuantizeOp quantize = mlir::dyn_cast<ttir::QuantizeOp>(
          dequantize.getInput().getDefiningOp());
      if (!quantize) {
        return failure();
      }
      // check that dequantize and quantize have the same quantization scales
      // and zero points check both UniformQuantizedType and
      // UniformQuantizedPerAxisType.
      auto dequantizeElementType =
          mlir::cast<RankedTensorType>(dequantize.getInput().getType())
              .getElementType();
      auto quantizeElementType =
          mlir::cast<RankedTensorType>(quantize.getResult().getType())
              .getElementType();
      if (dequantizeElementType != quantizeElementType) {
        return failure();
      }
    }
    return success();
  }

  // Computes effective output scale given input and weight quantized types.
  // Handles:
  // 1. Per-tensor input + per-tensor weight
  // 2. Per-tensor input + per-channel weight
  // 3. Per-channel input + per-channel weight (must match axis length)
  // Returns scale[i] = input_scale * weight_scale[i] for each channel.
  FailureOr<std::pair<SmallVector<double>, SmallVector<int64_t>>>
  computeOutputScalesAndZeroPoint(quant::QuantizedType inputType,
                                  quant::QuantizedType weightType) const {
    SmallVector<double> scales;
    SmallVector<int64_t> zeroPoints;
    double inputScale;
    int64_t inputZeroPoint;
    if (quant::UniformQuantizedType uniformType =
            dyn_cast<quant::UniformQuantizedType>(inputType)) {
      inputScale = uniformType.getScale();
      inputZeroPoint = uniformType.getZeroPoint();
    } else if (quant::UniformQuantizedPerAxisType perAxisType =
                   dyn_cast<quant::UniformQuantizedPerAxisType>(inputType)) {
      return failure();
    }

    if (quant::UniformQuantizedType uniformType =
            dyn_cast<quant::UniformQuantizedType>(weightType)) {
      if (uniformType.getZeroPoint() != inputZeroPoint) {
        return failure();
      }
      double weightScale = uniformType.getScale();
      scales.push_back(inputScale * weightScale);
      zeroPoints.push_back(inputZeroPoint);
      return std::make_pair(scales, zeroPoints);
    }

    if (quant::UniformQuantizedPerAxisType perAxisType =
            dyn_cast<quant::UniformQuantizedPerAxisType>(weightType)) {
      auto weightZeroPoints = perAxisType.getZeroPoints();
      for (auto zeroPoint : weightZeroPoints) {
        if (zeroPoint != inputZeroPoint) {
          return failure();
        }
        zeroPoints.push_back(inputZeroPoint);
      }
      auto weightScales = perAxisType.getScales();
      for (auto scale : weightScales) {
        scales.push_back(inputScale * scale);
      }
      return std::make_pair(scales, zeroPoints);
    }

    return failure();
  }

  LogicalResult matchAndRewrite(ttir::ConvolutionOp op,
                                PatternRewriter &rewriter) const override {
    // call the helper function for both op.getInput and op.getWeight.
    SmallVector<Value> ops = {op.getInput(), op.getWeight()};
    if (failed(checkQuantizeDequantizePair(ops))) {
      return failure();
    }

    // check the quantization is i8:f32.
    auto quantInputOp = mlir::dyn_cast<ttir::QuantizeOp>(
        op.getInput().getDefiningOp()->getOperand(0).getDefiningOp());
    auto quantWeightOp = mlir::dyn_cast<ttir::QuantizeOp>(
        op.getWeight().getDefiningOp()->getOperand(0).getDefiningOp());
    auto quantInputType = mlir::dyn_cast<mlir::quant::QuantizedType>(
        mlir::cast<RankedTensorType>(quantInputOp.getOutput().getType())
            .getElementType());
    auto quantWeightType = mlir::dyn_cast<mlir::quant::QuantizedType>(
        mlir::cast<RankedTensorType>(quantWeightOp.getOutput().getType())
            .getElementType());
    if (!quantInputType || !quantWeightType) {
      return failure();
    }
    if (quantInputType.getStorageType().getIntOrFloatBitWidth() != 8 ||
        quantWeightType.getStorageType().getIntOrFloatBitWidth() != 8) {
      return failure();
    }

    // rewrite the convolution op to be quantized.
    // create the output quantized type, whose scale is input * weight and
    // storage type is i32.
    auto storageType =
        IntegerType::get(rewriter.getContext(), 32, IntegerType::Signed);
    mlir::FailureOr<std::pair<int64_t, int64_t>> storageTypeMinMax =
        mlir::tt::ttir::utils::getStorageTypeMinMax(storageType, op->getLoc());
    if (mlir::failed(storageTypeMinMax)) {
      return failure();
    }
    auto [storageTypeMin, storageTypeMax] = *storageTypeMinMax;
    mlir::FailureOr<std::pair<SmallVector<double>, SmallVector<int64_t>>>
        scalesAndZeroPoints =
            computeOutputScalesAndZeroPoint(quantInputType, quantWeightType);
    if (mlir::failed(scalesAndZeroPoints)) {
      return failure();
    }
    mlir::quant::QuantizedType quantOutputType;
    if (scalesAndZeroPoints->first.size() == 1) {
      quantOutputType = quant::UniformQuantizedType::get(
          quantInputType.getFlags(), storageType,
          quantInputType.getExpressedType(), scalesAndZeroPoints->first[0],
          scalesAndZeroPoints->second[0], storageTypeMin, storageTypeMax);
    } else {
      // get the axis from the per-axis weight.
      quant::UniformQuantizedPerAxisType perAxisWeightType =
          dyn_cast<quant::UniformQuantizedPerAxisType>(quantWeightType);
      quantOutputType = quant::UniformQuantizedPerAxisType::get(
          perAxisWeightType.getFlags(), storageType,
          perAxisWeightType.getExpressedType(), scalesAndZeroPoints->first,
          scalesAndZeroPoints->second,
          perAxisWeightType.getQuantizedDimension(), storageTypeMin,
          storageTypeMax);
    }
    // output shape and encoding is same as original op output shape.
    auto oldConvOutputType = cast<RankedTensorType>(op->getResult(0).getType());
    auto quantConvOutputType =
        RankedTensorType::get(oldConvOutputType.getShape(), quantOutputType,
                              oldConvOutputType.getEncoding());
    auto quantConv =
        mlir::tt::ttir::utils::createDPSOp<mlir::tt::ttir::ConvolutionOp>(
            rewriter, op->getLoc(), quantConvOutputType,
            op.getInput().getDefiningOp()->getOperand(0),
            op.getWeight().getDefiningOp()->getOperand(0), op.getBias(),
            op.getInputDilationAttr(), op.getWeightDilationAttr(),
            op.getWindowStridesAttr(), op.getPaddingAttr(),
            op.getWindowReversalAttr(), op.getConvolutionLayoutAttr(),
            op.getFeatureGroupCountAttr(), op.getBatchGroupCountAttr());
    quantConv->setAttr("ttir.skip_qdq_commute", rewriter.getUnitAttr());
    // now dequantize the convolution.
    auto dequantize =
        mlir::tt::ttir::utils::createDPSOp<mlir::tt::ttir::DequantizeOp>(
            rewriter, op->getLoc(), oldConvOutputType,
            quantConv.getOperation()->getResult(0));
    rewriter.replaceOp(op, dequantize);
    return success();
  }
};

} // namespace

class TTIRQuantDequantConversion
    : public impl::TTIRQuantDequantConversionBase<TTIRQuantDequantConversion> {
public:
  using impl::TTIRQuantDequantConversionBase<
      TTIRQuantDequantConversion>::TTIRQuantDequantConversionBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // Rewrite the quantized convolution.
    patterns.add<QuantizeConvolutionRewriter>(&getContext());
    // Register the QDQ commutation pattern and apply greedily to the module.
    patterns.add<CommuteQuantizeAboveQuantizableOpRewriter>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace mlir::tt::ttir
