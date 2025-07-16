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

// Rewrites op(DQ(x)) into DQ(op(x)) if the op supports quantized execution.
// If the op does not support quantized execution, inserts DQ → op → Q
// sandwich instead.
class CommuteDequantizeBelowQuantizableOpRewriter
    : public TTIRCommuteOpInterfaceRewritePattern<ttir::DequantizeOp,
                                                  QuantizableOpInterface,
                                                  CommuteDirection::DOWNWARDS> {
public:
  using TTIRCommuteOpInterfaceRewritePattern<
      ttir::DequantizeOp, QuantizableOpInterface,
      CommuteDirection::DOWNWARDS>::TTIRCommuteOpInterfaceRewritePattern;

private:
  mutable RankedTensorType oldOpType;
  mutable llvm::SmallVector<Value> quantOperands;

  bool isCommuteDownwardsViable(QuantizableOpInterface op,
                                ttir::DequantizeOp dequantOp) const override {
    // Require that this operand is one of the inputs to the op
    // and that its result has only one user (to avoid duplicating work)
    if (op->hasAttr("ttir.skip_qdq_commute")) {
      return false;
    }
    return dequantOp->hasOneUse();
  }

  bool
  isCommuteDownwardsFavorable(QuantizableOpInterface op,
                              ttir::DequantizeOp dequantOp) const override {
    // Collects all the operands and calls op.isQuantizedRewriteFavorable
    quantOperands.clear();
    oldOpType = mlir::cast<RankedTensorType>(op->getResult(0).getType());
    DestinationStyleOpInterface dps =
        mlir::cast<mlir::DestinationStyleOpInterface>(op.getOperation());
    for (mlir::OpOperand *operand : dps.getDpsInputOperands()) {
      auto dq = op->getOperand(operand->getOperandNumber())
                    .getDefiningOp<ttir::DequantizeOp>();
      if (dq) {
        quantOperands.push_back(dq.getOperand(0));
      } else {
        quantOperands.push_back(op->getOperand(operand->getOperandNumber()));
      }
    }
    return op.isQuantizedRewriteFavorable(quantOperands);
  }

  void
  performCommuteDownwardsRewrite(QuantizableOpInterface op,
                                 ttir::DequantizeOp dequantOp,
                                 PatternRewriter &rewriter) const override {
    // Call rewriteWithQuantizedInputs which returns the new operation.
    // If the op is successfully rewritten in quantized form, dequantize the
    // output and rewrite. We expect rewriteWithQuantizedOutput to identify
    // whether the quantOperands set is sufficient to rewrite the op in
    // quantized form.
    DestinationStyleOpInterface dps =
        mlir::cast<mlir::DestinationStyleOpInterface>(op.getOperation());
    Operation *newOp = op.rewriteWithQuantizedInputs(rewriter, quantOperands,
                                                     dps.getDpsInits());
    llvm::SmallVector<mlir::Value> newResults;
    if (newOp) {
      // Op successfully rewritten in quantized form. For every output of the
      // old op, replace it with the corresponding output of the new op.
      for (auto [oldResult, newResult] :
           llvm::zip(op->getResults(), newOp->getResults())) {
        RankedTensorType oldType =
            mlir::cast<RankedTensorType>(oldResult.getType());
        RankedTensorType newResultType =
            mlir::cast<RankedTensorType>(newResult.getType());
        if (mlir::dyn_cast<mlir::quant::QuantizedType>(
                newResultType.getElementType())) {
          // It's quantized, so insert a DequantizeOp
          auto newDequant =
              mlir::tt::ttir::utils::createDPSOp<mlir::tt::ttir::DequantizeOp>(
                  rewriter, op->getLoc(), oldType, newResult);
          newResults.push_back(newDequant);
        } else {
          // It's already floating-point or non-quantized
          newResults.push_back(newResult);
        }
      }
    } else {
      // Op could not be quantized directly — fall back to inserting:
      //   Dequantize(Quantize(op(Dequantize(...))))
      // For every output of the old op, replace it with
      // Dequantize(Quantize(Orig_Output))
      Operation *fallbackOp = rewriter.clone(*op.getOperation());
      fallbackOp->setAttr("ttir.skip_qdq_commute", rewriter.getUnitAttr());
      for (auto result : fallbackOp->getResults()) {
        // the quantize's output type is the same shape as the op output type,
        // just quantized (type taken from dequantOp)
        // TODO(anuhsing): enable multiple dequant types
        RankedTensorType originalType =
            mlir::cast<RankedTensorType>(result.getType());
        quant::QuantizedType quantType = mlir::dyn_cast<quant::QuantizedType>(
            dequantOp.getInput().getType().getElementType());
        RankedTensorType quantizeType = RankedTensorType::get(
            originalType.getShape(), quantType, originalType.getEncoding());
        // create quantize op
        mlir::tt::ttir::QuantizeOp quantize =
            mlir::tt::ttir::utils::createDPSOp<mlir::tt::ttir::QuantizeOp>(
                rewriter, op->getLoc(), quantizeType, result);
        // now dequantize op
        mlir::tt::ttir::DequantizeOp dequantize =
            mlir::tt::ttir::utils::createDPSOp<mlir::tt::ttir::DequantizeOp>(
                rewriter, op->getLoc(), originalType, quantize);
        newResults.push_back(dequantize);
      }
    }
    rewriter.replaceOp(op, newResults);
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
    // patterns.add<QuantizeConvolutionRewriter>(&getContext());
    // Register the QDQ commutation pattern and apply greedily to the module.
    // patterns.add<CommuteQuantizeAboveQuantizableOpRewriter>(&getContext());
    patterns.add<CommuteDequantizeBelowQuantizableOpRewriter>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    // clean up attribute
    getOperation()->walk(
        [](Operation *op) { op->removeAttr("ttir.skip_qdq_commute"); });
  }
};

} // namespace mlir::tt::ttir
