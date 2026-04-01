// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOLegalizeComposite.h"

#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"

#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_LEGALIZESTABLEHLOCOMPOSITETOTTIR
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {

template <typename TargetOp>
class StableHLOToTTIRCompositeOpConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompositeOp> {

  using OpConversionPattern<mlir::stablehlo::CompositeOp>::OpConversionPattern;

public:
  StableHLOToTTIRCompositeOpConversionPattern(MLIRContext *context,
                                              llvm::StringRef opName)
      : OpConversionPattern<mlir::stablehlo::CompositeOp>(context),
        opName(opName) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                  mlir::stablehlo::CompositeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getName() != opName) {
      return rewriter.notifyMatchFailure(
          srcOp, ("CompositeOp must be " + std::string(opName) + ".").c_str());
    }
    if (srcOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "CompositeOp must have exactly one result.");
    }

    auto outputType =
        mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

    auto compositeAttrs = srcOp.getCompositeAttributes();
    SmallVector<NamedAttribute> namedAttrs;
    if (compositeAttrs) {
      for (const auto &attr : compositeAttrs) {
        namedAttrs.push_back(attr);
      }
    }

    rewriter.replaceOpWithNewOp<TargetOp>(srcOp, outputType,
                                          adaptor.getOperands(), namedAttrs);
    return success();
  }

private:
  std::string opName;
};

// Special handling for all tenstorrent.topk* composite ops.
// Three different composite ops are supported:
// - tenstorrent.topk: generated when both the values and indices are needed in
// the graph.
// - tenstorrent.topk_indices: generated when only the indices are needed in the
// graph.
// - tenstorrent.topk_values: generated when only the values are needed in the
// graph.
class TenstorrentTopKConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompositeOp> {
public:
  TenstorrentTopKConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CompositeOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                  mlir::stablehlo::CompositeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!supportedOpNames.contains(srcOp.getName())) {
      return failure();
    }
    if (adaptor.getOperands().size() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, srcOp.getName() +
                     " composite op must have exactly one input operand.");
    }

    bool isTopKWithValues = srcOp.getName() == "tenstorrent.topk_values";
    bool isTopKWithIndices = srcOp.getName() == "tenstorrent.topk_indices";
    bool isTopKWithBoth = srcOp.getName() == "tenstorrent.topk";

    if (isTopKWithBoth && srcOp->getNumResults() != 2) {
      return rewriter.notifyMatchFailure(
          srcOp,
          "tenstorrent.topk composite op must have exactly two results.");
    }
    if (isTopKWithValues && srcOp->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp,
          "tenstorrent.topk_values composite op must have exactly one result.");
    }
    if (isTopKWithIndices && srcOp->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "tenstorrent.topk_indices composite "
                                         "op must have exactly one result.");
    }

    SmallVector<RankedTensorType, 2> resultTypes;
    if (isTopKWithBoth) {
      resultTypes = {
          mlir::cast<RankedTensorType>(srcOp.getResult(0).getType()),
          mlir::cast<RankedTensorType>(srcOp.getResult(1).getType())};
    } else {
      resultTypes = {
          mlir::cast<RankedTensorType>(srcOp.getResult(0).getType())};
    }

    DictionaryAttr compositeAttrs = srcOp.getCompositeAttributes();
    IntegerAttr kAttr = IntegerAttr::get(rewriter.getIntegerType(32), 1);
    IntegerAttr dimAttr = IntegerAttr::get(rewriter.getIntegerType(32), -1);
    BoolAttr sortedAttr = BoolAttr::get(rewriter.getContext(), false);
    BoolAttr largestAttr = BoolAttr::get(rewriter.getContext(), true);

    if (compositeAttrs) {
      if (auto attr = compositeAttrs.getAs<IntegerAttr>("k")) {
        int64_t val = attr.getInt();
        if (!llvm::isInt<32>(val)) {
          return rewriter.notifyMatchFailure(
              srcOp, "k value is too large for i32: " + Twine(val));
        }
        kAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(val));
      }
      if (auto attr = compositeAttrs.getAs<IntegerAttr>("dim")) {
        int64_t val = attr.getInt();
        if (!llvm::isInt<32>(val)) {
          return rewriter.notifyMatchFailure(
              srcOp, "dim value is too large for i32: " + Twine(val));
        }
        dimAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(val));
      }
      if (auto attr = compositeAttrs.getAs<BoolAttr>("largest")) {
        largestAttr = attr;
      }
      if (auto attr = compositeAttrs.getAs<BoolAttr>("sorted")) {
        sortedAttr = attr;
      }
    } else {
      return rewriter.notifyMatchFailure(
          srcOp,
          "tenstorrent.topk composite op must have composite_attributes.");
    }

    auto inputType =
        mlir::cast<RankedTensorType>(adaptor.getOperands()[0].getType());
    RankedTensorType valuesType, indicesType;

    if (isTopKWithBoth) {
      valuesType = resultTypes[0];
      indicesType = resultTypes[1];
    } else if (isTopKWithValues) {
      valuesType = resultTypes[0];
      indicesType =
          RankedTensorType::get(valuesType.getShape(), rewriter.getI32Type());
    } else {
      auto indicesResultType = resultTypes[0];
      valuesType = RankedTensorType::get(indicesResultType.getShape(),
                                         inputType.getElementType());
      indicesType = indicesResultType;
    }

    auto input = adaptor.getOperands()[0];
    auto topKOp = rewriter.create<ttir::TopKOp>(
        srcOp.getLoc(), valuesType, indicesType, input, kAttr, dimAttr,
        largestAttr, sortedAttr);

    if (isTopKWithBoth) {
      rewriter.replaceOp(srcOp, {topKOp.getValues(), topKOp.getIndices()});
    } else if (isTopKWithValues) {
      rewriter.replaceOp(srcOp, {topKOp.getValues()});
    } else {
      rewriter.replaceOp(srcOp, {topKOp.getIndices()});
    }

    return success();
  }

private:
  llvm::SmallSet<llvm::StringRef, 3> supportedOpNames = {
      "tenstorrent.topk",
      "tenstorrent.topk_indices",
      "tenstorrent.topk_values",
  };
};

// Special handling for tenstorrent.uniform -> ttir.rand, as
// it requires extracting values from operands and translating them to
// attributes, and because ttir.rand is a non-DPS op.
class TenstorrentUniformToRandConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompositeOp> {

public:
  TenstorrentUniformToRandConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CompositeOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                  mlir::stablehlo::CompositeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getName() != "tenstorrent.uniform") {
      return failure();
    }

    auto outputType =
        mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

    DictionaryAttr compositeAttrs = srcOp.getCompositeAttributes();

    // Extract shape attribute.
    auto sizeAttr = mlir::cast<ArrayAttr>(compositeAttrs.get("shape"));

    // Extract low and high from constant operands.
    auto lowOp =
        adaptor.getOperands()[1].getDefiningOp<mlir::stablehlo::ConstantOp>();
    auto highOp =
        adaptor.getOperands()[2].getDefiningOp<mlir::stablehlo::ConstantOp>();

    assert(lowOp && "low operand must be a ConstantOp");
    assert(highOp && "high operand must be a ConstantOp");

    auto lowValue = mlir::cast<DenseFPElementsAttr>(lowOp.getValue());
    auto highValue = mlir::cast<DenseFPElementsAttr>(highOp.getValue());

    assert(lowValue.getNumElements() == 1 &&
           "Expected low operand to be a scalar constant");
    assert(highValue.getNumElements() == 1 &&
           "Expected high operand to be a scalar constant");

    auto lowAttr = rewriter.getF32FloatAttr(lowValue.getValues<float>()[0]);
    auto highAttr = rewriter.getF32FloatAttr(highValue.getValues<float>()[0]);

    // Proceed with default seed = 0 for now, because in tt-metal it will
    // actually generate different random numbers on each execution, which we
    // agreed is acceptable for training use cases for now. This workaround is
    // needed because seed is of tensor type in StableHLO, but float in tt-metal
    // and actual conversion can't be done.
    auto seedAttr = rewriter.getUI32IntegerAttr(0);

    rewriter.replaceOpWithNewOp<ttir::RandOp>(
        srcOp, outputType, sizeAttr, TypeAttr::get(outputType.getElementType()),
        lowAttr, highAttr, seedAttr);
    return success();
  }
};

// Special handling for tenstorrent.rms_norm -> ttir.rms_norm
// Converts normalized_shape tensor attribute to DenseI64ArrayAttr
// and sets operandSegmentSizes for AttrSizedOperandSegments
class TenstorrentRMSNormConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompositeOp> {

public:
  TenstorrentRMSNormConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CompositeOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                  mlir::stablehlo::CompositeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getName() != "tenstorrent.rms_norm") {
      return failure();
    }

    if (srcOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "CompositeOp must have exactly one result.");
    }

    auto outputType =
        mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

    DictionaryAttr compositeAttrs = srcOp.getCompositeAttributes();

    auto normalizedShapeAttr = compositeAttrs.get("normalized_shape");
    SmallVector<int64_t> normalizedShapeVec;

    if (auto denseAttr =
            mlir::dyn_cast<DenseIntElementsAttr>(normalizedShapeAttr)) {
      for (auto val : denseAttr.getValues<int64_t>()) {
        normalizedShapeVec.push_back(val);
      }
    } else if (auto arrayAttr =
                   mlir::dyn_cast<ArrayAttr>(normalizedShapeAttr)) {
      for (auto attr : arrayAttr) {
        normalizedShapeVec.push_back(mlir::cast<IntegerAttr>(attr).getInt());
      }
    } else {
      return rewriter.notifyMatchFailure(
          srcOp, "normalized_shape must be a dense tensor or array attribute");
    }

    auto normalizedShapeDenseAttr =
        rewriter.getDenseI64ArrayAttr(normalizedShapeVec);

    auto epsilonAttr = compositeAttrs.get("epsilon");

    SmallVector<NamedAttribute> namedAttrs;
    namedAttrs.push_back(
        rewriter.getNamedAttr("normalized_shape", normalizedShapeDenseAttr));
    if (epsilonAttr) {
      namedAttrs.push_back(rewriter.getNamedAttr("epsilon", epsilonAttr));
    }

    // ttir.rms_norm has AttrSizedOperandSegments: [input, weight, bias, output]
    size_t numOperands = adaptor.getOperands().size();
    SmallVector<int32_t> segmentSizes;
    if (numOperands == 3) { // input, weight, bias
      segmentSizes = {1, 1, 1};
    } else if (numOperands == 2) { // input, weight
      segmentSizes = {1, 1, 0};
    } else { // input
      segmentSizes = {1, 0, 0};
    }

    namedAttrs.push_back(rewriter.getNamedAttr(
        "operandSegmentSizes", rewriter.getDenseI32ArrayAttr(segmentSizes)));

    rewriter.replaceOpWithNewOp<ttir::RMSNormOp>(
        srcOp, outputType, adaptor.getOperands(), namedAttrs);
    return success();
  }
};

// Converts stablehlo.custom_call @tenstorrent.rms_norm -> ttir.rms_norm.
// This handles composites that were converted to custom_calls by
// FlattenGenericCompositesPass because they have custom sharding rules.
// Attributes are carried in the "tt.composite_attributes" discardable attr.
class CustomCallRMSNormConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CustomCallOp> {

public:
  CustomCallRMSNormConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CustomCallOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CustomCallOp srcOp,
                  mlir::stablehlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getCallTargetNameAttr() != "tenstorrent.rms_norm") {
      return failure();
    }

    if (srcOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "CustomCallOp must have exactly one result.");
    }

    auto outputType =
        mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

    auto compositeAttrs = mlir::dyn_cast_or_null<DictionaryAttr>(
        srcOp->getDiscardableAttr("tt.composite_attributes"));
    if (!compositeAttrs) {
      return rewriter.notifyMatchFailure(
          srcOp, "missing tt.composite_attributes on custom_call");
    }

    // Determine the number of normalized dimensions from the stored attribute.
    // The actual dimension sizes come from the (possibly sharded) input tensor,
    // since UpdateGlobalToLocalShapes may have rewritten types to local shapes.
    auto normalizedShapeAttr = compositeAttrs.get("normalized_shape");
    int64_t numNormalizedDims = 0;

    if (auto denseAttr =
            mlir::dyn_cast<DenseIntElementsAttr>(normalizedShapeAttr)) {
      numNormalizedDims = denseAttr.getNumElements();
    } else if (auto arrayAttr =
                   mlir::dyn_cast<ArrayAttr>(normalizedShapeAttr)) {
      numNormalizedDims = arrayAttr.size();
    } else {
      return rewriter.notifyMatchFailure(
          srcOp, "normalized_shape must be a dense tensor or array attribute");
    }

    auto inputType =
        mlir::cast<RankedTensorType>(srcOp.getOperand(0).getType());
    int64_t inputRank = inputType.getRank();
    if (numNormalizedDims > inputRank) {
      return rewriter.notifyMatchFailure(
          srcOp, "normalized_shape has more dims than input rank");
    }

    // Use actual trailing dimensions from the input tensor.
    SmallVector<int64_t> normalizedShapeVec;
    for (int64_t i = inputRank - numNormalizedDims; i < inputRank; ++i) {
      normalizedShapeVec.push_back(inputType.getDimSize(i));
    }

    auto normalizedShapeDenseAttr =
        rewriter.getDenseI64ArrayAttr(normalizedShapeVec);

    auto epsilonAttr = compositeAttrs.get("epsilon");

    SmallVector<NamedAttribute> namedAttrs;
    namedAttrs.push_back(
        rewriter.getNamedAttr("normalized_shape", normalizedShapeDenseAttr));
    if (epsilonAttr) {
      namedAttrs.push_back(rewriter.getNamedAttr("epsilon", epsilonAttr));
    }

    // ttir.rms_norm has AttrSizedOperandSegments: [input, weight, bias]
    size_t numOperands = adaptor.getOperands().size();
    SmallVector<int32_t> segmentSizes;
    if (numOperands == 3) { // input, weight, bias
      segmentSizes = {1, 1, 1};
    } else if (numOperands == 2) { // input, weight
      segmentSizes = {1, 1, 0};
    } else { // input
      segmentSizes = {1, 0, 0};
    }

    namedAttrs.push_back(rewriter.getNamedAttr(
        "operandSegmentSizes", rewriter.getDenseI32ArrayAttr(segmentSizes)));

    rewriter.replaceOpWithNewOp<ttir::RMSNormOp>(
        srcOp, outputType, adaptor.getOperands(), namedAttrs);
    return success();
  }
};

// Special handling for tenstorrent.layer_norm -> ttir.layer_norm
// Converts normalized_shape tensor attribute to DenseI64ArrayAttr
// and sets operandSegmentSizes for AttrSizedOperandSegments
class TenstorrentLayerNormConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompositeOp> {

public:
  TenstorrentLayerNormConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CompositeOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                  mlir::stablehlo::CompositeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getName() != "tenstorrent.layer_norm") {
      return failure();
    }

    if (srcOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "CompositeOp must have exactly one result.");
    }

    auto outputType =
        mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

    DictionaryAttr compositeAttrs = srcOp.getCompositeAttributes();

    auto normalizedShapeAttr = compositeAttrs.get("normalized_shape");
    SmallVector<int64_t> normalizedShapeVec;

    if (auto denseAttr =
            mlir::dyn_cast<DenseIntElementsAttr>(normalizedShapeAttr)) {
      for (auto val : denseAttr.getValues<int64_t>()) {
        normalizedShapeVec.push_back(val);
      }
    } else if (auto arrayAttr =
                   mlir::dyn_cast<ArrayAttr>(normalizedShapeAttr)) {
      for (auto attr : arrayAttr) {
        normalizedShapeVec.push_back(mlir::cast<IntegerAttr>(attr).getInt());
      }
    } else {
      return rewriter.notifyMatchFailure(
          srcOp, "normalized_shape must be a dense tensor or array attribute");
    }

    auto normalizedShapeDenseAttr =
        rewriter.getDenseI64ArrayAttr(normalizedShapeVec);

    auto epsilonAttr = compositeAttrs.get("epsilon");

    SmallVector<NamedAttribute> namedAttrs;
    namedAttrs.push_back(
        rewriter.getNamedAttr("normalized_shape", normalizedShapeDenseAttr));
    if (epsilonAttr) {
      namedAttrs.push_back(rewriter.getNamedAttr("epsilon", epsilonAttr));
    }

    // ttir.layer_norm has AttrSizedOperandSegments: [input, weight, bias]
    size_t numOperands = adaptor.getOperands().size();
    SmallVector<int32_t> segmentSizes;
    if (numOperands == 3) { // input, weight, bias
      segmentSizes = {1, 1, 1};
    } else if (numOperands == 2) { // input, weight
      segmentSizes = {1, 1, 0};
    } else { // input only
      segmentSizes = {1, 0, 0};
    }

    namedAttrs.push_back(rewriter.getNamedAttr(
        "operandSegmentSizes", rewriter.getDenseI32ArrayAttr(segmentSizes)));

    rewriter.replaceOpWithNewOp<ttir::LayerNormOp>(
        srcOp, outputType, adaptor.getOperands(), namedAttrs);
    return success();
  }
};

class ShardyAllSliceToTTIRMeshPartitionConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompositeOp> {
public:
  ShardyAllSliceToTTIRMeshPartitionConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CompositeOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                  mlir::stablehlo::CompositeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getName() != "sdy.all_slice") {
      return failure();
    }

    if (srcOp->getNumOperands() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp,
          "sdy.all_slice composite op must have exactly one input operand");
    }

    DictionaryAttr compositeAttrs = srcOp.getCompositeAttributes();
    auto maybeOutShardingAttr = compositeAttrs.get("out_sharding");
    if (!maybeOutShardingAttr) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "out_sharding attribute is required");
    }
    // Extract the out_sharding attribute
    mlir::ModuleOp moduleOp = srcOp->getParentOfType<mlir::ModuleOp>();
    mlir::sdy::MeshOp globalMeshOp = shardy_utils::getMeshOps(moduleOp)[0];
    mlir::sdy::TensorShardingAttr outShardingAttr =
        mlir::cast<mlir::sdy::TensorShardingAttr>(maybeOutShardingAttr);

    // Calculate the attributes for the ttir.mesh_shard op.
    llvm::Expected<mlir::tt::shardy_utils::ShardyMeshSharding>
        shardyMeshSharding =
            mlir::tt::shardy_utils::ShardyMeshSharding::generate(
                globalMeshOp.getMeshAttr(), outShardingAttr,
                mlir::tt::ttcore::ShardStatus::Unsharded,
                ttcore::MeshShardDirection::FullToShard);
    if (auto err = shardyMeshSharding.takeError()) {
      return rewriter.notifyMatchFailure(
          srcOp, "Error trying to parse shardy annotation when legalizing "
                 "sdy.all_slice composite op.");
    }
    auto shardDims = shardyMeshSharding->getShardDims();
    llvm::SmallVector<int32_t> tensorDims;
    llvm::SmallVector<uint32_t> clusterAxes;
    for (auto [dimIdx, dim] : llvm::enumerate(shardDims)) {
      if (dim >= 0) {
        tensorDims.push_back(static_cast<int32_t>(dim));
        clusterAxes.push_back(static_cast<uint32_t>(dimIdx));
      }
    }
    rewriter.setInsertionPoint(srcOp);

    mlir::Value currInput = adaptor.getOperands().front();
    auto meshShape = shardyMeshSharding->getMeshShape();
    // Replace the composite op with 1 or more ttir.mesh_partition ops.
    for (size_t i = 0; i < tensorDims.size(); ++i) {
      auto currInputType = mlir::cast<RankedTensorType>(currInput.getType());
      llvm::SmallVector<int64_t> newShape(currInputType.getShape().begin(),
                                          currInputType.getShape().end());
      if (static_cast<size_t>(tensorDims[i]) >= newShape.size()) {
        return rewriter.notifyMatchFailure(
            srcOp, "Invalid mesh partition dimension index.");
      }
      if (static_cast<size_t>(clusterAxes[i]) >= meshShape.size()) {
        return rewriter.notifyMatchFailure(srcOp, "Invalid mesh axis index.");
      }
      // Compute new shape for the result tensor, with original dimension
      // divided by mesh axis size
      int64_t meshAxisSize = meshShape[clusterAxes[i]];
      if (newShape[tensorDims[i]] == ShapedType::kDynamic ||
          newShape[tensorDims[i]] % meshAxisSize != 0) {
        return rewriter.notifyMatchFailure(
            srcOp,
            "Dimension size must be static and divisible by mesh axis size.");
      }
      newShape[tensorDims[i]] = newShape[tensorDims[i]] / meshAxisSize;
      auto resultType =
          mlir::RankedTensorType::get(newShape, currInputType.getElementType(),
                                      currInputType.getEncoding());
      currInput = rewriter.create<ttir::MeshPartitionOp>(
          srcOp->getLoc(), resultType, currInput,
          rewriter.getSI32IntegerAttr(tensorDims[i]),
          rewriter.getUI32IntegerAttr(clusterAxes[i]));
    }
    rewriter.replaceOp(srcOp, currInput);
    return success();
  }
};

// Special handling for tenstorrent.group_norm -> ttir.group_norm
// Extracts num_groups (int) and epsilon from composite attributes
// and sets operandSegmentSizes for AttrSizedOperandSegments
class TenstorrentGroupNormConversionPattern
    : public OpConversionPattern<mlir::stablehlo::CompositeOp> {

public:
  TenstorrentGroupNormConversionPattern(MLIRContext *context)
      : OpConversionPattern<mlir::stablehlo::CompositeOp>(context) {}

  LogicalResult
  matchAndRewrite(mlir::stablehlo::CompositeOp srcOp,
                  mlir::stablehlo::CompositeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (srcOp.getName() != "tenstorrent.group_norm") {
      return failure();
    }

    if (srcOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "CompositeOp must have exactly one result.");
    }

    auto outputType =
        mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

    DictionaryAttr compositeAttrs = srcOp.getCompositeAttributes();

    // Extract num_groups as a scalar integer attribute.
    auto numGroupsAttr = compositeAttrs.get("num_groups");
    IntegerAttr numGroupsIntAttr;
    if (auto intAttr = mlir::dyn_cast_or_null<IntegerAttr>(numGroupsAttr)) {
      numGroupsIntAttr = rewriter.getI64IntegerAttr(intAttr.getInt());
    } else if (auto denseAttr = mlir::dyn_cast_or_null<DenseIntElementsAttr>(
                   numGroupsAttr)) {
      // Handle case where num_groups comes as a single-element dense tensor.
      assert(denseAttr.getNumElements() == 1 &&
             "Expected num_groups to be a single-element dense tensor");
      numGroupsIntAttr =
          rewriter.getI64IntegerAttr((*denseAttr.getValues<int64_t>().begin()));
    } else {
      return rewriter.notifyMatchFailure(
          srcOp, "num_groups must be an integer attribute");
    }

    auto epsilonAttr = compositeAttrs.get("epsilon");

    auto channelDimAttr = compositeAttrs.get("channel_dim");
    assert(channelDimAttr && "channel_dim must be present");

    SmallVector<NamedAttribute> namedAttrs;
    namedAttrs.push_back(rewriter.getNamedAttr("num_groups", numGroupsIntAttr));
    if (epsilonAttr) {
      namedAttrs.push_back(rewriter.getNamedAttr("epsilon", epsilonAttr));
    }
    if (channelDimAttr) {
      namedAttrs.push_back(
          rewriter.getNamedAttr("channel_dim", channelDimAttr));
    }

    // ttir.group_norm has AttrSizedOperandSegments:
    //   [input, input_mask, weight, bias]
    // From StableHLO composite, we expect operands in order:
    //   input, [weight], [bias] (no input_mask from frontend).
    size_t numOperands = adaptor.getOperands().size();
    SmallVector<int32_t> segmentSizes;
    if (numOperands == 3) { // input, weight, bias
      segmentSizes = {1, 0, 1, 1};
    } else if (numOperands == 2) { // input, weight
      segmentSizes = {1, 0, 1, 0};
    } else { // input only
      segmentSizes = {1, 0, 0, 0};
    }

    namedAttrs.push_back(rewriter.getNamedAttr(
        "operandSegmentSizes", rewriter.getDenseI32ArrayAttr(segmentSizes)));

    rewriter.replaceOpWithNewOp<ttir::GroupNormOp>(
        srcOp, outputType, adaptor.getOperands(), namedAttrs);
    return success();
  }
};

struct LegalizeStableHLOCompositeToTTIR
    : public ttir::impl::LegalizeStableHLOCompositeToTTIRBase<
          LegalizeStableHLOCompositeToTTIR> {
  void runOnOperation() final {
    MLIRContext *context = &getContext();

    ConversionTarget target(*context);
    target.addLegalDialect<ttir::TTIRDialect>();
    // StableHLO is intentionally not marked as either legal or illegal.

    RewritePatternSet patterns(context);
    populateStableHLOCompositeLegalizationPatterns(context, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Verify that all custom_calls converted from composites with custom
    // sharding rules were lowered to TTIR ops. These were converted from
    // composites by FlattenGenericCompositesPass so Shardy could propagate
    // shardings at their boundary. If they survive unlowered, something is
    // wrong.
    bool hasUnlowered = false;
    getOperation().walk([&](mlir::stablehlo::CustomCallOp op) {
      if (op->hasAttr("tt.composite_attributes")) {
        op.emitError()
            << "custom_call '" << op.getCallTargetName()
            << "' (converted from composite with custom sharding rule) must be "
               "lowered to a TTIR op, but no lowering pattern matched";
        hasUnlowered = true;
      }
    });
    if (hasUnlowered) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>>
createLegalizeStableHLOCompositeToTTIRPass() {
  return std::make_unique<LegalizeStableHLOCompositeToTTIR>();
}

void populateStableHLOCompositeLegalizationPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<StableHLOToTTIRCompositeOpConversionPattern<ttir::GeluOp>>(
      context, "tenstorrent.gelu");
  patterns.add<StableHLOToTTIRCompositeOpConversionPattern<ttir::GeluOp>>(
      context, "tenstorrent.gelu_tanh");
  patterns.add<TenstorrentRMSNormConversionPattern>(context);
  patterns.add<CustomCallRMSNormConversionPattern>(context);
  patterns.add<TenstorrentLayerNormConversionPattern>(context);
  patterns.add<TenstorrentGroupNormConversionPattern>(context);
  patterns.add<TenstorrentUniformToRandConversionPattern>(context);
  patterns.add<TenstorrentTopKConversionPattern>(context);
  patterns.add<ShardyAllSliceToTTIRMeshPartitionConversionPattern>(context);
}
} // namespace mlir::tt
