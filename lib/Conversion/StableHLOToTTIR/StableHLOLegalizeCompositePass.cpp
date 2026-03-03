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
#include "stablehlo/dialect/StablehloOps.h"

#include <shardy/dialect/sdy/ir/dialect.h>

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
  patterns.add<TenstorrentLayerNormConversionPattern>(context);
  patterns.add<TenstorrentUniformToRandConversionPattern>(context);
  patterns.add<ShardyAllSliceToTTIRMeshPartitionConversionPattern>(context);
}
} // namespace mlir::tt
