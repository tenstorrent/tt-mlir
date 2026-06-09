// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpModel/TTNN/TTNNOutputTensorInference.h"

#ifdef TTMLIR_ENABLE_OPMODEL

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/Conversion.h"

#include "llvm/Support/raw_ostream.h"

#include <ttnn/tensor/tensor_spec.hpp>

// Forward declarations of internal helpers from TTNNOpModel.cpp
namespace mlir::tt::ttnn::op_model {
llvm::Expected<::ttnn::TensorSpec> getPrepareConv2dWeightsOpOutputTensorSpec(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    uint32_t in_channels, uint32_t out_channels, uint32_t batch_size,
    uint32_t input_height, uint32_t input_width,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    uint32_t groups, std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig, bool hasBias,
    bool transpose);

llvm::Expected<::ttnn::TensorSpec>
getPrepareMoEComputeW0W1WeightsOpOutputTensorSpec(
    llvm::ArrayRef<int64_t> w0Shape, TTNNLayoutAttr w0Layout,
    llvm::ArrayRef<int64_t> w1Shape, TTNNLayoutAttr w1Layout,
    std::optional<llvm::ArrayRef<int64_t>> bias0Shape,
    std::optional<TTNNLayoutAttr> bias0Layout,
    std::optional<llvm::ArrayRef<int64_t>> bias1Shape,
    std::optional<TTNNLayoutAttr> bias1Layout, uint32_t hiddenSize,
    uint32_t intermediateSize, std::optional<uint32_t> bhRingSize);

llvm::Expected<::ttnn::TensorSpec>
getPrepareMoEComputeW2WeightsOpOutputTensorSpec(
    llvm::ArrayRef<int64_t> w2Shape, TTNNLayoutAttr w2Layout,
    std::optional<llvm::ArrayRef<int64_t>> bias2Shape,
    std::optional<TTNNLayoutAttr> bias2Layout, uint32_t hiddenSize,
    uint32_t intermediateSize, std::optional<uint32_t> bhRingSize);

llvm::Expected<std::vector<::ttnn::TensorSpec>>
getMoeComputeOpOutputTensorSpecs(
    llvm::ArrayRef<int64_t> tilizeInputShape, TTNNLayoutAttr tilizeInputLayout,
    llvm::ArrayRef<int64_t> indicesShape, TTNNLayoutAttr indicesLayout,
    llvm::ArrayRef<int64_t> scoresShape, TTNNLayoutAttr scoresLayout,
    llvm::ArrayRef<int64_t> mappingShape, TTNNLayoutAttr mappingLayout,
    llvm::ArrayRef<int64_t> w0w1Shape, TTNNLayoutAttr w0w1Layout,
    llvm::ArrayRef<int64_t> w2Shape, TTNNLayoutAttr w2Layout, uint32_t layerId,
    uint32_t outputHeightShardDim, uint32_t intermediateSize, bool hasBias,
    ttcore::MoEActivationFunction activation,
    std::optional<uint32_t> bhRingSize);
} // namespace mlir::tt::ttnn::op_model

#endif // TTMLIR_ENABLE_OPMODEL

namespace mlir::tt::ttnn::op_model {

mlir::RankedTensorType
getPreparedConv2dWeightsOutputTensor(Conv2dOp *op,
                                     Conv2dConfigAttr conv2dConfig) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto input = op->getInput().getType();
  auto weight = op->getWeight().getType();
  auto inputLayout = mlir::cast<TTNNLayoutAttr>(input.getEncoding());
  auto weightLayout = mlir::cast<TTNNLayoutAttr>(weight.getEncoding());

  llvm::Expected<::ttnn::TensorSpec> outputTensorSpec =
      getPrepareConv2dWeightsOpOutputTensorSpec(
          input.getShape(), inputLayout, weight.getShape(), weightLayout,
          op->getInChannels(), op->getOutChannels(), op->getBatchSize(),
          op->getInputHeight(), op->getInputWidth(), op->getKernelSize(),
          op->getStride(), op->getPadding(), op->getDilation(), op->getGroups(),
          conv2dConfig, op->getConv2dSliceConfigAttr(),
          op->getBias() != nullptr,
          /* transpose */ false);

  if (!outputTensorSpec) {
    llvm::errs() << llvm::toString(outputTensorSpec.takeError());
    assert(false && "Failed to calculate conv2d prepared weights shape.");
  }

  // Convert back to RankedTensorType
  auto deviceGrid =
      ttcore::lookupDevice(op->getOperation()).getWorkerGrid().getShape();

  auto outputLayout = conversion::getLayoutAttrFromTensorSpec(
      op->getContext(), outputTensorSpec.get(), deviceGrid);

  auto shape = outputTensorSpec.get().logical_shape();

  return mlir::RankedTensorType::get(
      llvm::SmallVector<int64_t>(shape.cbegin(), shape.cend()),
      outputLayout.getScalarElementType(), outputLayout);
#else
  assert(false &&
         "Cannot calculate conv2d prepared weights shape without op model");
#endif
}

mlir::RankedTensorType
getPreparedConvTranspose2dWeightsOutputTensor(ConvTranspose2dOp *op,
                                              Conv2dConfigAttr conv2dConfig) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto input = op->getInput().getType();
  auto weight = op->getWeight().getType();
  auto inputLayout = mlir::cast<TTNNLayoutAttr>(input.getEncoding());
  auto weightLayout = mlir::cast<TTNNLayoutAttr>(weight.getEncoding());

  llvm::Expected<::ttnn::TensorSpec> outputTensorSpec =
      getPrepareConv2dWeightsOpOutputTensorSpec(
          input.getShape(), inputLayout, weight.getShape(), weightLayout,
          op->getInChannels(), op->getOutChannels(), op->getBatchSize(),
          op->getInputHeight(), op->getInputWidth(), op->getKernelSize(),
          op->getStride(), op->getPadding(), op->getDilation(), op->getGroups(),
          conv2dConfig, op->getConv2dSliceConfig(), op->getBias() != nullptr,
          /* transpose */ true);

  if (!outputTensorSpec) {
    llvm::errs() << llvm::toString(outputTensorSpec.takeError());
    assert(false &&
           "Failed to calculate conv_transpose2d prepared weights shape.");
  }

  // Convert back to RankedTensorType
  auto deviceGrid =
      ttcore::lookupDevice(op->getOperation()).getWorkerGrid().getShape();

  auto outputLayout = conversion::getLayoutAttrFromTensorSpec(
      op->getContext(), outputTensorSpec.get(), deviceGrid);

  auto shape = outputTensorSpec.get().logical_shape();

  return mlir::RankedTensorType::get(
      llvm::SmallVector<int64_t>(shape.cbegin(), shape.cend()),
      outputLayout.getScalarElementType(), outputLayout);
#else
  assert(false && "Cannot calculate conv_transpose2d prepared weights shape "
                  "without op model");
#endif
}

mlir::RankedTensorType
getPreparedMoEComputeW0W1WeightsOutputType(PrepareMoEComputeW0W1WeightsOp *op) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto w0Type = mlir::cast<mlir::RankedTensorType>(op->getW0().getType());
  auto w1Type = mlir::cast<mlir::RankedTensorType>(op->getW1().getType());
  auto w0Layout = mlir::cast<TTNNLayoutAttr>(w0Type.getEncoding());
  auto w1Layout = mlir::cast<TTNNLayoutAttr>(w1Type.getEncoding());

  std::optional<llvm::ArrayRef<int64_t>> bias0Shape, bias1Shape;
  std::optional<TTNNLayoutAttr> bias0Layout, bias1Layout;
  if (auto b0 = op->getBias_0()) {
    auto t = mlir::cast<mlir::RankedTensorType>(b0.getType());
    bias0Shape = t.getShape();
    bias0Layout = mlir::cast<TTNNLayoutAttr>(t.getEncoding());
  }
  if (auto b1 = op->getBias_1()) {
    auto t = mlir::cast<mlir::RankedTensorType>(b1.getType());
    bias1Shape = t.getShape();
    bias1Layout = mlir::cast<TTNNLayoutAttr>(t.getEncoding());
  }
  auto specOrErr = getPrepareMoEComputeW0W1WeightsOpOutputTensorSpec(
      w0Type.getShape(), w0Layout, w1Type.getShape(), w1Layout, bias0Shape,
      bias0Layout, bias1Shape, bias1Layout, op->getHiddenSize(),
      op->getIntermediateSize(), op->getBhRingSize());
  if (!specOrErr) {
    llvm::errs() << llvm::toString(specOrErr.takeError());
    assert(false &&
           "Failed to calculate prepare_moe_compute_w0_w1_weights shape.");
  }

  auto deviceGrid =
      ttcore::lookupDevice(op->getOperation()).getWorkerGrid().getShape();
  auto outLayout = conversion::getLayoutAttrFromTensorSpec(
      op->getContext(), specOrErr.get(), deviceGrid);

  auto shape = specOrErr.get().logical_shape();

  return mlir::RankedTensorType::get(
      llvm::SmallVector<int64_t>(shape.cbegin(), shape.cend()),
      outLayout.getScalarElementType(), outLayout);
#else
  assert(false &&
         "Cannot calculate prepare_moe_compute_w0_w1_weights shape without "
         "op model");
#endif
}

mlir::RankedTensorType
getPreparedMoEComputeW2WeightsOutputType(PrepareMoEComputeW2WeightsOp *op) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto w2Type = mlir::cast<mlir::RankedTensorType>(op->getW2().getType());
  auto w2Layout = mlir::cast<TTNNLayoutAttr>(w2Type.getEncoding());

  std::optional<llvm::ArrayRef<int64_t>> bias2Shape;
  std::optional<TTNNLayoutAttr> bias2Layout;
  if (auto b2 = op->getBias_2()) {
    auto t = mlir::cast<mlir::RankedTensorType>(b2.getType());
    bias2Shape = t.getShape();
    bias2Layout = mlir::cast<TTNNLayoutAttr>(t.getEncoding());
  }
  auto specOrErr = getPrepareMoEComputeW2WeightsOpOutputTensorSpec(
      w2Type.getShape(), w2Layout, bias2Shape, bias2Layout, op->getHiddenSize(),
      op->getIntermediateSize(), op->getBhRingSize());
  if (!specOrErr) {
    llvm::errs() << llvm::toString(specOrErr.takeError());
    assert(false &&
           "Failed to calculate prepare_moe_compute_w2_weights shape.");
  }

  auto deviceGrid =
      ttcore::lookupDevice(op->getOperation()).getWorkerGrid().getShape();

  auto outLayout = conversion::getLayoutAttrFromTensorSpec(
      op->getContext(), specOrErr.get(), deviceGrid);

  auto shape = specOrErr.get().logical_shape();

  return mlir::RankedTensorType::get(
      llvm::SmallVector<int64_t>(shape.cbegin(), shape.cend()),
      outLayout.getScalarElementType(), outLayout);
#else
  assert(false && "Cannot calculate prepare_moe_compute_w2_weights shape "
                  "without op model");
#endif
}

llvm::SmallVector<mlir::RankedTensorType>
getMoeComputeOutputTypes(MoeComputeOp *op) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto operand = [](mlir::Value v) {
    auto t = mlir::cast<mlir::RankedTensorType>(v.getType());
    return std::make_pair(t.getShape(),
                          mlir::cast<TTNNLayoutAttr>(t.getEncoding()));
  };
  auto [tilizeInputShape, tilizeInputLayout] =
      operand(op->getTilizeInputTensor());
  auto [indicesShape, indicesLayout] =
      operand(op->getTilizeExpertIndicesTensor());
  auto [scoresShape, scoresLayout] = operand(op->getTilizeExpertScoresTensor());
  auto [mappingShape, mappingLayout] =
      operand(op->getTilizeExpertMappingTensor());
  auto [w0w1Shape, w0w1Layout] = operand(op->getMatmulW0W1Tensor());
  auto [w2Shape, w2Layout] = operand(op->getMatmulW2Tensor());

  std::optional<uint32_t> bhRingSize = op->getBhRingSize();

  auto specsOrErr = getMoeComputeOpOutputTensorSpecs(
      tilizeInputShape, tilizeInputLayout, indicesShape, indicesLayout,
      scoresShape, scoresLayout, mappingShape, mappingLayout, w0w1Shape,
      w0w1Layout, w2Shape, w2Layout, op->getLayerId(),
      op->getOutputHeightShardDim(), op->getIntermediateSize(),
      op->getHasBias(), op->getActivationFunction(), bhRingSize);
  if (!specsOrErr) {
    llvm::errs() << llvm::toString(specsOrErr.takeError());
    assert(false && "Failed to calculate moe_compute output shapes.");
  }

  auto deviceGrid =
      ttcore::lookupDevice(op->getOperation()).getWorkerGrid().getShape();
  llvm::SmallVector<mlir::RankedTensorType> outputTypes;
  for (const auto &spec : specsOrErr.get()) {
    auto outLayout = conversion::getLayoutAttrFromTensorSpec(op->getContext(),
                                                             spec, deviceGrid);
    auto shape = spec.logical_shape();
    outputTypes.push_back(mlir::RankedTensorType::get(
        llvm::SmallVector<int64_t>(shape.cbegin(), shape.cend()),
        outLayout.getScalarElementType(), outLayout));
  }
  return outputTypes;
#else
  assert(false &&
         "Cannot calculate moe_compute output shapes without op model");
#endif
}

} // namespace mlir::tt::ttnn::op_model
