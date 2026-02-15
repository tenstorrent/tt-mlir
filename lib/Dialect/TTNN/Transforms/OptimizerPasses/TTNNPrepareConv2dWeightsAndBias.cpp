// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <type_traits>

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"
#include "ttmlir/OpModel/TTNN/TTNNOutputTensorInference.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNPREPARECONV2DWEIGHTSANDBIAS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNPrepareConv2dWeightsAndBias
    : public impl::TTNNPrepareConv2dWeightsAndBiasBase<
          TTNNPrepareConv2dWeightsAndBias> {

public:
  using impl::TTNNPrepareConv2dWeightsAndBiasBase<
      TTNNPrepareConv2dWeightsAndBias>::TTNNPrepareConv2dWeightsAndBiasBase;

  // Insert PrepareConv2dWeightsOp and PrepareConv2dBiasOp before every Conv2dOp
  // and PrepareConvTranspose2dWeightsOp and PrepareConvTranspose2dBiasOp before
  // every ConvTranspose2dOp that prepares weights and bias for convolution.
  // This is a prerequisite for const evaluation, which will improve performance
  // by eliminating the need for preprocessing the weights and bias on the
  // host/device.
  void runOnOperation() final {
#ifndef TTMLIR_ENABLE_OPMODEL
    llvm::llvm_unreachable_internal(
        "TTNNPrepareConv2dWeightsAndBias pass requires OpModel support to be "
        "enabled.");
#else
    // Device lifecycle is managed by OpModelDeviceWrapperPass in the pipeline,
    // but for standalone pass usage, the guard opens/closes it.
    op_model::ScopedSingletonDeviceGuard deviceGuard(getOperation());

    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    moduleOp.walk([&](ttnn::Conv2dOp convOp) {
      processConvOp<ttnn::Conv2dOp, ttnn::PrepareConv2dWeightsOp,
                    ttnn::PrepareConv2dBiasOp>(convOp, moduleOp, rewriter);
    });

    moduleOp.walk([&](ttnn::ConvTranspose2dOp convOp) {
      processConvOp<ttnn::ConvTranspose2dOp,
                    ttnn::PrepareConvTranspose2dWeightsOp,
                    ttnn::PrepareConvTranspose2dBiasOp>(convOp, moduleOp,
                                                        rewriter);
    });
#endif // TTMLIR_ENABLE_OPMODEL
  }

private:
  // Unified helper to get prepared bias type - works for any conv op.
  ::mlir::RankedTensorType getPreparedBiasType(Value bias,
                                               Type newElementType) {
    auto oldType = mlir::cast<mlir::RankedTensorType>(bias.getType());
    auto oldLayout = mlir::cast<ttnn::TTNNLayoutAttr>(oldType.getEncoding());

    auto newLayout = ttnn::TTNNLayoutAttr::get(
        &getContext(), oldType.getShape(),
        ttcore::TileType::get(newElementType), BufferType::DRAM,
        oldLayout.getGrid(),
        ttnn::TensorMemoryLayoutAttr::get(
            &getContext(), ttnn::TensorMemoryLayout::Interleaved));

    return mlir::RankedTensorType::get(oldType.getShape(), newElementType,
                                       newLayout);
  }

  ::mlir::RankedTensorType
  getPreparedWeightsType(ttnn::Conv2dOp conv2dOp,
                         ttnn::Conv2dConfigAttr conv2dConfig) {
    return op_model::getPreparedConv2dWeightsOutputTensor(&conv2dOp,
                                                          conv2dConfig);
  }

  ::mlir::RankedTensorType
  getPreparedWeightsType(ttnn::ConvTranspose2dOp convTranspose2dOp,
                         ttnn::Conv2dConfigAttr conv2dConfig) {
    return op_model::getPreparedConvTranspose2dWeightsOutputTensor(
        &convTranspose2dOp, conv2dConfig);
  }

  template <typename ConvOp, typename PrepareWeightsOp, typename PrepareBiasOp>
  void processConvOp(ConvOp convOp, ModuleOp moduleOp, IRRewriter &rewriter) {
    ttnn::TTNNLayoutAttr weightLayoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
        convOp.getWeight().getType().getEncoding());
    assert(weightLayoutAttr.getBufferType() == ttnn::BufferType::SystemMemory &&
           weightLayoutAttr.getLayout() == ttnn::Layout::RowMajor &&
           "Weight must be in system memory and row-major layout when "
           "calling TTNNPrepareConv2dWeightsAndBias pass.");

    if (convOp.getBias()) {
      ttnn::TTNNLayoutAttr biasLayoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
          convOp.getBias().getType().getEncoding());
      assert(biasLayoutAttr.getBufferType() == ttnn::BufferType::SystemMemory &&
             biasLayoutAttr.getLayout() == ttnn::Layout::RowMajor &&
             "Bias must be in system memory and row-major layout when "
             "calling TTNNPrepareConv2dWeightsAndBias pass");
    }

    ttnn::TTNNLayoutAttr inputLayoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
        convOp.getInput().getType().getEncoding());

    ttnn::TTNNLayoutAttr outputLayoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
        convOp.getResult().getType().getEncoding());

    ttcore::GridAttr deviceGrid =
        ttcore::lookupDevice(moduleOp).getWorkerGrid();
    ttnn::MemoryConfigAttr inputMemConfigAttr =
        ttnn::MemoryConfigAttr::get(inputLayoutAttr, deviceGrid);

    // Input and output dtype attr on prepare api is used to specify
    // input/output dtype for the conv2d operation.
    auto inputDtypeAttr = mlir::tt::ttcore::DataTypeAttr::get(
        &getContext(), inputLayoutAttr.getDataType());
    auto outputDtypeAttr = mlir::tt::ttcore::DataTypeAttr::get(
        &getContext(), outputLayoutAttr.getDataType());

    // When weight dtype is set in conv2d config prepare API will typecast the
    // output into desired dtype. Until we have some different use case we
    // will put prepared bias/weight into the same dtype as input.
    auto conv2dConfig = convOp.getConv2dConfigAttr()
                            ? convOp.getConv2dConfigAttr()
                            : Conv2dConfigAttr::get(&getContext());

    Type inputElementType = inputLayoutAttr.getScalarElementType();
    conv2dConfig = conv2dConfig.withWeightsDtype(inputDtypeAttr.getValue());

    rewriter.setInsertionPoint(convOp);

    // Create prepare weights op - differs slightly between Conv2d and
    // ConvTranspose2d.
    Value preparedWeights = createPrepareWeightsOp<ConvOp, PrepareWeightsOp>(
        convOp, rewriter, inputMemConfigAttr, inputLayoutAttr, inputDtypeAttr,
        outputDtypeAttr, conv2dConfig);

    // Create prepare bias op if bias exists.
    Value preparedBias;
    if (convOp.getBias()) {
      preparedBias = createPrepareBiasOp<ConvOp, PrepareBiasOp>(
          convOp, rewriter, inputMemConfigAttr, inputLayoutAttr,
          inputElementType, inputDtypeAttr, outputDtypeAttr, conv2dConfig);
    }

    // Update conv op to use prepared weights and bias.
    rewriter.modifyOpInPlace(convOp, [&]() {
      convOp.getWeightMutable().assign(preparedWeights);

      if (convOp.getBias()) {
        convOp.getBiasMutable().assign(preparedBias);
      }

      // Since we are updating the weight and bias output dtype we must
      // update the conv2d config attr as well, since metal uses it to
      // determine if weight and bias are already prepared.
      convOp.setConv2dConfigAttr(conv2dConfig);
    });
  }

  template <typename ConvOp, typename PrepareWeightsOp>
  Value createPrepareWeightsOp(ConvOp convOp, IRRewriter &rewriter,
                               ttnn::MemoryConfigAttr inputMemConfigAttr,
                               ttnn::TTNNLayoutAttr inputLayoutAttr,
                               mlir::tt::ttcore::DataTypeAttr inputDtypeAttr,
                               mlir::tt::ttcore::DataTypeAttr outputDtypeAttr,
                               ttnn::Conv2dConfigAttr conv2dConfig) {
    if constexpr (std::is_same_v<ConvOp, ttnn::Conv2dOp>) {
      return rewriter.create<PrepareWeightsOp>(
          ttmlir::utils::appendLocationSuffix(convOp.getLoc(),
                                              "_prepare_conv2d_weight"),
          getPreparedWeightsType(convOp, conv2dConfig), convOp.getWeight(),
          inputMemConfigAttr,
          rewriter.getAttr<ttnn::LayoutAttr>(inputLayoutAttr.getLayout()),
          /*weights_format=*/rewriter.getStringAttr("OIHW"),
          convOp.getInChannelsAttr(), convOp.getOutChannelsAttr(),
          convOp.getBatchSizeAttr(), convOp.getInputHeightAttr(),
          convOp.getInputWidthAttr(), convOp.getKernelSizeAttr(),
          convOp.getStrideAttr(), convOp.getPaddingAttr(),
          convOp.getDilationAttr(),
          rewriter.getBoolAttr(convOp.getBias() != nullptr),
          convOp.getGroupsAttr(), convOp.getDevice(), inputDtypeAttr,
          outputDtypeAttr, conv2dConfig, convOp.getComputeConfigAttr(),
          convOp.getConv2dSliceConfigAttr());
    } else {
      return rewriter.create<PrepareWeightsOp>(
          ttmlir::utils::appendLocationSuffix(
              convOp.getLoc(), "_prepare_conv_transpose2d_weight"),
          getPreparedWeightsType(convOp, conv2dConfig), convOp.getWeight(),
          inputMemConfigAttr,
          rewriter.getAttr<ttnn::LayoutAttr>(inputLayoutAttr.getLayout()),
          /*weights_format=*/rewriter.getStringAttr("IOHW"),
          convOp.getInChannelsAttr(), convOp.getOutChannelsAttr(),
          convOp.getBatchSizeAttr(), convOp.getInputHeightAttr(),
          convOp.getInputWidthAttr(), convOp.getKernelSizeAttr(),
          convOp.getStrideAttr(), convOp.getPaddingAttr(),
          convOp.getDilationAttr(),
          rewriter.getBoolAttr(convOp.getBias() != nullptr),
          convOp.getGroupsAttr(), convOp.getDevice(), inputDtypeAttr,
          outputDtypeAttr, conv2dConfig, convOp.getComputeConfigAttr(),
          convOp.getConv2dSliceConfigAttr(),
          /*mirror_kernel=*/rewriter.getBoolAttr(true));
    }
  }

  template <typename ConvOp, typename PrepareBiasOp>
  Value createPrepareBiasOp(ConvOp convOp, IRRewriter &rewriter,
                            ttnn::MemoryConfigAttr inputMemConfigAttr,
                            ttnn::TTNNLayoutAttr inputLayoutAttr,
                            Type inputElementType,
                            mlir::tt::ttcore::DataTypeAttr inputDtypeAttr,
                            mlir::tt::ttcore::DataTypeAttr outputDtypeAttr,
                            ttnn::Conv2dConfigAttr conv2dConfig) {
    if constexpr (std::is_same_v<ConvOp, ttnn::Conv2dOp>) {
      return rewriter.create<PrepareBiasOp>(
          ttmlir::utils::appendLocationSuffix(convOp.getLoc(),
                                              "_prepare_conv2d_bias"),
          getPreparedBiasType(convOp.getBias(), inputElementType),
          convOp.getBias(), inputMemConfigAttr,
          rewriter.getAttr<ttnn::LayoutAttr>(inputLayoutAttr.getLayout()),
          convOp.getInChannelsAttr(), convOp.getOutChannelsAttr(),
          convOp.getBatchSizeAttr(), convOp.getInputHeightAttr(),
          convOp.getInputWidthAttr(), convOp.getKernelSizeAttr(),
          convOp.getStrideAttr(), convOp.getPaddingAttr(),
          convOp.getDilationAttr(), convOp.getGroupsAttr(), convOp.getDevice(),
          inputDtypeAttr, outputDtypeAttr, conv2dConfig,
          convOp.getComputeConfigAttr(), convOp.getConv2dSliceConfigAttr());
    } else {
      return rewriter.create<PrepareBiasOp>(
          ttmlir::utils::appendLocationSuffix(convOp.getLoc(),
                                              "_prepare_conv_transpose2d_bias"),
          getPreparedBiasType(convOp.getBias(), inputElementType),
          convOp.getBias(), inputMemConfigAttr,
          rewriter.getAttr<ttnn::LayoutAttr>(inputLayoutAttr.getLayout()),
          convOp.getInChannelsAttr(), convOp.getOutChannelsAttr(),
          convOp.getBatchSizeAttr(), convOp.getInputHeightAttr(),
          convOp.getInputWidthAttr(), convOp.getKernelSizeAttr(),
          convOp.getStrideAttr(), convOp.getPaddingAttr(),
          convOp.getDilationAttr(), convOp.getGroupsAttr(), convOp.getDevice(),
          inputDtypeAttr, outputDtypeAttr, conv2dConfig,
          convOp.getComputeConfigAttr(), convOp.getConv2dSliceConfigAttr());
    }
  }
};
} // namespace mlir::tt::ttnn
