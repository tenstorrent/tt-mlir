// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"
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
  // that prepares weights and bias for convolution. This is a prerequisite for
  // const evaluation, which will improve performance by eliminating the need
  // for preprocessing the weights and bias on the host/device.
  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    moduleOp.walk([&](ttnn::Conv2dOp conv2dOp) {
      ttnn::TTNNLayoutAttr weightLayoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
          conv2dOp.getWeight().getType().getEncoding());
      assert(weightLayoutAttr.getBufferType() ==
                 ttnn::BufferType::SystemMemory &&
             weightLayoutAttr.getLayout() == ttnn::Layout::RowMajor &&
             "Weight must be in system memory and row-major layout when "
             "calling TTNNPrepareConv2dWeightsAndBias pass.");

      ttnn::TTNNLayoutAttr biasLayoutAttr;
      if (conv2dOp.getBias()) {
        biasLayoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
            conv2dOp.getBias().getType().getEncoding());
        assert(biasLayoutAttr.getBufferType() ==
                   ttnn::BufferType::SystemMemory &&
               biasLayoutAttr.getLayout() == ttnn::Layout::RowMajor &&
               "Bias must be in system memory and row-major layout when "
               "calling TTNNPrepareConv2dWeightsAndBias pass");
      }

      ttnn::TTNNLayoutAttr inputLayoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
          conv2dOp.getInput().getType().getEncoding());
      ttnn::TTNNLayoutAttr outputLayoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
          conv2dOp.getResult().getType().getEncoding());

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
      auto conv2dConfig = conv2dOp.getConv2dConfigAttr()
                              ? conv2dOp.getConv2dConfigAttr()
                              : Conv2dConfigAttr::get(&getContext());

      Type inputElementType = inputLayoutAttr.getScalarElementType();
      conv2dConfig = conv2dConfig.withWeightsDtype(inputDtypeAttr.getValue());

      rewriter.setInsertionPoint(conv2dOp);
      ttnn::PrepareConv2dWeightsOp prepareConv2dWeightsOp =
          rewriter.create<ttnn::PrepareConv2dWeightsOp>(
              ttmlir::utils::appendLocationSuffix(conv2dOp.getLoc(),
                                                  "_prepare_conv2d_weight"),
              getPreparedWeightsType(conv2dOp, conv2dConfig),
              conv2dOp.getWeight(), inputMemConfigAttr,
              rewriter.getAttr<ttnn::LayoutAttr>(inputLayoutAttr.getLayout()),
              rewriter.getStringAttr("OIHW"), conv2dOp.getInChannelsAttr(),
              conv2dOp.getOutChannelsAttr(), conv2dOp.getBatchSizeAttr(),
              conv2dOp.getInputHeightAttr(), conv2dOp.getInputWidthAttr(),
              conv2dOp.getKernelSizeAttr(), conv2dOp.getStrideAttr(),
              conv2dOp.getPaddingAttr(), conv2dOp.getDilationAttr(),
              rewriter.getBoolAttr(conv2dOp.getBias() != nullptr),
              conv2dOp.getGroupsAttr(), conv2dOp.getDevice(), inputDtypeAttr,
              outputDtypeAttr, conv2dConfig);

      ttnn::PrepareConv2dBiasOp prepareConv2dBiasOp;
      if (conv2dOp.getBias()) {
        prepareConv2dBiasOp = rewriter.create<ttnn::PrepareConv2dBiasOp>(
            ttmlir::utils::appendLocationSuffix(conv2dOp.getLoc(),
                                                "_prepare_conv2d_bias"),
            getPreparedBiasType(conv2dOp, inputElementType), conv2dOp.getBias(),
            inputMemConfigAttr,
            rewriter.getAttr<ttnn::LayoutAttr>(inputLayoutAttr.getLayout()),
            conv2dOp.getInChannelsAttr(), conv2dOp.getOutChannelsAttr(),
            conv2dOp.getBatchSizeAttr(), conv2dOp.getInputHeightAttr(),
            conv2dOp.getInputWidthAttr(), conv2dOp.getKernelSizeAttr(),
            conv2dOp.getStrideAttr(), conv2dOp.getPaddingAttr(),
            conv2dOp.getDilationAttr(), conv2dOp.getGroupsAttr(),
            conv2dOp.getDevice(), inputDtypeAttr, outputDtypeAttr,
            conv2dConfig);
      }

      // Update only the weight and bias since PrepareConv2dWeightsOp and
      // PrepareConv2dBiasOp will change the shape and layout of the weight and
      // bias.
      rewriter.modifyOpInPlace(conv2dOp, [&]() {
        conv2dOp.getWeightMutable().assign(prepareConv2dWeightsOp);

        if (conv2dOp.getBias()) {
          conv2dOp.getBiasMutable().assign(prepareConv2dBiasOp);
        }

        // Since we are updating the weight and bias output dtype we must
        // update the conv2d config attr as well, since metal uses it to
        // determine if weight and bias are already prepared.
        conv2dOp.setConv2dConfigAttr(conv2dConfig);
      });
    });

#ifdef TTMLIR_ENABLE_OPMODEL
    // Explicitly close device, leaving it open causes issues in frontend
    // runtime.
    // This will be removed once we switch to virtual device:
    // https://github.com/tenstorrent/tt-metal/issues/14000
    op_model::SingletonDeviceContext::closeInstance();
#endif
  }

private:
  ::mlir::RankedTensorType
  getPreparedWeightsType(ttnn::Conv2dOp conv2dOp,
                         ttnn::Conv2dConfigAttr conv2dConfig) {
    // We use graph capture to retrieve the output type of the PrepareConv2dOp
    // for now until metal exposes an API.
    return op_model::getPreparedConv2dWeightsOutputTensor(&conv2dOp,
                                                          conv2dConfig);
  }

  ::mlir::RankedTensorType getPreparedBiasType(ttnn::Conv2dOp conv2dOp,
                                               Type newElementType) {
    // Prepared bias will retain the shape of the original bias, and it will
    // have a <DRAM, interleaved, tile> memory layout.
    auto oldType =
        mlir::cast<mlir::RankedTensorType>(conv2dOp.getBias().getType());
    auto oldLayout = mlir::cast<ttnn::TTNNLayoutAttr>(oldType.getEncoding());

    auto newLayout = ttnn::TTNNLayoutAttr::get(
        &getContext(), oldType.getShape(),
        ttcore::TileType::get(newElementType), BufferType::DRAM,
        oldLayout.getGrid(),
        ttnn::TensorMemoryLayoutAttr::get(
            &getContext(), ttnn::TensorMemoryLayout::Interleaved));

    return mlir::RankedTensorType::get(oldType.getShape(), newElementType,
                                       newLayout);
    ;
  }
};
} // namespace mlir::tt::ttnn
