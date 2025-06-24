// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"
#include "ttmlir/Utils.h"
#include "llvm/Support/raw_ostream.h"

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
      rewriter.setInsertionPoint(conv2dOp);
      mlir::RankedTensorType inputType = conv2dOp.getInput().getType();
      ttnn::TTNNLayoutAttr inputLayoutAttr =
          mlir::cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding());
      GridAttr deviceGrid = lookupDevice(moduleOp).getWorkerGrid();
      ttnn::MemoryConfigAttr inputMemConfigAttr =
          rewriter.getAttr<ttnn::MemoryConfigAttr>(
              inputLayoutAttr.getMemLayout(),
              rewriter.getAttr<ttnn::BufferTypeAttr>(
                  inputLayoutAttr.getBufferType()),
              utils::createShardSpecIfNeeded(inputLayoutAttr, deviceGrid));
      auto inputDtypeAttr = mlir::tt::ttcore::DataTypeAttr::get(
          &getContext(), inputLayoutAttr.getDataType());

      mlir::RankedTensorType weightType = conv2dOp.getWeight().getType();
      ttnn::TTNNLayoutAttr weightLayoutAttr =
          mlir::cast<ttnn::TTNNLayoutAttr>(weightType.getEncoding());
      auto weightDtypeAttr = rewriter.getAttr<ttcore::DataTypeAttr>(
          weightLayoutAttr.getDataType());
      assert(weightLayoutAttr.getBufferType() ==
                 ttnn::BufferType::SystemMemory &&
             weightLayoutAttr.getLayout() == ttnn::Layout::RowMajor &&
             "Weight must be in system memory and row-major layout when "
             "calling TTNNPrepareConv2dWeightsAndBias pass.");

      ttnn::PrepareConv2dWeightsOp prepareConv2dWeightsOp =
          rewriter.create<ttnn::PrepareConv2dWeightsOp>(
              ttmlir::utils::appendLocationSuffix(conv2dOp.getLoc(),
                                                  "_prepare_conv2d_weight"),
              getPreparedWeightsType(conv2dOp), conv2dOp.getWeight(),
              inputMemConfigAttr,
              rewriter.getAttr<ttnn::LayoutAttr>(inputLayoutAttr.getLayout()),
              rewriter.getStringAttr("OIHW"), conv2dOp.getInChannelsAttr(),
              conv2dOp.getOutChannelsAttr(), conv2dOp.getBatchSizeAttr(),
              conv2dOp.getInputHeightAttr(), conv2dOp.getInputWidthAttr(),
              conv2dOp.getKernelSizeAttr(), conv2dOp.getStrideAttr(),
              conv2dOp.getPaddingAttr(), conv2dOp.getDilationAttr(),
              rewriter.getBoolAttr(conv2dOp.getBias() != nullptr),
              conv2dOp.getGroupsAttr(), conv2dOp.getDevice(), inputDtypeAttr,
              weightDtypeAttr, conv2dOp.getConv2dConfigAttr());

      ttnn::PrepareConv2dBiasOp prepareConv2dBiasOp;
      if (conv2dOp.getBias()) {
        mlir::RankedTensorType biasType = conv2dOp.getBias().getType();
        ttnn::TTNNLayoutAttr biasLayoutAttr =
            mlir::cast<ttnn::TTNNLayoutAttr>(biasType.getEncoding());
        auto biasDtypeAttr = rewriter.getAttr<ttcore::DataTypeAttr>(
          biasLayoutAttr.getDataType());
        assert(biasLayoutAttr.getBufferType() ==
                   ttnn::BufferType::SystemMemory &&
               biasLayoutAttr.getLayout() == ttnn::Layout::RowMajor &&
               "Bias must be in system memory and row-major layout when "
               "calling TTNNPrepareConv2dWeightsAndBias pass");

        // PrepareConv2dBias requires Conv2dConfig to be created and weights
        // dtype to be set.
        auto conv2dConfig = conv2dOp.getConv2dConfigAttr()
                                ? conv2dOp.getConv2dConfigAttr()
                                : Conv2dConfigAttr::get(&getContext());
        conv2dConfig = conv2dConfig.withWeightsDtype(elementTypeToDataType(
            conv2dOp.getWeight().getType().getElementType()));

        prepareConv2dBiasOp = rewriter.create<ttnn::PrepareConv2dBiasOp>(
            ttmlir::utils::appendLocationSuffix(conv2dOp.getLoc(),
                                                "_prepare_conv2d_bias"),
            getPreparedBiasType(conv2dOp), conv2dOp.getBias(),
            inputMemConfigAttr,
            rewriter.getAttr<ttnn::LayoutAttr>(inputLayoutAttr.getLayout()),
            conv2dOp.getInChannelsAttr(), conv2dOp.getOutChannelsAttr(),
            conv2dOp.getBatchSizeAttr(), conv2dOp.getInputHeightAttr(),
            conv2dOp.getInputWidthAttr(), conv2dOp.getKernelSizeAttr(),
            conv2dOp.getStrideAttr(), conv2dOp.getPaddingAttr(),
            conv2dOp.getDilationAttr(), conv2dOp.getGroupsAttr(),
            conv2dOp.getDevice(), inputDtypeAttr, biasDtypeAttr,  conv2dConfig);
      }

      // Update only the weight and bias since PrepareConv2dWeightsOp and
      // PrepareConv2dBiasOp will change the shape and layout of the weight and bias.
      rewriter.modifyOpInPlace(conv2dOp, [&]() {
        conv2dOp.getWeightMutable().assign(prepareConv2dWeightsOp);

        if (conv2dOp.getBias()) {
          conv2dOp.getBiasMutable().assign(prepareConv2dBiasOp);
        }
      });
    });

#ifdef TTMLIR_ENABLE_OPMODEL
    // Explicitly close device, leaving it open causes issues in frontend
    // runtime.
    // This will be removed once we switch to virtual device:
    // https://github.com/tenstorrent/tt-metal/issues/14000
    mlir::tt::op_model::ttnn::SingletonDeviceContext::closeInstance();
#endif
  }

private:
  ::mlir::RankedTensorType getPreparedWeightsType(ttnn::Conv2dOp conv2dOp) {
    // We use graph capture to retrieve the output type of the PrepareConv2dOp
    // for now until metal exposes an API.
    return op_model::ttnn::getPreparedConv2dWeightsOutputTensor(&conv2dOp);
  }

  ::mlir::RankedTensorType getPreparedBiasType(ttnn::Conv2dOp conv2dOp) {
    // Prepared bias will retain the shape of the original bias, and it will
    // have a <DRAM, interleaved, tile> memory layout.
    auto oldType =
        mlir::cast<mlir::RankedTensorType>(conv2dOp.getBias().getType());
    auto oldLayout = mlir::cast<ttnn::TTNNLayoutAttr>(oldType.getEncoding());

    auto newLayout = ttnn::TTNNLayoutAttr::get(
        &getContext(), oldType.getShape(),
        TileType::get(oldType.getElementType()), BufferType::DRAM,
        oldLayout.getGrid(),
        ttnn::TensorMemoryLayoutAttr::get(
            &getContext(), ttnn::TensorMemoryLayout::Interleaved));

    return mlir::RankedTensorType::get(oldType.getShape(),
                                       oldType.getElementType(), newLayout);
    ;
  }
};
} // namespace mlir::tt::ttnn
