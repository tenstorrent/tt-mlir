// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNPREPARECONV2DWEIGHTS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNPrepareConv2dWeights
    : public impl::TTNNPrepareConv2dWeightsBase<TTNNPrepareConv2dWeights> {

public:
  using impl::TTNNPrepareConv2dWeightsBase<
      TTNNPrepareConv2dWeights>::TTNNPrepareConv2dWeightsBase;

  // Insert PrepareConv2dWeightsOp before every Conv2dOp that prepares weights
  // for convolution. This is a prerequisite for const evaluation, which will
  // improve performance by eliminating the need for preprocessing the weights
  // on the host/device.
  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    moduleOp.walk([&](ttnn::Conv2dOp conv2dOp) {
      rewriter.setInsertionPoint(conv2dOp);
      ttnn::ToLayoutOp hostRMWeight = createToLayoutOpHostRM(
          rewriter, conv2dOp.getWeight(), conv2dOp.getLoc());
      // Modify conv2d op to use the hostRMWeight. We need this to calculate
      // prepared tensor.
      rewriter.modifyOpInPlace(conv2dOp, [&]() {
        conv2dOp.getWeightMutable().assign(hostRMWeight);
      });

      ttnn::PrepareConv2dWeightsOp prepareConv2dWeightsOp =
          createPrepareConv2dWeightsOp(rewriter, conv2dOp);

      // Update only the weight operand since PrepareConv2dWeightsOp will change
      // the shape and layout of the weight
      rewriter.modifyOpInPlace(conv2dOp, [&]() {
        conv2dOp.getWeightMutable().assign(prepareConv2dWeightsOp);
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
  PrepareConv2dWeightsOp createPrepareConv2dWeightsOp(OpBuilder &builder,
                                                      Conv2dOp conv2dOp) {
    TTNNLayoutAttr inputLayoutAttr =
        mlir::cast<TTNNLayoutAttr>(conv2dOp.getInput().getType().getEncoding());
    MemoryConfigAttr inputMemConfigAttr =
        MemoryConfigAttr::get(inputLayoutAttr);

    Location newLocation = ttmlir::utils::appendLocationSuffix(
        conv2dOp.getLoc(), "_prepare_conv2d");

    RankedTensorType prepareType = getPreparedWeightsType(conv2dOp);
    return builder.create<ttnn::PrepareConv2dWeightsOp>(
        newLocation, prepareType, conv2dOp.getWeight(), inputMemConfigAttr,
        builder.getAttr<ttnn::LayoutAttr>(inputLayoutAttr.getLayout()),
        builder.getStringAttr("OIHW"), conv2dOp.getInChannelsAttr(),
        conv2dOp.getOutChannelsAttr(), conv2dOp.getBatchSizeAttr(),
        conv2dOp.getInputHeightAttr(), conv2dOp.getInputWidthAttr(),
        conv2dOp.getKernelSizeAttr(), conv2dOp.getStrideAttr(),
        conv2dOp.getPaddingAttr(), conv2dOp.getDilationAttr(),
        builder.getBoolAttr(conv2dOp.getBias() != nullptr),
        conv2dOp.getGroupsAttr(), conv2dOp.getDevice(),
        conv2dOp.getConv2dConfigAttr());
  }

  ToLayoutOp createToLayoutOpHostRM(OpBuilder &builder,
                                    TypedValue<RankedTensorType> weight,
                                    Location loc) {
    const Layout layout = Layout::RowMajor;
    const BufferType bufferType = BufferType::SystemMemory;
    MLIRContext *context = builder.getContext();
    auto weightLayoutAttr =
        mlir::cast<TTNNLayoutAttr>(weight.getType().getEncoding());

    // Create to_layout output encoding by setting RM and System Memory.
    utils::RankedTensorTypeFactory::Params params;
    params.bufferType = bufferType;
    params.layout = layout;
    RankedTensorType toLayoutType =
        utils::RankedTensorTypeFactory::create(weight.getType(), params);
    Location newLoc = ttmlir::utils::appendLocationSuffix(loc, "_to_layout");

    MemoryConfigAttr toLayoutMemConifg = ttnn::MemoryConfigAttr::get(
        cast<TTNNLayoutAttr>(toLayoutType.getEncoding()));

    return builder.create<ToLayoutOp>(
        newLoc, toLayoutType, weight, LayoutAttr::get(context, layout),
        DataTypeAttr::get(context, weightLayoutAttr.getDataType()),
        toLayoutMemConifg, /*device=*/nullptr);
  }

  ::mlir::RankedTensorType getPreparedWeightsType(ttnn::Conv2dOp conv2dOp) {
    // We use graph capture to retrieve the output type of the PrepareConv2dOp
    // for now until metal exposes an API.
    return op_model::ttnn::getPreparedConv2dWeightsOutputTensor(&conv2dOp);
  }
};
} // namespace mlir::tt::ttnn
