// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
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
      mlir::RankedTensorType inputType = conv2dOp.getInput().getType();

      GridAttr deviceGrid = lookupDevice(moduleOp).getWorkerGrid();

      ttnn::TTNNLayoutAttr inputLayoutAttr =
          mlir::cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding());
      ttnn::MemoryConfigAttr inputMemConfigAttr =
          rewriter.getAttr<ttnn::MemoryConfigAttr>(
              inputLayoutAttr.getMemLayout(),
              rewriter.getAttr<ttnn::BufferTypeAttr>(
                  inputLayoutAttr.getBufferType()),
              utils::createShardSpecIfNeeded(inputLayoutAttr, deviceGrid));

      rewriter.setInsertionPoint(conv2dOp);
      ttnn::PrepareConv2dWeightsOp prepareConv2dWeightsOp =
          rewriter.create<ttnn::PrepareConv2dWeightsOp>(
              ttmlir::utils::appendLocationSuffix(conv2dOp.getLoc(),
                                                  "_prepare_conv2d"),
              getPreparedWeightsType(conv2dOp), conv2dOp.getWeight(),
              inputMemConfigAttr,
              rewriter.getAttr<ttnn::LayoutAttr>(inputLayoutAttr.getLayout()),
              rewriter.getStringAttr("OIHW"), conv2dOp.getInChannelsAttr(),
              conv2dOp.getOutChannelsAttr(), conv2dOp.getBatchSizeAttr(),
              conv2dOp.getInputHeightAttr(), conv2dOp.getInputWidthAttr(),
              conv2dOp.getKernelSizeAttr(), conv2dOp.getStrideAttr(),
              conv2dOp.getPaddingAttr(), conv2dOp.getDilationAttr(),
              rewriter.getBoolAttr(conv2dOp.getBias() != nullptr),
              conv2dOp.getGroupsAttr(), conv2dOp.getDevice(),
              conv2dOp.getConv2dConfigAttr());

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
    // mlir::tt::op_model::ttnn::SingletonDeviceContext::closeInstance();
#endif
  }

private:
  ::mlir::RankedTensorType getPreparedWeightsType(ttnn::Conv2dOp conv2dOp) {
    // We use graph capture to retrieve the output type of the PrepareConv2dOp
    // for now until metal exposes an API.
    return op_model::ttnn::getPreparedConv2dWeightsOutputTensor(&conv2dOp);
  }
};
} // namespace mlir::tt::ttnn
