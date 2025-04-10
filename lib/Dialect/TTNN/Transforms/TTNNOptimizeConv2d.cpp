// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNOPTIMIZECONV2D
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNOptimizeConv2d
    : public impl::TTNNOptimizeConv2dBase<TTNNOptimizeConv2d> {

public:
  using impl::TTNNOptimizeConv2dBase<
      TTNNOptimizeConv2d>::TTNNOptimizeConv2dBase;

  // Insert PrepareConv2dWeightsOp before every Conv2dOp that prepares weights
  // for convolution. This is a prerequisite for constevaluation, which will
  // improve performance by eliminating the need for preprocessing the weights
  // on the host/device.
  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    moduleOp.walk([&](ttnn::Conv2dOp conv2dOp) {
      mlir::RankedTensorType inputType = conv2dOp.getInput().getType();

      ttnn::TTNNLayoutAttr inputLayoutAttr =
          mlir::cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding());
      ttnn::MemoryConfigAttr inputMemConfigAttr =
          rewriter.getAttr<ttnn::MemoryConfigAttr>(
              rewriter.getAttr<ttnn::BufferTypeAttr>(
                  inputLayoutAttr.getBufferType()),
              rewriter.getAttr<ttnn::ShardSpecAttr>(
                  rewriter.getAttr<ttnn::ShapeAttr>(
                      inputLayoutAttr.getShardShape())),
              inputLayoutAttr.getMemLayout());

      rewriter.setInsertionPoint(conv2dOp);
      ttnn::PrepareConv2dWeightsOp prepareConv2dWeightsOp =
          rewriter.create<ttnn::PrepareConv2dWeightsOp>(
              conv2dOp.getLoc(), getPreparedWeightsType(conv2dOp),
              conv2dOp.getWeight(), inputMemConfigAttr,
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
      IRMapping mapper;
      mapper.map(conv2dOp.getWeight(), prepareConv2dWeightsOp.getResult());

      rewriter.replaceOp(conv2dOp,
                         rewriter.clone(*conv2dOp.getOperation(), mapper));
    });
  }

private:
  ::mlir::RankedTensorType getPreparedWeightsType(ttnn::Conv2dOp conv2dOp) {
    return op_model::ttnn::getPreparedConv2dWeightsOutputTensor(&conv2dOp);
  }
};
} // namespace mlir::tt::ttnn
