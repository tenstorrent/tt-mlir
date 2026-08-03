// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/BFPDtypeParser.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNCONV2DWEIGHTDTYPECONVERSION
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Post-analysis pass: changes conv2d_config.weights_dtype to the target BFP
// format on every Conv2dOp in the module.
//
// Runs AFTER TTNNOptimizer / TTNNGreedyMemoryLayoutPropagation (which select
// actBlockH, sharding, and double-buffer configurations using BF16 L1 estimates)
// and BEFORE TTNNPrepareConv2dWeightsAndBias (which reads weights_dtype and
// packs the weights into the target format on the host).
//
// Because the optimizer validated configs against BF16 CB footprints (larger),
// the actual runtime CB with BF8/BF4 weights is smaller — no CB clash possible.
// Output activations remain in BF16; only weight DRAM storage is compressed.
class TTNNConv2dWeightDtypeConversionPass
    : public impl::TTNNConv2dWeightDtypeConversionBase<
          TTNNConv2dWeightDtypeConversionPass> {
public:
  using impl::TTNNConv2dWeightDtypeConversionBase<
      TTNNConv2dWeightDtypeConversionPass>::TTNNConv2dWeightDtypeConversionBase;

  void runOnOperation() final {
    if (targetDtype == BFPDtype::None) {
      return;
    }

    ttcore::DataType dtype = bfpDtypeToDataType(targetDtype);
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    auto convertOp = [&](auto convOp) {
      auto conv2dConfig = convOp.getConv2dConfigAttr()
                              ? convOp.getConv2dConfigAttr()
                              : Conv2dConfigAttr::get(&getContext());

      if (conv2dConfig.getWeightsDtype().has_value() &&
          conv2dConfig.getWeightsDtype().value() == dtype) {
        return;
      }

      rewriter.modifyOpInPlace(convOp, [&]() {
        convOp.setConv2dConfigAttr(conv2dConfig.withWeightsDtype(dtype));
      });
    };

    moduleOp.walk([&](ttnn::Conv2dOp op) { convertOp(op); });
    moduleOp.walk([&](ttnn::ConvTranspose2dOp op) { convertOp(op); });
  }
};

} // namespace
} // namespace mlir::tt::ttnn
