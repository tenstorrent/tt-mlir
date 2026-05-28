// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNREFRESHCONV3DPREPAREWEIGHTS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {
// Post-optimizer pass that reconciles the eagerly-emitted
// PrepareConv3dWeightsOp (created in TTIRToTTNN with c_in_block = TILE_WIDTH =
// 32) against the c_in_block chosen by the optimizer and stored on the
// Conv3dOp's Conv3dConfigAttr. If they diverge, the prepare op's c_in_block
// attribute is rewritten in place.
//
// The prepared-weight tensor shape is `[c_in_aligned * kT * kH * kW,
// out_channels]` regardless of c_in_block — numCInBlocks * cInBlock collapses
// to c_in_aligned. So this pass only mutates the attribute, not the SSA value
// type. The runtime implementation of prepare_conv3d_weights reads c_in_block
// to drive the internal data shuffle but produces the same flat output shape.
class TTNNRefreshConv3dPrepareWeights
    : public impl::TTNNRefreshConv3dPrepareWeightsBase<
          TTNNRefreshConv3dPrepareWeights> {
public:
  using impl::TTNNRefreshConv3dPrepareWeightsBase<
      TTNNRefreshConv3dPrepareWeights>::TTNNRefreshConv3dPrepareWeightsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    moduleOp.walk([&](Conv3dOp convOp) {
      Conv3dConfigAttr config = convOp.getConv3dConfigAttr();
      if (!config || !config.getCInBlock().has_value()) {
        return; // Optimizer did not pick a c_in_block; leave the prepare op
                // alone — the TTIRToTTNN default already matches.
      }
      uint32_t chosenCInBlock = config.getCInBlock().value();

      auto prepareOp =
          convOp.getWeight().getDefiningOp<PrepareConv3dWeightsOp>();
      if (!prepareOp) {
        // Weight is not produced by a prepare op (e.g. const-evaluation may
        // have folded it). Nothing to refresh.
        return;
      }

      if (static_cast<uint32_t>(prepareOp.getCInBlock()) == chosenCInBlock) {
        return; // Already aligned.
      }

      prepareOp.setCInBlock(static_cast<int32_t>(chosenCInBlock));
    });
  }
};
} // namespace
} // namespace mlir::tt::ttnn
