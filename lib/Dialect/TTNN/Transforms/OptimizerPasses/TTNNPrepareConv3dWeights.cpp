// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/OpModel/TTNN/TTNNOutputTensorInference.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNPREPARECONV3DWEIGHTS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {
// Post-optimizer pass that materializes a PrepareConv3dWeightsOp before each
// Conv3dOp. The TTIRToTTNN conversion leaves Conv3dOp consuming its raw 5D
// weight; this pass picks up the c_in_block chosen by the optimizer (via
// Conv3dConfigAttr) and inserts the prepare op with that block size. The
// prepared weight's shape is invariant in c_in_block, so the shape is fully
// determined by op attributes.
//
// Mirrors TTNNPrepareConv2dWeightsAndBias: by deferring prepare-op creation
// until after the optimizer has chosen a config, we avoid the layout/c_in_block
// mistypings that the (deleted) TTNNRefreshConv3dPrepareWeights pass was
// patching up post-hoc.
class TTNNPrepareConv3dWeights
    : public impl::TTNNPrepareConv3dWeightsBase<TTNNPrepareConv3dWeights> {
public:
  using impl::TTNNPrepareConv3dWeightsBase<
      TTNNPrepareConv3dWeights>::TTNNPrepareConv3dWeightsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    moduleOp.walk([&](Conv3dOp convOp) { processConvOp(convOp, rewriter); });
  }

private:
  void processConvOp(Conv3dOp convOp, IRRewriter &rewriter) {
    // If the weight is already 2D, a prepare op (from a previous run of this
    // pass or an earlier pass) is already in place, so don't re-prepare. But
    // the conv3d runtime kernel still requires the weight in TILE layout, and
    // the weight-layout workaround was relaxed to dtype-only; nothing else
    // enforces tile here. So ensure the tile layout holds rather than assuming
    // it, then return.
    if (convOp.getWeight().getType().getRank() == 2) {
      mlir::TypedValue<RankedTensorType> weight = convOp.getWeight();
      auto weightLayout =
          mlir::cast<TTNNLayoutAttr>(weight.getType().getEncoding());
      if (weightLayout.getLayout() != Layout::Tile) {
        rewriter.setInsertionPoint(convOp);
        ToLayoutOp toTileOp = utils::createToLayoutOp(
            convOp, weight, rewriter, Layout::Tile,
            weightLayout.getBufferType(), weightLayout.getMemLayout(),
            weightLayout.getDataType(), "_prepare_conv3d_weight_to_tile");
        rewriter.modifyOpInPlace(convOp, [&]() {
          convOp.getWeightMutable().assign(toTileOp.getResult());
        });
      }
      return;
    }

    constexpr int32_t TILE_WIDTH =
        static_cast<int32_t>(ttcore::TileType::getDefaultShape()[1]);
    constexpr int32_t ALIGNMENT = TILE_WIDTH;

    // c_in_block is read straight from the op's Conv3dConfigAttr. TTIRToTTNN
    // attaches a complete config (and the optimizer only refines it), so the
    // field is always present by the time this pass runs. This pass does not
    // reason about tt-metal defaults or modify the config — it just consumes
    // the c_in_block already chosen and prepares the weight with it.
    Conv3dConfigAttr config = convOp.getConv3dConfigAttr();
    assert(config && config.getCInBlock() &&
           "Conv3dOp must carry a Conv3dConfigAttr with c_in_block (set by "
           "TTIRToTTNN) before TTNNPrepareConv3dWeights runs");
    int32_t cInBlock = static_cast<int32_t>(*config.getCInBlock());

    // The prepare op's MLIR result type must match what tt-metal's runtime
    // produces: a ROW_MAJOR 2D tensor. The helper builds exactly that.
    mlir::RankedTensorType preparedWeightType =
        op_model::getPreparedConv3dWeightsOutputTensor(&convOp);

    rewriter.setInsertionPoint(convOp);

    // tt-metal's prepare_conv3d_weights runtime kernel expects its input
    // weight to be ROW_MAJOR in SystemMemory. The optimizer's layout
    // propagation may have re-typed the raw 5D weight (e.g. to TILE in
    // DRAM); since the prepare op didn't exist at workaround time, the
    // input-operand workaround couldn't fire. Insert the to_layout here.
    mlir::TypedValue<RankedTensorType> rawWeight = convOp.getWeight();
    auto rawWeightLayout =
        mlir::cast<TTNNLayoutAttr>(rawWeight.getType().getEncoding());
    if (rawWeightLayout.getLayout() != Layout::RowMajor ||
        rawWeightLayout.getBufferType() != BufferType::SystemMemory) {
      rawWeight =
          utils::createToLayoutOp(convOp, rawWeight, rewriter, Layout::RowMajor,
                                  BufferType::SystemMemory,
                                  /*targetTensorMemoryLayout=*/nullptr,
                                  rawWeightLayout.getDataType(),
                                  "_prepare_conv3d_weight_to_row_major")
              .getResult();
    }

    auto prepareOp = rewriter.create<PrepareConv3dWeightsOp>(
        ttmlir::utils::appendLocationSuffix(convOp.getLoc(),
                                            "_prepare_conv3d_weight"),
        preparedWeightType, rawWeight,
        rewriter.getI32IntegerAttr(convOp.getGroups()),
        rewriter.getI32IntegerAttr(cInBlock),
        rewriter.getI32IntegerAttr(ALIGNMENT), convOp.getDevice());

    // tt-metal's conv3d kernel consumes the weight in TILE layout. The
    // prepare op outputs ROW_MAJOR (matching runtime), so insert a to_layout
    // op converting to TILE before the conv3d. Without this, the workaround
    // pass — which already ran — has not been able to enforce the tile
    // constraint on the weight (its input was the raw 5D weight).
    auto preparedLayout =
        mlir::cast<TTNNLayoutAttr>(preparedWeightType.getEncoding());
    ToLayoutOp toTileOp = utils::createToLayoutOp(
        convOp, prepareOp.getResult(), rewriter, Layout::Tile,
        preparedLayout.getBufferType(), preparedLayout.getMemLayout(),
        preparedLayout.getDataType(), "_prepare_conv3d_weight_to_tile");

    rewriter.modifyOpInPlace(convOp, [&]() {
      convOp.getWeightMutable().assign(toTileOp.getResult());
    });
  }
};
} // namespace
} // namespace mlir::tt::ttnn
