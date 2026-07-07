// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TTNNPixelUnshuffleL1Opt
//
// For each ttnn.pixel_unshuffle whose result is in DRAM, rebuilds it so the
// result goes directly to L1 interleaved (8×8 grid).  Sets memory_config=L1
// on the op itself and updates its result type.  All existing users (permute,
// deallocate, etc.) are re-pointed to the new L1 result unchanged — no
// to_memory_config is inserted.
//
// Root cause of the CB clash (Block C):
//   pixel_unshuffle writing to DRAM leaves the L1 allocator in a state where
//   the downstream HEIGHT_SHARDED conv's internal reshard buffer lands inside
//   the matmul static-CB region.  Writing pixel_unshuffle directly to L1
//   restores the same allocator state as the passing manual
//   reshape→permute→reshape chain and avoids the clash.

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/Builders.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNPIXELUNSHUFFLEL1OPT
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNPixelUnshuffleL1OptPass
    : public impl::TTNNPixelUnshuffleL1OptBase<TTNNPixelUnshuffleL1OptPass> {
public:
  using impl::TTNNPixelUnshuffleL1OptBase<
      TTNNPixelUnshuffleL1OptPass>::TTNNPixelUnshuffleL1OptBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    SmallVector<ttnn::PixelUnshuffleOp, 8> ops;
    moduleOp.walk([&](ttnn::PixelUnshuffleOp op) { ops.push_back(op); });

    for (ttnn::PixelUnshuffleOp pixOp : ops) {
      Value result = pixOp.getResult();
      auto resultType = mlir::cast<RankedTensorType>(result.getType());
      auto layout = mlir::dyn_cast<TTNNLayoutAttr>(resultType.getEncoding());

      if (!layout)
        continue;

      // Already writing to L1 directly — nothing to do.
      if (layout.getBufferType() == BufferType::L1)
        continue;

      // Build the L1-interleaved output type for pixel_unshuffle.
      // Use the full Wormhole 8×8 compute grid so each core holds a smaller
      // L1 slice, matching the allocator footprint of the original manual
      // reshape→permute→reshape chain.
      auto l1Layout = TTNNLayoutAttr::Builder(resultType)
                          .setBufferType(BufferType::L1)
                          .setGridShape({8, 8})
                          .build();
      auto l1Type = utils::RankedTensorTypeFactory::create(resultType, l1Layout);
      auto l1MemCfg = MemoryConfigAttr::get(l1Layout);

      // Rebuild pixel_unshuffle with memory_config=L1 and L1 output type.
      OpBuilder builder(pixOp);
      auto newPixOp = builder.create<ttnn::PixelUnshuffleOp>(
          pixOp.getLoc(), l1Type, pixOp.getInput(),
          pixOp.getDownscaleFactorAttr(), pixOp.getChannelOrderAttr(), l1MemCfg);

      // Re-point all users (permute, deallocate, etc.) to the new L1 result.
      result.replaceAllUsesWith(newPixOp.getResult());

      // Erase the old DRAM pixel_unshuffle.
      pixOp.erase();
    }
  }
};

} // namespace mlir::tt::ttnn
