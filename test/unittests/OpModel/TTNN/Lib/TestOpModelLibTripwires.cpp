// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Tripwire tests: pass today, expected to fail when a specific tt-metal
// change lands.  Each test's comment names the metal issue/PR to track.
// Kept in a separate target so it compiles fast and the whole file can be
// removed once the upstream change arrives.

#include "OpModelFixture.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace mlir::tt::ttnn::op_model {

class OpModelTripwireTest : public OpModelFixture {};

// PagedUpdateCacheOp operand 1 must be L1 height-sharded with virtual grid
// {numUsers, 1} = {input1.shape[1], 1}; other grids silently produced
// PCC=0 for upper users
// (https://github.com/tenstorrent/tt-metal/issues/44923).
//
// tt-metal PR #45016 adds the matching TT_FATAL
// (`grid.num_cores() == padded_shape()[1]`).  Until that uplift, metal
// still accepts PCC-degrading grids: this test feeds {4, 1} for
// numUsers = 8 and asserts OpModel accepts it today.  When the uplift
// lands, EXPECT_TRUE flips to fail -- signal to delete this file + target
// and drop the operand-1 rule in PagedUpdateCacheRuleBook (metal will
// enforce it from its side).
TEST_F(OpModelTripwireTest, PagedUpdateCacheOpWrongGrid) {
  const llvm::SmallVector<int64_t> cacheShape = {8, 4, 32, 256};
  const llvm::SmallVector<int64_t> inputShape = {1, 8, 12, 256};
  const llvm::SmallVector<int64_t> updateIndexShape = {8};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);

  // numUsers = inputShape[1] = 8; the only valid HS grid in tt-mlir IR is
  // {8, 1}.  We use {4, 1} (num_cores = 4) so the new TT_FATAL rejects it
  // once #45016 is uplifted.
  const llvm::SmallVector<int64_t> wrongVirtualGrid = {4, 1};

  const TTNNLayoutAttr cacheLayout = CreateTiledLayout(
      cacheShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr inputLayoutWrongGrid =
      CreateTiledLayout(inputShape, BufferType::L1,
                        TensorMemoryLayout::HeightSharded, wrongVirtualGrid);
  const TTNNLayoutAttr updateIndexLayout = CreateRowMajorLayoutInt32(
      updateIndexShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  auto constraintsExp = OpModel<PagedUpdateCacheOp>::getOpConstraints(
      workerGrid, cacheShape, cacheLayout, inputShape, inputLayoutWrongGrid,
      updateIndexShape, updateIndexLayout, std::nullopt, std::nullopt, false,
      cacheLayout);
  const bool ok = static_cast<bool>(constraintsExp);
  if (!ok) {
    llvm::consumeError(constraintsExp.takeError());
  }
  EXPECT_TRUE(ok);
}

} // namespace mlir::tt::ttnn::op_model
