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
// tt-metal PR #45016 added the matching TT_FATAL
// (`grid.num_cores() == padded_shape()[1]`), which has now been uplifted.
// metal rejects PCC-degrading grids, so OpModel surfaces them as a failed
// constraint query: this test feeds {4, 1} for numUsers = 8 and asserts the
// query fails.
//
// TODO(#44923): with metal enforcing this from its side, the operand-1 rule
// in PagedUpdateCacheRuleBook is now redundant and can be dropped, along with
// this file + target; deferred to a follow-up cleanup.
TEST_F(OpModelTripwireTest, PagedUpdateCacheOpWrongGrid) {
  const llvm::SmallVector<int64_t> cacheShape = {8, 4, 32, 256};
  const llvm::SmallVector<int64_t> inputShape = {1, 8, 12, 256};
  const llvm::SmallVector<int64_t> updateIndexShape = {8};

  // numUsers = inputShape[1] = 8; the only valid HS grid in tt-mlir IR is
  // {8, 1}.  We use {4, 1} (num_cores = 4) so metal's TT_FATAL rejects it.
  const llvm::SmallVector<int64_t> wrongVirtualGrid = {4, 1};

  const TTNNLayoutAttr cacheLayout = CreateTiledLayout(
      cacheShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr inputLayoutWrongGrid =
      CreateTiledLayout(inputShape, BufferType::L1,
                        TensorMemoryLayout::HeightSharded, wrongVirtualGrid);
  const TTNNLayoutAttr updateIndexLayout = CreateRowMajorLayoutInt32(
      updateIndexShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  auto constraintsExp = OpModel<PagedUpdateCacheOp>::getOpConstraints(
      cacheShape, cacheLayout, inputShape, inputLayoutWrongGrid,
      updateIndexShape, updateIndexLayout, std::nullopt, std::nullopt, false,
      cacheLayout);
  const bool ok = static_cast<bool>(constraintsExp);
  if (!ok) {
    llvm::consumeError(constraintsExp.takeError());
  }
  EXPECT_FALSE(ok);
}

// tt-metal's group_norm hangs on a ROW_MAJOR input, yet its constraint query
// reports row-major as legal, so OpModel<GroupNormOp>::getOpConstraints applies
// a workaround that rejects row-major up front
// (https://github.com/tenstorrent/tt-metal/issues/47972).
//
// This queries tt-metal's *unguarded* verdict via getOpConstraintsRaw and
// asserts it still reports row-major as legal. Once tt-metal#47972 fixes the
// query to reject row-major, this flips to a failed query and the EXPECT_TRUE
// trips — the signal to delete the workaround (and getOpConstraintsRaw) from
// TTNNOpModel and this test.
TEST_F(OpModelTripwireTest, GroupNormRowMajorInputStillReportedLegal) {
  const llvm::SmallVector<int64_t> inputShape = {1, 1, 64, 480};
  const llvm::SmallVector<int64_t> outputShape = {1, 1, 64, 480};
  const llvm::SmallVector<int64_t> inputMaskShape = {1, 8, 32, 96};
  const llvm::SmallVector<int64_t> weightShape = {1, 1, 15, 32};
  const llvm::SmallVector<int64_t> biasShape = {1, 1, 15, 32};

  // The input layout under test: ROW_MAJOR (the layout the workaround rejects).
  const TTNNLayoutAttr inputLayoutRowMajor = CreateRowMajorLayout(
      inputShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr inputMaskLayout = CreateTiledLayout(
      inputMaskShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr weightLayout = CreateRowMajorLayout(
      weightShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr biasLayout = CreateRowMajorLayout(
      biasShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  auto constraintsExp =
      OpModel<GroupNormOp>::getOpConstraintsSkippingRowMajorWorkaround(
          inputShape, inputLayoutRowMajor, inputMaskShape, inputMaskLayout,
          weightShape, weightLayout, biasShape, biasLayout, /*numGroups=*/8,
          llvm::APFloat(1e-5f), outputLayout);
  const bool ok = static_cast<bool>(constraintsExp);
  if (!ok) {
    llvm::consumeError(constraintsExp.takeError());
  }
  EXPECT_TRUE(ok);
}

} // namespace mlir::tt::ttnn::op_model
