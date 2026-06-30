// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Tripwire tests: pass today, expected to fail when a specific tt-metal
// change lands.  Each test's comment should name the metal issue/PR it
// tracks.  Kept in a separate target so it compiles fast and the file can be
// removed (or individual tripwires dropped) once the upstream change arrives.
//
// This file is currently an empty placeholder -- add a TEST_F here when a new
// tt-metal behavior needs a tripwire.  The PagedUpdateCacheOpWrongGrid
// tripwire lived here until tt-metal #45016 landed; see git history for an
// example of the pattern.

#include "OpModelFixture.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

namespace mlir::tt::ttnn::op_model {

class OpModelTripwireTest : public OpModelFixture {};

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

  auto constraintsExp = OpModel<GroupNormOp>::getOpConstraintsRaw(
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
