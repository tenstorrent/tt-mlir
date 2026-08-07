// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/LayoutFilterUtils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include "gtest/gtest.h"

using namespace mlir::tt::ttnn;
using namespace mlir::tt::ttnn::layout_filter_utils;

// Tests for the reshape output-layout divergence guard: a candidate sharded
// reshape-output layout is only legal when ttnn::reshape returns that exact
// memory config instead of substituting one of its own.
//
// Every case below is a case from the on-device sweep recorded in issue #8020,
// with its measured outcome named in the comment. `out` is 1x256x32x32 bf16
// (physical 8192x32 == 256x1 tiles) unless noted. The sweep ran these on
// 1x256x8x32, which has the same physical extent; the logical shape is stated
// tile-aligned here so that each case isolates the substitution rule it names
// instead of tripping rule P (see TileAlignment tests at the bottom).
class ReshapeDivergenceGuardTest : public ::testing::Test {
public:
  mlir::MLIRContext context;
  mlir::OpBuilder builder = mlir::OpBuilder(&context);

  void SetUp() override {
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
  }

  // A CoreRangeSet from {startX, startY, endX, endY} rectangles.
  CoreRangeSetAttr
  makeCoreRangeSet(llvm::ArrayRef<std::array<int64_t, 4>> rects) {
    llvm::SmallVector<CoreRangeAttr> ranges;
    for (const auto &r : rects) {
      ranges.push_back(
          CoreRangeAttr::get(&context, CoreCoordAttr::get(&context, r[0], r[1]),
                             CoreCoordAttr::get(&context, r[2], r[3])));
    }
    return CoreRangeSetAttr::get(&context, ranges);
  }

  CoreRangeSetAttr coreGrid(int64_t width, int64_t height) {
    return makeCoreRangeSet({{0, 0, width - 1, height - 1}});
  }

  // An L1 sharded tiled layout. `gridShape` ([H, W]) drives the derived
  // per-core shard shape; `crs` is the core placement the shard spec declares.
  // They are set independently so that specs tt-metal would renegotiate (an
  // over-provisioned shard shape, a non-rectangular grid) can be expressed.
  TTNNLayoutAttr shardedLayout(llvm::ArrayRef<int64_t> tensorShape,
                               TensorMemoryLayout memLayout,
                               llvm::ArrayRef<int64_t> gridShape,
                               CoreRangeSetAttr crs) {
    return TTNNLayoutAttr::Builder(
               &context, tensorShape,
               mlir::tt::ttcore::TileType::get(builder.getBF16Type()))
        .setBufferType(BufferType::L1)
        .setMemoryLayout(memLayout)
        .setGridShape(gridShape)
        .setCoreRangeSet(crs)
        .build();
  }

  TTNNLayoutAttr dramInterleavedLayout(llvm::ArrayRef<int64_t> tensorShape) {
    return TTNNLayoutAttr::Builder(
               &context, tensorShape,
               mlir::tt::ttcore::TileType::get(builder.getBF16Type()))
        .setBufferType(BufferType::DRAM)
        .setMemoryLayout(TensorMemoryLayout::Interleaved)
        .build();
  }

  // Row-major (untilized) counterparts of the two layouts above: a scalar
  // element type instead of a tile.
  TTNNLayoutAttr rowMajorShardedLayout(llvm::ArrayRef<int64_t> tensorShape,
                                       TensorMemoryLayout memLayout,
                                       llvm::ArrayRef<int64_t> gridShape,
                                       CoreRangeSetAttr crs) {
    return TTNNLayoutAttr::Builder(&context, tensorShape, builder.getBF16Type())
        .setBufferType(BufferType::L1)
        .setMemoryLayout(memLayout)
        .setGridShape(gridShape)
        .setCoreRangeSet(crs)
        .build();
  }

  TTNNLayoutAttr
  dramInterleavedRowMajorLayout(llvm::ArrayRef<int64_t> tensorShape) {
    return TTNNLayoutAttr::Builder(&context, tensorShape, builder.getBF16Type())
        .setBufferType(BufferType::DRAM)
        .setMemoryLayout(TensorMemoryLayout::Interleaved)
        .build();
  }
};

// A NULL hint and any interleaved candidate are never renegotiated.
TEST_F(ReshapeDivergenceGuardTest, InterleavedAndNullAreAlwaysHonored) {
  llvm::SmallVector<int64_t> inShape = {512, 512};
  llvm::SmallVector<int64_t> outShape = {1, 256, 32, 32};

  EXPECT_TRUE(reshapeOutputSpecIsHonored(
      TTNNLayoutAttr(), dramInterleavedLayout(inShape), inShape, outShape));
  EXPECT_TRUE(reshapeOutputSpecIsHonored(dramInterleavedLayout(outShape),
                                         dramInterleavedLayout(inShape),
                                         inShape, outShape));
}

// Rule H (reshape.cpp:177-211): HEIGHT_SHARDED shard shapes are always
// re-derived as round32(div_up(phys_h, num_cores)) over the declared grid.
TEST_F(ReshapeDivergenceGuardTest, HeightShardedShardShapeMustMatchRecompute) {
  llvm::SmallVector<int64_t> inShape = {512, 512};
  llvm::SmallVector<int64_t> outShape = {1, 256, 32, 32};
  TTNNLayoutAttr input = dramInterleavedLayout(inShape);

  // h64_4x1_matches_recompute: 4x1-tile shards over 64 cores == recompute.
  EXPECT_TRUE(reshapeOutputSpecIsHonored(
      shardedLayout(outShape, TensorMemoryLayout::HeightSharded, {64, 1},
                    coreGrid(8, 8)),
      input, inShape, outShape));

  // h64_8x1_over2x: 8x1-tile shards over the same 64 cores. tt-metal returns
  // 4x1 instead; measured outcome was a device hang.
  EXPECT_FALSE(reshapeOutputSpecIsHonored(
      shardedLayout(outShape, TensorMemoryLayout::HeightSharded, {32, 1},
                    coreGrid(8, 8)),
      input, inShape, outShape));

  // h8col_32x1_exact: a single column of 8 cores, exact shards.
  EXPECT_TRUE(reshapeOutputSpecIsHonored(
      shardedLayout(outShape, TensorMemoryLayout::HeightSharded, {8, 1},
                    coreGrid(1, 8)),
      input, inShape, outShape));
}

// Rule N (reshape.cpp:121-130): a non-rectangular grid is silently downgraded
// to INTERLEAVED.
TEST_F(ReshapeDivergenceGuardTest, NonRectangularGridIsRejected) {
  llvm::SmallVector<int64_t> inShape = {512, 512};
  llvm::SmallVector<int64_t> outShape = {1, 256, 32, 32};
  TTNNLayoutAttr input = dramInterleavedLayout(inShape);

  // nonrect54_height_5x1: 54 cores in a 8x7 bounding box.
  EXPECT_FALSE(reshapeOutputSpecIsHonored(
      shardedLayout(outShape, TensorMemoryLayout::HeightSharded, {54, 1},
                    makeCoreRangeSet({{0, 0, 7, 5}, {0, 6, 5, 6}})),
      input, inShape, outShape));

  // rect40_two_ranges_height_7x1: 40 cores spelled as two ranges that do fill
  // their 8x5 bounding box -- honored.
  EXPECT_TRUE(reshapeOutputSpecIsHonored(
      shardedLayout(outShape, TensorMemoryLayout::HeightSharded, {40, 1},
                    makeCoreRangeSet({{0, 0, 7, 2}, {0, 3, 7, 4}})),
      input, inShape, outShape));
}

// Rule B (reshape.cpp:137-152): an explicit BLOCK_SHARDED spec is passed
// through only when its grid covers the output's physical extent -- with `>=`,
// so over-provisioning is fine.
TEST_F(ReshapeDivergenceGuardTest, BlockShardedGridMustCoverOutput) {
  llvm::SmallVector<int64_t> inShape = {512, 512};
  llvm::SmallVector<int64_t> outShape = {1, 256, 32, 32};
  TTNNLayoutAttr input = dramInterleavedLayout(inShape);

  // b_full_32x1_over8xw: 8x8 cores x 32x1-tile shards covers 256x1 tiles
  // exactly in height and 8x over in width -- honored verbatim.
  EXPECT_TRUE(reshapeOutputSpecIsHonored(
      shardedLayout(outShape, TensorMemoryLayout::BlockSharded, {8, 1},
                    coreGrid(8, 8)),
      input, inShape, outShape));

  // b_full_16x1_undercover: 8 core rows x 16-tile shards = 128 < 256 tiles of
  // height. tt-metal replaces the grid; measured outcome was divergence.
  EXPECT_FALSE(reshapeOutputSpecIsHonored(
      shardedLayout(outShape, TensorMemoryLayout::BlockSharded, {16, 1},
                    coreGrid(8, 8)),
      input, inShape, outShape));
}

// Rule V (reshape.cpp:613-624): on the `this_is_view` fast path the requested
// memory config is discarded entirely and the input's spec is returned, so a
// candidate is only honored when it already matches the input's memory config.
TEST_F(ReshapeDivergenceGuardTest, ViewPathRequiresTheInputsMemoryConfig) {
  // View-compatible shapes: same last dim, both second-to-last tile aligned.
  llvm::SmallVector<int64_t> inShape = {8192, 32};
  llvm::SmallVector<int64_t> outShape = {1, 256, 32, 32};
  TTNNLayoutAttr input = shardedLayout(
      inShape, TensorMemoryLayout::HeightSharded, {8, 1}, coreGrid(1, 8));
  ASSERT_TRUE(reshapeShapesAllowView(inShape, outShape, /*inputIsTiled=*/true));

  // view_in_h8col_32x1__out_h64_4x1: a legal, recompute-matching 64-core
  // candidate -- but the view path returns the input's 8-core spec instead.
  EXPECT_FALSE(reshapeOutputSpecIsHonored(
      shardedLayout(outShape, TensorMemoryLayout::HeightSharded, {64, 1},
                    coreGrid(8, 8)),
      input, inShape, outShape));

  // Same memory config as the input: the view returns exactly that.
  EXPECT_TRUE(reshapeOutputSpecIsHonored(
      shardedLayout(outShape, TensorMemoryLayout::HeightSharded, {8, 1},
                    coreGrid(1, 8)),
      input, inShape, outShape));

  // A 64-core candidate is honored once the shapes take reshape off the view
  // path (same volume, but the last dim changes).
  llvm::SmallVector<int64_t> nonViewOutShape = {256, 1024};
  ASSERT_FALSE(
      reshapeShapesAllowView(inShape, nonViewOutShape, /*inputIsTiled=*/true));
  EXPECT_TRUE(reshapeOutputSpecIsHonored(
      shardedLayout(nonViewOutShape, TensorMemoryLayout::HeightSharded, {64, 1},
                    coreGrid(8, 8)),
      input, inShape, nonViewOutShape));
}

// Rule W: WIDTH_SHARDED shards keep the full physical height and split the
// width across the grid.
TEST_F(ReshapeDivergenceGuardTest, WidthShardedShardShapeMustMatchRecompute) {
  llvm::SmallVector<int64_t> inShape = {512, 512};
  llvm::SmallVector<int64_t> outShape = {1, 256, 32, 32};
  TTNNLayoutAttr input = dramInterleavedLayout(inShape);

  // w64_256x1: one tile of width does not divide over 64 cores, so recompute
  // lands on 256x1 tiles on the full grid -- which is what is declared.
  EXPECT_TRUE(reshapeOutputSpecIsHonored(
      shardedLayout(outShape, TensorMemoryLayout::WidthSharded, {1, 64},
                    coreGrid(8, 8)),
      input, inShape, outShape));
}

// The row-major reshape path never honors a requested output config at all
// (reshape.cpp:244 passes explicit_memory_config=false). `ttnn::reshape`
// neither tilizes nor untilizes, so it is the *input's* tile-ness that decides
// whether the op lands there: a tiled sharded candidate on a row-major input
// describes an output that cannot exist. Testing only the candidate lets that
// pair through, and it is what declared the one-core BLOCK_SHARDED L1 spec the
// n150 LLM decode/prefill layer tests tripped over at runtime.
TEST_F(ReshapeDivergenceGuardTest, RowMajorReshapePathIsNeverHonored) {
  llvm::SmallVector<int64_t> inShape = {512, 512};
  llvm::SmallVector<int64_t> outShape = {1, 256, 32, 32};

  // Control: on a tiled input the covering block-sharded candidate is honored.
  TTNNLayoutAttr tiledCandidate = shardedLayout(
      outShape, TensorMemoryLayout::BlockSharded, {8, 1}, coreGrid(8, 8));
  EXPECT_TRUE(reshapeOutputSpecIsHonored(
      tiledCandidate, dramInterleavedLayout(inShape), inShape, outShape));

  // The same candidate on a row-major input: the output is row-major whatever
  // the candidate claims, so the spec is renegotiated.
  EXPECT_FALSE(reshapeOutputSpecIsHonored(
      tiledCandidate, dramInterleavedRowMajorLayout(inShape), inShape,
      outShape));

  // A row-major sharded candidate is rejected from either side.
  TTNNLayoutAttr rowMajorCandidate = rowMajorShardedLayout(
      outShape, TensorMemoryLayout::BlockSharded, {8, 1}, coreGrid(8, 8));
  EXPECT_FALSE(reshapeOutputSpecIsHonored(
      rowMajorCandidate, dramInterleavedLayout(inShape), inShape, outShape));
  EXPECT_FALSE(reshapeOutputSpecIsHonored(
      rowMajorCandidate, dramInterleavedRowMajorLayout(inShape), inShape,
      outShape));

  // The shape from the real failure: `32x1 -> 32` off a row-major DRAM operand,
  // block-sharded onto a single core. (Rule P would reject this one too -- a
  // rank-1 output has a logical height of 1 -- but the row-major path is the
  // reason it was observed to fail.)
  llvm::SmallVector<int64_t> llmInShape = {32, 1};
  llvm::SmallVector<int64_t> llmOutShape = {32};
  ASSERT_FALSE(reshapeShapesAllowView(llmInShape, llmOutShape,
                                      /*inputIsTiled=*/false));
  EXPECT_FALSE(reshapeOutputSpecIsHonored(
      shardedLayout(llmOutShape, TensorMemoryLayout::BlockSharded, {1, 1},
                    coreGrid(1, 1)),
      dramInterleavedRowMajorLayout(llmInShape), llmInShape, llmOutShape));
}

// Rule P: a sharded candidate whose *logical* inner 2-D is not a multiple of
// the tile shape part-fills its tiles, and the rest of every shard is implicit
// padding the runtime never fills (`ttnn::reshape` is called with no
// `pad_value`). Such an output is returned with the requested config, so none
// of the substitution rules see it -- and yet a consumer re-dispatched over
// those shards miscomputes: on segformer this took the model from PCC 0.9967
// to 0.4851.
TEST_F(ReshapeDivergenceGuardTest, NonTileAlignedLogicalShapeIsRejected) {
  llvm::SmallVector<int64_t> inShape = {512, 512};

  // The two output shapes have the *same* physical extent -- 8192x32, 256x1
  // tiles -- and take the same candidate layout. Only the logical shape
  // differs: 32 real rows of every 32 versus 8 of every 32.
  llvm::SmallVector<int64_t> alignedOut = {1, 256, 32, 32};
  llvm::SmallVector<int64_t> paddedOut = {1, 256, 8, 32};

  TTNNLayoutAttr input = dramInterleavedLayout(inShape);
  EXPECT_TRUE(reshapeOutputSpecIsHonored(
      shardedLayout(alignedOut, TensorMemoryLayout::HeightSharded, {64, 1},
                    coreGrid(8, 8)),
      input, inShape, alignedOut));
  EXPECT_FALSE(reshapeOutputSpecIsHonored(
      shardedLayout(paddedOut, TensorMemoryLayout::HeightSharded, {64, 1},
                    coreGrid(8, 8)),
      input, inShape, paddedOut));

  // Interleaved candidates are unaffected -- they never carry a shard spec.
  EXPECT_TRUE(reshapeOutputSpecIsHonored(dramInterleavedLayout(paddedOut),
                                         input, inShape, paddedOut));

  // The shapes from the segformer failure (chisel RCA §3): the head of the
  // chain is `1x16x16x256` off the stage-3 (16x16) decode-head branch.
  llvm::SmallVector<int64_t> stage3In = {1, 256, 256};
  llvm::SmallVector<int64_t> stage3Out = {1, 16, 16, 256}; // 16x8 tiles
  EXPECT_FALSE(reshapeOutputSpecIsHonored(
      shardedLayout(stage3Out, TensorMemoryLayout::BlockSharded, {8, 8},
                    coreGrid(8, 8)),
      dramInterleavedLayout(stage3In), stage3In, stage3Out));

  // Its tile-aligned sibling, same branch geometry, is still admitted.
  llvm::SmallVector<int64_t> alignedStageOut = {1, 16, 32, 256};
  EXPECT_TRUE(reshapeOutputSpecIsHonored(
      shardedLayout(alignedStageOut, TensorMemoryLayout::BlockSharded, {8, 8},
                    coreGrid(8, 8)),
      dramInterleavedLayout(stage3In), stage3In, alignedStageOut));

  // A non-tile-aligned inner *width* is rejected on the same grounds.
  llvm::SmallVector<int64_t> narrowOut = {1, 256, 32, 24};
  EXPECT_FALSE(reshapeOutputSpecIsHonored(
      shardedLayout(narrowOut, TensorMemoryLayout::HeightSharded, {64, 1},
                    coreGrid(8, 8)),
      input, inShape, narrowOut));
}

// The guard is expressed as a config filter, and must leave NULL-hint configs
// alone.
TEST_F(ReshapeDivergenceGuardTest, FilterDropsOnlyDivergentConfigs) {
  llvm::SmallVector<int64_t> inShape = {512, 512};
  llvm::SmallVector<int64_t> outShape = {1, 256, 32, 32};
  std::vector<OpConfig> configs = {
      OpConfig(TTNNLayoutAttr()),
      OpConfig(dramInterleavedLayout(outShape)),
      OpConfig(shardedLayout(outShape, TensorMemoryLayout::HeightSharded,
                             {64, 1}, coreGrid(8, 8))),
      OpConfig(shardedLayout(outShape, TensorMemoryLayout::HeightSharded,
                             {32, 1}, coreGrid(8, 8))),
  };

  std::vector<OpConfig> kept = filterConfigs(
      configs, reshapeOutputSpecIsHonoredFilter(dramInterleavedLayout(inShape),
                                                inShape, outShape));
  ASSERT_EQ(kept.size(), 3u);
  EXPECT_FALSE(kept[0].outputLayout);
  EXPECT_EQ(kept[1].outputLayout, configs[1].outputLayout);
  EXPECT_EQ(kept[2].outputLayout, configs[2].outputLayout);
}
