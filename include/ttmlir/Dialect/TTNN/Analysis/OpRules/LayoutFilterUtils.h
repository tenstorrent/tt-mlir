// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_LAYOUTFILTERUTILS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_LAYOUTFILTERUTILS_H

#include "ttmlir/Dialect/TTNN/Analysis/OpModelStrategy.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace mlir::tt::ttnn::layout_filter_utils {

/// Reject all sharded layouts. Returns true if the layout should be kept.
inline bool rejectAllSharded(TTNNLayoutAttr layout) {
  auto ml = layout.getMemLayout();
  return !(ml && isShardedMemoryLayout(ml.getValue()));
}

/// Require DRAM-interleaved: reject sharded and L1 layouts.
inline bool requireDRAMInterleaved(TTNNLayoutAttr layout) {
  auto ml = layout.getMemLayout();
  if (ml && isShardedMemoryLayout(ml.getValue())) {
    return false;
  }
  return isDRAMBufferType(layout.getBufferType());
}

/// Reject L1-interleaved: keep DRAM (any) and L1-sharded, reject
/// L1-interleaved. Used for inputs whose kernel requires DRAM when not sharded
/// (e.g. Q in sdpa_decode: "Q tensor buffer type must be DRAM when not
/// sharded").
inline bool rejectL1Interleaved(TTNNLayoutAttr layout) {
  auto ml = layout.getMemLayout();
  bool isSharded = ml && isShardedMemoryLayout(ml.getValue());
  if (isSharded) {
    return true;
  }
  return isDRAMBufferType(layout.getBufferType());
}

/// Accept only ROW_MAJOR candidates. Pair with generatesRowMajorInputSiblings()
/// to drop the tiled originals once the RM siblings are synthesized.
inline bool requireRowMajor(TTNNLayoutAttr layout) {
  return layout.getLayout() == Layout::RowMajor;
}

/// Reject width-sharded layouts. Returns true if the layout should be kept.
inline bool rejectWidthSharded(TTNNLayoutAttr layout) {
  auto ml = layout.getMemLayout();
  return !(ml && ml.getValue() == TensorMemoryLayout::WidthSharded);
}

/// Geometry of a sharded layout's core grid: the number of cores it actually
/// uses and the extent of its bounding box (X = columns, Y = rows, matching
/// tt-metal's CoreCoord convention).
struct ShardGridGeometry {
  int64_t numCores = 0;
  int64_t bboxWidth = 0;
  int64_t bboxHeight = 0;

  /// tt-metal only accepts a grid whose cores fill its bounding box; anything
  /// else makes `bounding_box()` silently include cores the tensor does not
  /// occupy.
  bool isRectangular() const {
    return numCores > 0 && numCores == bboxWidth * bboxHeight;
  }
};

/// Core-grid geometry of a sharded layout. All-zero for an empty core range
/// set (which is not a usable grid).
inline ShardGridGeometry getShardGridGeometry(TTNNLayoutAttr layout) {
  ttnn::CoreRangeSetAttr ranges = layout.getCoreRangeSet();
  assert(ranges && "sharded layout must have a valid core range set");

  if (ranges.getCoreRanges().empty()) {
    return ShardGridGeometry{};
  }

  int64_t minX = std::numeric_limits<int64_t>::max();
  int64_t minY = std::numeric_limits<int64_t>::max();
  int64_t maxX = std::numeric_limits<int64_t>::min();
  int64_t maxY = std::numeric_limits<int64_t>::min();
  int64_t numCores = 0;
  for (const auto &coreRange : ranges.getCoreRanges()) {
    auto start = coreRange.getStartCoord();
    auto end = coreRange.getEndCoord();

    int64_t sizeX = static_cast<int64_t>(end.getX() - start.getX() + 1);
    int64_t sizeY = static_cast<int64_t>(end.getY() - start.getY() + 1);

    numCores += sizeX * sizeY;

    minX = std::min(minX, static_cast<int64_t>(start.getX()));
    minY = std::min(minY, static_cast<int64_t>(start.getY()));
    maxX = std::max(maxX, static_cast<int64_t>(end.getX()));
    maxY = std::max(maxY, static_cast<int64_t>(end.getY()));
  }

  return ShardGridGeometry{numCores, maxX - minX + 1, maxY - minY + 1};
}

/// Whether a sharded layout's core grid forms a full rectangular bounding
/// box (num_cores == bbox_num_cores). Interleaved layouts return true.
inline bool isFullBboxSharded(TTNNLayoutAttr layout) {
  auto ml = layout.getMemLayout();
  if (!isShardedMemoryLayout(ml.getValue())) {
    return true;
  }

  return getShardGridGeometry(layout).isRectangular();
}

//===----------------------------------------------------------------------===//
// Reshape output-layout divergence guard
//
// `ttnn::reshape` does not always return the memory config it is handed: four
// host-side negotiations in
// `ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp` can
// substitute a different one. When that happens the IR's declared output
// layout is a lie -- the consumer is dispatched against a layout the tensor
// does not have -- and the measured outcomes range from a silently wrong
// consumer result to a device hang (5 of 11 divergent configs in an on-device
// sweep hung, each needing `tt-smi -r`). None of them is caught at compile
// time: the device op validates only storage/dtype, and the runtime's
// expected-vs-actual memory-config check is DEBUG_ASSERT-only.
//
// The predicate below rejects any sharded reshape-output candidate that
// tt-metal would not return verbatim. It was validated on device against the
// measured "config honored" bit over 97 reshape configs (40 synthetic boundary
// cases + 57 real ones extracted from a compiled vision model) with 0
// mispredictions. See https://github.com/tenstorrent/tt-mlir/issues/8020.
//
// It also rejects one class tt-metal *does* return verbatim but whose consumer
// then miscomputes: a sharded output whose logical inner 2-D extent is not
// tile-aligned, so its shards are part implicit padding (rule P below).
//===----------------------------------------------------------------------===//

/// Shape-side of tt-metal's `this_is_view` fast path (reshape.cpp:611-619):
/// the last dim is unchanged, and for a tiled input the second-to-last dim is
/// either unchanged or tile-aligned on both sides (no padding change).
inline bool reshapeShapesAllowView(llvm::ArrayRef<int64_t> inShape,
                                   llvm::ArrayRef<int64_t> outShape,
                                   bool inputIsTiled) {
  if (inShape.empty() || outShape.empty()) {
    return false;
  }

  // Condition 1: last dimension must be unchanged.
  if (inShape.back() != outShape.back()) {
    return false;
  }

  // Condition 2: for tiled layout, second-to-last dim must be unchanged or
  // both second-to-last dims must be tile-aligned (no padding change).
  if (inputIsTiled) {
    int64_t inSecondLast =
        inShape.size() >= 2 ? inShape[inShape.size() - 2] : 1;
    int64_t outSecondLast =
        outShape.size() >= 2 ? outShape[outShape.size() - 2] : 1;
    if (inSecondLast != outSecondLast && !(outSecondLast % TILE_HEIGHT == 0 &&
                                           inSecondLast % TILE_HEIGHT == 0)) {
      return false;
    }
  }

  return true;
}

/// Whether two layouts describe the same tt-metal memory config: same buffer
/// type, same memory layout, same core placement and same per-core shard shape
/// (in scalar elements). Deliberately shape-agnostic -- the layouts being
/// compared belong to tensors of different logical shapes, so attribute
/// equality would never hold.
inline bool sameMemoryConfig(TTNNLayoutAttr lhs, TTNNLayoutAttr rhs) {
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs.getBufferType() != rhs.getBufferType() ||
      lhs.getMemLayout() != rhs.getMemLayout()) {
    return false;
  }
  if (lhs.getCoreRangeSet() != rhs.getCoreRangeSet()) {
    return false;
  }
  return lhs.getScalarShardShape() == rhs.getScalarShardShape();
}

/// Whether `ttnn::reshape` returns `candidate` verbatim as the output memory
/// config, i.e. whether the compiler may declare it. Interleaved candidates
/// are always honored. For a sharded candidate all four substitution rules
/// must be cleared:
///
///   V  the `this_is_view` fast path (reshape.cpp:613-624) discards the
///      requested config entirely -- `PerformView` takes no memory-config
///      argument -- and returns the *input's* spec. Note that `this_is_view`
///      compares only the `is_sharded()` / `is_l1()` booleans, never the
///      sharding kind, grid or shard shape, so it fires on layout changes it
///      then drops.
///   N  a non-rectangular shard grid falls back to INTERLEAVED
///      (reshape.cpp:121-130).
///   H/W for HEIGHT/WIDTH_SHARDED the caller's shard *shape* is always
///      re-derived as `round32(div_up(phys_dim, num_cores))` over the
///      preserved grid (reshape.cpp:177-211, tensor_spec.cpp:179-197); the
///      explicit-config passthrough at reshape.cpp:140 is gated on
///      BLOCK_SHARDED and so cannot preserve it.
///   B  a BLOCK_SHARDED spec is passed through only when it is tile-aligned
///      *and* its grid covers the output's physical extent
///      (reshape.cpp:137-152, `covers_output` uses `>=`, so over-provisioned
///      grids are honored); otherwise a smaller derived grid replaces it.
///
/// and two further classes are rejected outright:
///
///   P  an output whose *logical* inner 2-D extent is not tile-aligned, i.e.
///      whose shards are part padding (see below), and
///   R  the row-major reshape path, on either side, because it never honors an
///      explicit output config at all (reshape.cpp:244).
///
/// `inShape` / `outShape` are the reshape's logical input and output shapes and
/// `inputLayout` the layout of its operand.
inline bool reshapeOutputSpecIsHonored(TTNNLayoutAttr candidate,
                                       TTNNLayoutAttr inputLayout,
                                       llvm::ArrayRef<int64_t> inShape,
                                       llvm::ArrayRef<int64_t> outShape) {
  // A NULL hint means "backend picks", and an interleaved output config is
  // never renegotiated.
  if (!candidate) {
    return true;
  }
  TensorMemoryLayoutAttr memLayout = candidate.getMemLayout();
  if (!memLayout || !isShardedMemoryLayout(memLayout.getValue())) {
    return true;
  }
  if (!inputLayout) {
    return false;
  }

  // Rule V: when tt-metal takes the view path the declared config is dropped
  // and the input's spec is returned, so the two must already agree.
  // `this_is_view` requires matching is_sharded()/is_l1() as well; a sharded
  // candidate can only reach it from a sharded input.
  const bool inputIsSharded = inputLayout.hasShardedTensorMemoryLayout();
  const bool sameMemorySpace = isL1BufferType(candidate.getBufferType()) ==
                               isL1BufferType(inputLayout.getBufferType());
  if (inputIsSharded && sameMemorySpace &&
      reshapeShapesAllowView(inShape, outShape, inputLayout.isTiled()) &&
      !sameMemoryConfig(candidate, inputLayout)) {
    return false;
  }

  // Rule P: a sharded output whose *logical* inner 2-D extent is not a
  // multiple of the tile shape only part-fills its tiles -- the remaining
  // rows/columns are implicit padding, and the runtime calls `ttnn::reshape`
  // with no `pad_value`, so tt-metal's tile-padding fill never runs over them.
  // An interleaved consumer tolerates that; a consumer re-dispatched over such
  // shards does not. Measured on segformer: enabling sharded reshape outputs
  // (#8020) moved the decode head's 16x16 branch -- reshapes carrying 16 real
  // rows of every 32 -- onto block-sharded L1, and the `ttnn.linear` reading
  // them returned a materially different result from a bitwise-identical
  // input, taking the model from PCC 0.9967 to 0.4851 end to end.
  //
  // This is a property of the *logical* shape, not of the memory layout: all
  // of those outputs are `TILE`-layout and tile-aligned in their shard shapes,
  // so the tile-alignment tests further down cannot see them.
  const int64_t logicalWidth = outShape.empty() ? 1 : outShape.back();
  const int64_t logicalHeight =
      outShape.size() >= 2 ? outShape[outShape.size() - 2] : 1;
  if (logicalHeight % TILE_HEIGHT != 0 || logicalWidth % TILE_WIDTH != 0) {
    return false;
  }

  // Rule N: a non-rectangular grid is downgraded to INTERLEAVED.
  ShardGridGeometry grid = getShardGridGeometry(candidate);
  if (!grid.isRectangular()) {
    return false;
  }

  // Rules H/W/B compare the declared per-core shard shape against what
  // tt-metal derives, in tiles. Row-major sharded outputs are rejected
  // outright: their shard shape is re-derived through a different
  // (alignment-dependent) path that this predicate does not model, and the
  // explicit-config passthrough is unavailable on the row-major reshape path
  // (reshape.cpp:244 passes explicit_memory_config=false).
  //
  // The *input's* tile-ness is what decides this, not only the candidate's:
  // `ttnn::reshape` neither tilizes nor untilizes, so the output is row-major
  // exactly when the input is (tt-mlir relies on the same invariant in
  // `TTNNLayout.cpp`'s `shouldTilizeResult`). A tiled candidate on a row-major
  // input therefore describes an output that cannot exist -- the op model
  // reports the row-major layout tt-metal will really produce and
  // `TTNNOperationValidationAndFallback` rewrites the result type to it,
  // carrying the sharded memory config over onto the row-major path. Testing
  // the candidate alone lets that pair through; measured on the n150 LLM
  // decode/prefill layer tests it is what declares a BLOCK_SHARDED L1 spec
  // (e.g. shard [1, 32] on one core) that the device then rebuilds through its
  // ND-shard path, tripping the runtime's expected-vs-actual memory-config
  // check.
  if (!inputLayout.isTiled() || !candidate.isTiled()) {
    return false;
  }
  llvm::SmallVector<int64_t> outTiles = candidate.getTiledShape(outShape);
  llvm::SmallVector<int64_t> shardTiles = candidate.getShardShape();
  assert(outTiles.size() >= 2 && shardTiles.size() >= 2);
  const int64_t tilesH = outTiles[outTiles.size() - 2];
  const int64_t tilesW = outTiles[outTiles.size() - 1];
  const int64_t shardH = shardTiles[shardTiles.size() - 2];
  const int64_t shardW = shardTiles[shardTiles.size() - 1];

  switch (memLayout.getValue()) {
  case TensorMemoryLayout::HeightSharded:
    // round32(div_up(phys_h, ncores)) / 32 == div_up(tiles_h, ncores), and the
    // shard keeps the full physical width.
    return shardH ==
               static_cast<int64_t>(llvm::divideCeil(tilesH, grid.numCores)) &&
           shardW == tilesW;
  case TensorMemoryLayout::WidthSharded:
    return shardW ==
               static_cast<int64_t>(llvm::divideCeil(tilesW, grid.numCores)) &&
           shardH == tilesH;
  case TensorMemoryLayout::BlockSharded:
    // Tile alignment holds by construction for a tiled shard shape; only the
    // covers_output test can fail. tt-mlir layouts are always ROW_MAJOR
    // oriented, so rows map to Y and columns to X.
    return grid.bboxHeight * shardH >= tilesH &&
           grid.bboxWidth * shardW >= tilesW;
  default:
    return false;
  }
}

/// Filter rejecting sharded reshape-output candidates that `ttnn::reshape`
/// would not return verbatim. See `reshapeOutputSpecIsHonored`.
inline LayoutFilterFn
reshapeOutputSpecIsHonoredFilter(TTNNLayoutAttr inputLayout,
                                 llvm::ArrayRef<int64_t> inShape,
                                 llvm::ArrayRef<int64_t> outShape) {
  llvm::SmallVector<int64_t> in(inShape);
  llvm::SmallVector<int64_t> out(outShape);
  return [inputLayout, in, out](TTNNLayoutAttr candidate) -> bool {
    return reshapeOutputSpecIsHonored(candidate, inputLayout, in, out);
  };
}

/// Allow only a specific sharding type (plus interleaved). Returns a filter
/// function that rejects sharded layouts whose type doesn't match.
inline LayoutFilterFn
allowOnlyShardingType(TensorMemoryLayout allowedSharding) {
  return [allowedSharding](TTNNLayoutAttr layout) -> bool {
    auto ml = layout.getMemLayout();
    if (!ml || !isShardedMemoryLayout(ml.getValue())) {
      return true; // interleaved — keep
    }
    return ml.getValue() == allowedSharding;
  };
}

/// Filter legalConfigs by an output-layout predicate. Configs with a NULL
/// output layout are always kept (NULL hint means "backend picks"). Keeps any
/// config whose output layout passes `keep`.
inline std::vector<OpConfig>
filterConfigs(const std::vector<OpConfig> &legalConfigs, LayoutFilterFn keep) {
  std::vector<OpConfig> result;
  for (const auto &config : legalConfigs) {
    if (!config.outputLayout || keep(config.outputLayout)) {
      result.push_back(config);
    }
  }
  return result;
}

/// Non-sharded output hints (common pattern for many ops).
inline OutputHints
nonShardedOutputHints(const std::vector<OpConfig> &legalConfigs) {
  return OutputHints{filterConfigs(legalConfigs, rejectAllSharded), {}};
}

/// All non-width-sharded output hints (interleaved + block/height sharded).
inline OutputHints
nonWidthShardedOutputHints(const std::vector<OpConfig> &legalConfigs) {
  return OutputHints{filterConfigs(legalConfigs, rejectWidthSharded), {}};
}

/// NULL-hint-only output (backend decides from inputs, no fallbacks).
inline OutputHints nullHintOnly() {
  return OutputHints{{OpConfig(TTNNLayoutAttr())}, {}};
}

/// DRAM-interleaved output configs only (drops sharded and L1-interleaved
/// configs). Useful for ops whose downstream consumers require DRAM input.
inline OutputHints
dramInterleavedOnlyOutputHints(const std::vector<OpConfig> &legalConfigs) {
  std::vector<OpConfig> result;
  for (const auto &cfg : legalConfigs) {
    if (!cfg.outputLayout) {
      result.push_back(cfg);
      continue;
    }
    if (!isDRAMBufferType(cfg.outputLayout.getBufferType())) {
      continue;
    }
    auto memLayout = cfg.outputLayout.getMemLayout();
    if (memLayout && isShardedMemoryLayout(memLayout.getValue())) {
      continue;
    }
    result.push_back(cfg);
  }
  return OutputHints{result, {}};
}

} // namespace mlir::tt::ttnn::layout_filter_utils

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_LAYOUTFILTERUTILS_H
