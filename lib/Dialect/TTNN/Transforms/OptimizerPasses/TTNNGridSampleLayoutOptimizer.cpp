// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TTNNGridSampleLayoutOptimizer
//
// The compiler normally inserts an unnecessary TILE round-trip for the
// GridSample grid tensor. Since grid_sample mandates ROW_MAJOR input and
// the grid arrives as a contiguous ROW_MAJOR tensor, the TILE conversion
// is pure overhead with no functional purpose.
//
// Before:
//   BlockArg (ROW_MAJOR DRAM, 5D)
//     ↓ to_layout(TILE DRAM)           — 16× size inflation
//     ↓ reshape(TILE DRAM, 4D)         — DRAM copy at inflated size
//     ↓ [to_memory_config(DRAM)]       — optional intermediate step
//     ↓ to_layout(ROW_MAJOR DRAM, 4D)  — untilize back to ROW_MAJOR
//     ↓ grid_sample                    — reads grid from DRAM
//
// After:
//   BlockArg (ROW_MAJOR DRAM, 5D)
//     ↓ reshape(ROW_MAJOR DRAM, 4D)          — zero-copy logical view
//     ↓ to_memory_config(L1 HEIGHT_SHARD, 4D) — grid resident in L1
//     ↓ grid_sample                           — reads grid from L1 ✓
//
// Why the reshape must precede the L1 shard:
//   The 5D shard page (last-dim stride × 2B) is too small to satisfy the
//   device L1 alignment requirement (shard page ≥ 16 B).  After the 4D
//   reshape the shard page grows to (last-dim × 2B ≥ 32 B), satisfying
//   the constraint.  The DRAM reshape is a zero-copy reinterpretation of
//   contiguous memory — no DRAM bandwidth is consumed.
//
// Per-core L1 footprint after sharding:
//   shard_rows × shard_cols × element_size = 4 KB/core  ← fits in L1 ✓
//
// Per GridSampleOp, this pass eliminates three kernel dispatches —
//   TilizeWithValPadding + ReshapeView(TILE) + Untilize —
// and moves the grid to L1 HEIGHT_SHARDED, reducing DRAM reads during
// spatial sampling.

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNGRIDSAMPLELAYOUTOPTIMIZER
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

static void eraseDeallocsOf(Value v) {
  SmallVector<Operation *, 4> toErase;
  for (Operation *user : v.getUsers())
    if (mlir::isa<ttnn::DeallocateOp>(user))
      toErase.push_back(user);
  for (Operation *op : toErase)
    op->erase();
}

static void eraseWithDeallocs(Operation *op) {
  if (op->getNumResults() > 0)
    eraseDeallocsOf(op->getResult(0));
  op->erase();
}

// Matches a layout-changing op whose result has page layout `wantLayout`,
// returning its input Value (null on mismatch).
//
// This pass runs before TTNNDecomposeLayouts, so layout changes are still
// represented as the aggregate ttnn.to_tensor_spec op rather than the
// decomposed ttnn.to_layout / to_device / to_memory_config triple. Accept both
// forms so the pattern matches regardless of where in the pipeline it runs.
static Value matchLayoutChange(Operation *op, Layout wantLayout) {
  Value input;
  if (auto toLayout = mlir::dyn_cast_or_null<ttnn::ToLayoutOp>(op)) {
    input = toLayout.getInput();
  } else if (auto toSpec =
                 mlir::dyn_cast_or_null<ttnn::ToTensorSpecOp>(op)) {
    input = toSpec.getInput();
  } else {
    return nullptr;
  }
  auto lo = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
      mlir::cast<RankedTensorType>(op->getResult(0).getType()).getEncoding());
  if (!lo || lo.getLayout() != wantLayout)
    return nullptr;
  return input;
}

} // namespace

class TTNNGridSampleLayoutOptimizerPass
    : public impl::TTNNGridSampleLayoutOptimizerBase<
          TTNNGridSampleLayoutOptimizerPass> {
public:
  using impl::TTNNGridSampleLayoutOptimizerBase<
      TTNNGridSampleLayoutOptimizerPass>::TTNNGridSampleLayoutOptimizerBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = &getContext();
    [[maybe_unused]] int scanned = 0, optimized = 0, l1Sharded = 0;

    SmallVector<ttnn::GridSampleOp, 16> gsOps;
    moduleOp.walk([&](ttnn::GridSampleOp op) { gsOps.push_back(op); });

    for (ttnn::GridSampleOp gsOp : gsOps) {
      ++scanned;

      Value gridVal = gsOp.getGrid();
      auto *gridDef = gridVal.getDefiningOp();
      if (!gridDef) continue;

      // ── STEP 1: grid must come from a layout change to ROW_MAJOR ──────────
      Operation *rmLayoutOp = gridDef;
      Value beforeRm = matchLayoutChange(rmLayoutOp, Layout::RowMajor);
      if (!beforeRm) {
        continue;
      }

      // ── STEP 2: optionally skip to_memory_config(DRAM) if present ──────────
      // Note: newer TTNN encodes the destination memory config in the result
      // type's TTNNLayoutAttr rather than as an inline `memory_config`
      // attribute on the op, so read the buffer type from the result type and
      // fall back to the attribute only when present (older IR).
      ttnn::ToMemoryConfigOp memCfgOp = nullptr;
      if (auto *defOp = beforeRm.getDefiningOp()) {
        if (auto mco = mlir::dyn_cast<ttnn::ToMemoryConfigOp>(defOp)) {
          std::optional<BufferType> bufType;
          if (auto mcoCfg = mco.getMemoryConfig()) {
            bufType =
                mlir::cast<BufferTypeAttr>(mcoCfg->getBufferType()).getValue();
          } else if (auto lo = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
                         mlir::cast<RankedTensorType>(
                             mco.getResult().getType())
                             .getEncoding())) {
            bufType = lo.getBufferType();
          }
          if (bufType && *bufType == BufferType::DRAM) {
            memCfgOp = mco;
            beforeRm = mco.getInput();
          }
        }
      }

      // ── STEP 3: must be from a ReshapeOp with TILE layout ──────────────────
      auto tileReshape =
          mlir::dyn_cast_or_null<ttnn::ReshapeOp>(beforeRm.getDefiningOp());
      if (!tileReshape) {
        continue;
      }
      {
        auto lo = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
            mlir::cast<RankedTensorType>(tileReshape.getResult().getType())
                .getEncoding());
        if (!lo || lo.getLayout() != Layout::Tile) {
          continue;
        }
      }

      // ── STEP 4: Reshape input must come from a layout change to TILE ──────
      Operation *tileLayoutOp = tileReshape.getInput().getDefiningOp();
      Value lutArg = matchLayoutChange(tileLayoutOp, Layout::Tile);
      if (!lutArg) {
        continue;
      }

      // ── STEP 5: original LUT must be ROW_MAJOR DRAM/SystemMemory ───────────
      auto lutRtt  = mlir::cast<RankedTensorType>(lutArg.getType());
      auto lutLo   = mlir::dyn_cast_or_null<TTNNLayoutAttr>(lutRtt.getEncoding());
      if (!lutLo || lutLo.getLayout() != Layout::RowMajor) {
        continue;
      }
      if (lutLo.getBufferType() != BufferType::DRAM &&
          lutLo.getBufferType() != BufferType::SystemMemory) {
        continue;
      }

      // ── All checks passed — build optimized IR ─────────────────────────────

      // Reshape target shape (4D, same logical shape as TILE reshape output).
      auto reshapeResultShape =
          mlir::cast<RankedTensorType>(tileReshape.getResult().getType())
              .getShape();

      // DRAM ROW_MAJOR layout for the reshape output (4D).
      auto rmDramLayout =
          TTNNLayoutAttr::Builder(ctx, reshapeResultShape,
                                  lutLo.getScalarElementType())
              .setBufferType(BufferType::DRAM)
              .setLayout(Layout::RowMajor)
              .setMemoryLayout(TensorMemoryLayout::Interleaved)
              .build();
      auto rmDramType = RankedTensorType::get(
          reshapeResultShape, lutRtt.getElementType(), rmDramLayout);

      // Insert ops right before grid_sample.
      OpBuilder builder(gsOp);

      // ── Step A: Free ROW_MAJOR reshape (zero DRAM copy) ───────────────────
      // BlockArg (5D, DRAM ROW_MAJOR) → 4D DRAM ROW_MAJOR.
      // This is a logical view: merging last two dims in contiguous memory
      // costs no bandwidth.
      auto newReshape = builder.create<ttnn::ReshapeOp>(
          gsOp.getLoc(), rmDramType, lutArg,
          tileReshape.getShapeAttr());

      // ── Step B: Shard 4D grid to L1 HEIGHT_SHARDED ────────────────────────
      // 4D shard (128 rows × 16 cols × 2B = 32B page) satisfies L1 16B
      // alignment.  The 5D LUT cannot be sharded directly (4B page fails).
      //
      // Total rows after reshape = 1 × 128 × 64 = 8192; 8192 / 64 = 128/core.
      Value gridForSample = newReshape.getResult();
      {
        int64_t totalRows = 1;
        for (size_t i = 0; i + 1 < reshapeResultShape.size(); ++i)
          totalRows *= reshapeResultShape[i];

        constexpr int64_t NUM_CORES_H = 8, NUM_CORES_W = 8;
        constexpr int64_t NUM_CORES   = NUM_CORES_H * NUM_CORES_W;

        if (totalRows % NUM_CORES == 0) {
          auto coreRangeSet = CoreRangeSetAttr::get(
              ctx, CoreRangeAttr::get(ctx,
                                      CoreCoordAttr::get(ctx, 0, 0),
                                      CoreCoordAttr::get(ctx,
                                                         NUM_CORES_H - 1,
                                                         NUM_CORES_W - 1)));

          // Build L1 HEIGHT_SHARDED layout from the DRAM reshape output type.
          auto l1Layout =
              TTNNLayoutAttr::Builder(rmDramType)
                  .setBufferType(BufferType::L1)
                  .setMemoryLayout(TensorMemoryLayout::HeightSharded)
                  .setGridShape({NUM_CORES, 1})
                  .setCoreRangeSet(coreRangeSet)
                  .build();

          auto l1Type = utils::RankedTensorTypeFactory::create(
              rmDramType, l1Layout);

          auto toL1 = builder.create<ttnn::ToMemoryConfigOp>(
              gsOp.getLoc(), l1Type, newReshape.getResult());
          gridForSample = toL1.getResult();
          ++l1Sharded;
        }
      }

      // Move any DeallocateOp for lutArg that precedes the new reshape to after.
      for (Operation *user : lutArg.getUsers()) {
        if (mlir::isa<ttnn::DeallocateOp>(user)) {
          if (user->isBeforeInBlock(newReshape))
            user->moveAfter(newReshape);
          break;
        }
      }

      // Re-point grid_sample to the L1 grid.
      gsOp.getGridMutable().assign(gridForSample);

      // Erase the old dead chain.
      eraseWithDeallocs(rmLayoutOp);
      if (memCfgOp)
        eraseWithDeallocs(memCfgOp);
      eraseWithDeallocs(tileReshape);
      eraseWithDeallocs(tileLayoutOp);

      ++optimized;
    }

    (void)l1Sharded;
  }
};

} // namespace mlir::tt::ttnn
