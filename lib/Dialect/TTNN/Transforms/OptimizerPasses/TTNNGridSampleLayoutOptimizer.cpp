// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TTNNGridSampleLayoutOptimizer
//
// Before this pass the compiler inserts an unnecessary TILE round-trip for the
// GridSample LUT (grid) tensor, keeping everything in DRAM:
//
//   BlockArg (ROW_MAJOR DRAM 5D)  ~256 KB
//     ↓ to_layout(TILE DRAM)           ← 16× inflation → ~4 MB
//     ↓ reshape(TILE DRAM 4D)          ← DRAM copy of 4 MB
//     ↓ [to_memory_config(DRAM)]       ← Block D only
//     ↓ to_layout(ROW_MAJOR DRAM 4D)   ← untilize of 4 MB
//     ↓ grid_sample                    ← reads grid from DRAM
//
// This pass replaces the chain with a free ROW_MAJOR reshape directly into
// L1 HEIGHT_SHARDED so the grid_sample reads from L1:
//
//   BlockArg (ROW_MAJOR DRAM 5D)  ~256 KB
//     ↓ reshape(ROW_MAJOR DRAM 4D)          ← FREE view (zero copy, page-view)
//     ↓ to_memory_config(L1 HEIGHT_SHARD 4D) ← 256KB→L1, shard=(128×16)/core
//     ↓ grid_sample                          ← reads grid from L1 ✓
//
// Why the reshape must happen before the L1 shard:
//   The 5D LUT shard (1024 rows × 2 cols × 2B = 4B page) fails the Wormhole
//   L1 alignment requirement (shard page ≥ 16B).  The 4D reshaped shard
//   (128 rows × 16 cols × 2B = 32B page) satisfies it.  The DRAM reshape
//   is a zero-copy logical reinterpretation (no DRAM bandwidth consumed).
//
// Per-core L1 usage after shard:
//   Shard = (128 rows × 16 cols × 2B) = 4,096 bytes = 4 KB/core  ← fits ✓
//
// Savings (Block B: 4 cameras, Block D: 1 camera):
//   Block B: TilizeWithValPadding(−1.53ms) + Reshape(−0.83ms) + Untilize(−0.30ms)
//            = ~2.66 ms (~43% speedup for Block B)
//   Block D: TilizeWithValPadding(−0.45ms) + Reshape partial(−0.25ms)
//            = ~0.70 ms (~36% speedup for Block D)

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

      // ── STEP 1: grid input must come from to_layout(ROW_MAJOR) ─────────────
      Value gridVal = gsOp.getGrid();
      auto *gridDef = gridVal.getDefiningOp();
      if (!gridDef) continue;

      auto rmToLayout = mlir::dyn_cast<ttnn::ToLayoutOp>(gridDef);
      if (!rmToLayout) continue;
      {
        auto lo = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
            mlir::cast<RankedTensorType>(rmToLayout.getResult().getType())
                .getEncoding());
        if (!lo || lo.getLayout() != Layout::RowMajor) continue;
      }

      // ── STEP 2: optionally skip to_memory_config(DRAM) (Block D path) ──────
      Value beforeRm = rmToLayout.getInput();
      ttnn::ToMemoryConfigOp memCfgOp = nullptr;
      if (auto *defOp = beforeRm.getDefiningOp()) {
        if (auto mco = mlir::dyn_cast<ttnn::ToMemoryConfigOp>(defOp)) {
          auto mcoCfg = mco.getMemoryConfig();
          if (mcoCfg &&
              mlir::cast<BufferTypeAttr>(mcoCfg.getBufferType()).getValue() ==
                  BufferType::DRAM) {
            memCfgOp = mco;
            beforeRm = mco.getInput();
          }
        }
      }

      // ── STEP 3: must be from a ReshapeOp with TILE layout ──────────────────
      auto tileReshape =
          mlir::dyn_cast_or_null<ttnn::ReshapeOp>(beforeRm.getDefiningOp());
      if (!tileReshape) continue;
      {
        auto lo = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
            mlir::cast<RankedTensorType>(tileReshape.getResult().getType())
                .getEncoding());
        if (!lo || lo.getLayout() != Layout::Tile) continue;
      }

      // ── STEP 4: Reshape input must come from to_layout(TILE) ───────────────
      auto tileToLayout = mlir::dyn_cast_or_null<ttnn::ToLayoutOp>(
          tileReshape.getInput().getDefiningOp());
      if (!tileToLayout) continue;
      {
        auto lo = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
            mlir::cast<RankedTensorType>(tileToLayout.getResult().getType())
                .getEncoding());
        if (!lo || lo.getLayout() != Layout::Tile) continue;
      }

      // ── STEP 5: original LUT must be ROW_MAJOR DRAM/SystemMemory ───────────
      Value lutArg = tileToLayout.getInput();
      auto lutRtt  = mlir::cast<RankedTensorType>(lutArg.getType());
      auto lutLo   = mlir::dyn_cast_or_null<TTNNLayoutAttr>(lutRtt.getEncoding());
      if (!lutLo || lutLo.getLayout() != Layout::RowMajor) continue;
      if (lutLo.getBufferType() != BufferType::DRAM &&
          lutLo.getBufferType() != BufferType::SystemMemory)
        continue;

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
          tileReshape.getShapeAttr(), /*memory_config=*/nullptr);

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
          auto l1MemCfg = MemoryConfigAttr::get(l1Layout);

          auto toL1 = builder.create<ttnn::ToMemoryConfigOp>(
              gsOp.getLoc(), l1Type, newReshape.getResult(), l1MemCfg);

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
      eraseWithDeallocs(rmToLayout);
      if (memCfgOp)
        eraseWithDeallocs(memCfgOp);
      eraseWithDeallocs(tileReshape);
      eraseWithDeallocs(tileToLayout);

      ++optimized;
    }

    (void)l1Sharded;
  }
};


} // namespace mlir::tt::ttnn