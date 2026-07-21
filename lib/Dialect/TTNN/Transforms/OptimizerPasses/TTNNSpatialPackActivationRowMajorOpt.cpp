// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TTNNSpatialPackActivationRowMajorOpt
//
// Handles two spatial-packing patterns produced by TTIRSpatialRowGroupPackingOpt:
//
// ── NHWC path (process) ───────────────────────────────────────────────────────
// Problem: TTNNDecomposeLayouts propagates TILE backward through the spatial
// activation packing chain (to_layout(TILE) → reshape → permute → reshape)
// because ttnn.linear needs TILE input.  This tilizes the activation BEFORE
// the reshapes, which is wrong.  Additionally the optimizer inserts L1
// intermediates around linear.
//
//   BEFORE (wrong):
//     to_layout(%arg0, TILE)           DRAM TILE    (188 MB with C=3→32)
//     reshape([N,C*K,H/K,W])           DRAM TILE    (TILE copy, 28 MB)
//     permute({0,2,3,1})               DRAM TILE    (17.7 MB write)
//     reshape([N,1,H/K*W,C*K])         DRAM TILE
//     to_memory_config(L1_sharded)     ← L1 bounce
//     to_memory_config(DRAM)
//     linear(...)                      L1_sharded output
//     reshape([N,H/K,W,OC*K])          L1_sharded
//     to_memory_config(L1_interleaved) ← L1 intermediate
//     permute({0,3,1,2})               L1_interleaved
//     reshape([N,OC,H,W])              L1_interleaved
//     to_memory_config(DRAM_TILE)      final output
//
//   AFTER (correct — all DRAM, ROW_MAJOR for reshapes/permutes):
//     reshape([N,C*K,H/K,W])           DRAM ROW_MAJOR  ← free view
//     permute({0,2,3,1})               DRAM ROW_MAJOR  (17.7 MB only)
//     reshape([N,1,H/K*W,C*K])         DRAM ROW_MAJOR  ← free view
//     to_layout(TILE)                  DRAM TILE       (tilize 17.7 MB)
//     to_memory_config(DRAM)           DRAM TILE       ← no L1 bounce
//     linear(...)                      L1_sharded      (matmul config unchanged)
//     reshape([N,H/K,W,OC*K])          L1_sharded      (linear output, keep)
//     to_memory_config(DRAM_ROW_MAJOR) DRAM ROW_MAJOR  ← L1→DRAM + untilize
//     permute({0,3,1,2})               DRAM ROW_MAJOR
//     reshape([N,OC,H,W])              DRAM ROW_MAJOR  ← free view
//     to_memory_config(DRAM_TILE)      DRAM TILE       final output
//
// ── NCHW path (processNCHW) ───────────────────────────────────────────────────
// TTIRSpatialRowGroupPackingOpt now emits NCHW-native linear (no permutes):
//   reshape [N,C,H,W] → [N,1,IC*K,P]  (single reshape, no permute)
//   linear(W^T[1,1,OC*K,IC*K], act[N,1,IC*K,P]) → [N,1,OC*K,P]
//   reshape [N,1,OC*K,P] → [N,OC,H,W]  (no permute)
//
// TTNNDecomposeLayouts still tilizes before the NCHW reshape (wrong), and the
// output reshape in TILE crosses tile boundaries (~3.7ms).  This pass fixes both.
//
//   BEFORE (wrong):
//     to_layout(%arg0, TILE)           DRAM TILE
//     reshape([N,C,H,W]→[N,1,IC*K,P]) DRAM TILE  (crosses tile boundaries!)
//     ...
//     linear(W^T, act)                 [N,1,OC*K,P]
//     reshape([N,1,OC*K,P]→[N,OC,H,W]) TILE      (catastrophic ~3.7ms!)
//
//   AFTER (correct):
//     reshape([N,C,H,W]→[N,1,IC*K,P]) DRAM ROW_MAJOR  ← free view
//     to_layout(TILE)                  DRAM TILE       (tilize 17.7 MB)
//     ...
//     linear(W^T, act)                 [N,1,OC*K,P]
//     to_layout(ROW_MAJOR)             DRAM ROW_MAJOR  ← untilize (new op)
//     reshape([N,1,OC*K,P]→[N,OC,H,W]) DRAM ROW_MAJOR ← free view
//     to_memory_config(DRAM_TILE)      DRAM TILE       final output

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/Builders.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNSPATIALPACKACTIVATIONROWMAJOROPT
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Build a DRAM ROW_MAJOR interleaved type from shape/element.
// Uses the (ctx, shape, elemType) Builder form to build from scratch so that
// any L1/sharded grid attributes from the source type are NOT carried over.
// Starting from an existing L1 8×8-grid type would produce an invalid
// DRAM-interleaved layout with a non-1×1 grid.
static RankedTensorType mkDRAMRowMajorTy(RankedTensorType ref) {
  MLIRContext *ctx = ref.getContext();
  auto lo = TTNNLayoutAttr::Builder(ctx, ref.getShape(), ref.getElementType())
                .setBufferType(BufferType::DRAM)
                .setLayout(Layout::RowMajor)
                .setMemoryLayout(TensorMemoryLayout::Interleaved)
                .build();
  return RankedTensorType::get(ref.getShape(), ref.getElementType(), lo);
}

// Build a DRAM TILE interleaved type from shape/element (from scratch).
static RankedTensorType mkDRAMTileTy(RankedTensorType ref) {
  MLIRContext *ctx = ref.getContext();
  auto lo = TTNNLayoutAttr::Builder(ctx, ref.getShape(), ref.getElementType())
                .setBufferType(BufferType::DRAM)
                .setLayout(Layout::Tile)
                .setMemoryLayout(TensorMemoryLayout::Interleaved)
                .build();
  return RankedTensorType::get(ref.getShape(), ref.getElementType(), lo);
}


// Update a ToMemoryConfigOp to use a new result type (and matching attribute).
static void updateMemoryConfig(ttnn::ToMemoryConfigOp mcOp,
                                RankedTensorType newTy) {
  auto newLo = mlir::cast<TTNNLayoutAttr>(newTy.getEncoding());
  mcOp.getResult().setType(newTy);
  mcOp->setAttr("memory_config", MemoryConfigAttr::get(newLo));
}

class TTNNSpatialPackActivationRowMajorOptPass
    : public impl::TTNNSpatialPackActivationRowMajorOptBase<
          TTNNSpatialPackActivationRowMajorOptPass> {
public:
  using impl::TTNNSpatialPackActivationRowMajorOptBase<
      TTNNSpatialPackActivationRowMajorOptPass>::
      TTNNSpatialPackActivationRowMajorOptBase;

  void runOnOperation() final {
    ModuleOp mod = getOperation();
    SmallVector<ttnn::ToLayoutOp, 8> candidates;
    mod.walk([&](ttnn::ToLayoutOp op) {
      // Only consider ops that tilize (ROW_MAJOR → TILE).
      auto resultLo = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
          mlir::cast<RankedTensorType>(op.getResult().getType()).getEncoding());
      auto inputLo = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
          mlir::cast<RankedTensorType>(op.getInput().getType()).getEncoding());
      if (!resultLo || !inputLo)
        return;
      if (resultLo.getLayout() != Layout::Tile)
        return;
      if (inputLo.getLayout() != Layout::RowMajor)
        return;
      candidates.push_back(op);
    });

    for (ttnn::ToLayoutOp toTileOp : candidates) {
      process(toTileOp);
      processNCHW(toTileOp);
    }
  }

private:
  void process(ttnn::ToLayoutOp toTileOp) {
    // ── Match: to_layout(TILE) → reshape → permute → reshape ─────────────────
    ttnn::ReshapeOp r0;
    for (Operation *u : toTileOp.getResult().getUsers()) {
      if (mlir::isa<ttnn::DeallocateOp>(u))
        continue;
      r0 = mlir::dyn_cast<ttnn::ReshapeOp>(u);
      if (!r0)
        return;
    }
    if (!r0)
      return;

    ttnn::PermuteOp perm;
    for (Operation *u : r0.getResult().getUsers()) {
      if (mlir::isa<ttnn::DeallocateOp>(u))
        continue;
      perm = mlir::dyn_cast<ttnn::PermuteOp>(u);
      if (!perm)
        return;
    }
    if (!perm)
      return;

    ttnn::ReshapeOp r1;
    for (Operation *u : perm.getResult().getUsers()) {
      if (mlir::isa<ttnn::DeallocateOp>(u))
        continue;
      r1 = mlir::dyn_cast<ttnn::ReshapeOp>(u);
      if (!r1)
        return;
    }
    if (!r1)
      return;

    // Verify r0 is the channel-expansion reshape: [N,C,H,W] → [N,C*K,H/K,W]
    auto r0InTy  = mlir::cast<RankedTensorType>(r0.getInput().getType());
    auto r0OutTy = mlir::cast<RankedTensorType>(r0.getResult().getType());
    if (r0InTy.getRank() != 4 || r0OutTy.getRank() != 4)
      return;
    if (r0OutTy.getDimSize(1) <= r0InTy.getDimSize(1))
      return;

    // Verify permute is NCHW→NHWC: {0,2,3,1}
    auto permVec = perm.getPermutation();
    if (permVec.size() != 4 ||
        permVec[0] != 0 || permVec[1] != 2 ||
        permVec[2] != 3 || permVec[3] != 1)
      return;

    // Verify r1 flattens spatial: [N,H,W,C*K] → [N,1,H*W,C*K]
    auto r1OutTy = mlir::cast<RankedTensorType>(r1.getResult().getType());
    if (r1OutTy.getRank() != 4 || r1OutTy.getDimSize(1) != 1)
      return;

    // ── Transform 1: move to_layout(TILE) to AFTER the reshape chain ─────────
    Value rowMajorInput = toTileOp.getInput();

    // Build DRAM ROW_MAJOR from scratch — discards any L1/sharded grid.
    auto mkRMTy = [](RankedTensorType ref) -> RankedTensorType {
      return mkDRAMRowMajorTy(ref);
    };

    auto r0OutRM   = mkRMTy(r0OutTy);
    auto permOutRM = mkRMTy(mlir::cast<RankedTensorType>(perm.getResult().getType()));
    auto r1OutRM   = mkRMTy(r1OutTy);

    r0.getInputMutable().assign(rowMajorInput);
    r0.getResult().setType(r0OutRM);
    perm.getResult().setType(permOutRM);
    r1.getResult().setType(r1OutRM);

    toTileOp->moveAfter(r1);
    toTileOp.getInputMutable().assign(r1.getResult());

    auto origTileTy = mlir::cast<RankedTensorType>(toTileOp.getResult().getType());
    auto tileLo = TTNNLayoutAttr::Builder(origTileTy)
                      .setBufferType(BufferType::DRAM)
                      .setLayout(Layout::Tile)
                      .setMemoryLayout(TensorMemoryLayout::Interleaved)
                      .build();
    auto newTileTy = utils::RankedTensorTypeFactory::create(r1OutRM, tileLo);
    toTileOp.getResult().setType(newTileTy);

    SmallPtrSet<Operation *, 2> exceptions{toTileOp};
    r1.getResult().replaceAllUsesExcept(toTileOp.getResult(), exceptions);

    // ── Transform 2: remove L1 bounce before linear ───────────────────────────
    // The optimizer inserts: to_layout(TILE,DRAM) → to_memory_config(L1_sharded)
    //                         → to_memory_config(DRAM) → linear
    // Change to_memory_config(L1) → DRAM TILE so nothing goes to L1.
    ttnn::LinearOp linearOp;

    for (Operation *usr : llvm::make_early_inc_range(toTileOp.getResult().getUsers())) {
      if (mlir::isa<ttnn::DeallocateOp>(usr))
        continue;

      // Direct linear user (no L1 bounce in this path)
      if (auto lin = mlir::dyn_cast<ttnn::LinearOp>(usr)) {
        linearOp = lin;
        continue;
      }

      auto mcOp = mlir::dyn_cast<ttnn::ToMemoryConfigOp>(usr);
      if (!mcOp)
        continue;

      auto lo = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
          mlir::cast<RankedTensorType>(mcOp.getResult().getType()).getEncoding());
      if (!lo)
        continue;

      if (lo.getBufferType() == BufferType::L1) {
        // Change L1 to_memory_config → DRAM TILE interleaved
        auto newTy = mkDRAMTileTy(
            mlir::cast<RankedTensorType>(mcOp.getResult().getType()));
        updateMemoryConfig(mcOp, newTy);
      }

      // Trace to the downstream to_memory_config(DRAM) → linear
      for (Operation *u2 : mcOp.getResult().getUsers()) {
        if (mlir::isa<ttnn::DeallocateOp>(u2))
          continue;
        if (auto lin = mlir::dyn_cast<ttnn::LinearOp>(u2)) {
          linearOp = lin;
          continue;
        }
        if (auto mc2 = mlir::dyn_cast<ttnn::ToMemoryConfigOp>(u2)) {
          for (Operation *u3 : mc2.getResult().getUsers()) {
            if (mlir::isa<ttnn::DeallocateOp>(u3))
              continue;
            if (auto lin = mlir::dyn_cast<ttnn::LinearOp>(u3))
              linearOp = lin;
          }
        }
      }
    }

    if (!linearOp)
      return;

    // ── Transform 3: fix output unpack to DRAM ROW_MAJOR ─────────────────────
    // Pattern after linear:
    //   linear → reshape(r_out) → to_memory_config(L1_interleaved, mc_l1)
    //         → permute(perm_out, {0,3,1,2}) → reshape(r_final)
    //         → to_memory_config(DRAM_TILE, mc_dram_final)
    //
    // Transform:
    //   Change mc_l1            → DRAM ROW_MAJOR
    //   Change perm_out result  → DRAM ROW_MAJOR
    //   Change r_final result   → DRAM ROW_MAJOR
    //   (mc_dram_final stays — it tilizes the ROW_MAJOR result for downstream)
    //
    // Note on ReshapeViewDeviceOperation: r_out reshapes L1_sharded TILE data.
    // While this appears as a device op (0.7ms), reordering mc_l1 before r_out
    // to make it a ROW_MAJOR free view is counterproductive: perm_out then
    // permutes within DRAM (non-contiguous output) making r_final a larger
    // device op (~0.9ms). The original order keeps r_final as a free view
    // because perm_out writes L1→DRAM contiguously. Keep this order.

    // Find r_out (reshape immediately after linear)
    ttnn::ReshapeOp r_out;
    for (Operation *u : linearOp.getResult().getUsers()) {
      if (mlir::isa<ttnn::DeallocateOp>(u))
        continue;
      r_out = mlir::dyn_cast<ttnn::ReshapeOp>(u);
      break;
    }
    if (!r_out)
      return;

    // Find mc_l1 (to_memory_config after r_out — should be L1 interleaved)
    ttnn::ToMemoryConfigOp mc_l1;
    for (Operation *u : r_out.getResult().getUsers()) {
      if (mlir::isa<ttnn::DeallocateOp>(u))
        continue;
      mc_l1 = mlir::dyn_cast<ttnn::ToMemoryConfigOp>(u);
      break;
    }
    if (!mc_l1)
      return;
    {
      auto lo = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
          mlir::cast<RankedTensorType>(mc_l1.getResult().getType()).getEncoding());
      if (!lo || lo.getBufferType() != BufferType::L1)
        return;
    }

    // Find perm_out (permute after mc_l1)
    ttnn::PermuteOp perm_out;
    for (Operation *u : mc_l1.getResult().getUsers()) {
      if (mlir::isa<ttnn::DeallocateOp>(u))
        continue;
      perm_out = mlir::dyn_cast<ttnn::PermuteOp>(u);
      break;
    }
    if (!perm_out)
      return;
    // Verify NHWC→NCHW: {0,3,1,2}
    {
      auto pv = perm_out.getPermutation();
      if (pv.size() != 4 || pv[0] != 0 || pv[1] != 3 || pv[2] != 1 || pv[3] != 2)
        return;
    }

    // Find r_final (reshape after perm_out)
    ttnn::ReshapeOp r_final;
    for (Operation *u : perm_out.getResult().getUsers()) {
      if (mlir::isa<ttnn::DeallocateOp>(u))
        continue;
      r_final = mlir::dyn_cast<ttnn::ReshapeOp>(u);
      break;
    }
    if (!r_final)
      return;

    // Apply: change mc_l1 → DRAM ROW_MAJOR, update perm_out and r_final types
    auto mc_l1_newTy = mkDRAMRowMajorTy(
        mlir::cast<RankedTensorType>(mc_l1.getResult().getType()));
    updateMemoryConfig(mc_l1, mc_l1_newTy);

    perm_out.getResult().setType(
        mkDRAMRowMajorTy(mlir::cast<RankedTensorType>(perm_out.getResult().getType())));

    r_final.getResult().setType(
        mkDRAMRowMajorTy(mlir::cast<RankedTensorType>(r_final.getResult().getType())));

    // Ensure downstream consumers of r_final see DRAM TILE, not DRAM ROW_MAJOR.
    //
    // In block_C the compiler inserts a to_memory_config(DRAM TILE) after
    // r_final (mc_dram_final) which converts ROW_MAJOR → TILE before the UV
    // depthwise conv. In the full BEV model (multiple cameras, larger shapes)
    // this op may be absent. Without it, conv2d_DRAM internally allocates an
    // L1 tilize buffer at L1_UNRESERVED_BASE (0x98000 = 622592), which clashes
    // with the DramHeight kernel's own static CBs → TT_THROW at runtime.
    //
    // If mc_dram_final is already present, leave it unchanged (it already
    // converts to DRAM TILE). Otherwise, insert a new to_memory_config(DRAM
    // TILE) so all downstream users of r_final get TILE format.
    bool hasDramTileFinal = false;
    for (Operation *u : r_final.getResult().getUsers()) {
      if (mlir::isa<ttnn::DeallocateOp>(u))
        continue;
      auto mc = mlir::dyn_cast<ttnn::ToMemoryConfigOp>(u);
      if (!mc)
        continue;
      auto lo = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
          mlir::cast<RankedTensorType>(mc.getResult().getType()).getEncoding());
      if (lo && lo.getLayout() == Layout::Tile &&
          lo.getBufferType() == BufferType::DRAM) {
        hasDramTileFinal = true;
        break;
      }
    }

    if (!hasDramTileFinal) {
      // Insert to_memory_config(DRAM TILE) after r_final.
      auto rFinalTy = mlir::cast<RankedTensorType>(r_final.getResult().getType());
      auto tileTy   = mkDRAMTileTy(rFinalTy);
      auto tileLo   = mlir::cast<TTNNLayoutAttr>(tileTy.getEncoding());
      auto tileMemCfg = MemoryConfigAttr::get(tileLo);

      OpBuilder b(r_final->getContext());
      b.setInsertionPointAfter(r_final);
      auto mcTile = b.create<ttnn::ToMemoryConfigOp>(
          r_final.getLoc(), tileTy, r_final.getResult(), tileMemCfg);

      // Redirect all uses of r_final (except the new mc op itself) to mcTile.
      SmallPtrSet<Operation *, 1> except{mcTile.getOperation()};
      r_final.getResult().replaceAllUsesExcept(mcTile.getResult(), except);
    }
  }

  // ── NCHW-native path ───────────────────────────────────────────────────────
  // Handles the pattern emitted by TTIRSpatialRowGroupPackingOpt NCHW variant:
  //   to_layout(TILE) → reshape([N,C,H,W]→[N,1,IC*K,P])  (no permute)
  //   linear(W^T, act) → [N,1,OC*K,P]
  //   reshape([N,1,OC*K,P]→[N,OC,H,W])
  //
  // Discriminant vs. NHWC process(): the immediate reshape after to_layout has
  // dim[1]==1 (NCHW-flat [N,1,IC*K,P]).  The NHWC path has dim[1]==C*K>1.
  void processNCHW(ttnn::ToLayoutOp toTileOp) {
    // ── Match: to_layout(TILE) → rFlat([N,C,H,W]→[N,1,IC*K,P]) ─────────────
    ttnn::ReshapeOp rFlat;
    for (Operation *u : toTileOp.getResult().getUsers()) {
      if (mlir::isa<ttnn::DeallocateOp>(u))
        continue;
      rFlat = mlir::dyn_cast<ttnn::ReshapeOp>(u);
      if (!rFlat)
        return;
    }
    if (!rFlat)
      return;

    // Guard: output dim[1] == 1 (NCHW-flat: [N,1,IC*K,P])
    auto rFlatOutTy = mlir::cast<RankedTensorType>(rFlat.getResult().getType());
    if (rFlatOutTy.getRank() != 4 || rFlatOutTy.getDimSize(1) != 1)
      return;

    // Guard: input dim[1] != 1 (original has channels, not already NCHW-flat)
    auto rFlatInTy = mlir::cast<RankedTensorType>(rFlat.getInput().getType());
    if (rFlatInTy.getRank() != 4 || rFlatInTy.getDimSize(1) == 1)
      return;

    // ── Transform 1: move to_layout(TILE) to AFTER rFlat ─────────────────────
    // Before: to_layout(TILE, [N,C,H,W]) → rFlat (TILE reshape — crosses tiles)
    // After:  rFlat (ROW_MAJOR free view) → to_layout(TILE, [N,1,IC*K,P])
    Value rowMajorInput = toTileOp.getInput();

    auto rFlatOutRM = mkDRAMRowMajorTy(rFlatOutTy);
    rFlat.getInputMutable().assign(rowMajorInput);
    rFlat.getResult().setType(rFlatOutRM);

    toTileOp->moveAfter(rFlat);
    toTileOp.getInputMutable().assign(rFlat.getResult());

    auto origTileTy = mlir::cast<RankedTensorType>(toTileOp.getResult().getType());
    auto tileLo = TTNNLayoutAttr::Builder(origTileTy)
                      .setBufferType(BufferType::L1)
                      .setLayout(Layout::Tile)
                      .setMemoryLayout(TensorMemoryLayout::Interleaved)
                      .build();
    auto newTileTy = utils::RankedTensorTypeFactory::create(rFlatOutRM, tileLo);
    toTileOp.getResult().setType(newTileTy);

    SmallPtrSet<Operation *, 2> exceptions{toTileOp};
    rFlat.getResult().replaceAllUsesExcept(toTileOp.getResult(), exceptions);

    // ── Transform 2: remove L1 bounce + fuse matmul+add → linear ──────────────
    // The NCHW path lowers ttir::LinearOp(W^T, act, bias) to ttnn::MatmulOp +
    // ttnn::AddOp because the lowering only produces a fused ttnn::LinearOp when
    // the constant weight is the SECOND operand.  In our NCHW arrangement the
    // weight is first, so we detect the matmul+add pattern here and fuse it.
    ttnn::LinearOp linearOp;
    ttnn::MatmulOp matmulOp;

    for (Operation *usr :
         llvm::make_early_inc_range(toTileOp.getResult().getUsers())) {
      if (mlir::isa<ttnn::DeallocateOp>(usr))
        continue;

      if (auto lin = mlir::dyn_cast<ttnn::LinearOp>(usr)) {
        linearOp = lin;
        continue;
      }

      if (auto mm = mlir::dyn_cast<ttnn::MatmulOp>(usr)) {
        matmulOp = mm;
        continue;
      }

      auto mcOp = mlir::dyn_cast<ttnn::ToMemoryConfigOp>(usr);
      if (!mcOp)
        continue;

      auto lo = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
          mlir::cast<RankedTensorType>(mcOp.getResult().getType())
              .getEncoding());
      if (!lo)
        continue;

      // Do NOT convert L1 mc → DRAM: activation stays in L1 for the
      // L1-resident matmul path (tilize to L1, linear reads L1).

      for (Operation *u2 : mcOp.getResult().getUsers()) {
        if (mlir::isa<ttnn::DeallocateOp>(u2))
          continue;
        if (auto lin = mlir::dyn_cast<ttnn::LinearOp>(u2)) {
          linearOp = lin;
          continue;
        }
        if (auto mm = mlir::dyn_cast<ttnn::MatmulOp>(u2)) {
          matmulOp = mm;
          continue;
        }
        if (auto mc2 = mlir::dyn_cast<ttnn::ToMemoryConfigOp>(u2)) {
          for (Operation *u3 : mc2.getResult().getUsers()) {
            if (mlir::isa<ttnn::DeallocateOp>(u3))
              continue;
            if (auto lin = mlir::dyn_cast<ttnn::LinearOp>(u3))
              linearOp = lin;
            if (auto mm = mlir::dyn_cast<ttnn::MatmulOp>(u3))
              matmulOp = mm;
          }
        }
      }
    }

    // Convert MatmulOp + downstream AddOp(bias) → fused LinearOp(with bias).
    // This eliminates the separate ttnn.add, producing reshape→linear→reshape
    // with no extra ops in the TTNN IR.
    if (matmulOp && !linearOp) {
      // Find the bias AddOp directly after matmulOp before creating LinearOp.
      ttnn::AddOp biasAddOp;
      Value biasValue;
      for (Operation *u : matmulOp.getResult().getUsers()) {
        if (mlir::isa<ttnn::DeallocateOp>(u))
          continue;
        if (auto add = mlir::dyn_cast<ttnn::AddOp>(u)) {
          Value lhs = add.getLhs(), rhs = add.getRhs();
          biasValue = (lhs == matmulOp.getResult()) ? rhs : lhs;
          biasAddOp = add;
          break;
        }
      }

      auto linearResultTy =
          mlir::cast<RankedTensorType>(matmulOp.getResult().getType());
      OpBuilder ob(matmulOp);
      linearOp = ob.create<ttnn::LinearOp>(
          matmulOp.getLoc(), linearResultTy,
          matmulOp.getA(), matmulOp.getB(),
          biasValue,
          matmulOp.getTransposeA(), matmulOp.getTransposeB(),
          matmulOp.getMatmulProgramConfigAttr(),
          matmulOp.getActivationAttr(),
          matmulOp.getComputeConfigAttr());

      matmulOp.getResult().replaceAllUsesWith(linearOp.getResult());
      matmulOp.erase();

      // Remove the separate AddOp — bias is now fused into LinearOp.
      if (biasAddOp) {
        biasAddOp.getResult().replaceAllUsesWith(linearOp.getResult());
        biasAddOp.erase();
      }

      llvm::errs() << "[SpatialPackOpt] processNCHW: MatmulOp → LinearOp"
                   << (biasAddOp ? " (bias fused)" : " (no bias)")
                   << " act=" << linearOp.getB().getType()
                   << " w="   << linearOp.getA().getType() << "\n";
    }

    if (!linearOp)
      return;

    // ── Transform 3: insert untilize BEFORE output reshape ───────────────────
    // The output reshape [N,1,OC*K,P]→[N,OC,H,W] on TILE data is catastrophic
    // (~2.3 ms) because it crosses tile boundaries. Inserting to_layout(RM, L1)
    // before the reshape converts it to a cheap ROW_MAJOR re-stride (~180 µs).
    //
    // Pattern:  linear → [optional mc (ShardedToInterleaved)] → rOut(TILE reshape)
    //           → [optional mc_final(DRAM TILE)]
    // After:    linear → [mc] → toRM(RM, L1) → rOut(RM reshape→DRAM RM)
    //           → [mc_final(DRAM TILE)]  (inserted if absent)

    ttnn::ReshapeOp rOut;
    Value rOutSrc;

    for (Operation *u : linearOp.getResult().getUsers()) {
      if (mlir::isa<ttnn::DeallocateOp>(u))
        continue;
      if (auto r = mlir::dyn_cast<ttnn::ReshapeOp>(u)) {
        rOut    = r;
        rOutSrc = linearOp.getResult();
        break;
      }
      if (auto mc = mlir::dyn_cast<ttnn::ToMemoryConfigOp>(u)) {
        for (Operation *u2 : mc.getResult().getUsers()) {
          if (mlir::isa<ttnn::DeallocateOp>(u2))
            continue;
          if (auto r = mlir::dyn_cast<ttnn::ReshapeOp>(u2)) {
            rOut    = r;
            rOutSrc = mc.getResult();
            break;
          }
        }
      }
      if (rOut)
        break;
    }
    if (!rOut)
      return;

    // If rOutSrc is still in L1 (e.g., ShardedToInterleaved chose L1 interleaved
    // rather than DRAM TILE), redirect its defining mc to DRAM TILE first.
    // This releases the L1 buffer before the rm_reshape_interleaved kernel
    // allocates its CBs, preventing the CB address clash seen on Block C.
    auto rOutSrcTy = mlir::cast<RankedTensorType>(rOutSrc.getType());
    auto rOutSrcLo = mlir::dyn_cast_or_null<TTNNLayoutAttr>(rOutSrcTy.getEncoding());
    if (rOutSrcLo && rOutSrcLo.getBufferType() == BufferType::L1) {
      if (auto mcOp =
              mlir::dyn_cast_or_null<ttnn::ToMemoryConfigOp>(rOutSrc.getDefiningOp())) {
        auto dramTileTy = mkDRAMTileTy(rOutSrcTy);
        updateMemoryConfig(mcOp, dramTileTy);
        rOutSrcTy = dramTileTy;
        llvm::errs() << "[SpatialPackOpt] processNCHW: redirected ShardedToInterleaved "
                        "mc from L1 → DRAM TILE to release L1 before reshape\n";
      }
    }

    // Insert to_layout(ROW_MAJOR, DRAM interleaved) BEFORE rOut.
    // rOutSrc is TILE (DRAM or L1); untilize to DRAM RM so the reshape reads
    // from DRAM — avoids the CB clash where rm_reshape_interleaved CBs overlap
    // with any still-live L1 buffer.
    auto dramRMUnTy = mkDRAMRowMajorTy(rOutSrcTy);
    auto layoutAttr = ttnn::LayoutAttr::get(rOut->getContext(), Layout::RowMajor);
    OpBuilder ob(rOut->getContext());
    ob.setInsertionPoint(rOut);
    auto toRM = ob.create<ttnn::ToLayoutOp>(rOut.getLoc(), dramRMUnTy, rOutSrc,
                                             layoutAttr,
                                             /*dtype=*/nullptr,
                                             /*memory_config=*/nullptr);

    // Redirect rOut to consume toRM's DRAM RM output; make rOut output DRAM RM
    // (cheap re-stride — no tile boundary crossing, no CB clash).
    rOut.getInputMutable().assign(toRM.getResult());
    auto rOutTy  = mlir::cast<RankedTensorType>(rOut.getResult().getType());
    auto dramRMTy = mkDRAMRowMajorTy(rOutTy);
    rOut.getResult().setType(dramRMTy);

    llvm::errs() << "[SpatialPackOpt] processNCHW: inserted untilize(DRAM RM)"
                    " before output reshape → DRAM ROW_MAJOR re-stride\n";

    // Ensure downstream consumers of rOut see DRAM TILE (same guard as NHWC).
    bool hasDramTileFinal = false;
    for (Operation *u : rOut.getResult().getUsers()) {
      if (mlir::isa<ttnn::DeallocateOp>(u))
        continue;
      auto mc = mlir::dyn_cast<ttnn::ToMemoryConfigOp>(u);
      if (!mc)
        continue;
      auto lo = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
          mlir::cast<RankedTensorType>(mc.getResult().getType()).getEncoding());
      if (lo && lo.getLayout() == Layout::Tile &&
          lo.getBufferType() == BufferType::DRAM) {
        hasDramTileFinal = true;
        break;
      }
    }

    if (!hasDramTileFinal) {
      auto tileTy    = mkDRAMTileTy(dramRMTy);
      auto tileLo    = mlir::cast<TTNNLayoutAttr>(tileTy.getEncoding());
      auto tileMemCfg = MemoryConfigAttr::get(tileLo);

      OpBuilder b(rOut->getContext());
      b.setInsertionPointAfter(rOut);
      auto mcTile = b.create<ttnn::ToMemoryConfigOp>(
          rOut.getLoc(), tileTy, rOut.getResult(), tileMemCfg);

      SmallPtrSet<Operation *, 1> except{mcTile.getOperation()};
      rOut.getResult().replaceAllUsesExcept(mcTile.getResult(), except);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttnn
