// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/TileMatmulUtils.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::d2m::utils {

bool hasTileMatmulOp(linalg::GenericOp linalgOp) {
  return linalgOp
      ->walk([](d2m::TileMatmulOp) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

FailureOr<TileMatmulBlockOp> convertTileMatmulLinalgToBlock(
    RewriterBase &rewriter, GenericOp genericOp, linalg::GenericOp linalgOp,
    Region &region,
    llvm::function_ref<LogicalResult(RewriterBase &, Region &, Operation *)>
        dstInsertionCallback) {

  // Verify preconditions
  if (linalgOp.getInputs().size() != 2) {
    return linalgOp.emitError("Expected exactly 2 inputs for tile_matmul");
  }
  if (linalgOp.getOutputs().size() != 1) {
    return linalgOp.emitError("Expected exactly 1 output for tile_matmul");
  }

  // Extract operands before converting
  Value inputAMemref = linalgOp.getInputs()[0];
  Value inputBMemref = linalgOp.getInputs()[1];
  Value outputCMemref = linalgOp.getOutputs()[0];
  Location loc = linalgOp.getLoc();

  // Convert linalg.generic to affine loops
  rewriter.setInsertionPoint(linalgOp);
  auto linalgLoops = linalg::linalgOpToAffineLoops(rewriter, linalgOp);
  if (failed(linalgLoops)) {
    return failure();
  }

  // Erase the original linalg operation
  rewriter.eraseOp(linalgOp);

  // Get the outermost loop (or nullptr if no loops created)
  Operation *outermostLoop =
      !linalgLoops.value().empty() ? linalgLoops.value().front() : nullptr;

  // Invoke callback to insert DST operations in the affine loop region
  if (failed(dstInsertionCallback(rewriter, region, outermostLoop))) {
    return failure();
  }

  // Create the tile_matmul_block operation BEFORE deleting the loops
  // (while the operands are still accessible), but position it where
  // the loops currently are so it remains valid after loop deletion.
  TileMatmulBlockOp blockOp;
  if (!linalgLoops.value().empty()) {
    // Position before the outermost loop
    rewriter.setInsertionPoint(linalgLoops.value().front());
  } else {
    // No loops were created, insert at end of region
    rewriter.setInsertionPointToEnd(&region.front());
  }
  blockOp = rewriter.create<d2m::TileMatmulBlockOp>(
      loc, inputAMemref, inputBMemref, outputCMemref);

  // Delete the temporary affine loops (they were only needed for DST
  // insertion)
  if (!linalgLoops.value().empty()) {
    for (Operation *loopOp : llvm::reverse(linalgLoops.value())) {
      rewriter.eraseOp(loopOp);
    }
  }

  return blockOp;
}

LogicalResult processTileMatmulLinalgOps(
    RewriterBase &rewriter, GenericOp genericOp, Region &region,
    llvm::function_ref<LogicalResult(RewriterBase &, Region &, Operation *)>
        dstInsertionCallback) {
  // Process linalg.generic ops that contain tile_matmul operations.
  // These are left unconverted by LinalgToAffine when useTileMatmul=false.
  Block &block = region.front();
  WalkResult walkResult = block.walk([&](linalg::GenericOp linalgOp) {
    if (hasTileMatmulOp(linalgOp)) {
      // Only handle tile_matmul when not in explicit datamovement form.
      // Explicit datamovement form should be converted by LinalgToAffine pass.
      if (!genericOp.isExplicitDatamovementForm()) {
        // Convert linalg to tile_matmul_block with DST allocation callback.
        auto result = convertTileMatmulLinalgToBlock(
            rewriter, genericOp, linalgOp, region, dstInsertionCallback);
        if (failed(result)) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
    }

    // All other linalg ops should have been converted by LinalgToAffine pass.
    // If we encounter one in non-datamovement form, that's an error.
    if (!genericOp.isExplicitDatamovementForm()) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    return genericOp.emitError(
        "linalg.generic operations were not converted to affine loops");
  }
  return success();
}

} // namespace mlir::tt::d2m::utils
