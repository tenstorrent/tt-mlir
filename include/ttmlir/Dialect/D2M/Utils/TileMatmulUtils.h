// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_TILEMATMULUTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_TILEMATMULUTILS_H

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir {
namespace linalg {
class GenericOp;
} // namespace linalg
class Region;
} // namespace mlir

namespace mlir::tt::d2m {

class TileMatmulBlockOp;
class GenericOp;

namespace utils {

/// Check if a linalg.generic operation contains tile_matmul operations.
///
/// \param linalgOp The linalg.generic operation to check.
/// \return true if the operation contains any tile_matmul ops.
bool hasTileMatmulOp(linalg::GenericOp linalgOp);

/// Convert linalg.generic with tile_matmul to tile_matmul_block with DST
/// operations.
///
/// This utility handles the special case where tile_matmul operations are
/// represented as linalg.generic ops (when useTileMatmul=false). The
/// transformation proceeds as follows:
///
/// 1. Convert linalg.generic to temporary affine loops.
/// 2. Invoke the provided callback to insert DST acquire/release/copy
///    operations in the affine loop region.
/// 3. Delete the temporary affine loops.
/// 4. Emit a tile_matmul_block operation with the original operands.
///
/// This design allows different passes (e.g., InsertDstRegisterAccess,
/// InsertDstRegisterGC) to share the same linalg→affine→tile_matmul_block
/// transformation while providing their own DST allocation strategies via the
/// callback.
///
/// The callback signature is:
///   LogicalResult(RewriterBase &rewriter, Region &region, Operation
///   *outermostLoop)
///
/// Where:
/// - rewriter: The pattern rewriter for IR modifications.
/// - region: The region containing the affine loops (where DST ops should be
/// inserted).
/// - outermostLoop: The outermost affine loop (or nullptr if conversion
/// failed).
///
/// Example usage:
/// \code
///   auto result = convertTileMatmulLinalgToBlock(
///       rewriter, genericOp, linalgOp, region,
///       [&](RewriterBase &rewriter, Region &region, Operation *outerLoop) {
///         return success(insertMyDstAllocation(rewriter, region, outerLoop));
///       });
///   if (failed(result)) {
///     // Handle error
///   }
/// \endcode
///
/// \param rewriter The pattern rewriter for IR modifications.
/// \param genericOp The parent d2m.generic operation.
/// \param linalgOp The linalg.generic operation containing tile_matmul.
/// \param region The region containing the linalg operation.
/// \param dstInsertionCallback Callback to insert DST operations in the affine
/// region.
///
/// \return success() and the created tile_matmul_block on success,
/// failure() otherwise.
FailureOr<TileMatmulBlockOp> convertTileMatmulLinalgToBlock(
    RewriterBase &rewriter, GenericOp genericOp, linalg::GenericOp linalgOp,
    Region &region,
    llvm::function_ref<LogicalResult(RewriterBase &, Region &, Operation *)>
        dstInsertionCallback);

/// Process all linalg.generic operations containing tile_matmul in a region.
///
/// This higher-level utility walks the region and processes all linalg.generic
/// operations that contain tile_matmul ops. For each such operation, it calls
/// convertTileMatmulLinalgToBlock with the provided DST insertion callback.
///
/// This function handles the common pattern used by DST allocation passes:
/// - Walk the region to find linalg.generic ops with tile_matmul
/// - Skip explicit datamovement forms (handled by LinalgToAffine)
/// - Convert remaining ops using the provided DST allocation callback
/// - Report errors for unexpected unconverted linalg ops
///
/// \param rewriter The pattern rewriter for IR modifications.
/// \param genericOp The parent d2m.generic operation.
/// \param region The region to process.
/// \param dstInsertionCallback Callback to insert DST operations in the affine
/// region.
///
/// \return success() if all operations were processed successfully,
/// failure() otherwise.
LogicalResult processTileMatmulLinalgOps(
    RewriterBase &rewriter, GenericOp genericOp, Region &region,
    llvm::function_ref<LogicalResult(RewriterBase &, Region &, Operation *)>
        dstInsertionCallback);

/// Insert DST allocation for tile_matmul operations in an affine loop region.
///
/// This shared utility implements DST allocation for matmul operations,
/// usable by both InsertDstRegisterAccess (legacy) and InsertDstRegisterGC
/// (graph coloring) passes. It handles:
///
/// 1. Identifying DST accesses from the affine loads/stores in the region
/// 2. Creating acquire_dst with appropriate shape
/// 3. Generating prologue loops to copy input data to DST
/// 4. Generating epilogue loops to pack output data from DST to CB
/// 5. Creating release_dst at the end of the region
///
/// This function is designed to be called from the dstInsertionCallback of
/// processTileMatmulLinalgOps, operating on the temporary affine loop region
/// before the loops are deleted and replaced with TileMatmulBlockOp.
///
/// \param rewriter The pattern rewriter for IR modifications.
/// \param genericOp The parent d2m.generic operation.
/// \param region The region containing the affine loops.
/// \param outermostLoop The outermost affine loop (for placement context).
/// \param totalDstTiles Total number of DST tiles available.
///
/// \return success() if DST allocation was inserted successfully,
/// failure() otherwise.
LogicalResult insertMatmulDstAllocation(RewriterBase &rewriter,
                                        GenericOp genericOp, Region &region,
                                        Operation *outermostLoop,
                                        unsigned totalDstTiles);

} // namespace utils
} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_UTILS_TILEMATMULUTILS_H
