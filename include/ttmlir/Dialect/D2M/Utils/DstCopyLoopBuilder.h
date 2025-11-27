// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_DSTCOPYLOOPBUILDER_H
#define TTMLIR_DIALECT_D2M_UTILS_DSTCOPYLOOPBUILDER_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Dialect/D2M/Utils/LoopSemantics.h"
#include "ttmlir/Dialect/D2M/Utils/SelectiveLoopBuilder.h"

namespace mlir::tt::d2m {
class GenericOp;
} // namespace mlir::tt::d2m

namespace mlir::tt::d2m::utils {

/// Configuration for DST copy loop generation.
///
/// Specifies the context needed to generate prologue (L1→DST) or
/// epilogue (DST→L1) loops.
struct DstCopyConfig {
  /// The GenericOp containing the operation
  GenericOp genericOp;

  /// The source loop nest to base the copy loops on
  affine::AffineForOp sourceLoop;

  /// DST buffer to copy to/from
  mlir::Value dstBuffer;

  /// DST slice index (for multi-slice DST buffers)
  int64_t dstSlice;

  /// Index of the output operand being copied
  /// (relative to outputs, not total operands)
  unsigned outputOperandIndex;
};

/// Result of DST copy loop generation.
///
/// Contains the generated loop nest, guard, and mappings.
struct DstCopyLoopResult {
  /// The generated loop nest
  affine::AffineForOp loopNest;

  /// The if-op guard (if any) protecting the copy.
  /// Null if no guard was needed.
  scf::IfOp guardOp;

  /// Mapping from original to cloned values
  mlir::IRMapping irMapping;
};

/// Builder for DST copy loops (prologue/epilogue).
///
/// This class generates filtered loop nests for copying data between L1
/// and DST registers. It uses LoopSemanticsAnalyzer to determine which
/// dimensions to include and SelectiveLoopBuilder to construct the loops.
///
/// Example usage:
/// \code
///   DstCopyConfig config{
///       .genericOp = genericOp,
///       .sourceLoop = matmulLoopNest,
///       .dstBuffer = dstMemRef,
///       .dstSlice = 0,
///       .outputOperandIndex = 0  // First (and only) output
///   };
///
///   DstCopyLoopBuilder builder(rewriter, config);
///
///   // Generate prologue (L1 → DST)
///   auto prologueResult = builder.generatePrologue(loc, srcLoadOp);
///   // Creates: for i { for j { if (k != 0) { load L1, store DST } } }
///
///   // Generate epilogue (DST → L1)
///   auto epilogueResult = builder.generateEpilogue(loc, dstStoreOp);
///   // Creates: for i { for j { if (k != 0) { load DST, store L1 } } }
/// \endcode
class DstCopyLoopBuilder {
public:
  /// Construct builder with configuration
  DstCopyLoopBuilder(mlir::RewriterBase &rewriter, const DstCopyConfig &config);

  /// Generate prologue loop (L1 → DST initialization).
  ///
  /// Creates a filtered loop nest (only dimensions used by the output)
  /// that loads from L1 and stores to DST. Guards are inserted for
  /// non-participating reduction dimensions.
  ///
  /// \param loc Location for generated operations
  /// \param srcLoad The original load operation to base the copy on
  /// \return The generated loop structure or failure
  mlir::FailureOr<DstCopyLoopResult>
  generatePrologue(mlir::Location loc, affine::AffineLoadOp srcLoad);

  /// Generate epilogue loop (DST → L1 writeback).
  ///
  /// Creates a filtered loop nest (only dimensions used by the output)
  /// that loads from DST and stores to L1. Guards are inserted for
  /// non-participating reduction dimensions.
  ///
  /// \param loc Location for generated operations
  /// \param dstStore The original store operation to base the copy on
  /// \return The generated loop structure or failure
  mlir::FailureOr<DstCopyLoopResult>
  generateEpilogue(mlir::Location loc, affine::AffineStoreOp dstStore);

private:
  mlir::RewriterBase &rewriter;
  DstCopyConfig config;
  LoopSemanticsAnalyzer semanticsAnalyzer;

  /// Create guard condition for non-participating reduction dimensions.
  ///
  /// Generates: if (dim0 != 0 || dim1 != 0 || ...) { ... }
  /// where dims are the non-participating dimensions.
  ///
  /// \param loc Location for generated operations
  /// \param guardDims Dimensions to guard on
  /// \return The created IfOp, or null if no guard needed
  scf::IfOp createGuard(mlir::Location loc,
                        const llvm::SmallVector<unsigned> &guardDims);

  /// Build affine map for DST access with slice index inserted.
  ///
  /// Takes the original map for the operand and inserts a constant
  /// expression for the DST slice at position 0.
  ///
  /// Example: (d0, d1) -> (d0, d1) becomes (d0, d1) -> (slice, d0, d1)
  ///
  /// \param originalMap The original affine map
  /// \return The modified map with slice dimension
  mlir::AffineMap buildDstAccessMap(mlir::AffineMap originalMap) const;

  /// Remap indices through IRMapping.
  ///
  /// Maps original induction variables to their cloned counterparts.
  ///
  /// \param originalIndices Original index values
  /// \param mapping The IRMapping from cloning
  /// \return Mapped index values
  llvm::SmallVector<mlir::Value>
  remapIndices(llvm::ArrayRef<mlir::Value> originalIndices,
               const mlir::IRMapping &mapping) const;
};

} // namespace mlir::tt::d2m::utils

#endif // TTMLIR_DIALECT_D2M_UTILS_DSTCOPYLOOPBUILDER_H
