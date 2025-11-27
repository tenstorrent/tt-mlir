// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_SELECTIVELOOPBUILDER_H
#define TTMLIR_DIALECT_D2M_UTILS_SELECTIVELOOPBUILDER_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Dialect/D2M/Utils/LoopSemantics.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::d2m::utils {

/// Configuration for selective loop building.
///
/// Specifies which dimensions to include when cloning a loop nest structure.
/// Empty dimensionsToInclude means include all dimensions.
struct LoopBuildConfig {
  /// The original loop nest to base the new loops on
  affine::AffineForOp sourceLoop;

  /// Which dimensions (by index) to include in the generated loop nest.
  /// Empty vector means include all dimensions.
  /// For example, {0, 1} means only include the outermost two loops.
  llvm::SmallVector<unsigned> dimensionsToInclude;

  /// Whether to clear loop bodies (remove non-loop operations).
  /// True for prologue/epilogue structure cloning where we only want
  /// the loop structure, not the computation.
  bool clearBodies = true;

  /// Whether to preserve AffineApplyOps in cleared bodies.
  /// Useful for maintaining index computations that might be needed.
  bool preserveAffineApply = true;
};

/// Result of selective loop building.
///
/// Contains the generated loop nest and mappings for integrating it
/// into the surrounding IR.
struct LoopBuildResult {
  /// The outermost loop of the generated nest
  affine::AffineForOp outermostLoop;

  /// Mapping from original values to cloned values.
  /// This includes induction variables and any other values cloned
  /// from the source loop nest.
  mlir::IRMapping irMapping;

  /// The innermost block where new operations should be inserted.
  /// This is the block of the deepest generated loop.
  mlir::Block *innermostBlock;

  /// Induction variables for the generated loops (in order from outer to inner)
  llvm::SmallVector<mlir::Value> inductionVars;
};

/// Builder for creating selective loop nests.
///
/// This class clones affine loop structures while optionally filtering
/// dimensions. It's used to create prologue/epilogue loops that only
/// iterate over the dimensions actually accessed by an operand.
///
/// Example usage:
/// \code
///   // Build a loop nest with only dimensions 0 and 1 (skip dimension 2)
///   LoopBuildConfig config{
///       .sourceLoop = originalMatmulLoopNest,
///       .dimensionsToInclude = {0, 1},  // Only i and j, not k
///       .clearBodies = true
///   };
///
///   SelectiveLoopBuilder builder(rewriter, config);
///   auto result = builder.build();
///
///   if (succeeded(result)) {
///     // result->outermostLoop is a 2D loop nest (i, j)
///     // result->innermostBlock is where to insert copy operations
///     // result->irMapping maps original induction vars to new ones
///   }
/// \endcode
class SelectiveLoopBuilder {
public:
  /// Construct builder with configuration
  SelectiveLoopBuilder(mlir::RewriterBase &rewriter,
                       const LoopBuildConfig &config);

  /// Build the filtered loop nest.
  ///
  /// Returns the generated loop structure or failure if the configuration
  /// is invalid or building fails.
  mlir::FailureOr<LoopBuildResult> build();

  /// Build a loop nest filtered to only dimensions accessed by an operand.
  ///
  /// Convenience factory method that uses OperandAccessInfo to determine
  /// which dimensions to include.
  static mlir::FailureOr<LoopBuildResult>
  buildForOperandAccess(mlir::RewriterBase &rewriter,
                        affine::AffineForOp sourceLoop,
                        const OperandAccessInfo &accessInfo);

  /// Build a loop nest with only parallel dimensions.
  ///
  /// Convenience factory method that filters to only parallel dimensions
  /// based on LoopDimensionInfo.
  static mlir::FailureOr<LoopBuildResult>
  buildParallelOnly(mlir::RewriterBase &rewriter,
                    affine::AffineForOp sourceLoop,
                    const LoopDimensionInfo &dimInfo);

private:
  mlir::RewriterBase &rewriter;
  LoopBuildConfig config;

  /// Validate configuration
  mlir::LogicalResult validate();

  /// Clone loop structure recursively, filtering dimensions.
  ///
  /// \param loop The current loop to potentially clone
  /// \param currentDepth The depth of this loop (0 = outermost)
  /// \param mapper The IRMapping to track original->cloned values
  /// \return The cloned loop, or nullptr if this dimension is filtered out
  affine::AffineForOp cloneLoopNestFiltered(affine::AffineForOp loop,
                                            unsigned currentDepth,
                                            mlir::IRMapping &mapper);

  /// Clear non-loop operations from cloned loop body.
  ///
  /// Removes all operations except loops, terminators, and optionally
  /// AffineApplyOps (if preserveAffineApply is true).
  void clearLoopBody(affine::AffineForOp loop);

  /// Find the innermost block in a loop nest
  mlir::Block *findInnermostBlock(affine::AffineForOp loop);

  /// Collect induction variables from a loop nest
  llvm::SmallVector<mlir::Value> collectInductionVars(affine::AffineForOp loop);

  /// Check if a dimension should be included based on config
  bool shouldIncludeDimension(unsigned depth) const;
};

} // namespace mlir::tt::d2m::utils

#endif // TTMLIR_DIALECT_D2M_UTILS_SELECTIVELOOPBUILDER_H
