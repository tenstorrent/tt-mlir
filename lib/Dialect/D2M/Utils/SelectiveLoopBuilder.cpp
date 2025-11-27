// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/SelectiveLoopBuilder.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::tt::d2m::utils {

SelectiveLoopBuilder::SelectiveLoopBuilder(RewriterBase &rewriter,
                                           const LoopBuildConfig &config)
    : rewriter(rewriter), config(config) {}

LogicalResult SelectiveLoopBuilder::validate() {
  if (!config.sourceLoop) {
    return failure();
  }

  // Validate that dimension indices are reasonable
  // (We'll check actual loop depth during cloning)
  if (!config.dimensionsToInclude.empty()) {
    for (unsigned dim : config.dimensionsToInclude) {
      // Just a sanity check - actual validation happens during cloning
      if (dim > 100) { // Arbitrary upper bound
        return failure();
      }
    }
  }

  return success();
}

bool SelectiveLoopBuilder::shouldIncludeDimension(unsigned depth) const {
  // If no filter specified, include all dimensions
  if (config.dimensionsToInclude.empty()) {
    return true;
  }

  // Check if this depth is in the include list
  return llvm::is_contained(config.dimensionsToInclude, depth);
}

affine::AffineForOp SelectiveLoopBuilder::cloneLoopNestFiltered(
    affine::AffineForOp loop, unsigned currentDepth, IRMapping &mapper) {
  // Check if we should include this dimension
  if (!shouldIncludeDimension(currentDepth)) {
    // Skip this loop dimension - look for nested loops to continue filtering
    affine::AffineForOp nestedLoop = nullptr;
    for (Operation &op : loop.getRegion().front().getOperations()) {
      if (auto nested = dyn_cast<affine::AffineForOp>(&op)) {
        nestedLoop = nested;
        break;
      }
    }

    if (nestedLoop) {
      // Continue filtering in the nested loop
      return cloneLoopNestFiltered(nestedLoop, currentDepth + 1, mapper);
    } else {
      // No nested loop and this dimension is filtered - we're done
      return nullptr;
    }
  }

  // Clone this loop
  auto *clonedOp = rewriter.clone(*loop.getOperation(), mapper);
  auto clonedLoop = cast<affine::AffineForOp>(clonedOp);

  // Find nested loop in the original
  affine::AffineForOp originalNestedLoop = nullptr;
  for (Operation &op : loop.getRegion().front().getOperations()) {
    if (auto nested = dyn_cast<affine::AffineForOp>(&op)) {
      originalNestedLoop = nested;
      break;
    }
  }

  if (originalNestedLoop) {
    // Recursively process nested loops
    Block &clonedBody = clonedLoop.getRegion().front();

    // Remove the cloned nested loop (we'll replace it with filtered version)
    affine::AffineForOp clonedNestedLoop = nullptr;
    for (Operation &op : clonedBody.getOperations()) {
      if (auto nested = dyn_cast<affine::AffineForOp>(&op)) {
        clonedNestedLoop = nested;
        break;
      }
    }

    if (clonedNestedLoop) {
      rewriter.eraseOp(clonedNestedLoop);
    }

    // Clone filtered nested loop
    rewriter.setInsertionPointToStart(&clonedBody);
    auto filteredNested =
        cloneLoopNestFiltered(originalNestedLoop, currentDepth + 1, mapper);

    if (!filteredNested) {
      // Nested loop was completely filtered out - that's ok
      // The body of clonedLoop is now empty (except terminator)
    }
  }

  return clonedLoop;
}

void SelectiveLoopBuilder::clearLoopBody(affine::AffineForOp loop) {
  Block &body = loop.getRegion().front();

  // First, recursively clear nested loops
  SmallVector<affine::AffineForOp> nestedLoops;
  for (Operation &op : body.getOperations()) {
    if (auto nested = dyn_cast<affine::AffineForOp>(&op)) {
      nestedLoops.push_back(nested);
    }
  }

  for (auto nested : nestedLoops) {
    clearLoopBody(nested);
  }

  // Collect operations to erase
  SmallVector<Operation *> toErase;
  for (Operation &op : body.getOperations()) {
    bool shouldKeep = false;

    // Always keep nested loops and terminators
    if (isa<affine::AffineForOp>(&op) || op.hasTrait<OpTrait::IsTerminator>()) {
      shouldKeep = true;
    }
    // Optionally keep AffineApplyOps
    else if (config.preserveAffineApply && isa<affine::AffineApplyOp>(&op)) {
      shouldKeep = true;
    }

    if (!shouldKeep) {
      toErase.push_back(&op);
    }
  }

  // Drop uses before erasing to avoid use-def issues
  for (Operation *op : toErase) {
    op->dropAllUses();
  }

  // Erase operations
  for (Operation *op : toErase) {
    rewriter.eraseOp(op);
  }
}

Block *SelectiveLoopBuilder::findInnermostBlock(affine::AffineForOp loop) {
  affine::AffineForOp current = loop;

  while (current) {
    affine::AffineForOp nested = nullptr;
    Block &body = current.getRegion().front();

    for (Operation &op : body.getOperations()) {
      if (auto nestedLoop = dyn_cast<affine::AffineForOp>(&op)) {
        nested = nestedLoop;
        break;
      }
    }

    if (!nested) {
      // No nested loop - this is the innermost
      return &body;
    }

    current = nested;
  }

  return nullptr;
}

SmallVector<Value>
SelectiveLoopBuilder::collectInductionVars(affine::AffineForOp loop) {
  SmallVector<Value> ivs;
  affine::AffineForOp current = loop;

  while (current) {
    ivs.push_back(current.getInductionVar());

    affine::AffineForOp nested = nullptr;
    Block &body = current.getRegion().front();

    for (Operation &op : body.getOperations()) {
      if (auto nestedLoop = dyn_cast<affine::AffineForOp>(&op)) {
        nested = nestedLoop;
        break;
      }
    }

    current = nested;
  }

  return ivs;
}

FailureOr<LoopBuildResult> SelectiveLoopBuilder::build() {
  // Validate configuration
  if (failed(validate())) {
    return failure();
  }

  // Clone with filtering
  IRMapping mapper;
  auto clonedLoop = cloneLoopNestFiltered(config.sourceLoop, 0, mapper);

  if (!clonedLoop) {
    // All dimensions were filtered out
    return failure();
  }

  // Clear loop bodies if requested
  if (config.clearBodies) {
    clearLoopBody(clonedLoop);
  }

  // Build result
  LoopBuildResult result;
  result.outermostLoop = clonedLoop;
  result.irMapping = std::move(mapper);
  result.innermostBlock = findInnermostBlock(clonedLoop);
  result.inductionVars = collectInductionVars(clonedLoop);

  return result;
}

FailureOr<LoopBuildResult> SelectiveLoopBuilder::buildForOperandAccess(
    RewriterBase &rewriter, affine::AffineForOp sourceLoop,
    const OperandAccessInfo &accessInfo) {
  // Convert participating dims to unsigned
  SmallVector<unsigned> dims;
  dims.reserve(accessInfo.participatingDims.size());
  for (int64_t dim : accessInfo.participatingDims) {
    dims.push_back(static_cast<unsigned>(dim));
  }

  LoopBuildConfig config{.sourceLoop = sourceLoop,
                         .dimensionsToInclude = dims,
                         .clearBodies = true,
                         .preserveAffineApply = true};

  SelectiveLoopBuilder builder(rewriter, config);
  return builder.build();
}

FailureOr<LoopBuildResult>
SelectiveLoopBuilder::buildParallelOnly(RewriterBase &rewriter,
                                        affine::AffineForOp sourceLoop,
                                        const LoopDimensionInfo &dimInfo) {
  // Collect parallel dimensions in order
  SmallVector<unsigned> parallelDims;
  for (unsigned i = 0; i < dimInfo.numDimensions; ++i) {
    if (dimInfo.isParallel(i)) {
      parallelDims.push_back(i);
    }
  }

  LoopBuildConfig config{.sourceLoop = sourceLoop,
                         .dimensionsToInclude = parallelDims,
                         .clearBodies = true,
                         .preserveAffineApply = true};

  SelectiveLoopBuilder builder(rewriter, config);
  return builder.build();
}

} // namespace mlir::tt::d2m::utils
