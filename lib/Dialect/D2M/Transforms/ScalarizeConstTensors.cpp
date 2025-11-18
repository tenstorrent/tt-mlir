// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MSCALARIZECONSTTENSORS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

// Helper function to create a scalar constant from an attribute
static Value createScalarConstant(OpBuilder &builder, Location loc,
                                  Attribute attr) {
  if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    return builder.create<arith::ConstantOp>(loc, floatAttr);
  }
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    return builder.create<arith::ConstantOp>(loc, intAttr);
  }
  return nullptr;
}

namespace {

// Struct to track where a constant value flows through the IR
struct ConstantUseChain {
  Operation *constOp;       // The d2m.full op
  Attribute splatValue;     // The scalar value
  GenericOp genericOp;      // The d2m.generic that uses it
  unsigned genericInputIdx; // Which input of the generic
};

// Trace forward from a constant op through d2m.to_layout to d2m.generic ops
static void
traceConstantToGenericOps(Operation *constOp, Attribute splatValue,
                          SmallVectorImpl<ConstantUseChain> &chains) {

  // Walk through all uses of the constant
  for (Operation *user : constOp->getUsers()) {
    Value currentValue = constOp->getResult(0);
    Operation *currentOp = user;

    // Trace through d2m.to_layout ops
    while (auto toLayoutOp = dyn_cast<ToLayoutOp>(currentOp)) {
      currentValue = toLayoutOp.getResult(0);
      if (currentValue.getUsers().empty()) {
        break;
      }
      // Move to next user
      currentOp = *currentValue.getUsers().begin();
    }

    // Check if we reached a d2m.generic op
    if (auto genericOp = dyn_cast<GenericOp>(currentOp)) {
      // Find which input operand this constant corresponds to
      for (auto [idx, input] : llvm::enumerate(genericOp.getInputs())) {
        if (input == currentValue) {
          chains.push_back(
              {constOp, splatValue, genericOp, static_cast<unsigned>(idx)});
        }
      }
    }
  }
}

// Helper to trace from a d2m.generic input to linalg.generic block arguments
static void
findLinalgBlockArgsForGenericInput(GenericOp genericOp,
                                   unsigned genericInputIdx,
                                   SmallVectorImpl<BlockArgument> &linalgArgs) {

  // Walk through the generic region to find linalg.generic ops
  for (Region &region : genericOp.getRegions()) {
    region.walk([&](linalg::GenericOp linalgOp) {
      Block *linalgBlock = linalgOp.getBody();

      // Check each linalg input to see if it traces back to our generic input
      for (auto [linalgArgIdx, linalgInput] :
           llvm::enumerate(linalgOp.getInputs())) {

        Value tracedValue = linalgInput;
        Operation *defOp = tracedValue.getDefiningOp();

        // Trace back through d2m.wait operations
        if (auto waitOp = dyn_cast_or_null<WaitOp>(defOp)) {
          tracedValue = waitOp.getCb();
        }

        // Check if this traces back to a generic block argument
        if (auto blockArg = dyn_cast<BlockArgument>(tracedValue)) {
          if (blockArg.getOwner()->getParentOp() == genericOp.getOperation() &&
              blockArg.getArgNumber() == genericInputIdx) {
            linalgArgs.push_back(linalgBlock->getArgument(linalgArgIdx));
          }
        }
      }
    });
  }
}

// Pattern to scalarize d2m.full constant tensor operands
class ScalarizeFullOpPattern : public OpRewritePattern<FullOp> {
public:
  using OpRewritePattern<FullOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FullOp fullOp,
                                PatternRewriter &rewriter) const override {
    Attribute splatValue = fullOp.getFillValue();
    if (!splatValue) {
      return failure();
    }

    // Track if any changes were made
    bool madeChanges = false;

    // Trace this constant forward to d2m.generic operations
    SmallVector<ConstantUseChain> chains;
    traceConstantToGenericOps(fullOp.getOperation(), splatValue, chains);

    if (chains.empty()) {
      return failure();
    }

    // Process each use chain
    for (const auto &chain : chains) {
      // Find linalg.generic block arguments that correspond to this input
      SmallVector<BlockArgument> linalgArgs;
      findLinalgBlockArgsForGenericInput(chain.genericOp, chain.genericInputIdx,
                                         linalgArgs);

      // For each linalg block argument, check if it can be scalarized
      for (BlockArgument arg : linalgArgs) {
        // Check if this argument is used by operations that support scalars
        bool canScalarize = false;
        for (Operation *user : arg.getUsers()) {
          if (auto computeOp =
                  dyn_cast<OperandLoadStoreRegisterOpInterface>(user)) {
            if (computeOp.supportsTileOrScalarRhs()) {
              // Check if the arg is used as the RHS (operand 1)
              if (user->getNumOperands() >= 2 && user->getOperand(1) == arg) {
                canScalarize = true;
                break;
              }
            }
          }
        }

        if (!canScalarize) {
          continue;
        }

        // Create the scalar constant at the beginning of the linalg block
        Block *linalgBlock = arg.getOwner();
        rewriter.setInsertionPointToStart(linalgBlock);
        Value scalarConst =
            createScalarConstant(rewriter, fullOp.getLoc(), splatValue);

        if (!scalarConst) {
          continue;
        }

        // Replace uses of the block argument in operations that support scalars
        for (Operation *user : llvm::make_early_inc_range(arg.getUsers())) {
          if (auto computeOp =
                  dyn_cast<OperandLoadStoreRegisterOpInterface>(user)) {
            if (computeOp.supportsTileOrScalarRhs() &&
                user->getNumOperands() >= 2 && user->getOperand(1) == arg) {
              rewriter.modifyOpInPlace(
                  user, [&]() { user->setOperand(1, scalarConst); });
              madeChanges = true;
            }
          }
        }
      }
    }

    return madeChanges ? success() : failure();
  }
};

// Pattern to clean up d2m.generic and linalg.generic ops with unused inputs
class CleanupUnusedGenericInputsPattern : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    bool madeChanges = false;

    // Collect linalg.generic ops that need to be modified
    SmallVector<linalg::GenericOp> linalgOpsToProcess;
    for (Region &region : genericOp.getRegions()) {
      region.walk([&](linalg::GenericOp linalgOp) {
        Block *linalgBlock = linalgOp.getBody();

        // Check if any input block arguments are unused
        for (unsigned i = 0; i < linalgOp.getNumDpsInputs(); ++i) {
          BlockArgument arg = linalgBlock->getArgument(i);
          if (arg.use_empty()) {
            linalgOpsToProcess.push_back(linalgOp);
            return;
          }
        }
      });
    }

    // Now process the collected ops outside the walk
    for (linalg::GenericOp linalgOp : linalgOpsToProcess) {
      SmallVector<unsigned> unusedInputIndices;
      Block *linalgBlock = linalgOp.getBody();

      // Find unused input block arguments
      for (unsigned i = 0; i < linalgOp.getNumDpsInputs(); ++i) {
        BlockArgument arg = linalgBlock->getArgument(i);
        if (arg.use_empty()) {
          unusedInputIndices.push_back(i);
        }
      }

      // Collect d2m.wait operations that feed unused inputs for later erasure
      SmallVector<WaitOp> waitOpsToErase;
      for (unsigned idx : unusedInputIndices) {
        Value input = linalgOp.getInputs()[idx];
        if (auto waitOp = input.getDefiningOp<WaitOp>()) {
          if (waitOp->hasOneUse()) {
            waitOpsToErase.push_back(waitOp);
          }
        }
      }

      // Build new inputs list excluding unused ones
      SmallVector<Value> newInputs;
      for (auto [idx, input] : llvm::enumerate(linalgOp.getInputs())) {
        if (!llvm::is_contained(unusedInputIndices, idx)) {
          newInputs.push_back(input);
        }
      }

      // Build new indexing maps excluding unused input maps
      SmallVector<AffineMap> newIndexingMaps;
      auto oldMaps = linalgOp.getIndexingMapsArray();
      for (unsigned i = 0; i < linalgOp.getNumDpsInputs(); ++i) {
        if (!llvm::is_contained(unusedInputIndices, i)) {
          newIndexingMaps.push_back(oldMaps[i]);
        }
      }
      // Add output maps
      for (unsigned i = linalgOp.getNumDpsInputs(); i < oldMaps.size(); ++i) {
        newIndexingMaps.push_back(oldMaps[i]);
      }

      // Create new linalg.generic with fewer inputs
      rewriter.setInsertionPoint(linalgOp);
      auto newLinalgOp = rewriter.create<linalg::GenericOp>(
          linalgOp.getLoc(), linalgOp.getResultTypes(), newInputs,
          linalgOp.getOutputs(), newIndexingMaps,
          linalgOp.getIteratorTypesArray());

      // Clone the region but with fewer block arguments
      Region &newRegion = newLinalgOp.getRegion();
      Block *newBlock = rewriter.createBlock(&newRegion);

      // Add block arguments for new inputs (using element types)
      for (Value input : newInputs) {
        auto tensorType = cast<RankedTensorType>(input.getType());
        newBlock->addArgument(tensorType.getElementType(), linalgOp.getLoc());
      }
      // Add block arguments for outputs (using element types)
      for (Value output : linalgOp.getOutputs()) {
        auto tensorType = cast<RankedTensorType>(output.getType());
        newBlock->addArgument(tensorType.getElementType(), linalgOp.getLoc());
      }

      // Build mapping from old block args to new ones
      IRMapping mapping;
      unsigned newArgIdx = 0;
      for (unsigned oldArgIdx = 0; oldArgIdx < linalgBlock->getNumArguments();
           ++oldArgIdx) {
        if (oldArgIdx < linalgOp.getNumDpsInputs() &&
            llvm::is_contained(unusedInputIndices, oldArgIdx)) {
          // Skip unused input arguments
          continue;
        }
        mapping.map(linalgBlock->getArgument(oldArgIdx),
                    newBlock->getArgument(newArgIdx++));
      }

      // Clone operations from old block to new block
      rewriter.setInsertionPointToStart(newBlock);
      for (Operation &op : linalgBlock->getOperations()) {
        rewriter.clone(op, mapping);
      }

      // Replace the old linalg.generic
      rewriter.replaceOp(linalgOp, newLinalgOp.getResults());

      // Now it's safe to erase the wait operations that were feeding unused
      // inputs
      for (WaitOp waitOp : waitOpsToErase) {
        rewriter.eraseOp(waitOp);
      }

      madeChanges = true;
    }

    // Now check if any d2m.generic block arguments are unused
    SmallVector<unsigned> unusedGenericInputIndices;
    for (Region &region : genericOp.getRegions()) {
      Block *block = &region.front();
      unsigned numInputs = genericOp.getInputs().size();

      for (unsigned i = 0; i < numInputs; ++i) {
        BlockArgument arg = block->getArgument(i);
        if (arg.use_empty()) {
          unusedGenericInputIndices.push_back(i);
        }
      }
    }

    if (unusedGenericInputIndices.empty() && !madeChanges) {
      return failure();
    }

    if (!unusedGenericInputIndices.empty()) {
      // Build new inputs list for d2m.generic
      SmallVector<Value> newGenericInputs;
      for (auto [idx, input] : llvm::enumerate(genericOp.getInputs())) {
        if (!llvm::is_contained(unusedGenericInputIndices, idx)) {
          newGenericInputs.push_back(input);
        }
      }

      // Build new indexing maps excluding unused input maps
      SmallVector<Attribute> newIndexingMaps;
      auto oldMaps = genericOp.getIndexingMaps();
      unsigned numInputs = genericOp.getInputs().size();
      for (unsigned i = 0; i < numInputs; ++i) {
        if (!llvm::is_contained(unusedGenericInputIndices, i)) {
          newIndexingMaps.push_back(oldMaps[i]);
        }
      }
      // Add output maps
      for (unsigned i = numInputs; i < oldMaps.size(); ++i) {
        newIndexingMaps.push_back(oldMaps[i]);
      }

      // Create new d2m.generic with fewer inputs
      rewriter.setInsertionPoint(genericOp);
      auto newGenericOp = rewriter.create<GenericOp>(
          genericOp.getLoc(), genericOp.getResultTypes(), newGenericInputs,
          genericOp.getOutputs(), genericOp.getGrid(),
          genericOp.getBlockFactors(), rewriter.getArrayAttr(newIndexingMaps),
          genericOp.getIteratorTypes(), genericOp.getThreads(),
          /*regions=*/1);

      // Clone regions but remove unused block arguments
      for (auto [oldRegion, newRegion] :
           llvm::zip(genericOp.getRegions(), newGenericOp.getRegions())) {
        Block *oldBlock = &oldRegion.front();

        // Build list of new block argument types (excluding unused inputs)
        SmallVector<Type> newBlockArgTypes;
        SmallVector<Location> newBlockArgLocs;
        for (unsigned i = 0; i < oldBlock->getNumArguments(); ++i) {
          if (i < numInputs &&
              llvm::is_contained(unusedGenericInputIndices, i)) {
            // Skip unused input arguments
            continue;
          }
          newBlockArgTypes.push_back(oldBlock->getArgument(i).getType());
          newBlockArgLocs.push_back(oldBlock->getArgument(i).getLoc());
        }

        // Create the new block with proper argument types
        Block *newBlock = rewriter.createBlock(
            &newRegion, newRegion.end(), newBlockArgTypes, newBlockArgLocs);

        // Build mapping skipping unused input arguments
        IRMapping mapping;
        unsigned newArgIdx = 0;
        for (unsigned oldArgIdx = 0; oldArgIdx < oldBlock->getNumArguments();
             ++oldArgIdx) {
          if (oldArgIdx < numInputs &&
              llvm::is_contained(unusedGenericInputIndices, oldArgIdx)) {
            // Skip unused input arguments
            continue;
          }
          mapping.map(oldBlock->getArgument(oldArgIdx),
                      newBlock->getArgument(newArgIdx++));
        }

        // Clone operations
        rewriter.setInsertionPointToStart(newBlock);
        for (Operation &op : oldBlock->without_terminator()) {
          rewriter.clone(op, mapping);
        }
        // Clone terminator if it exists
        if (oldBlock->mightHaveTerminator()) {
          rewriter.clone(*oldBlock->getTerminator(), mapping);
        }
      }

      // Replace the old d2m.generic
      rewriter.replaceOp(genericOp, newGenericOp.getResults());
      madeChanges = true;
    }

    return madeChanges ? success() : failure();
  }
};

class D2MScalarizeConstTensors
    : public impl::D2MScalarizeConstTensorsBase<D2MScalarizeConstTensors> {
public:
  using D2MScalarizeConstTensorsBase::D2MScalarizeConstTensorsBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ScalarizeFullOpPattern, CleanupUnusedGenericInputsPattern>(
        ctx);

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::d2m
