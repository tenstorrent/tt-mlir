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
      // NOTE: This only follows the first user. If a to_layout result has
      // multiple users, only one path will be traced. This is acceptable for
      // the current use case but may miss optimization opportunities.
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

// Helper to check if an operation uses a value as a scalarizable RHS operand
static bool isScalarRhsUse(Operation *user, Value arg) {
  if (auto computeOp = dyn_cast<OperandLoadStoreRegisterOpInterface>(user)) {
    if (computeOp.supportsTileOrScalarRhs()) {
      // Check if the arg is used as the RHS (operand 1)
      if (user->getNumOperands() >= 2 && user->getOperand(1) == arg) {
        return true;
      }
    }
  }
  return false;
}

// Helper to find unused input indices in a block
static SmallVector<unsigned> findUnusedInputIndices(Block *block,
                                                    unsigned numInputs) {
  SmallVector<unsigned> unusedIndices;
  for (unsigned i = 0; i < numInputs; ++i) {
    BlockArgument arg = block->getArgument(i);
    if (arg.use_empty()) {
      unusedIndices.push_back(i);
    }
  }
  return unusedIndices;
}

// Helper to build new inputs list excluding unused ones
template <typename RangeT>
static SmallVector<Value> buildNewInputsList(RangeT &&inputs,
                                             ArrayRef<unsigned> unusedIndices) {
  SmallVector<Value> newInputs;
  for (auto [idx, input] : llvm::enumerate(inputs)) {
    if (!llvm::is_contained(unusedIndices, idx)) {
      newInputs.push_back(input);
    }
  }
  return newInputs;
}

// Helper to build mapping from old block args to new ones, skipping unused
static IRMapping buildBlockArgMapping(Block *oldBlock, Block *newBlock,
                                      unsigned numInputs,
                                      ArrayRef<unsigned> unusedIndices) {
  IRMapping mapping;
  unsigned newArgIdx = 0;
  for (unsigned oldArgIdx = 0; oldArgIdx < oldBlock->getNumArguments();
       ++oldArgIdx) {
    if (oldArgIdx < numInputs && llvm::is_contained(unusedIndices, oldArgIdx)) {
      // Skip unused input arguments
      continue;
    }
    mapping.map(oldBlock->getArgument(oldArgIdx),
                newBlock->getArgument(newArgIdx++));
  }
  return mapping;
}

// Helper to rebuild linalg.generic with fewer inputs
static linalg::GenericOp
rebuildLinalgGenericWithFewerInputs(linalg::GenericOp linalgOp,
                                    ArrayRef<unsigned> unusedInputIndices,
                                    PatternRewriter &rewriter) {
  // Build new inputs list
  SmallVector<Value> newInputs =
      buildNewInputsList(linalgOp.getInputs(), unusedInputIndices);

  // Build new indexing maps excluding unused input maps
  SmallVector<AffineMap> newIndexingMaps;
  auto oldMaps = linalgOp.getIndexingMapsArray();
  unsigned numInputs = linalgOp.getNumDpsInputs();
  // Add input maps (excluding unused)
  for (unsigned i = 0; i < numInputs; ++i) {
    if (!llvm::is_contained(unusedInputIndices, i)) {
      newIndexingMaps.push_back(oldMaps[i]);
    }
  }
  // Add output maps
  for (unsigned i = numInputs; i < oldMaps.size(); ++i) {
    newIndexingMaps.push_back(oldMaps[i]);
  }

  // Create new linalg.generic with fewer inputs
  rewriter.setInsertionPoint(linalgOp);
  auto newLinalgOp = rewriter.create<linalg::GenericOp>(
      linalgOp.getLoc(), linalgOp.getResultTypes(), newInputs,
      linalgOp.getOutputs(), newIndexingMaps, linalgOp.getIteratorTypesArray());

  // Clone the region with fewer block arguments
  Block *linalgBlock = linalgOp.getBody();
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
  IRMapping mapping = buildBlockArgMapping(
      linalgBlock, newBlock, linalgOp.getNumDpsInputs(), unusedInputIndices);

  // Clone operations from old block to new block
  rewriter.setInsertionPointToStart(newBlock);
  for (Operation &op : linalgBlock->getOperations()) {
    rewriter.clone(op, mapping);
  }

  return newLinalgOp;
}

// Helper to rebuild d2m.generic with fewer inputs
static GenericOp
rebuildD2MGenericWithFewerInputs(GenericOp genericOp,
                                 ArrayRef<unsigned> unusedInputIndices,
                                 PatternRewriter &rewriter) {
  unsigned numInputs = genericOp.getInputs().size();

  // Build new inputs list
  SmallVector<Value> newGenericInputs =
      buildNewInputsList(genericOp.getInputs(), unusedInputIndices);

  // Build new indexing maps excluding unused input maps
  SmallVector<Attribute> newIndexingMaps;
  auto oldMaps = genericOp.getIndexingMaps();
  // Add input maps (excluding unused)
  for (unsigned i = 0; i < numInputs; ++i) {
    if (!llvm::is_contained(unusedInputIndices, i)) {
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
      genericOp.getOutputs(), genericOp.getGrid(), genericOp.getBlockFactors(),
      rewriter.getArrayAttr(newIndexingMaps), genericOp.getIteratorTypes(),
      genericOp.getThreads(),
      /*regions=*/1);

  // Clone regions but remove unused block arguments
  for (auto [oldRegion, newRegion] :
       llvm::zip(genericOp.getRegions(), newGenericOp.getRegions())) {
    Block *oldBlock = &oldRegion.front();

    // Build list of new block argument types (excluding unused inputs)
    SmallVector<Type> newBlockArgTypes;
    SmallVector<Location> newBlockArgLocs;
    for (unsigned i = 0; i < oldBlock->getNumArguments(); ++i) {
      if (i < numInputs && llvm::is_contained(unusedInputIndices, i)) {
        // Skip unused input arguments
        continue;
      }
      newBlockArgTypes.push_back(oldBlock->getArgument(i).getType());
      newBlockArgLocs.push_back(oldBlock->getArgument(i).getLoc());
    }

    // Create the new block with proper argument types
    Block *newBlock = rewriter.createBlock(&newRegion, newRegion.end(),
                                           newBlockArgTypes, newBlockArgLocs);

    // Build mapping skipping unused input arguments
    IRMapping mapping =
        buildBlockArgMapping(oldBlock, newBlock, numInputs, unusedInputIndices);

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

  return newGenericOp;
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

    // Track linalg.generic ops that may need cleanup
    SmallVector<linalg::GenericOp> linalgOpsToCleanup;
    // Track d2m.generic ops that may need cleanup
    SmallVector<GenericOp> genericOpsToCleanup;

    // Process each use chain
    for (const auto &chain : chains) {
      // Find linalg.generic block arguments that correspond to this input
      SmallVector<BlockArgument> linalgArgs;
      findLinalgBlockArgsForGenericInput(chain.genericOp, chain.genericInputIdx,
                                         linalgArgs);

      // Skip if no linalg arguments found for this input
      if (linalgArgs.empty()) {
        continue;
      }

      // For each linalg block argument, check if it can be scalarized
      for (BlockArgument arg : linalgArgs) {
        // Check if this argument can be scalarized
        bool canScalarize = false;
        for (Operation *user : arg.getUsers()) {
          if (isScalarRhsUse(user, arg)) {
            canScalarize = true;
            break;
          }
        }

        if (!canScalarize) {
          continue;
        }

        // Create the scalar constant at the beginning of the linalg block
        Block *linalgBlock = arg.getOwner();
        rewriter.setInsertionPointToStart(linalgBlock);

        // Create scalar constant from attribute
        Value scalarConst;
        if (auto floatAttr = dyn_cast<FloatAttr>(splatValue)) {
          scalarConst =
              rewriter.create<arith::ConstantOp>(fullOp.getLoc(), floatAttr);
        } else if (auto intAttr = dyn_cast<IntegerAttr>(splatValue)) {
          scalarConst =
              rewriter.create<arith::ConstantOp>(fullOp.getLoc(), intAttr);
        }

        if (!scalarConst) {
          continue;
        }

        // Replace uses of the block argument in operations that support scalars
        for (Operation *user : llvm::make_early_inc_range(arg.getUsers())) {
          if (isScalarRhsUse(user, arg)) {
            rewriter.modifyOpInPlace(
                user, [&]() { user->setOperand(1, scalarConst); });
            madeChanges = true;
          }
        }

        // If the argument is now unused, mark its linalg.generic for cleanup
        if (arg.use_empty()) {
          if (auto linalgOp =
                  dyn_cast<linalg::GenericOp>(linalgBlock->getParentOp())) {
            if (!llvm::is_contained(linalgOpsToCleanup, linalgOp)) {
              linalgOpsToCleanup.push_back(linalgOp);
            }
          }
        }
      }

      // Mark the d2m.generic for potential cleanup
      if (!llvm::is_contained(genericOpsToCleanup, chain.genericOp)) {
        genericOpsToCleanup.push_back(chain.genericOp);
      }
    }

    // Clean up linalg.generic ops with unused inputs
    for (linalg::GenericOp linalgOp : linalgOpsToCleanup) {
      Block *linalgBlock = linalgOp.getBody();
      SmallVector<unsigned> unusedInputIndices =
          findUnusedInputIndices(linalgBlock, linalgOp.getNumDpsInputs());

      if (unusedInputIndices.empty()) {
        continue;
      }

      // Collect wait ops feeding unused inputs for later erasure
      SmallVector<WaitOp> waitOpsToErase;
      for (unsigned idx : unusedInputIndices) {
        Value input = linalgOp.getInputs()[idx];
        if (auto waitOp = input.getDefiningOp<WaitOp>()) {
          if (waitOp->hasOneUse()) {
            waitOpsToErase.push_back(waitOp);
          }
        }
      }

      // Rebuild linalg.generic with fewer inputs
      auto newLinalgOp = rebuildLinalgGenericWithFewerInputs(
          linalgOp, unusedInputIndices, rewriter);

      // Replace the old linalg.generic
      rewriter.replaceOp(linalgOp, newLinalgOp.getResults());

      // Erase wait operations that were feeding unused inputs
      for (WaitOp waitOp : waitOpsToErase) {
        rewriter.eraseOp(waitOp);
      }

      madeChanges = true;
    }

    // Clean up d2m.generic ops with unused inputs
    for (GenericOp genericOp : genericOpsToCleanup) {
      if (!genericOp.getRegions().empty()) {
        Block *block = &genericOp.getRegions().front().front();
        unsigned numInputs = genericOp.getInputs().size();
        SmallVector<unsigned> unusedGenericInputIndices =
            findUnusedInputIndices(block, numInputs);

        if (!unusedGenericInputIndices.empty()) {
          // Rebuild d2m.generic with fewer inputs
          auto newGenericOp = rebuildD2MGenericWithFewerInputs(
              genericOp, unusedGenericInputIndices, rewriter);

          // Replace the old d2m.generic
          rewriter.replaceOp(genericOp, newGenericOp.getResults());
          madeChanges = true;
        }
      }
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
    patterns.add<ScalarizeFullOpPattern>(ctx);

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::d2m
