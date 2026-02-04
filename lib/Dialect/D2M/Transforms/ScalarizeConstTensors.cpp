// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <climits>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MSCALARIZECONSTTENSORS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

struct ConstantUseChain {
  GenericOp genericOp;
  unsigned genericInputIdx;
};

static Value getLayoutOrCastResult(Operation *op) {
  if (auto toLayoutOp = dyn_cast<ToLayoutOp>(op)) {
    return toLayoutOp.getResult(0);
  }
  if (auto viewLayoutOp = dyn_cast<ViewLayoutOp>(op)) {
    return viewLayoutOp.getResult();
  }
  if (auto castOp = dyn_cast<ttir::TTNNMetalLayoutCastOp>(op)) {
    return castOp.getResult();
  }
  return nullptr;
}

static void
traceConstantToGenericOps(Operation *constOp,
                          SmallVectorImpl<ConstantUseChain> &chains) {
  for (Operation *user : constOp->getUsers()) {
    Value currentValue = constOp->getResult(0);
    Operation *currentOp = user;

    while (Value result = getLayoutOrCastResult(currentOp)) {
      currentValue = result;
      if (currentValue.getUsers().empty()) {
        break;
      }
      // NOTE: This only follows the first user. If a result has
      // multiple users, only one path will be traced. This is acceptable for
      // the current use case but may miss optimization opportunities.
      for (Operation *user : currentValue.getUsers()) {
        // RemoteLoadOps are embedded within a GenericOp, so we skip them to try
        // to find the enclosing GenericOp.
        if (isa<RemoteLoadOp>(user)) {
          continue;
        }
        currentOp = user;
        break;
      }
    }

    if (auto genericOp = dyn_cast<GenericOp>(currentOp)) {
      for (auto [idx, input] : llvm::enumerate(genericOp.getInputs())) {
        if (input == currentValue) {
          chains.push_back({genericOp, static_cast<unsigned>(idx)});
        }
      }
    }
  }
}

static void
findLinalgBlockArgsForGenericInput(GenericOp genericOp,
                                   unsigned genericInputIdx,
                                   SmallVectorImpl<BlockArgument> &linalgArgs) {
  for (Region &region : genericOp.getRegions()) {
    region.walk([&](linalg::GenericOp linalgOp) {
      Block *linalgBlock = linalgOp.getBody();
      for (auto [linalgArgIdx, linalgInput] :
           llvm::enumerate(linalgOp.getInputs())) {
        if (auto remoteLoadOp = linalgInput.getDefiningOp<RemoteLoadOp>()) {
          if (remoteLoadOp.getMemref() ==
              genericOp.getOperand(genericInputIdx)) {
            linalgArgs.push_back(linalgBlock->getArgument(linalgArgIdx));
          }
        }
      }
    });
  }
}

static bool isScalarRhsUse(Operation *user, Value arg) {
  if (auto computeOp = dyn_cast<OperandLoadStoreRegisterOpInterface>(user)) {
    if (computeOp.supportsTileOrScalarRhs()) {
      // Check if the arg is used as the RHS (operand 1).
      if (user->getNumOperands() >= 2 && user->getOperand(1) == arg) {
        return true;
      }
    }
  }
  return false;
}

static SmallVector<unsigned> findScalarizedInputIndices(Block *block,
                                                        unsigned numInputs) {
  SmallVector<unsigned> scalarizedIndices;
  for (unsigned i = 0; i < numInputs; ++i) {
    BlockArgument arg = block->getArgument(i);
    if (arg.use_empty()) {
      scalarizedIndices.push_back(i);
    }
  }
  return scalarizedIndices;
}

// Build a list of unused input operands for a GenericOp.
static SmallVector<unsigned>
findUnusedGenericInputIndices(GenericOp genericOp) {
  SmallVector<unsigned> unusedIndices;

  auto isInputUsed = [&](Value input) {
    for (Region &region : genericOp.getRegions()) {
      for (Operation &op : region.getOps()) {
        if (llvm::is_contained(op.getOperands(), input)) {
          return true;
        }
      }
    }
    return false;
  };

  for (auto [idx, input] : llvm::enumerate(genericOp.getInputs())) {
    if (!isInputUsed(input)) {
      unusedIndices.push_back(idx);
    }
  }

  return unusedIndices;
}

template <typename RangeT>
static SmallVector<Value>
buildInputsWithoutScalarizedIndices(RangeT &&inputs,
                                    ArrayRef<unsigned> scalarizedIndices) {
  SmallVector<Value> newInputs;
  for (auto [idx, input] : llvm::enumerate(inputs)) {
    if (!llvm::is_contained(scalarizedIndices, idx)) {
      newInputs.push_back(input);
    }
  }
  return newInputs;
}

static IRMapping buildBlockArgMappingWithoutScalarizedIndices(
    Block *oldBlock, Block *newBlock, unsigned numInputs,
    ArrayRef<unsigned> scalarizedIndices) {
  IRMapping mapping;
  unsigned newArgIdx = 0;
  for (unsigned oldArgIdx = 0; oldArgIdx < oldBlock->getNumArguments();
       ++oldArgIdx) {
    if (oldArgIdx < numInputs &&
        llvm::is_contained(scalarizedIndices, oldArgIdx)) {
      continue;
    }
    mapping.map(oldBlock->getArgument(oldArgIdx),
                newBlock->getArgument(newArgIdx++));
  }
  return mapping;
}

static void updateTensorEmptyResultType(
    Operation *originalOp, Operation *clonedOp, GenericOp oldGenericOp,
    GenericOp newGenericOp,
    const llvm::DenseMap<uint64_t, uint64_t> &operandIndexRemap) {
  auto tensorEmptyOp = mlir::dyn_cast<mlir::tensor::EmptyOp>(clonedOp);
  auto originalEmptyOp = mlir::dyn_cast<mlir::tensor::EmptyOp>(originalOp);
  if (!tensorEmptyOp || !originalEmptyOp) {
    return;
  }

  Value associatedOperand = oldGenericOp.findAssocOperand(originalEmptyOp);
  if (!associatedOperand) {
    return;
  }

  // Find the operand index in the old generic op
  unsigned oldOperandIdx = UINT_MAX;
  for (unsigned i = 0; i < oldGenericOp->getNumOperands(); ++i) {
    if (oldGenericOp->getOperand(i) == associatedOperand) {
      oldOperandIdx = i;
      break;
    }
  }

  auto it = operandIndexRemap.find(oldOperandIdx);
  if (it == operandIndexRemap.end()) {
    return;
  }

  Value newCB = newGenericOp.findAssocCBByOperandIndex(clonedOp, it->second);
  auto cbType = mlir::dyn_cast<d2m::CBType>(newCB.getType());
  if (!cbType) {
    return;
  }

  auto tensorType = mlir::dyn_cast<RankedTensorType>(cbType.getUnderlying());
  if (tensorType) {
    tensorEmptyOp.getResult().setType(tensorType);
  }
}

static linalg::GenericOp rebuildLinalgGenericWithoutScalarizedInputs(
    linalg::GenericOp linalgOp, ArrayRef<unsigned> scalarizedInputIndices,
    PatternRewriter &rewriter) {
  SmallVector<Value> newInputs = buildInputsWithoutScalarizedIndices(
      linalgOp.getInputs(), scalarizedInputIndices);

  SmallVector<AffineMap> newIndexingMaps;
  auto oldMaps = linalgOp.getIndexingMapsArray();
  unsigned numInputs = linalgOp.getNumDpsInputs();
  for (unsigned i = 0; i < numInputs; ++i) {
    if (!llvm::is_contained(scalarizedInputIndices, i)) {
      newIndexingMaps.push_back(oldMaps[i]);
    }
  }
  for (unsigned i = numInputs; i < oldMaps.size(); ++i) {
    newIndexingMaps.push_back(oldMaps[i]);
  }

  rewriter.setInsertionPoint(linalgOp);
  auto newLinalgOp = rewriter.create<linalg::GenericOp>(
      linalgOp.getLoc(), linalgOp.getResultTypes(), newInputs,
      linalgOp.getOutputs(), newIndexingMaps, linalgOp.getIteratorTypesArray());

  Block *linalgBlock = linalgOp.getBody();
  Region &newRegion = newLinalgOp.getRegion();
  Block *newBlock = rewriter.createBlock(&newRegion);

  for (Value input : newInputs) {
    auto tensorType = cast<RankedTensorType>(input.getType());
    newBlock->addArgument(tensorType.getElementType(), linalgOp.getLoc());
  }
  for (Value output : linalgOp.getOutputs()) {
    auto tensorType = cast<RankedTensorType>(output.getType());
    newBlock->addArgument(tensorType.getElementType(), linalgOp.getLoc());
  }

  IRMapping mapping = buildBlockArgMappingWithoutScalarizedIndices(
      linalgBlock, newBlock, linalgOp.getNumDpsInputs(),
      scalarizedInputIndices);

  rewriter.setInsertionPointToStart(newBlock);
  for (Operation &op : linalgBlock->getOperations()) {
    rewriter.clone(op, mapping);
  }

  return newLinalgOp;
}

static GenericOp rebuildD2MGenericWithoutScalarizedInputs(
    GenericOp genericOp, ArrayRef<unsigned> scalarizedInputIndices,
    PatternRewriter &rewriter) {
  unsigned numInputs = genericOp.getInputs().size();
  unsigned numOutputs = genericOp.getOutputs().size();

  SmallVector<Value> newGenericInputs = buildInputsWithoutScalarizedIndices(
      genericOp.getInputs(), scalarizedInputIndices);

  SmallVector<Attribute> newIndexingMaps;
  auto oldMaps = genericOp.getIndexingMaps();
  for (unsigned i = 0; i < numInputs; ++i) {
    if (!llvm::is_contained(scalarizedInputIndices, i)) {
      newIndexingMaps.push_back(oldMaps[i]);
    }
  }
  for (unsigned i = numInputs; i < oldMaps.size(); ++i) {
    newIndexingMaps.push_back(oldMaps[i]);
  }

  // Build operand index remapping: old operand index -> new operand index
  llvm::DenseMap<uint64_t, uint64_t> operandIndexRemap;
  unsigned newOperandIdx = 0;
  // Map input operands (skip scalarized ones)
  for (unsigned oldIdx = 0; oldIdx < numInputs; ++oldIdx) {
    if (!llvm::is_contained(scalarizedInputIndices, oldIdx)) {
      operandIndexRemap[oldIdx] = newOperandIdx++;
    }
  }
  // Map output operands (these shift down by the number of removed inputs)
  for (unsigned oldIdx = numInputs; oldIdx < numInputs + numOutputs; ++oldIdx) {
    operandIndexRemap[oldIdx] = newOperandIdx++;
  }

  rewriter.setInsertionPoint(genericOp);
  auto newGenericOp = rewriter.create<GenericOp>(
      genericOp.getLoc(), genericOp.getResultTypes(), newGenericInputs,
      genericOp.getOutputs(), genericOp.getGrid(), genericOp.getBlockFactors(),
      rewriter.getArrayAttr(newIndexingMaps), genericOp.getIteratorTypes(),
      genericOp.getThreads(), genericOp.getScratchInputsAttr(),
      /*regions=*/1);

  for (auto [oldRegion, newRegion] :
       llvm::zip(genericOp.getRegions(), newGenericOp.getRegions())) {
    Block *oldBlock = &oldRegion.front();

    SmallVector<Type> newBlockArgTypes;
    SmallVector<Location> newBlockArgLocs;
    for (unsigned i = 0; i < oldBlock->getNumArguments(); ++i) {
      if (i < numInputs && llvm::is_contained(scalarizedInputIndices, i)) {
        continue;
      }
      newBlockArgTypes.push_back(oldBlock->getArgument(i).getType());
      newBlockArgLocs.push_back(oldBlock->getArgument(i).getLoc());
    }

    Block *newBlock = rewriter.createBlock(&newRegion, newRegion.end(),
                                           newBlockArgTypes, newBlockArgLocs);

    IRMapping mapping = buildBlockArgMappingWithoutScalarizedIndices(
        oldBlock, newBlock, numInputs, scalarizedInputIndices);

    rewriter.setInsertionPointToStart(newBlock);
    for (Operation &op : oldBlock->without_terminator()) {
      Operation *clonedOp = rewriter.clone(op, mapping);
      // Update tensor.empty result type to match the associated operand CB
      updateTensorEmptyResultType(&op, clonedOp, genericOp, newGenericOp,
                                  operandIndexRemap);
    }
    if (oldBlock->mightHaveTerminator()) {
      rewriter.clone(*oldBlock->getTerminator(), mapping);
    }
  }

  return newGenericOp;
}

class ScalarizeFullOpPattern : public OpRewritePattern<FullOp> {
public:
  using OpRewritePattern<FullOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FullOp fullOp,
                                PatternRewriter &rewriter) const override {
    Attribute splatValue = fullOp.getFillValue();
    bool madeChanges = false;

    SmallVector<ConstantUseChain> chains;
    traceConstantToGenericOps(fullOp.getOperation(), chains);

    if (chains.empty()) {
      return failure();
    }

    SmallVector<linalg::GenericOp> linalgOpsToCleanup;
    SmallVector<GenericOp> genericOpsToCleanup;

    for (const auto &chain : chains) {
      SmallVector<BlockArgument> linalgArgs;
      findLinalgBlockArgsForGenericInput(chain.genericOp, chain.genericInputIdx,
                                         linalgArgs);

      if (linalgArgs.empty()) {
        continue;
      }

      // Replace each candidate const linalg block argument with a scalar
      // constant if possible.
      for (BlockArgument arg : linalgArgs) {
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

        Block *linalgBlock = arg.getOwner();
        rewriter.setInsertionPointToStart(linalgBlock);

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

        for (Operation *user : llvm::make_early_inc_range(arg.getUsers())) {
          if (isScalarRhsUse(user, arg)) {
            rewriter.modifyOpInPlace(
                user, [&]() { user->setOperand(1, scalarConst); });
            madeChanges = true;
          }
        }

        if (arg.use_empty()) {
          if (auto linalgOp =
                  dyn_cast<linalg::GenericOp>(linalgBlock->getParentOp())) {
            if (!llvm::is_contained(linalgOpsToCleanup, linalgOp)) {
              linalgOpsToCleanup.push_back(linalgOp);
            }
          }
        }
      }

      if (!llvm::is_contained(genericOpsToCleanup, chain.genericOp)) {
        genericOpsToCleanup.push_back(chain.genericOp);
      }
    }

    // rewrite modified linalg.generic ops to remove unused inputs
    for (linalg::GenericOp linalgOp : linalgOpsToCleanup) {
      Block *linalgBlock = linalgOp.getBody();
      SmallVector<unsigned> scalarizedInputIndices =
          findScalarizedInputIndices(linalgBlock, linalgOp.getNumDpsInputs());

      if (scalarizedInputIndices.empty()) {
        continue;
      }

      // Collect remote load ops that will become dead after removing scalarized
      // inputs.
      SmallVector<RemoteLoadOp> remoteLoadOpsToErase;
      for (unsigned idx : scalarizedInputIndices) {
        Value input = linalgOp.getInputs()[idx];
        if (auto remoteLoadOp = input.getDefiningOp<RemoteLoadOp>()) {
          if (remoteLoadOp->hasOneUse()) {
            remoteLoadOpsToErase.push_back(remoteLoadOp);
          }
        }
      }

      auto newLinalgOp = rebuildLinalgGenericWithoutScalarizedInputs(
          linalgOp, scalarizedInputIndices, rewriter);

      rewriter.replaceOp(linalgOp, newLinalgOp.getResults());

      // Erase RemoteLoadOps after rebuilding linalg.generic
      for (RemoteLoadOp remoteLoadOp : remoteLoadOpsToErase) {
        rewriter.eraseOp(remoteLoadOp);
      }

      madeChanges = true;
    }

    // Rebuild linalg.generic ops to remove unused inputs
    for (GenericOp genericOp : genericOpsToCleanup) {
      SmallVector<unsigned> unusedInputIndices =
          findUnusedGenericInputIndices(genericOp);

      if (unusedInputIndices.empty()) {
        continue;
      }

      auto newGenericOp = rebuildD2MGenericWithoutScalarizedInputs(
          genericOp, unusedInputIndices, rewriter);

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
