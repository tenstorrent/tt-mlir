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

struct ConstantUseChain {
  GenericOp genericOp;
  unsigned genericInputIdx;
};

static Value getLayoutOrCastResult(Operation *op) {
  if (auto toLayoutOp = dyn_cast<ToLayoutOp>(op)) {
    return toLayoutOp.getResult(0);
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
      currentOp = *currentValue.getUsers().begin();
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
        Value tracedValue = linalgInput;
        Operation *defOp = tracedValue.getDefiningOp();

        if (auto waitOp = dyn_cast_or_null<WaitOp>(defOp)) {
          tracedValue = waitOp.getCb();
        }

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

  rewriter.setInsertionPoint(genericOp);
  auto newGenericOp = rewriter.create<GenericOp>(
      genericOp.getLoc(), genericOp.getResultTypes(), newGenericInputs,
      genericOp.getOutputs(), genericOp.getGrid(), genericOp.getBlockFactors(),
      rewriter.getArrayAttr(newIndexingMaps), genericOp.getIteratorTypes(),
      genericOp.getThreads(),
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
      rewriter.clone(op, mapping);
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

    for (linalg::GenericOp linalgOp : linalgOpsToCleanup) {
      Block *linalgBlock = linalgOp.getBody();
      SmallVector<unsigned> scalarizedInputIndices =
          findScalarizedInputIndices(linalgBlock, linalgOp.getNumDpsInputs());

      if (scalarizedInputIndices.empty()) {
        continue;
      }

      // Collect wait ops that will become dead after removing scalarized
      // inputs.
      SmallVector<WaitOp> waitOpsToErase;
      for (unsigned idx : scalarizedInputIndices) {
        Value input = linalgOp.getInputs()[idx];
        if (auto waitOp = input.getDefiningOp<WaitOp>()) {
          if (waitOp->hasOneUse()) {
            waitOpsToErase.push_back(waitOp);
          }
        }
      }

      auto newLinalgOp = rebuildLinalgGenericWithoutScalarizedInputs(
          linalgOp, scalarizedInputIndices, rewriter);

      rewriter.replaceOp(linalgOp, newLinalgOp.getResults());

      for (WaitOp waitOp : waitOpsToErase) {
        rewriter.eraseOp(waitOp);
      }

      madeChanges = true;
    }

    for (GenericOp genericOp : genericOpsToCleanup) {
      if (!genericOp.getRegions().empty()) {
        Block *block = &genericOp.getRegions().front().front();
        unsigned numInputs = genericOp.getInputs().size();
        SmallVector<unsigned> scalarizedInputIndices =
            findScalarizedInputIndices(block, numInputs);

        if (!scalarizedInputIndices.empty()) {
          auto newGenericOp = rebuildD2MGenericWithoutScalarizedInputs(
              genericOp, scalarizedInputIndices, rewriter);

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
