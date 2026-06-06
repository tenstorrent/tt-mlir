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
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <optional>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MSCALARIZECONSTTENSORS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

struct ConstantUseChain {
  GenericOp genericOp;
  unsigned genericInputIdx;
};

struct ScalarizationPlan {
  GenericOp fillGenericOp;
  Location loc;
  Attribute splatValue;
  SmallVector<ConstantUseChain> chains;
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

/// Splatted fill value when \p genericOp matches the lowered `ttir.full` shape
/// (one inputless `linalg.generic` in the outer region, body is tile_fill +
/// arith.constant for the scalar). Otherwise std::nullopt.
static std::optional<Attribute>
tryGetSplatAttrFromFillGeneric(GenericOp genericOp) {
  if (genericOp.getNumRegions() < 1) {
    return std::nullopt;
  }
  Block &outerBlock = genericOp.getRegion(0).front();
  linalg::GenericOp fillLinalg;
  for (Operation &op : outerBlock.without_terminator()) {
    auto lg = dyn_cast<linalg::GenericOp>(&op);
    if (!lg || lg.getNumDpsInputs() != 0) {
      continue;
    }
    if (fillLinalg) {
      return std::nullopt;
    }
    fillLinalg = lg;
  }
  if (!fillLinalg) {
    return std::nullopt;
  }

  Block *body = fillLinalg.getBody();
  TileFillOp tileFill;
  for (Operation &op : body->without_terminator()) {
    if (auto tf = dyn_cast<TileFillOp>(&op)) {
      if (tileFill) {
        return std::nullopt;
      }
      tileFill = tf;
    } else if (!isa<arith::ConstantOp>(&op)) {
      return std::nullopt;
    }
  }
  if (!tileFill) {
    return std::nullopt;
  }

  Value fillVal = tileFill.getValue();
  if (auto cst = fillVal.getDefiningOp<arith::ConstantOp>()) {
    return cst.getValue();
  }
  return std::nullopt;
}

static void traceValueToGenericChainsImpl(
    Value currentValue, SmallVectorImpl<ConstantUseChain> &chains,
    llvm::SmallPtrSetImpl<Operation *> &visitedLayoutOrCastOps) {
  for (Operation *user : currentValue.getUsers()) {
    // RemoteLoadOps are embedded within a GenericOp; the GenericOp itself is
    // the use chain endpoint we need for operand cleanup.
    if (isa<RemoteLoadOp>(user)) {
      continue;
    }

    if (Value result = getLayoutOrCastResult(user)) {
      if (visitedLayoutOrCastOps.insert(user).second) {
        traceValueToGenericChainsImpl(result, chains, visitedLayoutOrCastOps);
      }
      continue;
    }

    if (auto genericOp = dyn_cast<GenericOp>(user)) {
      for (auto [idx, input] : llvm::enumerate(genericOp.getInputs())) {
        if (input == currentValue) {
          chains.push_back({genericOp, static_cast<unsigned>(idx)});
        }
      }
    }
  }
}

static void
traceValueToGenericChains(Value producedValue,
                          SmallVectorImpl<ConstantUseChain> &chains) {
  llvm::SmallPtrSet<Operation *, 32> visitedLayoutOrCastOps;
  traceValueToGenericChainsImpl(producedValue, chains, visitedLayoutOrCastOps);
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
  auto computeOp = dyn_cast<OperandLoadStoreRegisterOpInterface>(user);
  if (!computeOp || !computeOp.supportsTileOrScalarRhs()) {
    return false;
  }
  return user->getNumOperands() >= 2 && user->getOperand(1) == arg;
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

// Indices of generic inputs not referenced by any op inside the generic's
// regions.
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

static linalg::GenericOp rebuildLinalgGenericWithoutScalarizedInputs(
    linalg::GenericOp linalgOp, ArrayRef<unsigned> scalarizedInputIndices,
    RewriterBase &rewriter) {
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
    RewriterBase &rewriter) {
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

  // Old operand index -> new index after dropping scalarized inputs.
  llvm::DenseMap<uint64_t, uint64_t> operandIndexRemap;
  unsigned newOperandIdx = 0;
  for (unsigned oldIdx = 0; oldIdx < numInputs; ++oldIdx) {
    if (!llvm::is_contained(scalarizedInputIndices, oldIdx)) {
      operandIndexRemap[oldIdx] = newOperandIdx++;
    }
  }
  for (unsigned oldIdx = numInputs; oldIdx < numInputs + numOutputs; ++oldIdx) {
    operandIndexRemap[oldIdx] = newOperandIdx++;
  }

  rewriter.setInsertionPoint(genericOp);
  auto newGenericOp = rewriter.create<GenericOp>(
      genericOp.getLoc(), genericOp.getResultTypes(), newGenericInputs,
      genericOp.getOutputs(), genericOp.getAdditionalArgs(),
      genericOp.getGrid(), genericOp.getBlockFactors(),
      rewriter.getArrayAttr(newIndexingMaps), genericOp.getIteratorTypes(),
      genericOp.getThreads(), genericOp.getFabricConnectionConfigAttr(),
      /*regions=*/1);

  for (auto [oldRegion, newRegion] :
       llvm::zip(genericOp.getRegions(), newGenericOp.getRegions())) {
    Block *oldBlock = &oldRegion.front();

    SmallVector<Type> newBlockArgTypes;
    SmallVector<Location> newBlockArgLocs;
    for (BlockArgument arg : oldBlock->getArguments()) {
      newBlockArgTypes.push_back(arg.getType());
      newBlockArgLocs.push_back(arg.getLoc());
    }

    Block *newBlock = rewriter.createBlock(&newRegion, newRegion.end(),
                                           newBlockArgTypes, newBlockArgLocs);

    IRMapping mapping;
    for (unsigned i = 0; i < oldBlock->getNumArguments(); ++i) {
      mapping.map(oldBlock->getArgument(i), newBlock->getArgument(i));
    }

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

template <typename OpT>
static void addUniqueOp(OpT op, SmallVectorImpl<OpT> &ops,
                        llvm::SmallPtrSetImpl<Operation *> &seen) {
  if (seen.insert(op.getOperation()).second) {
    ops.push_back(op);
  }
}

/// Replace splat tensor uses (via layout/cast chains) with scalar RHS constants
/// where supported. Cleanup is batched separately so each affected op is
/// rebuilt at most once.
static bool scalarizeSplatThroughUseChains(
    Location loc, Attribute splatValue, ArrayRef<ConstantUseChain> chains,
    RewriterBase &rewriter,
    SmallVectorImpl<linalg::GenericOp> &linalgOpsToCleanup,
    SmallVectorImpl<GenericOp> &genericOpsToCleanup,
    llvm::SmallPtrSetImpl<Operation *> &linalgOpsSeen,
    llvm::SmallPtrSetImpl<Operation *> &genericOpsSeen) {
  bool madeChanges = false;

  for (const auto &chain : chains) {
    SmallVector<BlockArgument> linalgArgs;
    findLinalgBlockArgsForGenericInput(chain.genericOp, chain.genericInputIdx,
                                       linalgArgs);

    if (linalgArgs.empty()) {
      continue;
    }

    for (BlockArgument arg : linalgArgs) {
      if (!llvm::any_of(arg.getUsers(), [&](Operation *user) {
            return isScalarRhsUse(user, arg);
          })) {
        continue;
      }

      Block *linalgBlock = arg.getOwner();
      rewriter.setInsertionPointToStart(linalgBlock);

      Value scalarConst;
      if (auto floatAttr = dyn_cast<FloatAttr>(splatValue)) {
        scalarConst = rewriter.create<arith::ConstantOp>(loc, floatAttr);
      } else if (auto intAttr = dyn_cast<IntegerAttr>(splatValue)) {
        scalarConst = rewriter.create<arith::ConstantOp>(loc, intAttr);
      }

      if (!scalarConst) {
        continue;
      }

      for (Operation *user : llvm::make_early_inc_range(arg.getUsers())) {
        if (isScalarRhsUse(user, arg)) {
          rewriter.modifyOpInPlace(user,
                                   [&]() { user->setOperand(1, scalarConst); });
          madeChanges = true;
        }
      }

      if (arg.use_empty()) {
        if (auto linalgOp =
                dyn_cast<linalg::GenericOp>(linalgBlock->getParentOp())) {
          addUniqueOp(linalgOp, linalgOpsToCleanup, linalgOpsSeen);
        }
      }
    }

    addUniqueOp(chain.genericOp, genericOpsToCleanup, genericOpsSeen);
  }

  return madeChanges;
}

static bool
cleanupScalarizedLinalgOps(ArrayRef<linalg::GenericOp> linalgOpsToCleanup,
                           RewriterBase &rewriter) {
  bool madeChanges = false;

  for (linalg::GenericOp linalgOp : linalgOpsToCleanup) {
    if (!linalgOp->getBlock()) {
      continue;
    }

    Block *linalgBlock = linalgOp.getBody();
    SmallVector<unsigned> scalarizedInputIndices =
        findScalarizedInputIndices(linalgBlock, linalgOp.getNumDpsInputs());

    if (scalarizedInputIndices.empty()) {
      continue;
    }

    // Capture RemoteLoads that will be dead after inputs are removed; erase
    // after replace.
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

    for (RemoteLoadOp remoteLoadOp : remoteLoadOpsToErase) {
      rewriter.eraseOp(remoteLoadOp);
    }

    madeChanges = true;
  }

  return madeChanges;
}

static bool hasNoResultUses(Operation *op) {
  return llvm::all_of(op->getResults(),
                      [](Value result) { return result.use_empty(); });
}

static bool isDeadScalarizationHelperOp(Operation *op) {
  return hasNoResultUses(op) &&
         isa<arith::ConstantOp, tensor::EmptyOp, BlockIndexOp, ToLayoutOp,
             ViewLayoutOp, ttir::TTNNMetalLayoutCastOp>(op);
}

static bool cleanupDeadOpsInGenericRegions(ArrayRef<GenericOp> genericOps,
                                           RewriterBase &rewriter) {
  bool madeAnyChanges = false;
  bool madeChanges = false;

  do {
    madeChanges = false;

    for (GenericOp genericOp : genericOps) {
      if (!genericOp->getBlock()) {
        continue;
      }

      SmallVector<Operation *> deadOps;
      for (Region &region : genericOp.getRegions()) {
        region.walk<WalkOrder::PostOrder>([&](Operation *op) {
          if (op == genericOp.getOperation() ||
              op->hasTrait<OpTrait::IsTerminator>()) {
            return;
          }
          if (isDeadScalarizationHelperOp(op)) {
            deadOps.push_back(op);
          }
        });
      }

      for (Operation *op : deadOps) {
        if (!op->getBlock() || !isDeadScalarizationHelperOp(op)) {
          continue;
        }
        rewriter.eraseOp(op);
        madeChanges = true;
        madeAnyChanges = true;
      }
    }
  } while (madeChanges);

  return madeAnyChanges;
}

static bool cleanupScalarizedGenericOps(ArrayRef<GenericOp> genericOpsToCleanup,
                                        RewriterBase &rewriter) {
  bool madeChanges = false;

  for (GenericOp genericOp : genericOpsToCleanup) {
    if (!genericOp->getBlock()) {
      continue;
    }

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

  return madeChanges;
}

static void
collectLayoutOrCastUsersPostOrder(Value value,
                                  SmallVectorImpl<Operation *> &ops,
                                  llvm::SmallPtrSetImpl<Operation *> &seen) {
  for (Operation *user : value.getUsers()) {
    Value result = getLayoutOrCastResult(user);
    if (!result) {
      continue;
    }

    if (!seen.insert(user).second) {
      continue;
    }
    collectLayoutOrCastUsersPostOrder(result, ops, seen);
    ops.push_back(user);
  }
}

static bool cleanupDeadScalarizedFillChains(ArrayRef<ScalarizationPlan> plans,
                                            RewriterBase &rewriter) {
  bool madeChanges = false;
  SmallVector<Operation *> maybeDeadOps;
  llvm::SmallPtrSet<Operation *, 32> seen;

  for (const ScalarizationPlan &plan : plans) {
    GenericOp fillGenericOp = plan.fillGenericOp;
    if (!fillGenericOp->getBlock()) {
      continue;
    }

    collectLayoutOrCastUsersPostOrder(fillGenericOp.getResult(0), maybeDeadOps,
                                      seen);
    if (seen.insert(fillGenericOp).second) {
      maybeDeadOps.push_back(fillGenericOp);
    }
  }

  for (Operation *op : maybeDeadOps) {
    if (!op->getBlock() || !isOpTriviallyDead(op)) {
      continue;
    }
    rewriter.eraseOp(op);
    madeChanges = true;
  }

  return madeChanges;
}

static SmallVector<ScalarizationPlan> collectScalarizationPlans(Operation *op) {
  SmallVector<ScalarizationPlan> plans;
  op->walk([&](GenericOp genericOp) {
    std::optional<Attribute> splatAttr =
        tryGetSplatAttrFromFillGeneric(genericOp);
    if (!splatAttr) {
      return;
    }

    SmallVector<ConstantUseChain> chains;
    traceValueToGenericChains(genericOp.getResult(0), chains);

    if (chains.empty()) {
      return;
    }

    plans.push_back(
        {genericOp, genericOp.getLoc(), *splatAttr, std::move(chains)});
  });
  return plans;
}

class D2MScalarizeConstTensors
    : public impl::D2MScalarizeConstTensorsBase<D2MScalarizeConstTensors> {
public:
  using D2MScalarizeConstTensorsBase::D2MScalarizeConstTensorsBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    SmallVector<ScalarizationPlan> plans =
        collectScalarizationPlans(getOperation());
    if (plans.empty()) {
      return;
    }

    IRRewriter rewriter(ctx);
    SmallVector<linalg::GenericOp> linalgOpsToCleanup;
    SmallVector<GenericOp> genericOpsToCleanup;
    llvm::SmallPtrSet<Operation *, 32> linalgOpsSeen;
    llvm::SmallPtrSet<Operation *, 32> genericOpsSeen;

    for (ScalarizationPlan &plan : plans) {
      scalarizeSplatThroughUseChains(
          plan.loc, plan.splatValue, plan.chains, rewriter, linalgOpsToCleanup,
          genericOpsToCleanup, linalgOpsSeen, genericOpsSeen);
    }

    cleanupScalarizedLinalgOps(linalgOpsToCleanup, rewriter);
    cleanupDeadOpsInGenericRegions(genericOpsToCleanup, rewriter);
    cleanupScalarizedGenericOps(genericOpsToCleanup, rewriter);
    cleanupDeadScalarizedFillChains(plans, rewriter);
  }
};

} // namespace

} // namespace mlir::tt::d2m
