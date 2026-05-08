// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MOPTIMIZEMASKS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Absence from the table is an untracked state. For compatibility checks this
// collapses to undef: it is safe for an undef target but not for a concrete
// required OOB value.
using OOBStateMap = llvm::DenseMap<Value, ttcore::OOBVal>;

static ttcore::OOBVal getConstantOOBVal(Attribute attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    if (intAttr.getValue().isZero()) {
      return ttcore::OOBVal::Zero;
    }
    if (intAttr.getValue().isOne()) {
      return ttcore::OOBVal::One;
    }
    return ttcore::OOBVal::Undef;
  }

  if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    APFloat value = floatAttr.getValue();
    if (value.isZero()) {
      return ttcore::OOBVal::Zero;
    }
    if (value.isInfinity()) {
      return value.isNegative() ? ttcore::OOBVal::NegInf : ttcore::OOBVal::Inf;
    }
    return value.convertToDouble() == 1.0 ? ttcore::OOBVal::One
                                          : ttcore::OOBVal::Undef;
  }

  return ttcore::OOBVal::Undef;
}

static ttcore::OOBVal getStateOrUndef(const OOBStateMap &states, Value value) {
  auto it = states.find(value);
  if (it != states.end()) {
    return it->second;
  }

  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    return getConstantOOBVal(constant.getValue());
  }

  return ttcore::OOBVal::Undef;
}

static ttcore::OOBVal negateOOBVal(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::Zero:
    return ttcore::OOBVal::Zero;
  case ttcore::OOBVal::Inf:
    return ttcore::OOBVal::NegInf;
  case ttcore::OOBVal::NegInf:
    return ttcore::OOBVal::Inf;
  case ttcore::OOBVal::One:
  case ttcore::OOBVal::Undef:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown OOB value");
}

static ttcore::OOBVal absOOBVal(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::Zero:
  case ttcore::OOBVal::One:
  case ttcore::OOBVal::Inf:
    return value;
  case ttcore::OOBVal::NegInf:
    return ttcore::OOBVal::Inf;
  case ttcore::OOBVal::Undef:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown OOB value");
}

static ttcore::OOBVal addOOBVals(ttcore::OOBVal lhs, ttcore::OOBVal rhs) {
  if (lhs == ttcore::OOBVal::Zero) {
    return rhs;
  }
  if (rhs == ttcore::OOBVal::Zero) {
    return lhs;
  }
  if (lhs == ttcore::OOBVal::Inf || rhs == ttcore::OOBVal::Inf) {
    return (lhs == ttcore::OOBVal::NegInf || rhs == ttcore::OOBVal::NegInf)
               ? ttcore::OOBVal::Undef
               : ttcore::OOBVal::Inf;
  }
  if (lhs == ttcore::OOBVal::NegInf || rhs == ttcore::OOBVal::NegInf) {
    return ttcore::OOBVal::NegInf;
  }
  return ttcore::OOBVal::Undef;
}

static ttcore::OOBVal subOOBVals(ttcore::OOBVal lhs, ttcore::OOBVal rhs) {
  if (rhs == ttcore::OOBVal::Zero) {
    return lhs;
  }
  if (lhs == ttcore::OOBVal::Zero) {
    return negateOOBVal(rhs);
  }
  if (lhs == ttcore::OOBVal::Inf) {
    return rhs == ttcore::OOBVal::Inf ? ttcore::OOBVal::Undef
                                      : ttcore::OOBVal::Inf;
  }
  if (lhs == ttcore::OOBVal::NegInf) {
    return rhs == ttcore::OOBVal::NegInf ? ttcore::OOBVal::Undef
                                         : ttcore::OOBVal::NegInf;
  }
  if (rhs == ttcore::OOBVal::Inf) {
    return ttcore::OOBVal::NegInf;
  }
  if (rhs == ttcore::OOBVal::NegInf) {
    return ttcore::OOBVal::Inf;
  }
  if (lhs == ttcore::OOBVal::One && rhs == ttcore::OOBVal::One) {
    return ttcore::OOBVal::Zero;
  }
  return ttcore::OOBVal::Undef;
}

static ttcore::OOBVal mulOOBVals(ttcore::OOBVal lhs, ttcore::OOBVal rhs) {
  // Treat zero as absorbing. This matches the padding use-case and is the
  // useful conservative fold even though exact hardware NaN behavior can vary.
  if (lhs == ttcore::OOBVal::Zero || rhs == ttcore::OOBVal::Zero) {
    return ttcore::OOBVal::Zero;
  }
  if (lhs == ttcore::OOBVal::One) {
    return rhs;
  }
  if (rhs == ttcore::OOBVal::One) {
    return lhs;
  }
  if (lhs == ttcore::OOBVal::Undef || rhs == ttcore::OOBVal::Undef) {
    return ttcore::OOBVal::Undef;
  }
  return lhs == rhs ? ttcore::OOBVal::Inf : ttcore::OOBVal::NegInf;
}

static ttcore::OOBVal divOOBVals(ttcore::OOBVal lhs, ttcore::OOBVal rhs) {
  if (rhs == ttcore::OOBVal::One) {
    return lhs;
  }
  if (lhs == ttcore::OOBVal::Zero && rhs != ttcore::OOBVal::Zero &&
      rhs != ttcore::OOBVal::Undef) {
    return ttcore::OOBVal::Zero;
  }
  if (lhs == ttcore::OOBVal::One && rhs == ttcore::OOBVal::One) {
    return ttcore::OOBVal::One;
  }
  return ttcore::OOBVal::Undef;
}

static ttcore::OOBVal maxOOBVals(ttcore::OOBVal lhs, ttcore::OOBVal rhs) {
  if (lhs == ttcore::OOBVal::Inf || rhs == ttcore::OOBVal::Inf) {
    return ttcore::OOBVal::Inf;
  }
  if (lhs == ttcore::OOBVal::NegInf) {
    return rhs;
  }
  if (rhs == ttcore::OOBVal::NegInf) {
    return lhs;
  }
  if (lhs == ttcore::OOBVal::Undef || rhs == ttcore::OOBVal::Undef) {
    return ttcore::OOBVal::Undef;
  }
  return (lhs == ttcore::OOBVal::One || rhs == ttcore::OOBVal::One)
             ? ttcore::OOBVal::One
             : ttcore::OOBVal::Zero;
}

static ttcore::OOBVal minOOBVals(ttcore::OOBVal lhs, ttcore::OOBVal rhs) {
  if (lhs == ttcore::OOBVal::NegInf || rhs == ttcore::OOBVal::NegInf) {
    return ttcore::OOBVal::NegInf;
  }
  if (lhs == ttcore::OOBVal::Inf) {
    return rhs;
  }
  if (rhs == ttcore::OOBVal::Inf) {
    return lhs;
  }
  if (lhs == ttcore::OOBVal::Undef || rhs == ttcore::OOBVal::Undef) {
    return ttcore::OOBVal::Undef;
  }
  return (lhs == ttcore::OOBVal::Zero || rhs == ttcore::OOBVal::Zero)
             ? ttcore::OOBVal::Zero
             : ttcore::OOBVal::One;
}

static ttcore::OOBVal inferElementwiseOOBVal(Operation *op,
                                             const OOBStateMap &states) {
  if (isa<TileTypecastOp>(op)) {
    return getStateOrUndef(states, op->getOperand(0));
  }
  if (isa<TileNegativeOp>(op)) {
    return negateOOBVal(getStateOrUndef(states, op->getOperand(0)));
  }
  if (isa<TileAbsOp>(op)) {
    return absOOBVal(getStateOrUndef(states, op->getOperand(0)));
  }
  if (isa<TileFillOp>(op)) {
    return getStateOrUndef(states, op->getOperand(0));
  }

  if (op->getNumOperands() < 2) {
    return ttcore::OOBVal::Undef;
  }

  ttcore::OOBVal lhs = getStateOrUndef(states, op->getOperand(0));
  ttcore::OOBVal rhs = getStateOrUndef(states, op->getOperand(1));
  if (isa<TileAddOp>(op)) {
    return addOOBVals(lhs, rhs);
  }
  if (isa<TileSubOp>(op)) {
    return subOOBVals(lhs, rhs);
  }
  if (isa<TileMulOp>(op)) {
    return mulOOBVals(lhs, rhs);
  }
  if (isa<TileDivOp>(op)) {
    return divOOBVals(lhs, rhs);
  }
  if (isa<TileMaximumOp>(op)) {
    return maxOOBVals(lhs, rhs);
  }
  if (isa<TileMinimumOp>(op)) {
    return minOOBVals(lhs, rhs);
  }
  if (isa<TileWhereOp>(op)) {
    ttcore::OOBVal trueValue = getStateOrUndef(states, op->getOperand(1));
    ttcore::OOBVal falseValue = getStateOrUndef(states, op->getOperand(2));
    return trueValue == falseValue ? trueValue : ttcore::OOBVal::Undef;
  }
  if (isa<TileMatmulOp>(op)) {
    ttcore::OOBVal accumulator = getStateOrUndef(states, op->getOperand(2));
    return (lhs == ttcore::OOBVal::Zero || rhs == ttcore::OOBVal::Zero)
               ? accumulator
               : ttcore::OOBVal::Undef;
  }

  return ttcore::OOBVal::Undef;
}

static void propagateRegionOp(Operation *op, OOBStateMap &states);

static void propagateLinalgGeneric(linalg::GenericOp linalgOp,
                                   OOBStateMap &states) {
  Block *body = linalgOp.getBody();
  if (!body) {
    return;
  }

  OOBStateMap bodyStates = states;
  unsigned blockArgIndex = 0;
  for (Value input : linalgOp.getInputs()) {
    if (blockArgIndex >= body->getNumArguments()) {
      break;
    }
    bodyStates[body->getArgument(blockArgIndex++)] =
        getStateOrUndef(states, input);
  }
  for (Value output : linalgOp.getOutputs()) {
    if (blockArgIndex >= body->getNumArguments()) {
      break;
    }
    bodyStates[body->getArgument(blockArgIndex++)] =
        getStateOrUndef(states, output);
  }

  for (Operation &bodyOp : body->getOperations()) {
    if (auto yieldOp = dyn_cast<linalg::YieldOp>(&bodyOp)) {
      for (auto [result, yieldValue] :
           llvm::zip(linalgOp.getResults(), yieldOp.getValues())) {
        states[result] = getStateOrUndef(bodyStates, yieldValue);
      }
      return;
    }
    propagateRegionOp(&bodyOp, bodyStates);
  }
}

static SmallVector<ttcore::OOBVal> propagateGenericRegion(Region &region,
                                                          OOBStateMap states) {
  SmallVector<ttcore::OOBVal> yieldedStates;
  if (region.empty()) {
    return yieldedStates;
  }

  Block &block = region.front();
  for (Operation &op : block.getOperations()) {
    if (auto yieldOp = dyn_cast<YieldOp>(&op)) {
      yieldedStates.clear();
      for (Value value : yieldOp.getValues()) {
        yieldedStates.push_back(getStateOrUndef(states, value));
      }
      continue;
    }
    propagateRegionOp(&op, states);
  }

  return yieldedStates;
}

static void seedGenericBlockArgs(GenericOp genericOp, Region &region,
                                 OOBStateMap &states) {
  if (region.empty()) {
    return;
  }

  Block &block = region.front();
  unsigned blockArgIndex = 0;
  for (Value input : genericOp.getInputs()) {
    if (blockArgIndex >= block.getNumArguments()) {
      return;
    }
    states[block.getArgument(blockArgIndex++)] = getStateOrUndef(states, input);
  }
  for (Value output : genericOp.getOutputs()) {
    if (blockArgIndex >= block.getNumArguments()) {
      return;
    }
    states[block.getArgument(blockArgIndex++)] =
        getStateOrUndef(states, output);
  }
  for (Value additionalArg : genericOp.getAdditionalArgs()) {
    if (blockArgIndex >= block.getNumArguments()) {
      return;
    }
    states[block.getArgument(blockArgIndex++)] =
        getStateOrUndef(states, additionalArg);
  }
}

static void propagateGenericOp(GenericOp genericOp, OOBStateMap &states) {
  SmallVector<SmallVector<ttcore::OOBVal>> regionStates;
  for (Region &region : genericOp->getRegions()) {
    OOBStateMap localStates = states;
    for (Value input : genericOp.getInputs()) {
      localStates[input] = getStateOrUndef(states, input);
    }
    for (Value output : genericOp.getOutputs()) {
      localStates[output] = getStateOrUndef(states, output);
    }
    for (Value additionalArg : genericOp.getAdditionalArgs()) {
      localStates[additionalArg] = getStateOrUndef(states, additionalArg);
    }
    seedGenericBlockArgs(genericOp, region, localStates);
    SmallVector<ttcore::OOBVal> yieldedStates =
        propagateGenericRegion(region, std::move(localStates));
    if (!yieldedStates.empty()) {
      regionStates.push_back(std::move(yieldedStates));
    }
  }

  for (auto [resultIndex, result] : llvm::enumerate(genericOp.getResults())) {
    ttcore::OOBVal merged = ttcore::OOBVal::Undef;
    bool initialized = false;
    for (ArrayRef<ttcore::OOBVal> perRegionStates : regionStates) {
      if (resultIndex >= perRegionStates.size()) {
        continue;
      }
      ttcore::OOBVal candidate = perRegionStates[resultIndex];
      if (!initialized) {
        merged = candidate;
        initialized = true;
        continue;
      }
      if (merged != candidate) {
        merged = ttcore::OOBVal::Undef;
        break;
      }
    }
    states[result] = initialized ? merged : ttcore::OOBVal::Undef;
  }
}

static void propagateRegionOp(Operation *op, OOBStateMap &states) {
  if (auto linalgOp = dyn_cast<linalg::GenericOp>(op)) {
    propagateLinalgGeneric(linalgOp, states);
    return;
  }
  if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
    states[constant.getResult()] = getConstantOOBVal(constant.getValue());
    return;
  }
  if (auto remoteLoad = dyn_cast<RemoteLoadOp>(op)) {
    if (remoteLoad->getNumResults() != 0) {
      states[remoteLoad->getResult(0)] =
          getStateOrUndef(states, remoteLoad.getMemref());
    }
    return;
  }

  if (op->getNumResults() == 0) {
    return;
  }

  ttcore::OOBVal resultState = inferElementwiseOOBVal(op, states);
  for (Value result : op->getResults()) {
    states[result] = resultState;
  }
}

static bool hasNoMaskablePadding(ShapedType shapedType,
                                 ArrayRef<int64_t> logicalShape) {
  auto tileType = dyn_cast<ttcore::TileType>(shapedType.getElementType());
  if (!tileType) {
    return true;
  }

  ttcore::DeviceLayoutInterface layout = ttcore::getDeviceLayout(shapedType);
  if (!layout || logicalShape.size() < 2) {
    return false;
  }

  if (auto metalLayout = dyn_cast<ttcore::MetalLayoutAttr>(layout)) {
    ArrayRef<int64_t> dimAlignments = metalLayout.getDimAlignments();
    if (logicalShape.size() != dimAlignments.size()) {
      return false;
    }
    for (auto [logicalDim, alignment] :
         llvm::zip(logicalShape, dimAlignments)) {
      if (ttmlir::utils::alignUp(logicalDim, alignment) != logicalDim) {
        return false;
      }
    }
    return true;
  }

  ArrayRef<int64_t> gridShape = layout.getGridShape(shapedType);
  ArrayRef<int64_t> shardShape = layout.getShardShape(shapedType);
  if (gridShape.size() < 2 || shardShape.size() < 2) {
    return false;
  }

  ArrayRef<int64_t> tileShape = tileType.getShape();
  int64_t physicalRows = gridShape[gridShape.size() - 2] *
                         shardShape[shardShape.size() - 2] * tileShape[0];
  int64_t physicalCols = gridShape[gridShape.size() - 1] *
                         shardShape[shardShape.size() - 1] * tileShape[1];
  return logicalShape[logicalShape.size() - 2] == physicalRows &&
         logicalShape[logicalShape.size() - 1] == physicalCols;
}

static bool isKnownCompatible(const OOBStateMap &states, Value value,
                              ttcore::OOBVal required) {
  if (required == ttcore::OOBVal::Undef) {
    return true;
  }

  return getStateOrUndef(states, value) == required;
}

static bool canReplaceMaskWithInput(MaskOp maskOp) {
  return maskOp.getInput().getType() == maskOp.getResult().getType();
}

class D2MOptimizeMasksPass
    : public impl::D2MOptimizeMasksBase<D2MOptimizeMasksPass> {
public:
  using impl::D2MOptimizeMasksBase<D2MOptimizeMasksPass>::D2MOptimizeMasksBase;

  void runOnOperation() override {
    OOBStateMap states;
    SmallVector<MaskOp> masksToErase;

    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto maskOp = dyn_cast<MaskOp>(op)) {
        ttcore::OOBVal required = maskOp.getFillValue();

        bool redundant = false;
        if (canReplaceMaskWithInput(maskOp)) {
          redundant = isKnownCompatible(states, maskOp.getInput(), required);

          auto inputType = dyn_cast<ShapedType>(maskOp.getInput().getType());
          if (!redundant && inputType) {
            redundant =
                hasNoMaskablePadding(inputType, maskOp.getLogicalShape());
          }
        }

        if (redundant) {
          maskOp.getResult().replaceAllUsesWith(maskOp.getInput());
          masksToErase.push_back(maskOp);
        } else {
          states[maskOp.getResult()] = required;
        }
        return;
      }

      if (auto genericOp = dyn_cast<GenericOp>(op)) {
        propagateGenericOp(genericOp, states);
        return;
      }

      if (auto viewLayout = dyn_cast<ViewLayoutOp>(op)) {
        states[viewLayout.getResult()] =
            getStateOrUndef(states, viewLayout.getInput());
        return;
      }

      if (auto toLayout = dyn_cast<ToLayoutOp>(op)) {
        states[toLayout.getResult(0)] = ttcore::OOBVal::Undef;
      }
    });

    for (MaskOp maskOp : masksToErase) {
      maskOp.erase();
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
