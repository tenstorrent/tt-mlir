// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <algorithm>
#include <optional>

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_STABLEHLOUNROLLSTATICWHILEPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

namespace {

static std::optional<int64_t> getScalarIntConstant(Value value) {
  auto constantOp = value.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!constantOp) {
    return std::nullopt;
  }

  auto denseAttr = dyn_cast<DenseIntElementsAttr>(constantOp.getValue());
  if (!denseAttr || denseAttr.getNumElements() != 1) {
    return std::nullopt;
  }

  return denseAttr.getSplatValue<APInt>().getSExtValue();
}

static Value createScalarIntConstant(Location loc, Type type, int64_t value,
                                     IRRewriter &rewriter) {
  auto tensorType = cast<RankedTensorType>(type);
  auto intType = cast<IntegerType>(tensorType.getElementType());
  APInt intValue(intType.getWidth(), value, /*isSigned=*/true);
  auto attr = DenseIntElementsAttr::get(tensorType, intValue);
  return rewriter.create<mlir::stablehlo::ConstantOp>(loc, attr);
}

static bool isOneStepInductionReturn(Value value, BlockArgument inductionArg) {
  auto addOp = value.getDefiningOp<mlir::stablehlo::AddOp>();
  if (!addOp) {
    return false;
  }

  Value maybeStep;
  if (addOp.getLhs() == inductionArg) {
    maybeStep = addOp.getRhs();
  } else if (addOp.getRhs() == inductionArg) {
    maybeStep = addOp.getLhs();
  } else {
    return false;
  }

  std::optional<int64_t> step = getScalarIntConstant(maybeStep);
  return step && *step == 1;
}

static LogicalResult getStaticTripCount(mlir::stablehlo::WhileOp whileOp,
                                        int64_t maxTripCount,
                                        unsigned &inductionIndex,
                                        int64_t &startValue,
                                        int64_t &tripCount) {
  Block &condBlock = whileOp.getCond().front();
  Block &bodyBlock = whileOp.getBody().front();

  auto condReturn = dyn_cast<mlir::stablehlo::ReturnOp>(
      condBlock.getTerminator());
  auto bodyReturn = dyn_cast<mlir::stablehlo::ReturnOp>(
      bodyBlock.getTerminator());
  if (!condReturn || !bodyReturn || condReturn.getNumOperands() != 1 ||
      bodyReturn.getNumOperands() != whileOp.getNumOperands()) {
    return failure();
  }

  auto compareOp =
      condReturn.getOperand(0).getDefiningOp<mlir::stablehlo::CompareOp>();
  if (!compareOp ||
      compareOp.getComparisonDirection() !=
          mlir::stablehlo::ComparisonDirection::LT) {
    return failure();
  }

  auto condInductionArg = dyn_cast<BlockArgument>(compareOp.getLhs());
  if (!condInductionArg || condInductionArg.getOwner() != &condBlock) {
    return failure();
  }

  std::optional<int64_t> limit = getScalarIntConstant(compareOp.getRhs());
  if (!limit) {
    return failure();
  }

  inductionIndex = condInductionArg.getArgNumber();
  std::optional<int64_t> start =
      getScalarIntConstant(whileOp->getOperand(inductionIndex));
  if (!start) {
    return failure();
  }

  BlockArgument bodyInductionArg = bodyBlock.getArgument(inductionIndex);
  if (!isOneStepInductionReturn(bodyReturn.getOperand(inductionIndex),
                                bodyInductionArg)) {
    return failure();
  }

  int64_t computedTripCount = std::max<int64_t>(0, *limit - *start);
  if (computedTripCount > maxTripCount) {
    return failure();
  }

  startValue = *start;
  tripCount = computedTripCount;
  return success();
}

static void unrollWhile(mlir::stablehlo::WhileOp whileOp,
                        unsigned inductionIndex, int64_t startValue,
                        int64_t tripCount, IRRewriter &rewriter) {
  Block &bodyBlock = whileOp.getBody().front();
  auto bodyReturn =
      cast<mlir::stablehlo::ReturnOp>(bodyBlock.getTerminator());

  SmallVector<Value> carried(whileOp.getOperands());
  rewriter.setInsertionPoint(whileOp);

  for (int64_t i = 0; i < tripCount; ++i) {
    IRMapping mapping;
    for (auto [index, arg] : llvm::enumerate(bodyBlock.getArguments())) {
      if (index == inductionIndex) {
        Value indexConstant = createScalarIntConstant(
            whileOp.getLoc(), arg.getType(), startValue + i, rewriter);
        mapping.map(arg, indexConstant);
        continue;
      }
      mapping.map(arg, carried[index]);
    }

    for (Operation &op : bodyBlock.without_terminator()) {
      rewriter.clone(op, mapping);
    }

    for (auto [index, returned] : llvm::enumerate(bodyReturn.getOperands())) {
      if (index == inductionIndex) {
        carried[index] = createScalarIntConstant(
            whileOp.getLoc(), returned.getType(), startValue + i + 1, rewriter);
        continue;
      }
      carried[index] = mapping.lookupOrDefault(returned);
    }
  }

  rewriter.replaceOp(whileOp, carried);
}

class StableHLOUnrollStaticWhilePass
    : public impl::StableHLOUnrollStaticWhilePassBase<
          StableHLOUnrollStaticWhilePass> {
public:
  using impl::StableHLOUnrollStaticWhilePassBase<
      StableHLOUnrollStaticWhilePass>::StableHLOUnrollStaticWhilePassBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();
    IRRewriter rewriter(context);

    bool changed = false;
    do {
      changed = false;
      module.walk([&](mlir::stablehlo::WhileOp whileOp) {
        unsigned inductionIndex = 0;
        int64_t startValue = 0;
        int64_t tripCount = 0;
        if (failed(getStaticTripCount(whileOp, maxTripCount, inductionIndex,
                                      startValue, tripCount))) {
          return WalkResult::advance();
        }
        unrollWhile(whileOp, inductionIndex, startValue, tripCount, rewriter);
        changed = true;
        return WalkResult::interrupt();
      });
    } while (changed);
  }
};

} // namespace
} // namespace mlir::tt::stablehlo
