// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/APInt.h"

#include <optional>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRWHILETRIPCOUNT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

std::optional<llvm::APInt> matchScalarIntConstant(Value value);

mlir::IntegerType getScalarIntType(Value value) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!tensorType) {
    return nullptr;
  }
  return mlir::dyn_cast<mlir::IntegerType>(tensorType.getElementType());
}

// Frontends express loop bounds as 32- or 64-bit integers; a cast to any other
// width is more likely to be changing the value than to be bookkeeping.
bool isSupportedCastType(mlir::IntegerType type) {
  return type && (type.getWidth() == 32 || type.getWidth() == 64);
}

// Looks through a typecast a frontend left around a scalar constant. Only
// integer casts that change width between 32 and 64 bits or change signedness
// qualify, and only when the constant survives them: everything downstream
// reads these values as signed, so a cast that drops bits or flips the sign
// would silently change the trip count.
// Clang's static analyzer incorrectly models destruction of optional<APInt>
// (llvm/llvm-project#119415).
// NOLINTBEGIN(clang-analyzer-cplusplus.NewDelete)
std::optional<llvm::APInt>
matchTypecastIntConstant(ttir::TypecastOp typecastOp) {
  mlir::IntegerType inputType = getScalarIntType(typecastOp.getInput());
  mlir::IntegerType resultType = getScalarIntType(typecastOp.getResult());
  if (!isSupportedCastType(inputType) || !isSupportedCastType(resultType)) {
    return std::nullopt;
  }

  std::optional<llvm::APInt> input =
      matchScalarIntConstant(typecastOp.getInput());
  if (!input) {
    return std::nullopt;
  }

  llvm::APInt converted = inputType.isUnsigned()
                              ? input->zextOrTrunc(resultType.getWidth())
                              : input->sextOrTrunc(resultType.getWidth());
  if (resultType.isUnsigned() && converted.isNegative()) {
    return std::nullopt;
  }
  if (converted.sextOrTrunc(inputType.getWidth()) != *input) {
    return std::nullopt;
  }
  return converted;
}
// NOLINTEND(clang-analyzer-cplusplus.NewDelete)

// Resolves `value` to a single integer, looking through the shape-only ops and
// casts a frontend leaves around a scalar constant. Returns nullopt for
// anything that is not a compile-time-known scalar integer.
std::optional<llvm::APInt> matchScalarIntConstant(Value value) {
  while (Operation *definingOp = value.getDefiningOp()) {
    if (isa<ttir::ReshapeOp, ttir::BroadcastOp>(definingOp)) {
      value = definingOp->getOperand(0);
      continue;
    }
    if (auto typecastOp = mlir::dyn_cast<ttir::TypecastOp>(definingOp)) {
      return matchTypecastIntConstant(typecastOp);
    }
    break;
  }

  if (auto constantOp = value.getDefiningOp<ttir::ConstantOp>()) {
    auto elements =
        mlir::dyn_cast<mlir::DenseIntElementsAttr>(constantOp.getValueAttr());
    if (!elements || !elements.isSplat()) {
      return std::nullopt;
    }
    return elements.getSplatValue<llvm::APInt>();
  }

  // Canonicalization rewrites splat ttir.constant into ttir.full, so by the
  // time this pass runs most scalars are in this form.
  if (auto fullOp = value.getDefiningOp<ttir::FullOp>()) {
    if (auto fillValue =
            mlir::dyn_cast<mlir::IntegerAttr>(fullOp.getFillValueAttr())) {
      return fillValue.getValue();
    }
    return std::nullopt;
  }

  return std::nullopt;
}

// Resolves a value used inside one of the loop's regions to a loop-invariant
// integer constant.
//
// Region block arguments are `inits ++ captures`. Only the captures are
// loop-invariant; an init changes every iteration, so it can never be treated
// as a constant here.
std::optional<llvm::APInt> matchLoopInvariantConstant(ttir::WhileOp whileOp,
                                                      Value value) {
  if (auto blockArg = mlir::dyn_cast<BlockArgument>(value)) {
    if (blockArg.getOwner()->getParentOp() != whileOp.getOperation()) {
      return std::nullopt;
    }
    unsigned index = blockArg.getArgNumber();
    unsigned numInits = whileOp.getInits().size();
    if (index < numInits) {
      return std::nullopt;
    }
    return matchScalarIntConstant(whileOp.getCaptures()[index - numInits]);
  }
  return matchScalarIntConstant(value);
}

struct InductionComparison {
  unsigned inductionIndex;
  llvm::APInt limit;
  bool ascending;
  bool inclusive;
};

// Matches a condition region of the shape
//   ^cond(...): %p = ttir.<cmp>(%arg_k, <invariant const>)
//               ttir.yield %p
// The induction variable may be on either side; canonicalization commutes
// comparisons, so `lt(%arg_k, %limit)` often reaches us as `gt(%limit,
// %arg_k)`.
std::optional<InductionComparison> matchConditionRegion(ttir::WhileOp whileOp) {
  Block &block = whileOp.getCondBlock();
  Operation *compareOp =
      whileOp.getCondYield().getOperands().front().getDefiningOp();
  if (!compareOp || compareOp->getBlock() != &block) {
    return std::nullopt;
  }

  // `ascending` is expressed relative to the left-hand operand.
  bool ascending;
  bool inclusive;
  if (isa<ttir::LessThanOp>(compareOp)) {
    ascending = true;
    inclusive = false;
  } else if (isa<ttir::LessEqualOp>(compareOp)) {
    ascending = true;
    inclusive = true;
  } else if (isa<ttir::GreaterThanOp>(compareOp)) {
    ascending = false;
    inclusive = false;
  } else if (isa<ttir::GreaterEqualOp>(compareOp)) {
    ascending = false;
    inclusive = true;
  } else {
    return std::nullopt;
  }

  auto isInductionArg = [&](Value value) -> BlockArgument {
    auto blockArg = mlir::dyn_cast<BlockArgument>(value);
    if (!blockArg || blockArg.getOwner() != &block ||
        blockArg.getArgNumber() >= whileOp.getInits().size()) {
      return nullptr;
    }
    return blockArg;
  };

  BlockArgument inductionArg = isInductionArg(compareOp->getOperand(0));
  Value limitValue = compareOp->getOperand(1);
  if (!inductionArg) {
    inductionArg = isInductionArg(compareOp->getOperand(1));
    limitValue = compareOp->getOperand(0);
    // Swapping the operands mirrors the comparison.
    ascending = !ascending;
  }
  if (!inductionArg) {
    return std::nullopt;
  }

  std::optional<llvm::APInt> limit =
      matchLoopInvariantConstant(whileOp, limitValue);
  if (!limit) {
    return std::nullopt;
  }

  return InductionComparison{inductionArg.getArgNumber(), *limit, ascending,
                             inclusive};
}

// Matches the body's update of the induction variable:
//   ttir.yield ..., ttir.add(%arg_k, <invariant const>), ...
std::optional<llvm::APInt> matchInductionStep(ttir::WhileOp whileOp,
                                              unsigned inductionIndex) {
  Block &block = whileOp.getBodyBlock();
  Value yielded = whileOp.getBodyYield().getOperands()[inductionIndex];

  auto addOp = yielded.getDefiningOp<ttir::AddOp>();
  if (!addOp || addOp->getBlock() != &block) {
    return std::nullopt;
  }

  // Addition is commutative, so canonicalization may have put the induction
  // variable on either side.
  auto isInductionArg = [&](Value value) {
    auto blockArg = mlir::dyn_cast<BlockArgument>(value);
    return blockArg && blockArg.getOwner() == &block &&
           blockArg.getArgNumber() == inductionIndex;
  };

  if (isInductionArg(addOp->getOperand(0))) {
    return matchLoopInvariantConstant(whileOp, addOp->getOperand(1));
  }
  if (isInductionArg(addOp->getOperand(1))) {
    return matchLoopInvariantConstant(whileOp, addOp->getOperand(0));
  }
  return std::nullopt;
}

// Computes how many times the loop body runs, given a counted loop's
// parameters. Returns nullopt when the loop is not provably counted, e.g. when
// the step moves away from the limit and the loop would never terminate.
std::optional<int64_t> computeTripCount(const InductionComparison &comparison,
                                        const llvm::APInt &start,
                                        const llvm::APInt &step) {
  // Widen to a common signed width so the arithmetic below cannot overflow the
  // narrow tensor element type (typically 32-bit after normalization).
  constexpr unsigned kWideBits = 128;
  llvm::APInt begin = start.sext(kWideBits);
  llvm::APInt limit = comparison.limit.sext(kWideBits);
  llvm::APInt stride = step.sext(kWideBits);

  if (stride.isZero()) {
    return std::nullopt;
  }
  if (comparison.ascending != stride.isStrictlyPositive()) {
    // The induction variable moves away from the limit: either the loop never
    // terminates or it never runs, and we cannot tell which without also
    // proving the initial comparison. Leave it data-dependent.
    return std::nullopt;
  }

  // Normalize to a strictly-less-than comparison over a positive stride.
  if (comparison.inclusive) {
    limit += comparison.ascending ? 1 : -1;
  }
  llvm::APInt distance =
      comparison.ascending ? (limit - begin) : (begin - limit);
  llvm::APInt absStride = comparison.ascending ? stride : -stride;

  if (!distance.isStrictlyPositive()) {
    return 0;
  }

  // Ceiling division: the final iteration may overshoot the limit.
  llvm::APInt count = (distance + absStride - 1).sdiv(absStride);
  if (!count.isSignedIntN(63)) {
    return std::nullopt;
  }
  return count.getSExtValue();
}

std::optional<int64_t> analyzeTripCount(ttir::WhileOp whileOp) {
  std::optional<InductionComparison> comparison = matchConditionRegion(whileOp);
  if (!comparison) {
    return std::nullopt;
  }

  std::optional<llvm::APInt> start =
      matchScalarIntConstant(whileOp.getInits()[comparison->inductionIndex]);
  if (!start) {
    return std::nullopt;
  }

  std::optional<llvm::APInt> step =
      matchInductionStep(whileOp, comparison->inductionIndex);
  if (!step) {
    return std::nullopt;
  }

  return computeTripCount(*comparison, *start, *step);
}

class TTIRWhileTripCount
    : public impl::TTIRWhileTripCountBase<TTIRWhileTripCount> {
public:
  using impl::TTIRWhileTripCountBase<
      TTIRWhileTripCount>::TTIRWhileTripCountBase;

  void runOnOperation() final {
    getOperation()->walk([&](ttir::WhileOp whileOp) {
      if (whileOp.getTripCount()) {
        return;
      }
      if (std::optional<int64_t> tripCount = analyzeTripCount(whileOp)) {
        whileOp.setTripCount(*tripCount);
      }
    });
  }
};

} // namespace
} // namespace mlir::tt::ttir
