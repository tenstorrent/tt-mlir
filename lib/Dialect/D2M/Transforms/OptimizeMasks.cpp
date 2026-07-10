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
#include "llvm/ADT/SmallPtrSet.h"

#include <limits>
#include <optional>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MOPTIMIZEMASKS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Tracks values whose padded elements are known in the OOB lattice.
using OOBStateMap = llvm::DenseMap<Value, ttcore::OOBVal>;

static bool isConcreteOOBVal(ttcore::OOBVal value) {
  return value != ttcore::OOBVal::Undef;
}

static ttcore::OOBVal getConstantOOBVal(Attribute attr) {
  if (!attr) {
    return ttcore::OOBVal::Undef;
  }

  if (auto denseAttr = dyn_cast<DenseElementsAttr>(attr)) {
    if (!denseAttr.isSplat()) {
      return ttcore::OOBVal::Undef;
    }
    return getConstantOOBVal(denseAttr.getSplatValue<Attribute>());
  }

  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    APInt value = intAttr.getValue();
    if (value.isZero()) {
      return ttcore::OOBVal::Zero;
    }
    if (value.isOne()) {
      return ttcore::OOBVal::One;
    }
    if (value == APInt::getSignedMinValue(value.getBitWidth())) {
      return ttcore::OOBVal::NegInf;
    }
    if (value == APInt::getSignedMaxValue(value.getBitWidth())) {
      return ttcore::OOBVal::Inf;
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

static std::optional<double> getConstantAsDouble(Attribute attr) {
  if (!attr) {
    return std::nullopt;
  }

  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    APInt value = intAttr.getValue();
    if (value.getBitWidth() > 63) {
      return std::nullopt;
    }
    return static_cast<double>(value.getSExtValue());
  }

  if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    return floatAttr.getValue().convertToDouble();
  }

  return std::nullopt;
}

static ttcore::OOBVal getStateOrUndef(const OOBStateMap &states, Value value) {
  auto it = states.find(value);
  if (it != states.end()) {
    return it->second;
  }

  // Constants do not need explicit state entries.
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    return getConstantOOBVal(constant.getValue());
  }

  // Absence from the state map is untracked. For compatibility checks this
  // collapses to undef: safe for an undef target, unsafe for a concrete fill.
  return ttcore::OOBVal::Undef;
}

static std::optional<double> getOOBValAsDouble(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::Zero:
    return 0.0;
  case ttcore::OOBVal::One:
    return 1.0;
  case ttcore::OOBVal::Inf:
    return std::numeric_limits<double>::infinity();
  case ttcore::OOBVal::NegInf:
    return -std::numeric_limits<double>::infinity();
  case ttcore::OOBVal::Undef:
    return std::nullopt;
  }
  llvm_unreachable("unknown OOB value");
}

static std::optional<bool> getOOBValAsBool(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::Zero:
    return false;
  case ttcore::OOBVal::One:
  case ttcore::OOBVal::Inf:
  case ttcore::OOBVal::NegInf:
    return true;
  case ttcore::OOBVal::Undef:
    return std::nullopt;
  }
  llvm_unreachable("unknown OOB value");
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

// These transfer functions intentionally stay in the five-value OOB lattice.
// If an operation produces a real value we cannot represent exactly, such as
// -1, 0.5, or 2, it collapses to undef rather than inventing a new state.
static ttcore::OOBVal recipOOBVal(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::One:
    return ttcore::OOBVal::One;
  case ttcore::OOBVal::Inf:
  case ttcore::OOBVal::NegInf:
    return ttcore::OOBVal::Zero;
  case ttcore::OOBVal::Zero:
  case ttcore::OOBVal::Undef:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown OOB value");
}

static ttcore::OOBVal expOOBVal(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::Zero:
    return ttcore::OOBVal::One;
  case ttcore::OOBVal::Inf:
    return ttcore::OOBVal::Inf;
  case ttcore::OOBVal::NegInf:
    return ttcore::OOBVal::Zero;
  case ttcore::OOBVal::One:
  case ttcore::OOBVal::Undef:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown OOB value");
}

static ttcore::OOBVal expm1OOBVal(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::Zero:
    return ttcore::OOBVal::Zero;
  case ttcore::OOBVal::Inf:
    return ttcore::OOBVal::Inf;
  case ttcore::OOBVal::One:
  case ttcore::OOBVal::NegInf:
  case ttcore::OOBVal::Undef:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown OOB value");
}

static ttcore::OOBVal logOOBVal(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::Zero:
    return ttcore::OOBVal::NegInf;
  case ttcore::OOBVal::One:
    return ttcore::OOBVal::Zero;
  case ttcore::OOBVal::Inf:
    return ttcore::OOBVal::Inf;
  case ttcore::OOBVal::NegInf:
  case ttcore::OOBVal::Undef:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown OOB value");
}

static ttcore::OOBVal log1pOOBVal(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::Zero:
    return ttcore::OOBVal::Zero;
  case ttcore::OOBVal::Inf:
    return ttcore::OOBVal::Inf;
  case ttcore::OOBVal::One:
  case ttcore::OOBVal::NegInf:
  case ttcore::OOBVal::Undef:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown OOB value");
}

static ttcore::OOBVal sqrtOOBVal(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::Zero:
  case ttcore::OOBVal::One:
  case ttcore::OOBVal::Inf:
    return value;
  case ttcore::OOBVal::NegInf:
  case ttcore::OOBVal::Undef:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown OOB value");
}

static ttcore::OOBVal rsqrtOOBVal(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::Zero:
    return ttcore::OOBVal::Inf;
  case ttcore::OOBVal::One:
    return ttcore::OOBVal::One;
  case ttcore::OOBVal::Inf:
    return ttcore::OOBVal::Zero;
  case ttcore::OOBVal::NegInf:
  case ttcore::OOBVal::Undef:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown OOB value");
}

static ttcore::OOBVal squareOOBVal(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::Zero:
  case ttcore::OOBVal::One:
    return value;
  case ttcore::OOBVal::Inf:
  case ttcore::OOBVal::NegInf:
    return ttcore::OOBVal::Inf;
  case ttcore::OOBVal::Undef:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown OOB value");
}

static ttcore::OOBVal sinOOBVal(ttcore::OOBVal value) {
  return value == ttcore::OOBVal::Zero ? ttcore::OOBVal::Zero
                                       : ttcore::OOBVal::Undef;
}

static ttcore::OOBVal cosOOBVal(ttcore::OOBVal value) {
  return value == ttcore::OOBVal::Zero ? ttcore::OOBVal::One
                                       : ttcore::OOBVal::Undef;
}

static ttcore::OOBVal acosOOBVal(ttcore::OOBVal value) {
  return value == ttcore::OOBVal::One ? ttcore::OOBVal::Zero
                                      : ttcore::OOBVal::Undef;
}

static ttcore::OOBVal tanhOOBVal(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::Zero:
    return ttcore::OOBVal::Zero;
  case ttcore::OOBVal::Inf:
    return ttcore::OOBVal::One;
  case ttcore::OOBVal::One:
  case ttcore::OOBVal::NegInf:
  case ttcore::OOBVal::Undef:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown OOB value");
}

static ttcore::OOBVal sigmoidOOBVal(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::Inf:
    return ttcore::OOBVal::One;
  case ttcore::OOBVal::NegInf:
    return ttcore::OOBVal::Zero;
  case ttcore::OOBVal::Zero:
  case ttcore::OOBVal::One:
  case ttcore::OOBVal::Undef:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown OOB value");
}

static ttcore::OOBVal reluOOBVal(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::Zero:
  case ttcore::OOBVal::One:
  case ttcore::OOBVal::Inf:
    return value;
  case ttcore::OOBVal::NegInf:
    return ttcore::OOBVal::Zero;
  case ttcore::OOBVal::Undef:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown OOB value");
}

static ttcore::OOBVal geluOOBVal(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::Zero:
    return ttcore::OOBVal::Zero;
  case ttcore::OOBVal::Inf:
    return ttcore::OOBVal::Inf;
  case ttcore::OOBVal::NegInf:
    return ttcore::OOBVal::Zero;
  case ttcore::OOBVal::One:
  case ttcore::OOBVal::Undef:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown OOB value");
}

static ttcore::OOBVal seluOOBVal(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::Zero:
    return ttcore::OOBVal::Zero;
  case ttcore::OOBVal::Inf:
    return ttcore::OOBVal::Inf;
  case ttcore::OOBVal::One:
  case ttcore::OOBVal::NegInf:
  case ttcore::OOBVal::Undef:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown OOB value");
}

static ttcore::OOBVal signOOBVal(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::Zero:
    return ttcore::OOBVal::Zero;
  case ttcore::OOBVal::One:
  case ttcore::OOBVal::Inf:
    return ttcore::OOBVal::One;
  case ttcore::OOBVal::NegInf:
  case ttcore::OOBVal::Undef:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown OOB value");
}

static ttcore::OOBVal signbitOOBVal(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::One:
  case ttcore::OOBVal::Inf:
    return ttcore::OOBVal::Zero;
  case ttcore::OOBVal::NegInf:
    return ttcore::OOBVal::One;
  case ttcore::OOBVal::Zero:
  case ttcore::OOBVal::Undef:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown OOB value");
}

static ttcore::OOBVal fracOOBVal(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::Zero:
  case ttcore::OOBVal::One:
    return ttcore::OOBVal::Zero;
  case ttcore::OOBVal::Inf:
  case ttcore::OOBVal::NegInf:
  case ttcore::OOBVal::Undef:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown OOB value");
}

static ttcore::OOBVal positivePredicateOOBVal(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::One:
  case ttcore::OOBVal::Inf:
    return ttcore::OOBVal::One;
  case ttcore::OOBVal::Zero:
  case ttcore::OOBVal::NegInf:
    return ttcore::OOBVal::Zero;
  case ttcore::OOBVal::Undef:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown OOB value");
}

static ttcore::OOBVal nonNegativePredicateOOBVal(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::Zero:
  case ttcore::OOBVal::One:
  case ttcore::OOBVal::Inf:
    return ttcore::OOBVal::One;
  case ttcore::OOBVal::NegInf:
    return ttcore::OOBVal::Zero;
  case ttcore::OOBVal::Undef:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown OOB value");
}

static ttcore::OOBVal negativePredicateOOBVal(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::NegInf:
    return ttcore::OOBVal::One;
  case ttcore::OOBVal::Zero:
  case ttcore::OOBVal::One:
  case ttcore::OOBVal::Inf:
    return ttcore::OOBVal::Zero;
  case ttcore::OOBVal::Undef:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown OOB value");
}

static ttcore::OOBVal nonPositivePredicateOOBVal(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::Zero:
  case ttcore::OOBVal::NegInf:
    return ttcore::OOBVal::One;
  case ttcore::OOBVal::One:
  case ttcore::OOBVal::Inf:
    return ttcore::OOBVal::Zero;
  case ttcore::OOBVal::Undef:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown OOB value");
}

// Keep IEEE-invalid or target-sensitive combinations as undef. Examples include
// `inf + -inf`, `inf - inf`, and `0 * inf`; preserving undef keeps explicit
// masks in place until OOB arithmetic can be made target-specific.
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
  if (lhs == ttcore::OOBVal::Zero || rhs == ttcore::OOBVal::Zero) {
    ttcore::OOBVal other = lhs == ttcore::OOBVal::Zero ? rhs : lhs;

    // Do not let zero absorb unknown or infinite padding. IEEE 754 defines
    // zero times infinity as NaN, and TT hardware behavior may vary by target
    // or data format. Until this is made target-specific, only claim the
    // product is definitely zero when the other OOB value is representably
    // finite in this lattice; otherwise preserve undef so a later explicit
    // mask is not optimized away.
    return other == ttcore::OOBVal::Zero || other == ttcore::OOBVal::One
               ? ttcore::OOBVal::Zero
               : ttcore::OOBVal::Undef;
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
  if (lhs == ttcore::OOBVal::Zero && isConcreteOOBVal(rhs) &&
      rhs != ttcore::OOBVal::Zero) {
    return ttcore::OOBVal::Zero;
  }
  if (lhs == ttcore::OOBVal::One &&
      (rhs == ttcore::OOBVal::Inf || rhs == ttcore::OOBVal::NegInf)) {
    return ttcore::OOBVal::Zero;
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

static ttcore::OOBVal powOOBVals(ttcore::OOBVal lhs, ttcore::OOBVal rhs) {
  if (rhs == ttcore::OOBVal::Zero) {
    return ttcore::OOBVal::One;
  }
  if (rhs == ttcore::OOBVal::One) {
    return lhs;
  }
  if (lhs == ttcore::OOBVal::One) {
    return ttcore::OOBVal::One;
  }
  return ttcore::OOBVal::Undef;
}

static ttcore::OOBVal bitwiseAndOOBVals(ttcore::OOBVal lhs,
                                        ttcore::OOBVal rhs) {
  if (lhs == ttcore::OOBVal::Zero || rhs == ttcore::OOBVal::Zero) {
    return ttcore::OOBVal::Zero;
  }
  if (lhs == ttcore::OOBVal::One && rhs == ttcore::OOBVal::One) {
    return ttcore::OOBVal::One;
  }
  return ttcore::OOBVal::Undef;
}

static ttcore::OOBVal bitwiseOrOOBVals(ttcore::OOBVal lhs, ttcore::OOBVal rhs) {
  if (lhs == ttcore::OOBVal::Zero) {
    return rhs;
  }
  if (rhs == ttcore::OOBVal::Zero) {
    return lhs;
  }
  if (lhs == ttcore::OOBVal::One && rhs == ttcore::OOBVal::One) {
    return ttcore::OOBVal::One;
  }
  return ttcore::OOBVal::Undef;
}

static ttcore::OOBVal bitwiseXorOOBVals(ttcore::OOBVal lhs,
                                        ttcore::OOBVal rhs) {
  if (lhs == ttcore::OOBVal::Zero) {
    return rhs;
  }
  if (rhs == ttcore::OOBVal::Zero) {
    return lhs;
  }
  if (lhs == ttcore::OOBVal::One && rhs == ttcore::OOBVal::One) {
    return ttcore::OOBVal::Zero;
  }
  return ttcore::OOBVal::Undef;
}

static ttcore::OOBVal shiftOOBVals(ttcore::OOBVal lhs, ttcore::OOBVal rhs) {
  if (lhs == ttcore::OOBVal::Zero) {
    return ttcore::OOBVal::Zero;
  }
  if (rhs == ttcore::OOBVal::Zero) {
    return lhs;
  }
  return ttcore::OOBVal::Undef;
}

enum class KnownZero {
  Unknown,
  Zero,
  NonZero,
};

static KnownZero getConstantKnownZero(Attribute attr) {
  if (!attr) {
    return KnownZero::Unknown;
  }

  if (auto denseAttr = dyn_cast<DenseElementsAttr>(attr)) {
    if (!denseAttr.isSplat()) {
      return KnownZero::Unknown;
    }
    return getConstantKnownZero(denseAttr.getSplatValue<Attribute>());
  }

  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    return intAttr.getValue().isZero() ? KnownZero::Zero : KnownZero::NonZero;
  }

  if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    APFloat value = floatAttr.getValue();
    if (value.isNaN()) {
      return KnownZero::Unknown;
    }
    return value.isZero() ? KnownZero::Zero : KnownZero::NonZero;
  }

  return KnownZero::Unknown;
}

static KnownZero getKnownZero(ttcore::OOBVal value) {
  switch (value) {
  case ttcore::OOBVal::Zero:
    return KnownZero::Zero;
  case ttcore::OOBVal::One:
  case ttcore::OOBVal::Inf:
  case ttcore::OOBVal::NegInf:
    return KnownZero::NonZero;
  case ttcore::OOBVal::Undef:
    return KnownZero::Unknown;
  }
  llvm_unreachable("unknown OOB value");
}

static ttcore::OOBVal getZeroPredicateOOBVal(KnownZero knownZero) {
  switch (knownZero) {
  case KnownZero::Zero:
    return ttcore::OOBVal::One;
  case KnownZero::NonZero:
    return ttcore::OOBVal::Zero;
  case KnownZero::Unknown:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown zero state");
}

static ttcore::OOBVal getNonZeroPredicateOOBVal(KnownZero knownZero) {
  switch (knownZero) {
  case KnownZero::Zero:
    return ttcore::OOBVal::Zero;
  case KnownZero::NonZero:
    return ttcore::OOBVal::One;
  case KnownZero::Unknown:
    return ttcore::OOBVal::Undef;
  }
  llvm_unreachable("unknown zero state");
}

static KnownZero inferKnownZero(Value value, const OOBStateMap &states,
                                unsigned depth = 0);

static KnownZero inferAddKnownZero(KnownZero lhsZero, KnownZero rhsZero,
                                   ttcore::OOBVal lhs, ttcore::OOBVal rhs) {
  if (lhsZero == KnownZero::Zero) {
    return rhsZero;
  }
  if (rhsZero == KnownZero::Zero) {
    return lhsZero;
  }

  if (lhsZero != KnownZero::NonZero || rhsZero != KnownZero::NonZero ||
      !isConcreteOOBVal(lhs) || !isConcreteOOBVal(rhs)) {
    return KnownZero::Unknown;
  }

  if ((lhs == ttcore::OOBVal::Inf && rhs == ttcore::OOBVal::NegInf) ||
      (lhs == ttcore::OOBVal::NegInf && rhs == ttcore::OOBVal::Inf)) {
    return KnownZero::Unknown;
  }
  return KnownZero::NonZero;
}

static KnownZero inferSubKnownZero(KnownZero lhsZero, KnownZero rhsZero,
                                   ttcore::OOBVal lhs, ttcore::OOBVal rhs) {
  if (rhsZero == KnownZero::Zero) {
    return lhsZero;
  }
  if (lhsZero == KnownZero::Zero) {
    return rhsZero == KnownZero::NonZero ? KnownZero::NonZero
                                         : KnownZero::Unknown;
  }

  if (lhs == ttcore::OOBVal::One && rhs == ttcore::OOBVal::One) {
    return KnownZero::Zero;
  }
  if (lhs == ttcore::OOBVal::Inf) {
    return rhs == ttcore::OOBVal::Inf ? KnownZero::Unknown : KnownZero::NonZero;
  }
  if (lhs == ttcore::OOBVal::NegInf) {
    return rhs == ttcore::OOBVal::NegInf ? KnownZero::Unknown
                                         : KnownZero::NonZero;
  }
  if (rhs == ttcore::OOBVal::Inf || rhs == ttcore::OOBVal::NegInf) {
    return lhs == ttcore::OOBVal::Undef ? KnownZero::Unknown
                                        : KnownZero::NonZero;
  }

  return KnownZero::Unknown;
}

static KnownZero inferMulKnownZero(KnownZero lhsZero, KnownZero rhsZero) {
  if (lhsZero == KnownZero::Zero || rhsZero == KnownZero::Zero) {
    return KnownZero::Zero;
  }
  if (lhsZero == KnownZero::NonZero && rhsZero == KnownZero::NonZero) {
    return KnownZero::NonZero;
  }
  return KnownZero::Unknown;
}

static KnownZero inferSelectKnownZero(KnownZero conditionZero,
                                      KnownZero trueValueZero,
                                      KnownZero falseValueZero) {
  if (conditionZero == KnownZero::Zero) {
    return falseValueZero;
  }
  if (conditionZero == KnownZero::NonZero) {
    return trueValueZero;
  }
  return trueValueZero == falseValueZero ? trueValueZero : KnownZero::Unknown;
}

static KnownZero inferKnownZero(Value value, const OOBStateMap &states,
                                unsigned depth) {
  ttcore::OOBVal state = getStateOrUndef(states, value);
  KnownZero knownZero = getKnownZero(state);
  if (knownZero != KnownZero::Unknown) {
    return knownZero;
  }

  Operation *op = value.getDefiningOp();
  if (!op) {
    return KnownZero::Unknown;
  }

  if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
    return getConstantKnownZero(constant.getValue());
  }

  if (depth >= 4) {
    return KnownZero::Unknown;
  }

  if (isa<TileLogicalNotOp, TileEqzOp>(op)) {
    return getKnownZero(getZeroPredicateOOBVal(
        inferKnownZero(op->getOperand(0), states, depth + 1)));
  }
  if (isa<TileNezOp>(op)) {
    return getKnownZero(getNonZeroPredicateOOBVal(
        inferKnownZero(op->getOperand(0), states, depth + 1)));
  }

  if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
    return inferSelectKnownZero(
        inferKnownZero(selectOp.getCondition(), states, depth + 1),
        inferKnownZero(selectOp.getTrueValue(), states, depth + 1),
        inferKnownZero(selectOp.getFalseValue(), states, depth + 1));
  }

  if (op->getNumOperands() < 2) {
    return KnownZero::Unknown;
  }

  KnownZero lhsZero = inferKnownZero(op->getOperand(0), states, depth + 1);
  KnownZero rhsZero = inferKnownZero(op->getOperand(1), states, depth + 1);
  ttcore::OOBVal lhs = getStateOrUndef(states, op->getOperand(0));
  ttcore::OOBVal rhs = getStateOrUndef(states, op->getOperand(1));

  if (isa<TileAddOp, arith::AddFOp, arith::AddIOp>(op)) {
    return inferAddKnownZero(lhsZero, rhsZero, lhs, rhs);
  }
  if (isa<TileSubOp, arith::SubFOp, arith::SubIOp>(op)) {
    return inferSubKnownZero(lhsZero, rhsZero, lhs, rhs);
  }
  if (isa<TileMulOp, arith::MulFOp, arith::MulIOp>(op)) {
    return inferMulKnownZero(lhsZero, rhsZero);
  }
  if (isa<TileWhereOp>(op)) {
    return inferSelectKnownZero(
        lhsZero, rhsZero, inferKnownZero(op->getOperand(2), states, depth + 1));
  }

  return KnownZero::Unknown;
}

static ttcore::OOBVal selectOOBVals(ttcore::OOBVal trueValue,
                                    ttcore::OOBVal falseValue) {
  return trueValue == falseValue ? trueValue : ttcore::OOBVal::Undef;
}

static ttcore::OOBVal selectOOBVals(ttcore::OOBVal condition,
                                    ttcore::OOBVal trueValue,
                                    ttcore::OOBVal falseValue) {
  if (std::optional<bool> conditionValue = getOOBValAsBool(condition)) {
    return *conditionValue ? trueValue : falseValue;
  }
  return selectOOBVals(trueValue, falseValue);
}

static ttcore::OOBVal clampOOBVal(ttcore::OOBVal value, Attribute minAttr,
                                  Attribute maxAttr) {
  std::optional<double> maybeValue = getOOBValAsDouble(value);
  std::optional<double> maybeMin = getConstantAsDouble(minAttr);
  std::optional<double> maybeMax = getConstantAsDouble(maxAttr);
  if (!maybeValue || !maybeMin || !maybeMax) {
    return ttcore::OOBVal::Undef;
  }

  double clamped = *maybeValue;
  if (clamped < *maybeMin) {
    clamped = *maybeMin;
  }
  if (clamped > *maybeMax) {
    clamped = *maybeMax;
  }

  if (clamped == 0.0) {
    return ttcore::OOBVal::Zero;
  }
  if (clamped == 1.0) {
    return ttcore::OOBVal::One;
  }
  if (clamped == std::numeric_limits<double>::infinity()) {
    return ttcore::OOBVal::Inf;
  }
  if (clamped == -std::numeric_limits<double>::infinity()) {
    return ttcore::OOBVal::NegInf;
  }
  return ttcore::OOBVal::Undef;
}

static ttcore::OOBVal inferElementwiseOOBVal(Operation *op,
                                             const OOBStateMap &states) {
  // Model each supported op as a transfer function over the OOB lattice.
  if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
    return selectOOBVals(getStateOrUndef(states, selectOp.getCondition()),
                         getStateOrUndef(states, selectOp.getTrueValue()),
                         getStateOrUndef(states, selectOp.getFalseValue()));
  }

  if (op->getNumOperands() == 0) {
    return ttcore::OOBVal::Undef;
  }

  ttcore::OOBVal input = getStateOrUndef(states, op->getOperand(0));
  if (isa<TileTypecastOp, TileBcastOp, TileTransposeOp, DstReinterpretCastOp>(
          op)) {
    return input;
  }
  if (isa<TileNegativeOp, arith::NegFOp>(op)) {
    return negateOOBVal(input);
  }
  if (isa<TileAbsOp>(op)) {
    return absOOBVal(input);
  }
  if (isa<TileRecipOp>(op)) {
    return recipOOBVal(input);
  }
  if (isa<TileExpOp, TileExp2Op>(op)) {
    return expOOBVal(input);
  }
  if (isa<TileExpm1Op>(op)) {
    return expm1OOBVal(input);
  }
  if (isa<TileLogOp>(op)) {
    return logOOBVal(input);
  }
  if (isa<TileLog1pOp>(op)) {
    return log1pOOBVal(input);
  }
  if (isa<TileSqrtOp>(op)) {
    return sqrtOOBVal(input);
  }
  if (isa<TileRsqrtOp>(op)) {
    return rsqrtOOBVal(input);
  }
  if (isa<TileSquareOp>(op)) {
    return squareOOBVal(input);
  }
  if (isa<TileSinOp, TileTanOp, TileAtanOp, TileAsinOp>(op)) {
    return sinOOBVal(input);
  }
  if (isa<TileCosOp>(op)) {
    return cosOOBVal(input);
  }
  if (isa<TileAcosOp>(op)) {
    return acosOOBVal(input);
  }
  if (isa<TileTanhOp>(op)) {
    return tanhOOBVal(input);
  }
  if (isa<TileSigmoidOp, TileHardsigmoidOp>(op)) {
    return sigmoidOOBVal(input);
  }
  if (isa<TileReluOp>(op)) {
    return reluOOBVal(input);
  }
  if (isa<TileGeluOp, TileSiluOp>(op)) {
    return geluOOBVal(input);
  }
  if (isa<TileSeluOp>(op)) {
    return seluOOBVal(input);
  }
  if (isa<TileSoftsignOp, TileErfOp>(op)) {
    return tanhOOBVal(input);
  }
  if (isa<TileErfcOp>(op)) {
    if (input == ttcore::OOBVal::Zero) {
      return ttcore::OOBVal::One;
    }
    if (input == ttcore::OOBVal::Inf) {
      return ttcore::OOBVal::Zero;
    }
    return ttcore::OOBVal::Undef;
  }
  if (isa<TileSignOp>(op)) {
    return signOOBVal(input);
  }
  if (isa<TileSignbitOp>(op)) {
    return signbitOOBVal(input);
  }
  if (isa<TileCeilOp, TileFloorOp, TileTruncOp>(op)) {
    return input;
  }
  if (isa<TileFracOp>(op)) {
    return fracOOBVal(input);
  }
  if (isa<TileLogicalNotOp, TileEqzOp>(op)) {
    return getZeroPredicateOOBVal(inferKnownZero(op->getOperand(0), states));
  }
  if (isa<TileNezOp>(op)) {
    return getNonZeroPredicateOOBVal(inferKnownZero(op->getOperand(0), states));
  }
  if (isa<TileGtzOp>(op)) {
    return positivePredicateOOBVal(input);
  }
  if (isa<TileGezOp>(op)) {
    return nonNegativePredicateOOBVal(input);
  }
  if (isa<TileLtzOp>(op)) {
    return negativePredicateOOBVal(input);
  }
  if (isa<TileLezOp>(op)) {
    return nonPositivePredicateOOBVal(input);
  }
  if (isa<TileClampScalarOp>(op)) {
    return clampOOBVal(input, op->getAttr("min"), op->getAttr("max"));
  }
  if (isa<TileFillOp>(op)) {
    return input;
  }

  if (op->getNumOperands() < 2) {
    return ttcore::OOBVal::Undef;
  }

  ttcore::OOBVal lhs = input;
  ttcore::OOBVal rhs = getStateOrUndef(states, op->getOperand(1));
  if (isa<TileAddOp, arith::AddFOp, arith::AddIOp>(op)) {
    return addOOBVals(lhs, rhs);
  }
  if (isa<TileSubOp, arith::SubFOp, arith::SubIOp>(op)) {
    return subOOBVals(lhs, rhs);
  }
  if (isa<TileMulOp, arith::MulFOp, arith::MulIOp>(op)) {
    return mulOOBVals(lhs, rhs);
  }
  if (isa<TileDivOp, arith::DivFOp, arith::DivSIOp, arith::DivUIOp>(op)) {
    return divOOBVals(lhs, rhs);
  }
  if (isa<TilePowOp>(op)) {
    return powOOBVals(lhs, rhs);
  }
  if (isa<TileMaximumOp, arith::MaximumFOp, arith::MaxNumFOp, arith::MaxSIOp,
          arith::MaxUIOp>(op)) {
    return maxOOBVals(lhs, rhs);
  }
  if (isa<TileMinimumOp, arith::MinimumFOp, arith::MinNumFOp, arith::MinSIOp,
          arith::MinUIOp>(op)) {
    return minOOBVals(lhs, rhs);
  }
  if (isa<TileBitwiseAndOp>(op)) {
    return bitwiseAndOOBVals(lhs, rhs);
  }
  if (isa<TileBitwiseOrOp>(op)) {
    return bitwiseOrOOBVals(lhs, rhs);
  }
  if (isa<TileBitwiseXorOp>(op)) {
    return bitwiseXorOOBVals(lhs, rhs);
  }
  if (isa<TileLogicalLeftShiftOp, TileLogicalRightShiftOp, TileRightShiftOp>(
          op)) {
    return shiftOOBVals(lhs, rhs);
  }
  if (isa<TileWhereOp>(op)) {
    return selectOOBVals(lhs, getStateOrUndef(states, op->getOperand(1)),
                         getStateOrUndef(states, op->getOperand(2)));
  }
  if (isa<TileMatmulOp>(op)) {
    ttcore::OOBVal accumulator = getStateOrUndef(states, op->getOperand(2));
    return (lhs == ttcore::OOBVal::Zero || rhs == ttcore::OOBVal::Zero)
               ? accumulator
               : ttcore::OOBVal::Undef;
  }
  if (isa<TileReduceSumOp, TileReduceMeanOp>(op)) {
    ttcore::OOBVal accumulator = getStateOrUndef(states, op->getOperand(2));
    return addOOBVals(mulOOBVals(lhs, rhs), accumulator);
  }
  if (isa<TileReduceMaxOp>(op)) {
    ttcore::OOBVal accumulator = getStateOrUndef(states, op->getOperand(2));
    return maxOOBVals(mulOOBVals(lhs, rhs), accumulator);
  }
  if (isa<TileSFPUReduceSumOp>(op)) {
    return addOOBVals(lhs, rhs);
  }
  if (isa<TileSFPUReduceMaxOp>(op)) {
    return maxOOBVals(lhs, rhs);
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

  // Scan the body with operands mapped to block args, then copy yielded states
  // back to the linalg results.
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
  // Each region gets a local state map seeded from the generic operands and
  // block args. Region yields become candidate states for the op results.
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
    // If multiple regions yield different OOB states for the same result, the
    // merged result is unknown.
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

  // Unhandled result-producing ops become undef through the elementwise table.
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
    // Metal layouts expose per-logical-dim alignments directly.
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
  // For generic device layouts, only the innermost tile matrix dimensions can
  // introduce maskable padding here.
  // The last two physical dimensions are logical matrix rows/cols:
  //   rows = gridY * shardRows * tileRows
  //   cols = gridX * shardCols * tileCols
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

static void eraseDeadTensorEmptyOps(ArrayRef<EmptyOp> emptyOps) {
  llvm::SmallPtrSet<Operation *, 8> seen;
  for (EmptyOp emptyOp : emptyOps) {
    if (!seen.insert(emptyOp).second) {
      continue;
    }

    // Tensor d2m.empty ops are destination placeholders. Once the mask using
    // one is erased, the placeholder should not be left for canonicalization.
    if (emptyOp->use_empty()) {
      emptyOp.erase();
    }
  }
}

class D2MOptimizeMasksPass
    : public impl::D2MOptimizeMasksBase<D2MOptimizeMasksPass> {
public:
  using impl::D2MOptimizeMasksBase<D2MOptimizeMasksPass>::D2MOptimizeMasksBase;

  void runOnOperation() override {
    OOBStateMap states;
    SmallVector<MaskOp> masksToErase;
    SmallVector<EmptyOp> maybeDeadMaskOutputs;

    // Pre-order scan: producer ops update `states`; each mask is checked
    // against the state known for its input at that point.
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
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
          if (auto output = maskOp.getOutput().getDefiningOp<EmptyOp>();
              output && isa<RankedTensorType>(output.getResult().getType())) {
            maybeDeadMaskOutputs.push_back(output);
          }
          maskOp.getResult().replaceAllUsesWith(maskOp.getInput());
          masksToErase.push_back(maskOp);
        } else {
          states[maskOp.getResult()] = required;
        }
        return WalkResult::advance();
      }

      if (auto genericOp = dyn_cast<GenericOp>(op)) {
        propagateGenericOp(genericOp, states);
        return WalkResult::skip();
      }

      if (auto viewLayout = dyn_cast<ViewLayoutOp>(op)) {
        states[viewLayout.getResult()] =
            getStateOrUndef(states, viewLayout.getInput());
        return WalkResult::advance();
      }

      if (auto toLayout = dyn_cast<ToLayoutOp>(op)) {
        states[toLayout.getResult(0)] = ttcore::OOBVal::Undef;
      }
      return WalkResult::advance();
    });

    for (MaskOp maskOp : masksToErase) {
      maskOp.erase();
    }
    eraseDeadTensorEmptyOps(maybeDeadMaskOutputs);
  }
};

} // namespace
} // namespace mlir::tt::d2m
