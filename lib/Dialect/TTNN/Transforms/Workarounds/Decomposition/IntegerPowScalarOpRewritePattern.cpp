// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/IntegerPowScalarOpRewritePattern.h"

#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Exponentiation by squaring needs ~2*log2(n) multiplies, so the cap is
// generous; it exists only to keep a pathological exponent from unrolling into
// an unreasonable amount of IR.
static constexpr int64_t kMaxPowUnrollExponent = 64;

LogicalResult IntegerPowScalarOpRewritePattern::matchAndRewrite(
    ttnn::PowScalarOp op, PatternRewriter &rewriter) const {
  // Only integer bases hit the tt-metal rejection; float pow is unaffected and
  // keeps using the native kernel.
  RankedTensorType inputType = op.getLhs().getType();
  RankedTensorType resultType = op.getResult().getType();
  if (!mlir::isa<IntegerType>(inputType.getElementType())) {
    return failure();
  }

  // The multiply chain reuses a single type for every intermediate, so bail if
  // the op isn't the shape-and-type-preserving form we expect.
  if (inputType != resultType) {
    return failure();
  }

  // A float exponent attribute on an integer tensor is not something this
  // decomposition can reason about; leave it alone.
  auto exponentAttr = mlir::dyn_cast<IntegerAttr>(op.getRhs());
  if (!exponentAttr) {
    return failure();
  }

  int64_t exponent = exponentAttr.getInt();
  // x**0 is 1, which needs a constant tensor rather than a multiply chain, so
  // it is left for tt-metal to reject. Negative exponents are not meaningful
  // in integer arithmetic and TTNN rejects them anyway.
  if (exponent < 1 || exponent > kMaxPowUnrollExponent) {
    return failure();
  }

  Location loc = op.getLoc();
  unsigned mulIndex = 0;
  auto multiply = [&](Value lhs, Value rhs) -> Value {
    return rewriter
        .create<ttnn::MultiplyOp>(
            ttmlir::utils::appendLocationSuffix(
                loc, "_int_pow_mul_" + std::to_string(mulIndex++)),
            resultType, lhs, rhs)
        .getResult();
  };

  // Exponentiation by squaring. For the overwhelmingly common x**2 this
  // reduces to a single multiply and no squaring step.
  Value result = nullptr;
  Value base = op.getLhs();
  for (int64_t remaining = exponent; remaining > 0; remaining >>= 1) {
    if (remaining & 1) {
      result = result ? multiply(result, base) : base;
    }
    if (remaining >> 1) {
      base = multiply(base, base);
    }
  }

  rewriter.replaceOp(op, result);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
