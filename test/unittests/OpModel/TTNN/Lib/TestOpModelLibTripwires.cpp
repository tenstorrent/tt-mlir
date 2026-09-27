// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Tripwire tests: pass today, expected to fail when a specific tt-metal
// change lands.  Each test's comment should name the metal issue/PR it
// tracks.  Kept in a separate target so it compiles fast and the file can be
// removed (or individual tripwires dropped) once the upstream change arrives.
//
// The PagedUpdateCacheOpWrongGrid tripwire lived here until tt-metal #45016
// landed; see git history for an example of the pattern.

#include "OpModelFixture.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace mlir::tt::ttnn::op_model {

class OpModelTripwireTest : public OpModelFixture {};

// tt-metal rejects integer inputs to float-only unary ops as of
// f0025ce5f30 ("#56938: Reject integer inputs for float-only unary ops",
// https://github.com/tenstorrent/tt-metal/pull/56980).  POWER and
// POWER_ITERATIVE are not in unary_op_supports_integer_dtype(), so an int32
// pow falls to that switch's `default: return false` and the query fails.
//
// tt-mlir works around this by rewriting pow_scalar(int, const n) into a
// chain of multiplies (IntegerPowScalarOpRewritePattern).
//
// tt-metal PR #56860 ("[Bug fix] ttnn.pow: compute integer pow exactly
// instead of reading int bits as float") implements the same
// exponentiation-by-squaring in metal's composite layer, and additionally
// handles n == 0 and n < 0, which our pattern deliberately bails on.  When
// that PR is uplifted this test starts failing.
//
// TODO(tt-metal#56860): when this trips, delete
// IntegerPowScalarOpRewritePattern (.cpp/.h), its CMakeLists entry, its
// registration in TTNNWorkaroundsPatterns.cpp, and restore the `ttnn.pow`
// CHECK lines in test/ttmlir/Silicon/StableHLO/n150/Binary/pow_op.mlir.
TEST_F(OpModelTripwireTest, PowScalarRejectsInt32) {
  const llvm::SmallVector<int64_t> inputShape = {32, 32};

  const TTNNLayoutAttr inputLayout = CreateTiledLayoutInt32(
      inputShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  // Exponent 2 is inside the range PR #56860 computes exactly, and is the
  // exponent the decomposition reduces to a single multiply.
  // Signless i32: IntegerAttr::getInt() asserts on signed/unsigned types, and
  // the decomposition pattern reads the exponent through getInt().
  mlir::Attribute exponent =
      IntegerAttr::get(IntegerType::get(&context, 32), /*value=*/2);

  auto constraintsExp = OpModel<PowScalarOp>::getOpConstraints(
      inputShape, inputLayout, exponent, inputLayout);
  const bool ok = static_cast<bool>(constraintsExp);
  if (!ok) {
    llvm::consumeError(constraintsExp.takeError());
  }
  EXPECT_FALSE(ok);
}

} // namespace mlir::tt::ttnn::op_model
