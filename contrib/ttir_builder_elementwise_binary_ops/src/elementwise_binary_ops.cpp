#include "contrib/ttir_builder_elementwise_binary_ops/include/elementwise_binary_ops.h"

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace tt::ttir;

namespace tt {
namespace ttir_builder {

// Helper function to create elementwise binary operations
Value createElementwiseBinaryOp(PatternRewriter &rewriter, Location loc, Value lhs, Value rhs,
                                ElementwiseBinaryOpKind opKind) {
  auto resultType = lhs.getType().dyn_cast<RankedTensorType>();
  if (!resultType) {
    return nullptr;
  }

  switch (opKind) {
    case ElementwiseBinaryOpKind::Add:
      return rewriter.create<AddOp>(loc, resultType, lhs, rhs, rewriter.getF32FloatAttr(1.0));
    case ElementwiseBinaryOpKind::Sub:
      return rewriter.create<SubOp>(loc, resultType, lhs, rhs, rewriter.getF32FloatAttr(1.0));
    case ElementwiseBinaryOpKind::Mul:
      return rewriter.create<MulOp>(loc, resultType, lhs, rhs);
    case ElementwiseBinaryOpKind::Div:
      return rewriter.create<DivOp>(loc, resultType, lhs, rhs);
    default:
      return nullptr;
  }
}

// Implementation for Add operation
Value createAddOp(PatternRewriter &rewriter, Location loc, Value lhs, Value rhs) {
  return createElementwiseBinaryOp(rewriter, loc, lhs, rhs, ElementwiseBinaryOpKind::Add);
}

// Implementation for Sub operation
Value createSubOp(PatternRewriter &rewriter, Location loc, Value lhs, Value rhs) {
  return createElementwiseBinaryOp(rewriter, loc, lhs, rhs, ElementwiseBinaryOpKind::Sub);
}

// Implementation for Mul operation
Value createMulOp(PatternRewriter &rewriter, Location loc, Value lhs, Value rhs) {
  return createElementwiseBinaryOp(rewriter, loc, lhs, rhs, ElementwiseBinaryOpKind::Mul);
}

// Implementation for Div operation
Value createDivOp(PatternRewriter &rewriter, Location loc, Value lhs, Value rhs) {
  return createElementwiseBinaryOp(rewriter, loc, lhs, rhs, ElementwiseBinaryOpKind::Div);
}

} // namespace ttir_builder
} // namespace tt
