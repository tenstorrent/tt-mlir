#ifndef TTIR_BUILDER_ELEMENTWISE_BINARY_OPS_H
#define TTIR_BUILDER_ELEMENTWISE_BINARY_OPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/PatternMatch.h"

namespace tt {
namespace ttir_builder {

/// Enum to represent different elementwise binary operations
enum class ElementwiseBinaryOpKind {
  Add,
  Sub,
  Mul,
  Div
};

/// Creates an elementwise binary operation based on the specified kind
mlir::Value createElementwiseBinaryOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
                                     mlir::Value lhs, mlir::Value rhs,
                                     ElementwiseBinaryOpKind opKind);

/// Creates an add operation
mlir::Value createAddOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
                        mlir::Value lhs, mlir::Value rhs);

/// Creates a subtract operation
mlir::Value createSubOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
                        mlir::Value lhs, mlir::Value rhs);

/// Creates a multiply operation
mlir::Value createMulOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
                        mlir::Value lhs, mlir::Value rhs);

/// Creates a divide operation
mlir::Value createDivOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
                        mlir::Value lhs, mlir::Value rhs);

} // namespace ttir_builder
} // namespace tt

#endif // TTIR_BUILDER_ELEMENTWISE_BINARY_OPS_H
