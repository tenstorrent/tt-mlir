# ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_COMPLEXRESHAPEPATTERN_H
# define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_COMPLEXRESHAPEPATTERN_H

# include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

# include "mlir/IR/PatternMatch.h"
# include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

class ComplexReshapePattern : public OpRewritePattern<ttnn::ReshapeOp> {

public:
  using OpRewritePattern<ttnn::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ReshapeOp srcOp,
                                PatternRewriter &rewriter) const override;

private:
  LogicalResult decomposeReshapeType1(ttnn::ReshapeOp srcOp,
                                      PatternRewriter &rewriter,
                                      mlir::RankedTensorType inputType,
                                      mlir::RankedTensorType outputType,
                                      llvm::ArrayRef<int64_t> inputShape,
                                      llvm::ArrayRef<int64_t> outputShape) const;

  LogicalResult decomposeReshapeType2(ttnn::ReshapeOp srcOp,
                                      PatternRewriter &rewriter,
                                      mlir::RankedTensorType inputType,
                                      mlir::RankedTensorType outputType,
                                      llvm::ArrayRef<int64_t> inputShape,
                                      llvm::ArrayRef<int64_t> outputShape) const;

};

} // namespace mlir::tt::ttnn::workarounds::decomposition

# endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_COMPLEXRESHAPEPATTERN_H