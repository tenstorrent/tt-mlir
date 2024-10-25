#include "ttmlir/Conversion/TTIRToTTMetal/TTIRMatmulToTTMetal.h"

namespace mlir::tt::ttmetal {

class TTIRToTTMetalMatmulRewriter : public OpRewritePattern<ttir::MatmulOp> {
public:
  using OpRewritePattern<ttir::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::MatmulOp op, PatternRewriter &rewriter) const final {
    auto operand_constraints = mlir::dyn_cast<mlir::ArrayAttr>(op.getOperand(3));
    assert(std::find(operand_constraints.begin(), operand_constraints.end(),
                     mlir::tt::OperandConstraint(5))); // TODO(jdesousa): Check for tile layout constriant tile == 5

    return success();
  }
};

} // namespace mlir::tt::ttmetal
