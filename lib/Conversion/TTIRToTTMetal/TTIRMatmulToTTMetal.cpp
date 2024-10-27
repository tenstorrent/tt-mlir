#include "ttmlir/Conversion/TTIRToTTMetal/TTIRMatmulToTTMetal.h"

namespace mlir::tt {
namespace ttmetal {

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;

class TTIRToTTMetalMatmulRewriter : public OpRewritePattern<ttir::MatmulOp> {
public:
  using OpRewritePattern<ttir::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::MatmulOp op, PatternRewriter &rewriter) const final {
    RankedTensorType tensorA = op.getA().getType();
    RankedTensorType tensorB = op.getB().getType();
    RankedTensorType outputTensor = op.getOutput().getType();
    ArrayAttr constraints = op.getOperandConstraints();

    // Operands must be DRAM OR L1 AND Tile Layout
    if ((std::find(constraints.begin(), constraints.end(), OperandConstraint::DRAM) == constraints.end() &&
        std::find(constraints.begin(), constraints.end(), OperandConstraint::L1) == constraints.end()) ||
        std::find(constraints.begin(), constraints.end(), OperandConstraint::Tile) == constraints.end()) {
          return failure();
    }

    uint32_t tensorARank = tensorA.getRank();
    uint32_t tensorBRank = tensorB.getRank();
    uint32_t outputTensorRank = outputTensor.getRank();

    // Input A must be tile aligned
    if ((tensorA.getShape()[tensorARank - 1] % TILE_WIDTH != 0 || tensorA.getShape()[tensorARank - 2] % TILE_HEIGHT != 0)) {
      return failure();
    }

    // Input B must be tile aligned
    if ((tensorB.getShape()[tensorBRank - 1] % TILE_WIDTH != 0 ||
         tensorB.getShape()[tensorBRank - 2] % TILE_HEIGHT != 0)) {
      return failure();
    }

    // Output must be tile aligned
    if ((outputTensor.getShape()[outputTensorRank - 1] % TILE_WIDTH != 0 ||
         outputTensor.getShape()[outputTensorRank - 2] % TILE_HEIGHT != 0)) {
      return failure();
    }

    return success();
  }
};
}
} // namespace mlir::tt::ttmetal
