#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

namespace mlir::tt {
namespace ttmetal {

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;

class TTIRToTTMetalMatmulRewriter : public OpRewritePattern<ttir::MatmulOp> {
public:
  using OpRewritePattern<ttir::MatmulOp>::OpRewritePattern;

  // Region generaterReaderDispatchRegion(ttir::MatmulOp op, PatternRewriter &rewriter) const {
  //   ttkernel::TilizeInitOp tilize_init = rewriter.create<ttkernel::TilizeInitOp>(op.getLoc());
  //   ttkernel::TilizeBlockOp tilize_block = rewriter.create<ttkernel::TilizeBlockOp>(op.getLoc());
  //   // ...
  //   Block *readerBlock = rewriter.createBlock(&metalDispatch.getRegion(0));
  //   OpBuilder readerBuilder(tensixBlock, tensixBlock->begin());
  // }

  LogicalResult matchAndRewrite(ttir::MatmulOp op, PatternRewriter &rewriter) const final {
    RankedTensorType tensorA = op.getA().getType();
    RankedTensorType tensorB = op.getB().getType();
    RankedTensorType outputTensor = op.getOutput().getType();
    // ArrayAttr constraints = op.getOperandConstraints();

    // // Operands must be DRAM OR L1 AND Tile Layout
    // if ((std::find(constraints.begin(), constraints.end(), OperandConstraint::DRAM) == constraints.end() &&
    //     std::find(constraints.begin(), constraints.end(), OperandConstraint::L1) == constraints.end()) ||
    //     std::find(constraints.begin(), constraints.end(), OperandConstraint::Tile) == constraints.end()) {
    //       return failure();
    // }

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

    llvm::errs() << "MatmulOp matched\n";

    auto outputTensorLayout = mlir::cast<LayoutAttr>(outputTensor.getEncoding());
    auto outputTensorGrid = outputTensorLayout.getGrid().getShape();

    llvm::SmallVector<Attribute> coreRanges;
    for (uint32_t i = 0; i < outputTensorGrid[0]; i++) {
      for (uint32_t j = 0; j < outputTensorGrid[1]; j++) {
        // auto grid = rewriter.getAttr<GridAttr>(llvm::ArrayRef<int64_t>{i, j},
        //                                        llvm::ArrayRef<int64_t>{1, 1});
        coreRanges.push_back(rewriter.getAttr<ttmetal::CoreRangeAttr>(
                                 llvm::ArrayRef<int64_t>{i, j},
                                 llvm::ArrayRef<int64_t>{1, 1}));
      }
    }

    SmallVector<Value> operands = {op.getA(), op.getB()};
    SmallVector<Value> outputs = {op.getOutput()};
    auto metalDispatch = rewriter.create<ttmetal::DispatchOp>(
        op.getLoc(), op->getResults().getTypes(), operands, outputs,
        rewriter.getArrayAttr(coreRanges),
        rewriter.getArrayAttr({}), coreRanges.size());
    // TODO(jdesousa): Generate kernel configs 
    // TODO(jdesousa): Generate core ranges

    rewriter.replaceOp(op, metalDispatch);
    return success();
  }
};

} // namespace ttmetal

} // namespace mlir::tt
