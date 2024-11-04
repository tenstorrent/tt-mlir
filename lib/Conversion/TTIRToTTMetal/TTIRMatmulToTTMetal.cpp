#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"
#include "ttmlir/Conversion/TTIRToTTMetal/TTIRToTTMetal.h"

namespace mlir::tt {
namespace ttmetal {

constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;

class TTIRToTTMetalMatmulRewriter : public OpRewritePattern<ttir::MatmulOp> {
public:
  using OpRewritePattern<ttir::MatmulOp>::OpRewritePattern;

  std::pair<SmallVector<Attribute, 5>, SmallVector<Attribute, 5>> generate2DMMAttributes(ArrayRef<int64_t> &gridShape, PatternRewriter &rewriter) const {
    SmallVector<Attribute, 5> coreRanges;
    SmallVector<Attribute, 5> kernelConfigs;
    
    // Compute (whole worker grid)
    coreRanges.push_back(rewriter.getAttr<ttmetal::CoreRangeAttr>(
        llvm::ArrayRef<int64_t>{0, 0}, llvm::ArrayRef<int64_t>{gridShape[0], gridShape[1]}));
    kernelConfigs.push_back(rewriter.getAttr<ttkernel::TensixConfigAttr>(
        ttkernel::MathFidelity::HiFi4, false, false));

    // in0 senders
    coreRanges.push_back(rewriter.getAttr<ttmetal::CoreRangeAttr>(
        llvm::ArrayRef<int64_t>{0, 0},
        llvm::ArrayRef<int64_t>{1, gridShape[1]}));
    kernelConfigs.push_back(rewriter.getAttr<ttkernel::NocConfigAttr>(
        ttkernel::NocIndex::Noc0));

    // in1 senders/writers
    coreRanges.push_back(rewriter.getAttr<ttmetal::CoreRangeAttr>(
        llvm::ArrayRef<int64_t>{0, 0},
        llvm::ArrayRef<int64_t>{gridShape[0], 1}));
    kernelConfigs.push_back(rewriter.getAttr<ttkernel::NocConfigAttr>(
        ttkernel::NocIndex::Noc1));

    // in0 receivers
    coreRanges.push_back(rewriter.getAttr<ttmetal::CoreRangeAttr>(
        llvm::ArrayRef<int64_t>{1, 0},
        llvm::ArrayRef<int64_t>{gridShape[0] - 1, gridShape[1]}));
    kernelConfigs.push_back(rewriter.getAttr<ttkernel::NocConfigAttr>(
        ttkernel::NocIndex::Noc0));

    // in1 receivers/writers
    coreRanges.push_back(rewriter.getAttr<ttmetal::CoreRangeAttr>(
        llvm::ArrayRef<int64_t>{0, 1},
        llvm::ArrayRef<int64_t>{gridShape[0], gridShape[1] - 1}));
    kernelConfigs.push_back(
        rewriter.getAttr<ttkernel::NocConfigAttr>(ttkernel::NocIndex::Noc1));

    return std::pair<SmallVector<Attribute, 5>, SmallVector<Attribute, 5>>{
        coreRanges, kernelConfigs};
  }

  void generateComputeBlock(ttmetal::DispatchOp &metalDispatch, PatternRewriter &rewriter, SmallVector<ttkernel::CBType, 3> &cbs) const {
    Block *computeBlock = rewriter.createBlock(&metalDispatch.getRegion(0));
    OpBuilder computeBuilder(computeBlock, computeBlock->begin());

    computeBlock->addArgument(cbs[0], metalDispatch.getLoc());
    computeBlock->addArgument(cbs[1], metalDispatch.getLoc());
    computeBlock->addArgument(cbs[2], metalDispatch.getLoc());

    // kernel here

    computeBuilder.create<ttkernel::ReturnOp>(metalDispatch.getLoc());
  }

  void generateReaderBlocks(ttmetal::DispatchOp &metalDispatch, PatternRewriter &rewriter, SmallVector<ttkernel::CBType, 3> &cbs) const {
    // generate 4 reader blocks, block 0 is the compute block, blocks 1-4 are
    // the reader blocks
    // TODO(jdesousa) add semaphore generation here
    for (int i = 1; i < 5; i++) {
      Block *readerBlock = rewriter.createBlock(&metalDispatch.getRegion(i));
      OpBuilder readerBuilder(readerBlock, readerBlock->begin());

      readerBlock->addArgument(cbs[0], metalDispatch.getLoc());
      readerBlock->addArgument(cbs[1], metalDispatch.getLoc());

      // kernels for each block here (use createDataMovementThread / buildNocAsyncTx, etc. (TTIRToTTMetal.cpp))

      readerBuilder.create<ttkernel::ReturnOp>(metalDispatch.getLoc());
    }
  }

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

    auto in0TensorLayout = mlir::cast<LayoutAttr>(tensorA.getEncoding());
    auto in1TensorLayout = mlir::cast<LayoutAttr>(tensorB.getEncoding());
    auto out0TensorLayout = mlir::cast<LayoutAttr>(outputTensor.getEncoding());
    auto outputTensorGrid = out0TensorLayout.getGrid().getShape();

    auto [coreRanges, kernelConfigs] = generate2DMMAttributes(outputTensorGrid, rewriter);

    SmallVector<Value> operands = {op.getA(), op.getB()};
    SmallVector<Value> outputs = {op.getOutput()};
    auto metalDispatch = rewriter.create<ttmetal::DispatchOp>(
        op.getLoc(), op->getResults().getTypes(), operands, outputs,
        rewriter.getArrayAttr(coreRanges),
        rewriter.getArrayAttr(kernelConfigs), coreRanges.size());

    std::int64_t in0BaseAddress = lookupAddress(op.getA());
    std::int64_t in1BaseAddress = lookupAddress(op.getB());
    std::int64_t out0BaseAddress = lookupAddress(op.getOutput());

    ttkernel::CBType in0CBTy = rewriter.getType<ttkernel::CBType>(
        ttkernel::CBPort::In0, in0BaseAddress,
        mlir::cast<MemRefType>(in0TensorLayout.getMemref()),
        in0TensorLayout.getElementSizeBytes(),
        /*num_buffers*/ 1);
    ttkernel::CBType in1CBTy = rewriter.getType<ttkernel::CBType>(
        ttkernel::CBPort::In1, in1BaseAddress,
        mlir::cast<MemRefType>(in1TensorLayout.getMemref()),
        in1TensorLayout.getElementSizeBytes(),
        /*num_buffers*/ 1);
    ttkernel::CBType out0CBTy = rewriter.getType<ttkernel::CBType>(
        ttkernel::CBPort::Out0, out0BaseAddress,
        mlir::cast<MemRefType>(out0TensorLayout.getMemref()),
        out0TensorLayout.getElementSizeBytes(),
        /*num_buffers*/ 1);

    SmallVector<ttkernel::CBType, 3> cbTypes = {in0CBTy, in1CBTy, out0CBTy};

    generateComputeBlock(metalDispatch, rewriter, cbTypes);
    generateReaderBlocks(metalDispatch, rewriter, cbTypes);
    rewriter.replaceOp(op, metalDispatch);
    return success();
  }
};

} // namespace ttmetal

} // namespace mlir::tt
