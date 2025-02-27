
#include "ttmlir/Conversion/TTIRToTTKernel/TTIRToTTKernel.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

namespace mlir::tt::ttkernel {

Value i32(int32_t value, OpBuilder &builder) {
  return builder
      .create<arith::ConstantOp>(builder.getUnknownLoc(),
                                 builder.getI32Type(),
                                 builder.getI32IntegerAttr(value))
      .getResult();
}

namespace {
class TTIRGenericRewriter : public OpRewritePattern<ttir::GenericOp> {
public:
  using OpRewritePattern<ttir::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::GenericOp op,
                                PatternRewriter &rewriter) const final {
    llvm::errs() << "ttir generic rewriter\n";

    for (auto &region : op->getRegions()) {
      assert(region.getBlocks().size() == 1 && "Expected single block in region, failing.");
      Block &block = region.getBlocks().front();
      assert(block.getNumArguments() == op.getNumOperands() && "Mismatch between number of operands and block arguments, failing.");
      llvm::SmallVector<ttkernel::CBType> cbTypes;
      for (uint32_t i = 0; i < block.getNumArguments(); i++) {
        auto memref = mlir::cast<MemRefType>(block.getArgument(i).getType());
        auto cbPort = ttkernel::symbolizeCBPort(i);
        assert(cbPort.has_value() && "Out of CBs, failing.");
        cbTypes.push_back(ttkernel::CBType::get(rewriter.getContext(),
                                                cbPort.value(), 0, memref));
      }
      for (uint32_t i = 0; i < block.getNumArguments(); i++) {
        rewriter.modifyOpInPlace(op, [&]() {
          block.getArgument(i).setType(cbTypes[i]);
        });
      }
    }

    //rewriter.create<ttmetal::EnqueueProgramOp>(op.getLoc())

    return success();
  };
};
} // namespace

// memref load rewriter
namespace {

class MemrefLoadRewriter : public OpRewritePattern<memref::LoadOp> {
public:
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp op,
                                PatternRewriter &rewriter) const final {

    // erase the load op for now? sfpu / fpu considerations
    // rewriter.eraseOp(op);
    return success();
    };

};

} // namespace
// memref store rewriter

namespace {

class MemrefStoreRewriter : public OpRewritePattern<memref::StoreOp> {
public:
  using OpRewritePattern<memref::StoreOp>::OpRewritePattern;

  static uint32_t getCbId(memref::StoreOp op) {
    memref::CollapseShapeOp collapseOp = llvm::cast<memref::CollapseShapeOp>(op.getMemref().getDefiningOp());
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(collapseOp.getSrc())) {
      return blockArg.getArgNumber();
    }
    assert(false && "Could not match collapse op src to block argument, cannot determine CB id. Failing.");
  }

  LogicalResult matchAndRewrite(memref::StoreOp op,
                                PatternRewriter &rewriter) const final {
    llvm::errs() << "memref store rewriter\n";
    OpBuilder builder(op);

    auto cbId = i32(getCbId(op), builder);
    auto numPages = op->getParentOfType<scf::ForOp>().getStep();
    // tile regs calls
    rewriter.create<ttkernel::CBReserveBackOp>(op.getLoc(), cbId, numPages);
    rewriter.create<ttkernel::PackTileOp>(op.getLoc(), i32(0, builder), cbId, i32(0, builder));
    rewriter.create<ttkernel::CBPushBackOp>(op.getLoc(), cbId, numPages);

    rewriter.eraseOp(op);
    return success();
  };
};

} // namespace

// tile ops rewriter

// ttir await rewriter

// ttir yield rewriter

} // namespace mlir::tt::ttkernel

namespace mlir::tt {

void populateTTIRToTTKernelPatterns(MLIRContext *ctx,
                                   RewritePatternSet &patterns,
                                   TypeConverter & /*typeConverter*/) {
  patterns.add<ttkernel::TTIRGenericRewriter>(ctx);
}

} // namespace mlir::tt