#include "llvm/ADT/ScopeExit.h"
#include <llvm/ADT/StringRef.h>
#include <mlir/Support/LogicalResult.h>

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/DialectConversion.h"

#include "../PassDetail.h"
#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

using namespace mlir;
using namespace mlir::emitc;
using namespace mlir::tt::ttnn;

template <typename SrcOp, typename Adaptor = typename SrcOp::Adaptor>
class DefaultOpConversionPattern : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  DefaultOpConversionPattern(MLIRContext *ctx)
      : OpConversionPattern<SrcOp>(ctx) {}

private:
  std::string getOpName(SrcOp op) const {
    auto name = op.getOperationName();
    if (name.starts_with("ttnn.")) {
      return "ttnn::" + name.drop_front(5).str();
    }
    return "ttnn::" + name.str();
  }

  LogicalResult
  matchAndRewrite(SrcOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<Attribute, 4> templateArguments;

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, srcOp->getResultTypes(), getOpName(srcOp), nullptr, nullptr,
        adaptor.getOperands());

    return success();
  }
};

void populateTTNNToEmitCPatterns(mlir::MLIRContext *ctx,
                                 mlir::RewritePatternSet &patterns) {
  // Device ops
  //
  patterns.add<DefaultOpConversionPattern<mlir::tt::ttnn::OpenDeviceOp>>(ctx);
  patterns.add<DefaultOpConversionPattern<mlir::tt::ttnn::CloseDeviceOp>>(ctx);

  // Memory ops
  //
  patterns.add<DefaultOpConversionPattern<mlir::tt::ttnn::ToMemoryConfigOp>>(
      ctx);

  // Tensor ops
  //
  patterns.add<DefaultOpConversionPattern<mlir::tt::ttnn::FullOp>>(ctx);

  // Math ops
  //
  patterns.add<DefaultOpConversionPattern<mlir::tt::ttnn::AddOp>>(ctx);
  patterns.add<DefaultOpConversionPattern<mlir::tt::ttnn::MultiplyOp>>(ctx);
}

namespace {

struct ConvertTTNNToEmitCPass
    : public mlir::tt::ttnn::ConvertTTNNToEmitCBase<ConvertTTNNToEmitCPass> {
  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addIllegalDialect<mlir::tt::ttnn::TTNNDialect>();

    RewritePatternSet patterns(&getContext());
    populateTTNNToEmitCPatterns(&getContext(), patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  };
};

} // namespace

namespace mlir::tt::ttnn {

std::unique_ptr<OperationPass<func::FuncOp>> createConvertTTNNToEmitCPass() {
  return std::make_unique<ConvertTTNNToEmitCPass>();
}

LogicalResult emitTTNNAsCpp(ModuleOp origOp, llvm::raw_ostream &os) {
  ModuleOp op = cast<ModuleOp>(origOp->clone());
  auto cleanupDispatchClone = llvm::make_scope_exit([&op] { op->erase(); });

  auto pm = PassManager::on<ModuleOp>(op.getContext());
  pm.addPass(createConvertTTNNToEmitCPass());

  if (pm.run(op).failed()) {
    return failure();
  }

  if (mlir::emitc::translateToCpp(op, os).failed()) {
    return failure();
  }

  return success();
}

} // namespace mlir::tt::ttnn
