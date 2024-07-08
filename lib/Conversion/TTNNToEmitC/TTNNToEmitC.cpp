#include "llvm/ADT/ScopeExit.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/DialectConversion.h"

#include "../PassDetail.h"
#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"

using namespace mlir;
using namespace mlir::emitc;
using namespace mlir::tt::ttnn;

void populateTTNNToEmitCPatterns(mlir::MLIRContext *ctx,
                                 mlir::RewritePatternSet &patterns) {
  //   // Insert patterns for TOSA data node ops.
  //   patterns.add<ConstOpConversion>(ctx);

  //   // Insert patterns for TOSA unary elementwise ops.
  //   patterns.add<GenericOpConversion<tosa::AbsOp>>(ctx, "emitc::tosa::abs");
  //   patterns.add<GenericOpConversion<tosa::CastOp>>(ctx, "emitc::tosa::cast",
  //                                                   /*explicitResultType=*/true);
  //   patterns.add<GenericOpConversion<tosa::CeilOp>>(ctx,
  //   "emitc::tosa::ceil"); patterns.add<GenericOpConversion<tosa::ClzOp>>(ctx,
  //   "emitc::tosa::clz"); patterns.add<ClampOpConversion>(ctx);
  //   patterns.add<GenericOpConversion<tosa::ExpOp>>(ctx, "emitc::tosa::exp");
  //   patterns.add<GenericOpConversion<tosa::FloorOp>>(ctx,
  //   "emitc::tosa::floor");
  //   patterns.add<GenericOpConversion<tosa::LogOp>>(ctx, "emitc::tosa::log");
  //   patterns.add<NegateOpConversion>(ctx);
  //   patterns.add<GenericOpConversion<tosa::ReciprocalOp>>(
  //       ctx, "emitc::tosa::reciprocal");
  //   patterns.add<RescaleOpConversion>(ctx);
  //   patterns.add<RsqrtOpConversion>(ctx);
  //   patterns.add<GenericOpConversion<tosa::TanhOp>>(ctx,
  //   "emitc::tosa::tanh");

  //   // Insert patterns for TOSA binary elementwise ops.
  //   patterns.add<CallOpBroadcastableConversion<tosa::AddOp>>(ctx,
  //                                                            "emitc::tosa::add");
  //   patterns.add<ArithmeticRightShiftOpConversion>(
  //       ctx, "emitc::tosa::arithmetic_right_shift");
  //   patterns.add<CallOpBroadcastableConversion<tosa::EqualOp>>(
  //       ctx, "emitc::tosa::equal", /*explicitResultType=*/true);
  //   patterns.add<CallOpBroadcastableConversion<tosa::GreaterEqualOp>>(
  //       ctx, "emitc::tosa::greater_equal", /*explicitResultType=*/true);
  //   patterns.add<CallOpBroadcastableConversion<tosa::LogicalLeftShiftOp>>(
  //       ctx, "emitc::tosa::logical_left_shift");
  //   patterns.add<CallOpBroadcastableConversion<tosa::MaximumOp>>(
  //       ctx, "emitc::tosa::maximum");
  //   patterns.add<CallOpBroadcastableConversion<tosa::MinimumOp>>(
  //       ctx, "emitc::tosa::minimum");
  //   patterns.add<MulOpConversion>(ctx, "emitc::tosa::mul");
  //   patterns.add<CallOpBroadcastableConversion<tosa::PowOp>>(ctx,
  //                                                            "emitc::tosa::pow");
  //   patterns.add<CallOpBroadcastableConversion<tosa::SubOp>>(ctx,
  //                                                            "emitc::tosa::sub");
  //   patterns.add<GenericOpConversion<tosa::TableOp>>(ctx,
  //   "emitc::tosa::table");

  //   // Insert patterns for TOSA ternary elementwise ops.
  //   patterns.add<SelectOpConversion>(ctx);

  //   // Insert patterns for other TOSA ops.
  //   patterns.add<ConcatOpConversion>(ctx);
  //   patterns.add<GenericConvOpConversion<tosa::Conv2DOp>>(ctx,
  //                                                         "emitc::tosa::conv2d");
  //   patterns.add<GenericConvOpConversion<tosa::DepthwiseConv2DOp>>(
  //       ctx, "emitc::tosa::depthwise_conv2d");
  //   patterns.add<GenericPoolOpConversion<tosa::AvgPool2dOp>>(
  //       ctx, "emitc::tosa::avg_pool2d");
  //   patterns.add<GenericPoolOpConversion<tosa::MaxPool2dOp>>(
  //       ctx, "emitc::tosa::max_pool2d");
  //   patterns.add<FullyConnectedOpConversion>(ctx,
  //   "emitc::tosa::fully_connected");
  //   patterns.add<GenericOpConversion<tosa::GatherOp>>(
  //       ctx, "emitc::tosa::gather",
  //       /*explicitResultType=*/true);
  //   patterns.add<MatMulOpConversion>(ctx);
  //   patterns.add<TileOpConversion>(ctx);
  //   patterns.add<ReduceOpConversion<tosa::ArgMaxOp>>(ctx,
  //   "emitc::tosa::argmax",
  //                                                    false);
  //   patterns.add<ReduceOpConversion<tosa::ReduceAllOp>>(
  //       ctx, "emitc::tosa::reduce_all", true);
  //   patterns.add<ReduceOpConversion<tosa::ReduceAnyOp>>(
  //       ctx, "emitc::tosa::reduce_any", true);
  //   patterns.add<ReduceOpConversion<tosa::ReduceMaxOp>>(
  //       ctx, "emitc::tosa::reduce_max", true);
  //   patterns.add<ReduceOpConversion<tosa::ReduceMinOp>>(
  //       ctx, "emitc::tosa::reduce_min", true);
  //   patterns.add<ReduceOpConversion<tosa::ReduceProdOp>>(
  //       ctx, "emitc::tosa::reduce_prod", true);
  //   patterns.add<ReduceOpConversion<tosa::ReduceSumOp>>(
  //       ctx, "emitc::tosa::reduce_sum", true);
  //   patterns.add<GenericOpConversion<tosa::ReshapeOp>>(
  //       ctx, "emitc::tosa::reshape",
  //       /*explicitResultType=*/true);
  //   patterns.add<SliceOpConversion>(ctx);
  //   patterns.add<PadOpConversion>(ctx);
  //   patterns.add<GenericOpConversion<tosa::TransposeOp>>(
  //       ctx, "emitc::tosa::transpose", /*explicitResultType=*/true);
}

namespace {

struct ConvertTTNNToEmitCPass
    : public mlir::tt::ttnn::ConvertTTNNToEmitCBase<ConvertTTNNToEmitCPass> {
  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<mlir::tt::ttnn::TTNNDialect>();

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
