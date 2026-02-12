// Elementwise Binary Operations Implementation for TT-MLIR
// Bounty Task: $150
// Issue: #4862

#include "ttmlir/Conversion/TTIRToTTMetal/TTIRToTTMetal.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace ttmlir;

namespace {

// Base pattern for elementwise binary operations
template <typename SrcOp, typename DstOp>
struct ElementwiseBinaryOpPattern : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SrcOp op, 
      typename SrcOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    // Get input and output values
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value output = adaptor.getOutput();
    
    // Type checking and conversion
    auto lhsType = lhs.getType().template dyn_cast<RankedTensorType>();
    auto rhsType = rhs.getType().template dyn_cast<RankedTensorType>();
    auto outType = output.getType().template dyn_cast<RankedTensorType>();
    
    if (!lhsType || !rhsType || !outType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    }
    
    // Verify shapes are compatible (broadcasting handled separately)
    if (lhsType.getShape() != rhsType.getShape() && 
        lhsType.getShape() != outType.getShape()) {
      return rewriter.notifyMatchFailure(op, "incompatible shapes");
    }
    
    // Create the destination operation
    auto newOp = rewriter.create<DstOp>(
        op.getLoc(),
        outType,
        lhs,
        rhs,
        output);
    
    // Copy over any attributes
    if (op.getOperation()->getAttrs().size() > 0) {
      newOp->setAttrs(op->getAttrs());
    }
    
    rewriter.replaceOp(op, newOp.getOperation()->getResults());
    return success();
  }
};

// Specific op patterns using the base pattern
using AddOpPattern = ElementwiseBinaryOpPattern<
    ttir::AddOp, 
    ttmetal::ElementwiseBinaryOp>;

using SubtractOpPattern = ElementwiseBinaryOpPattern<
    ttir::SubtractOp, 
    ttmetal::ElementwiseBinaryOp>;

using MultiplyOpPattern = ElementwiseBinaryOpPattern<
    ttir::MultiplyOp, 
    ttmetal::ElementwiseBinaryOp>;

using DivideOpPattern = ElementwiseBinaryOpPattern<
    ttir::DivideOp, 
    ttmetal::ElementwiseBinaryOp>;

// Additional binary operations
using GreaterThanOpPattern = ElementwiseBinaryOpPattern<
    ttir::GreaterThanOp, 
    ttmetal::ElementwiseBinaryOp>;

using LessThanOpPattern = ElementwiseBinaryOpPattern<
    ttir::LessThanOp, 
    ttmetal::ElementwiseBinaryOp>;

using EqualOpPattern = ElementwiseBinaryOpPattern<
    ttir::EqualOp, 
    ttmetal::ElementwiseBinaryOp>;

using MaximumOpPattern = ElementwiseBinaryOpPattern<
    ttir::MaximumOp, 
    ttmetal::ElementwiseBinaryOp>;

using MinimumOpPattern = ElementwiseBinaryOpPattern<
    ttir::MinimumOp, 
    ttmetal::ElementwiseBinaryOp>;

// Power operation (special handling for floating point)
struct PowerOpPattern : public OpConversionPattern<ttir::PowerOp> {
  using OpConversionPattern<ttir::PowerOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ttir::PowerOp op,
      ttir::PowerOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    Value base = adaptor.getLhs();
    Value exponent = adaptor.getRhs();
    Value output = adaptor.getOutput();
    
    auto baseType = base.getType().dyn_cast<RankedTensorType>();
    auto expType = exponent.getType().dyn_cast<RankedTensorType>();
    auto outType = output.getType().dyn_cast<RankedTensorType>();
    
    if (!baseType || !expType || !outType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    }
    
    // Verify element types support power operation
    auto baseElemType = baseType.getElementType();
    if (!baseElemType.isF32() && !baseElemType.isF64() && 
        !baseElemType.isInteger(32)) {
      return rewriter.notifyMatchFailure(op, "unsupported element type for power");
    }
    
    auto newOp = rewriter.create<ttmetal::ElementwiseBinaryOp>(
        op.getLoc(),
        outType,
        base,
        exponent,
        output);
    
    // Set power operation attribute
    newOp->setAttr("op_type", rewriter.getStringAttr("power"));
    
    rewriter.replaceOp(op, newOp.getOperation()->getResults());
    return success();
  }
};

// Remainder/Modulo operation
struct RemainderOpPattern : public OpConversionPattern<ttir::RemainderOp> {
  using OpConversionPattern<ttir::RemainderOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ttir::RemainderOp op,
      ttir::RemainderOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    Value dividend = adaptor.getLhs();
    Value divisor = adaptor.getRhs();
    Value output = adaptor.getOutput();
    
    auto dividendType = dividend.getType().dyn_cast<RankedTensorType>();
    auto divisorType = divisor.getType().dyn_cast<RankedTensorType>();
    
    if (!dividendType || !divisorType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    }
    
    // Verify element types are integers
    auto elemType = dividendType.getElementType();
    if (!elemType.isInteger(32) && !elemType.isInteger(64)) {
      return rewriter.notifyMatchFailure(op, "remainder requires integer types");
    }
    
    auto newOp = rewriter.create<ttmetal::ElementwiseBinaryOp>(
        op.getLoc(),
        output.getType(),
        dividend,
        divisor,
        output);
    
    newOp->setAttr("op_type", rewriter.getStringAttr("remainder"));
    
    rewriter.replaceOp(op, newOp.getOperation()->getResults());
    return success();
  }
};

} // namespace

// Populate conversion patterns
void populateTTIRToTTMetalElementwiseBinaryPatterns(
    RewritePatternSet &patterns) {
  
  patterns.add<
      AddOpPattern,
      SubtractOpPattern,
      MultiplyOpPattern,
      DivideOpPattern,
      GreaterThanOpPattern,
      LessThanOpPattern,
      EqualOpPattern,
      MaximumOpPattern,
      MinimumOpPattern,
      PowerOpPattern,
      RemainderOpPattern>(patterns.getContext());
}

// Conversion pass
struct TTIRToTTMetalElementwiseBinaryPass
    : public PassWrapper<TTIRToTTMetalElementwiseBinaryPass, OperationPass<ModuleOp>> {
  
  StringRef getArgument() const final {
    return "ttir-to-ttmetal-elementwise-binary";
  }
  
  StringRef getDescription() const final {
    return "Convert TTIR elementwise binary ops to TTMetal dialect";
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ttir::TTIRDialect>();
    registry.insert<ttmetal::TTMetalDialect>();
  }
  
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    ConversionTarget target(getContext());
    target.addLegalDialect<ttmetal::TTMetalDialect>();
    target.addIllegalDialect<ttir::TTIRDialect>();
    
    RewritePatternSet patterns(&getContext());
    populateTTIRToTTMetalElementwiseBinaryPatterns(patterns);
    
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createTTIRToTTMetalElementwiseBinaryPass() {
  return std::make_unique<TTIRToTTMetalElementwiseBinaryPass>();
}

} // namespace ttmlir