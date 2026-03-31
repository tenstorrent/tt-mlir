#include "contrib/ttir_builder_elementwise_binary_ops/include/elementwise_binary_ops.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "gtest/gtest.h"
#include <memory>

using namespace mlir;
using namespace tt::ttir_builder;

class ElementwiseBinaryOpsTest : public ::testing::Test {
protected:
  void SetUp() override {
    ctx.loadDialect<tt::ttir::TTIRDialect>();
    ctx.loadDialect<tensor::TensorDialect>();
    ctx.loadDialect<func::FuncDialect>();
    
    rewriter = std::make_unique<PatternRewriter>(ctx);
    loc = UnknownLoc::get(&ctx);
    
    // Create tensor types for testing
    RankedTensorType tensorType = RankedTensorType::get({2, 2}, Float32Type::get(&ctx));
    
    // Create dummy values for operands
    auto module = ModuleOp::create(loc);
    auto funcType = FunctionType::get(&ctx, {}, {});
    auto funcOp = func::FuncOp::create(loc, "test_func", funcType);
    module.push_back(funcOp);
    
    OpBuilder builder(&ctx);
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    
    auto tensor1 = builder.create<tensor::EmptyOp>(loc, tensorType);
    auto tensor2 = builder.create<tensor::EmptyOp>(loc, tensorType);
    
    operand1 = tensor1.getResult();
    operand2 = tensor2.getResult();
  }
  
  MLIRContext ctx;
  std::unique_ptr<PatternRewriter> rewriter;
  Location loc;
  Value operand1, operand2;
};

TEST_F(ElementwiseBinaryOpsTest, CreateAddOp) {
  auto result = createAddOp(*rewriter, loc, operand1, operand2);
  ASSERT_NE(result, nullptr);
  
  // Verify the result type matches input type
  EXPECT_EQ(result.getType(), operand1.getType());
}

TEST_F(ElementwiseBinaryOpsTest, CreateSubOp) {
  auto result = createSubOp(*rewriter, loc, operand1, operand2);
  ASSERT_NE(result, nullptr);
  
  // Verify the result type matches input type
  EXPECT_EQ(result.getType(), operand1.getType());
}

TEST_F(ElementwiseBinaryOpsTest, CreateMulOp) {
  auto result = createMulOp(*rewriter, loc, operand1, operand2);
  ASSERT_NE(result, nullptr);
  
  // Verify the result type matches input type
  EXPECT_EQ(result.getType(), operand1.getType());
}

TEST_F(ElementwiseBinaryOpsTest, CreateDivOp) {
  auto result = createDivOp(*rewriter, loc, operand1, operand2);
  ASSERT_NE(result, nullptr);
  
  // Verify the result type matches input type
  EXPECT_EQ(result.getType(), operand1.getType());
}

TEST_F(ElementwiseBinaryOpsTest, CreateElementwiseBinaryOpWithAdd) {
  auto result = createElementwiseBinaryOp(*rewriter, loc, operand1, operand2, ElementwiseBinaryOpKind::Add);
  ASSERT_NE(result, nullptr);
  
  // Verify the result type matches input type
  EXPECT_EQ(result.getType(), operand1.getType());
}

TEST_F(ElementwiseBinaryOpsTest, CreateElementwiseBinaryOpWithSub) {
  auto result = createElementwiseBinaryOp(*rewriter, loc, operand1, operand2, ElementwiseBinaryOpKind::Sub);
  ASSERT_NE(result, nullptr);
  
  // Verify the result type matches input type
  EXPECT_EQ(result.getType(), operand1.getType());
}

TEST_F(ElementwiseBinaryOpsTest, CreateElementwiseBinaryOpWithMul) {
  auto result = createElementwiseBinaryOp(*rewriter, loc, operand1, operand2, ElementwiseBinaryOpKind::Mul);
  ASSERT_NE(result, nullptr);
  
  // Verify the result type matches input type
  EXPECT_EQ(result.getType(), operand1.getType());
}

TEST_F(ElementwiseBinaryOpsTest, CreateElementwiseBinaryOpWithDiv) {
  auto result = createElementwiseBinaryOp(*rewriter, loc, operand1, operand2, ElementwiseBinaryOpKind::Div);
  ASSERT_NE(result, nullptr);
  
  // Verify the result type matches input type
  EXPECT_EQ(result.getType(), operand1.getType());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
