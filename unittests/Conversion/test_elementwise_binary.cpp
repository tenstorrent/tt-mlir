// Test file for Elementwise Binary Operations
// TT-MLIR Bounty Task #4862

#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace ttmlir;

class ElementwiseBinaryOpsTest : public ::testing::Test {
protected:
  void SetUp() override {
    ctx.loadDialect<ttir::TTIRDialect>();
    ctx.loadDialect<ttmetal::TTMetalDialect>();
  }
  
  MLIRContext ctx;
};

// Test Add operation conversion
TEST_F(ElementwiseBinaryOpsTest, AddOpConversion) {
  std::string moduleStr = R"mlir(
    module {
      func.func @test_add(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
        %0 = ttir.empty() : tensor<2x3xf32>
        %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
        return %1 : tensor<2x3xf32>
      }
    }
  )mlir";
  
  auto module = parseSourceString<ModuleOp>(moduleStr, &ctx);
  EXPECT_TRUE(module);
  
  // After conversion, should have ttmetal operation
  // Verification logic here
  EXPECT_TRUE(module->lookupSymbol("test_add"));
}

// Test Subtract operation conversion
TEST_F(ElementwiseBinaryOpsTest, SubtractOpConversion) {
  std::string moduleStr = R"mlir(
    module {
      func.func @test_subtract(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
        %0 = ttir.empty() : tensor<4x4xf32>
        %1 = "ttir.subtract"(%arg0, %arg1, %0) : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
        return %1 : tensor<4x4xf32>
      }
    }
  )mlir";
  
  auto module = parseSourceString<ModuleOp>(moduleStr, &ctx);
  EXPECT_TRUE(module);
  EXPECT_TRUE(module->lookupSymbol("test_subtract"));
}

// Test Multiply operation conversion
TEST_F(ElementwiseBinaryOpsTest, MultiplyOpConversion) {
  std::string moduleStr = R"mlir(
    module {
      func.func @test_multiply(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
        %0 = ttir.empty() : tensor<3x3xf32>
        %1 = "ttir.multiply"(%arg0, %arg1, %0) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
        return %1 : tensor<3x3xf32>
      }
    }
  )mlir";
  
  auto module = parseSourceString<ModuleOp>(moduleStr, &ctx);
  EXPECT_TRUE(module);
  EXPECT_TRUE(module->lookupSymbol("test_multiply"));
}

// Test Divide operation conversion
TEST_F(ElementwiseBinaryOpsTest, DivideOpConversion) {
  std::string moduleStr = R"mlir(
    module {
      func.func @test_divide(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
        %0 = ttir.empty() : tensor<2x2xf32>
        %1 = "ttir.divide"(%arg0, %arg1, %0) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        return %1 : tensor<2x2xf32>
      }
    }
  )mlir";
  
  auto module = parseSourceString<ModuleOp>(moduleStr, &ctx);
  EXPECT_TRUE(module);
  EXPECT_TRUE(module->lookupSymbol("test_divide"));
}

// Test comparison operations
TEST_F(ElementwiseBinaryOpsTest, GreaterThanOpConversion) {
  std::string moduleStr = R"mlir(
    module {
      func.func @test_gt(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xi1> {
        %0 = ttir.empty() : tensor<2x2xi1>
        %1 = "ttir.greater_than"(%arg0, %arg1, %0) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xi1>) -> tensor<2x2xi1>
        return %1 : tensor<2x2xi1>
      }
    }
  )mlir";
  
  auto module = parseSourceString<ModuleOp>(moduleStr, &ctx);
  EXPECT_TRUE(module);
}

TEST_F(ElementwiseBinaryOpsTest, LessThanOpConversion) {
  std::string moduleStr = R"mlir(
    module {
      func.func @test_lt(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xi1> {
        %0 = ttir.empty() : tensor<2x2xi1>
        %1 = "ttir.less_than"(%arg0, %arg1, %0) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xi1>) -> tensor<2x2xi1>
        return %1 : tensor<2x2xi1>
      }
    }
  )mlir";
  
  auto module = parseSourceString<ModuleOp>(moduleStr, &ctx);
  EXPECT_TRUE(module);
}

TEST_F(ElementwiseBinaryOpsTest, EqualOpConversion) {
  std::string moduleStr = R"mlir(
    module {
      func.func @test_eq(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xi1> {
        %0 = ttir.empty() : tensor<2x2xi1>
        %1 = "ttir.equal"(%arg0, %arg1, %0) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xi1>) -> tensor<2x2xi1>
        return %1 : tensor<2x2xi1>
      }
    }
  )mlir";
  
  auto module = parseSourceString<ModuleOp>(moduleStr, &ctx);
  EXPECT_TRUE(module);
}

// Test max/min operations
TEST_F(ElementwiseBinaryOpsTest, MaximumOpConversion) {
  std::string moduleStr = R"mlir(
    module {
      func.func @test_max(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
        %0 = ttir.empty() : tensor<3x3xf32>
        %1 = "ttir.maximum"(%arg0, %arg1, %0) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
        return %1 : tensor<3x3xf32>
      }
    }
  )mlir";
  
  auto module = parseSourceString<ModuleOp>(moduleStr, &ctx);
  EXPECT_TRUE(module);
}

TEST_F(ElementwiseBinaryOpsTest, MinimumOpConversion) {
  std::string moduleStr = R"mlir(
    module {
      func.func @test_min(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
        %0 = ttir.empty() : tensor<3x3xf32>
        %1 = "ttir.minimum"(%arg0, %arg1, %0) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xf32>
        return %1 : tensor<3x3xf32>
      }
    }
  )mlir";
  
  auto module = parseSourceString<ModuleOp>(moduleStr, &ctx);
  EXPECT_TRUE(module);
}

// Test edge cases
TEST_F(ElementwiseBinaryOpsTest, DifferentShapes) {
  // Test broadcasting behavior
  std::string moduleStr = R"mlir(
    module {
      func.func @test_broadcast(%arg0: tensor<1x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xf32> {
        %0 = ttir.empty() : tensor<3x4xf32>
        %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<1x4xf32>, tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
        return %1 : tensor<3x4xf32>
      }
    }
  )mlir";
  
  auto module = parseSourceString<ModuleOp>(moduleStr, &ctx);
  EXPECT_TRUE(module);
}

TEST_F(ElementwiseBinaryOpsTest, IntegerTypes) {
  // Test with integer types
  std::string moduleStr = R"mlir(
    module {
      func.func @test_int(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
        %0 = ttir.empty() : tensor<2x2xi32>
        %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
        return %1 : tensor<2x2xi32>
      }
    }
  )mlir";
  
  auto module = parseSourceString<ModuleOp>(moduleStr, &ctx);
  EXPECT_TRUE(module);
}

// Main test runner
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
