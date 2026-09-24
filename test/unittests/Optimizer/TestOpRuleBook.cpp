// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"

#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"

#include "gtest/gtest.h"

namespace mlir::tt::ttnn {
namespace {

TEST(OpRuleBookTest, OperandPolicyPreservesOpWideOverrides) {
  struct DisabledRuleBook : OpRuleBook {
    bool shouldExploreReshards() const override { return false; }
  };

  OpRuleBook defaultRules;
  DisabledRuleBook disabledRules;
  const OpRuleBook &rules = disabledRules;
  for (unsigned operandIdx = 0; operandIdx < 3; ++operandIdx) {
    EXPECT_TRUE(defaultRules.shouldExploreReshards(operandIdx));
    EXPECT_FALSE(rules.shouldExploreReshards(operandIdx));
  }
}

class OpRuleBookDispatchTest : public ::testing::Test {
protected:
  MLIRContext context;

  void SetUp() override { context.loadDialect<TTNNDialect>(); }

  void checkOperandPolicy(llvm::StringRef opName,
                          llvm::ArrayRef<bool> expected) {
    // Rule dispatch only needs the operation name. Avoid device-dependent
    // builders and backend validation in these policy tests.
    OperationState state(UnknownLoc::get(&context), opName);
    Operation *op = Operation::create(state);
    for (unsigned operandIdx = 0; operandIdx < expected.size(); ++operandIdx) {
      EXPECT_EQ(shouldExploreReshards(op, operandIdx), expected[operandIdx])
          << opName.str() << " operand " << operandIdx;
    }
    op->destroy();
  }
};

TEST_F(OpRuleBookDispatchTest, RotaryEmbeddingOnlyExploresInput) {
  checkOperandPolicy(RotaryEmbeddingOp::getOperationName(),
                     {true, false, false});
}

TEST_F(OpRuleBookDispatchTest, RotaryEmbeddingLlamaOnlyExploresInput) {
  checkOperandPolicy(RotaryEmbeddingLlamaOp::getOperationName(),
                     {true, false, false, false});
}

TEST_F(OpRuleBookDispatchTest, ExistingPoliciesArePreserved) {
  checkOperandPolicy(AddOp::getOperationName(), {true, true});
  checkOperandPolicy(MatmulOp::getOperationName(), {true, true});
  checkOperandPolicy(ReshapeOp::getOperationName(), {false});
  checkOperandPolicy(ScaledDotProductAttentionOp::getOperationName(),
                     {false, false, false});
}

} // namespace
} // namespace mlir::tt::ttnn
