// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/ConstevalForwardAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>

namespace mlir::tt::ttir {
namespace {

// Synthetic IR with mixed CONST/PARAM and INPUT block arguments:
//   %add_pp  = constArg0 + paramArg1            -> traces to const args
//   %add_pi  = constArg0 + inputArg2            -> mixed, does not trace
//   %add_ii  = inputArg2 + inputArg3            -> traces only to inputs
//   %two     = arith.constant ...               -> no block args at all
//   %use_two = %add_pp + %two                   -> still traces (const path)
constexpr llvm::StringRef kModuleText = R"MLIR(
module {
  func.func @f(
      %constArg0: i32 {ttcore.argument_type = #ttcore.argument_type<constant>},
      %paramArg1: i32 {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %inputArg2: i32 {ttcore.argument_type = #ttcore.argument_type<input>},
      %inputArg3: i32 {ttcore.argument_type = #ttcore.argument_type<input>}
  ) -> (i32, i32, i32, i32, i32) {
    %add_pp  = arith.addi %constArg0, %paramArg1 : i32
    %add_pi  = arith.addi %constArg0, %inputArg2 : i32
    %add_ii  = arith.addi %inputArg2, %inputArg3 : i32
    %two     = arith.constant 2 : i32
    %use_two = arith.addi %add_pp, %two : i32
    return %add_pp, %add_pi, %add_ii, %two, %use_two : i32, i32, i32, i32, i32
  }
}
)MLIR";

class ConstevalForwardAnalysisTest : public ::testing::Test {
protected:
  mlir::MLIRContext context;

  ConstevalForwardAnalysisTest() {
    context.loadDialect<mlir::arith::ArithDialect>();
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse() {
    return mlir::parseSourceString<mlir::ModuleOp>(kModuleText, &context);
  }

  static mlir::func::FuncOp getFunc(mlir::ModuleOp module) {
    auto funcs = module.getOps<mlir::func::FuncOp>();
    return *funcs.begin();
  }

  // Collect every Value in the function (block args + op results).
  static llvm::SmallVector<mlir::Value> collectAllValues(mlir::func::FuncOp f) {
    llvm::SmallVector<mlir::Value> out;
    for (mlir::BlockArgument arg : f.getArguments()) {
      out.push_back(arg);
    }
    f.walk([&](mlir::Operation *op) {
      for (mlir::Value r : op->getResults()) {
        out.push_back(r);
      }
    });
    return out;
  }
};

// The cached lookup must agree bit-for-bit with the public slow-path helper.
TEST_F(ConstevalForwardAnalysisTest, CachedLookupMatchesSlowPath) {
  auto module = parse();
  ASSERT_TRUE(module);
  mlir::func::FuncOp func = getFunc(*module);

  llvm::SmallVector<mlir::Value> values = collectAllValues(func);
  ASSERT_FALSE(values.empty());

  llvm::DenseMap<mlir::Value, bool> slowResults;
  for (mlir::Value v : values) {
    slowResults[v] = ttcore::valueTracesToConstantArgs(v);
  }

  // Sanity: synthetic module exercises both true and false branches.
  size_t numTrue = 0;
  size_t numFalse = 0;
  for (auto &kv : slowResults) {
    (kv.second ? numTrue : numFalse)++;
  }
  EXPECT_GT(numTrue, 0u);
  EXPECT_GT(numFalse, 0u);

  ConstevalForwardAnalysis analysis(func);
  for (mlir::Value v : values) {
    EXPECT_EQ(analysis.valueTracesToConstantArgs(v), slowResults[v])
        << "cached vs slow-path divergence for value index "
        << std::distance(values.begin(),
                         std::find(values.begin(), values.end(), v));
  }
}

// Spot-check exact answers on the small synthetic module.
TEST_F(ConstevalForwardAnalysisTest, ExpectedAnswers) {
  auto module = parse();
  ASSERT_TRUE(module);
  mlir::func::FuncOp func = getFunc(*module);

  ConstevalForwardAnalysis analysis(func);

  EXPECT_TRUE(analysis.valueTracesToConstantArgs(func.getArgument(0)));
  EXPECT_TRUE(analysis.valueTracesToConstantArgs(func.getArgument(1)));
  EXPECT_FALSE(analysis.valueTracesToConstantArgs(func.getArgument(2)));
  EXPECT_FALSE(analysis.valueTracesToConstantArgs(func.getArgument(3)));

  llvm::SmallVector<mlir::Operation *> arithOps;
  func.walk([&](mlir::Operation *op) {
    if (mlir::isa<mlir::arith::AddIOp, mlir::arith::ConstantOp>(op)) {
      arithOps.push_back(op);
    }
  });
  ASSERT_EQ(arithOps.size(), 5u);

  // add_pp: const + param  -> traces to const args.
  EXPECT_TRUE(analysis.valueTracesToConstantArgs(arithOps[0]->getResult(0)));
  // add_pi: const + input  -> mixed, does not trace.
  EXPECT_FALSE(analysis.valueTracesToConstantArgs(arithOps[1]->getResult(0)));
  // add_ii: input + input  -> traces only to inputs.
  EXPECT_FALSE(analysis.valueTracesToConstantArgs(arithOps[2]->getResult(0)));
  // arith.constant: no block args in chain at all -> false.
  EXPECT_FALSE(analysis.valueTracesToConstantArgs(arithOps[3]->getResult(0)));
  // use_two: add_pp + two -> chain reaches const args via add_pp.
  EXPECT_TRUE(analysis.valueTracesToConstantArgs(arithOps[4]->getResult(0)));
}

// IR mutations through a Listener-aware rewriter must invalidate the cache,
// and the next lookup must produce a correct answer (verified against the
// slow path on the mutated IR).
TEST_F(ConstevalForwardAnalysisTest, ListenerInvalidatesOnMutation) {
  auto module = parse();
  ASSERT_TRUE(module);
  mlir::func::FuncOp func = getFunc(*module);

  mlir::arith::AddIOp targetOp;
  func.walk([&](mlir::arith::AddIOp op) {
    if (!targetOp) {
      targetOp = op;
    }
  });
  ASSERT_TRUE(targetOp);

  ConstevalForwardAnalysis analysis(func);

  // Warm the cache.
  for (mlir::Value v : collectAllValues(func)) {
    (void)analysis.valueTracesToConstantArgs(v);
  }

  // Replace %add_pp = const + param with const + const (still const-traced).
  // The mutation flips dirty_; the next lookup rebuilds.
  mlir::IRRewriter rewriter(&context);
  rewriter.setListener(&analysis);
  rewriter.setInsertionPoint(targetOp);
  mlir::Value lhs = targetOp.getLhs();
  auto newOp =
      rewriter.create<mlir::arith::AddIOp>(targetOp.getLoc(), lhs, lhs);
  rewriter.replaceOp(targetOp, newOp.getResult());

  for (mlir::Value v : collectAllValues(func)) {
    EXPECT_EQ(analysis.valueTracesToConstantArgs(v),
              ttcore::valueTracesToConstantArgs(v));
  }
}

} // namespace
} // namespace mlir::tt::ttir
