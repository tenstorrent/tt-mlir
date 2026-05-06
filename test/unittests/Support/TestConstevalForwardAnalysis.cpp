// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>

namespace mlir::tt::ttcore {
namespace {

// Synthetic IR with mixed CONST/PARAM and INPUT block arguments. Each result
// exercises a different traceability case:
//   %add_pp  = constArg0 + paramArg1            -> traces to const args
//   %add_pi  = constArg0 + inputArg2            -> mixed, does not trace
//   %add_ii  = inputArg2 + inputArg3            -> traces only to inputs
//   %two     = arith.constant ...               -> no block args at all
//   %use_two = %add_pp + %two                   -> still traces (const path)
//
// `two` (a constant op) and `use_two` exercise the "no block args in chain"
// branch of the slow path.
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

// (1) The cached lookup must agree bit-for-bit with the slow recursive walk
// for every Value in a function. Capture slow-path answers without any active
// scope, then re-query with the analysis active and compare.
TEST_F(ConstevalForwardAnalysisTest, CachedLookupMatchesSlowPath) {
  auto module = parse();
  ASSERT_TRUE(module);
  mlir::func::FuncOp func = getFunc(*module);

  llvm::SmallVector<mlir::Value> values = collectAllValues(func);
  ASSERT_FALSE(values.empty());

  // Slow-path baseline (no active analysis).
  llvm::DenseMap<mlir::Value, bool> slowResults;
  for (mlir::Value v : values) {
    slowResults[v] = valueTracesToConstantArgs(v);
  }

  // Sanity: the synthetic module must produce a non-trivial mix of true and
  // false answers, otherwise the test does not actually exercise both
  // branches.
  size_t numTrue = 0;
  size_t numFalse = 0;
  for (auto &kv : slowResults) {
    (kv.second ? numTrue : numFalse)++;
  }
  EXPECT_GT(numTrue, 0u);
  EXPECT_GT(numFalse, 0u);

  // Cached answers must match.
  ConstevalForwardAnalysis analysis(func);
  ConstevalAnalysisScope scope(analysis);
  for (mlir::Value v : values) {
    EXPECT_EQ(valueTracesToConstantArgs(v), slowResults[v])
        << "cached vs slow-path divergence for value index "
        << std::distance(values.begin(),
                         std::find(values.begin(), values.end(), v));
  }
}

// (2) Specific spot checks — the synthetic module is small enough that we can
// pin down exactly which values must be true vs false.
TEST_F(ConstevalForwardAnalysisTest, ExpectedAnswers) {
  auto module = parse();
  ASSERT_TRUE(module);
  mlir::func::FuncOp func = getFunc(*module);

  // Block args.
  EXPECT_TRUE(valueTracesToConstantArgs(func.getArgument(0)));  // constant
  EXPECT_TRUE(valueTracesToConstantArgs(func.getArgument(1)));  // parameter
  EXPECT_FALSE(valueTracesToConstantArgs(func.getArgument(2))); // input
  EXPECT_FALSE(valueTracesToConstantArgs(func.getArgument(3))); // input

  // Walk results in declaration order.
  llvm::SmallVector<mlir::Operation *> arithOps;
  func.walk([&](mlir::Operation *op) {
    if (mlir::isa<mlir::arith::AddIOp, mlir::arith::ConstantOp>(op)) {
      arithOps.push_back(op);
    }
  });
  ASSERT_EQ(arithOps.size(), 5u);

  // add_pp: const + param  -> traces to const args.
  EXPECT_TRUE(valueTracesToConstantArgs(arithOps[0]->getResult(0)));
  // add_pi: const + input  -> mixed, does not trace.
  EXPECT_FALSE(valueTracesToConstantArgs(arithOps[1]->getResult(0)));
  // add_ii: input + input  -> traces only to inputs.
  EXPECT_FALSE(valueTracesToConstantArgs(arithOps[2]->getResult(0)));
  // arith.constant: no block args in chain at all -> false.
  EXPECT_FALSE(valueTracesToConstantArgs(arithOps[3]->getResult(0)));
  // use_two: add_pp + two  -> add_pp traces to const, two has no block args
  //                          (chain still reaches const args via add_pp).
  EXPECT_TRUE(valueTracesToConstantArgs(arithOps[4]->getResult(0)));
}

// (3) The cache must actually be reused across calls when the IR is
// untouched. We assert that rebuildCount stays at exactly 1 after many
// lookups inside an active scope with no IR mutations.
TEST_F(ConstevalForwardAnalysisTest, CacheReusedAcrossLookups) {
  auto module = parse();
  ASSERT_TRUE(module);
  mlir::func::FuncOp func = getFunc(*module);

  ConstevalForwardAnalysis analysis(func);
  ConstevalAnalysisScope scope(analysis);

  EXPECT_EQ(analysis.rebuildCount(), 0u);

  llvm::SmallVector<mlir::Value> values = collectAllValues(func);
  for (int i = 0; i < 100; ++i) {
    for (mlir::Value v : values) {
      (void)valueTracesToConstantArgs(v);
    }
  }

  // First lookup triggers exactly one rebuild; subsequent lookups hit the
  // cache.
  EXPECT_EQ(analysis.rebuildCount(), 1u);
}

// (4) IR mutations through a rewriter must invalidate the cache, and the
// next lookup must produce a correct answer (verified against the slow path
// on the mutated IR).
TEST_F(ConstevalForwardAnalysisTest, ListenerInvalidatesOnMutation) {
  auto module = parse();
  ASSERT_TRUE(module);
  mlir::func::FuncOp func = getFunc(*module);

  // Find the op to replace before constructing the scope.
  mlir::arith::AddIOp targetOp;
  func.walk([&](mlir::arith::AddIOp op) {
    if (!targetOp) {
      targetOp = op;
    }
  });
  ASSERT_TRUE(targetOp);

  size_t rebuildsAfterMutation = 0;
  llvm::SmallVector<mlir::Value> postValues;
  llvm::SmallVector<bool> cachedAnswers;
  {
    ConstevalForwardAnalysis analysis(func);
    ConstevalAnalysisScope scope(analysis);

    // Warm the cache: 1 rebuild expected.
    for (mlir::Value v : collectAllValues(func)) {
      (void)valueTracesToConstantArgs(v);
    }
    ASSERT_EQ(analysis.rebuildCount(), 1u);

    // Mutate IR through an IRRewriter so listener notifications fire. Replace
    // %add_pp = const + param with const + const (still all const-traced).
    // The structural change forces dirty_ to flip and the cache to rebuild
    // on next lookup.
    mlir::IRRewriter rewriter(&context);
    rewriter.setListener(&analysis);
    rewriter.setInsertionPoint(targetOp);
    mlir::Value lhs = targetOp.getLhs();
    auto newOp =
        rewriter.create<mlir::arith::AddIOp>(targetOp.getLoc(), lhs, lhs);
    rewriter.replaceOp(targetOp, newOp.getResult());

    // The notifications themselves do not rebuild — they only flip dirty_.
    // The rebuild happens on the next lookup.
    EXPECT_EQ(analysis.rebuildCount(), 1u);

    postValues = collectAllValues(func);
    cachedAnswers.reserve(postValues.size());
    for (mlir::Value v : postValues) {
      cachedAnswers.push_back(valueTracesToConstantArgs(v));
    }
    rebuildsAfterMutation = analysis.rebuildCount();
  }

  // The first post-mutation lookup must have rebuilt the cache.
  EXPECT_EQ(rebuildsAfterMutation, 2u);

  // Compute slow-path baseline OUTSIDE any scope so valueTracesToConstantArgs
  // takes the slow recursive path. Compare against the cached answers we
  // captured above.
  ASSERT_EQ(getActiveConstevalAnalysis(), nullptr);
  ASSERT_EQ(postValues.size(), cachedAnswers.size());
  for (size_t i = 0; i < postValues.size(); ++i) {
    EXPECT_EQ(cachedAnswers[i], valueTracesToConstantArgs(postValues[i]))
        << "cached vs slow-path divergence after mutation, value index " << i;
  }
}

// (5) Scope guard nesting and unwind: the active analysis pointer must be
// restored on destruction. Without this property the listener wiring around
// nested pattern applications would leak state across passes.
TEST_F(ConstevalForwardAnalysisTest, ScopeNestingRestoresPrevious) {
  auto module = parse();
  ASSERT_TRUE(module);
  mlir::func::FuncOp func = getFunc(*module);

  EXPECT_EQ(getActiveConstevalAnalysis(), nullptr);
  {
    ConstevalForwardAnalysis outer(func);
    ConstevalAnalysisScope outerScope(outer);
    EXPECT_EQ(getActiveConstevalAnalysis(), &outer);
    {
      ConstevalForwardAnalysis inner(func);
      ConstevalAnalysisScope innerScope(inner);
      EXPECT_EQ(getActiveConstevalAnalysis(), &inner);
    }
    EXPECT_EQ(getActiveConstevalAnalysis(), &outer);
  }
  EXPECT_EQ(getActiveConstevalAnalysis(), nullptr);
}

} // namespace
} // namespace mlir::tt::ttcore
