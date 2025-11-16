// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/GraphColoringStrategy.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>

namespace mlir::tt::d2m {

namespace gtest = ::testing;

// Test fixture for GraphColoringStrategy tests.
struct GraphColoringStrategyTest : public gtest::Test {
  void SetUp() override {
    ctx = std::make_unique<MLIRContext>();
    ctx->loadDialect<arith::ArithDialect>();
  }

  std::unique_ptr<MLIRContext> ctx;
};

//===----------------------------------------------------------------------===//
// Tests for InterferenceGraph
//===----------------------------------------------------------------------===//

TEST_F(GraphColoringStrategyTest, InterferenceGraphEmpty) {
  InterferenceGraph graph;

  EXPECT_EQ(graph.getNodes().size(), 0u);
}

TEST_F(GraphColoringStrategyTest, InterferenceGraphAddNodes) {
  InterferenceGraph graph;
  OpBuilder builder(ctx.get());

  auto tensorType = RankedTensorType::get({32, 32}, builder.getF32Type());
  auto val1 =
      builder
          .create<arith::ConstantOp>(builder.getUnknownLoc(), tensorType,
                                     builder.getZeroAttr(tensorType))
          .getResult();
  auto val2 =
      builder
          .create<arith::ConstantOp>(builder.getUnknownLoc(), tensorType,
                                     builder.getZeroAttr(tensorType))
          .getResult();

  graph.addNode(val1);
  graph.addNode(val2);

  auto nodes = graph.getNodes();
  EXPECT_EQ(nodes.size(), 2u);
  EXPECT_EQ(graph.degree(val1), 0u);
  EXPECT_EQ(graph.degree(val2), 0u);
}

TEST_F(GraphColoringStrategyTest, InterferenceGraphAddEdges) {
  InterferenceGraph graph;
  OpBuilder builder(ctx.get());

  auto tensorType = RankedTensorType::get({32, 32}, builder.getF32Type());
  auto val1 =
      builder
          .create<arith::ConstantOp>(builder.getUnknownLoc(), tensorType,
                                     builder.getZeroAttr(tensorType))
          .getResult();
  auto val2 =
      builder
          .create<arith::ConstantOp>(builder.getUnknownLoc(), tensorType,
                                     builder.getZeroAttr(tensorType))
          .getResult();
  auto val3 =
      builder
          .create<arith::ConstantOp>(builder.getUnknownLoc(), tensorType,
                                     builder.getZeroAttr(tensorType))
          .getResult();

  graph.addEdge(val1, val2);
  graph.addEdge(val2, val3);

  EXPECT_EQ(graph.degree(val1), 1u);
  EXPECT_EQ(graph.degree(val2), 2u);
  EXPECT_EQ(graph.degree(val3), 1u);

  auto neighbors1 = graph.neighbors(val1);
  auto neighbors2 = graph.neighbors(val2);
  auto neighbors3 = graph.neighbors(val3);

  EXPECT_EQ(neighbors1.size(), 1u);
  EXPECT_EQ(neighbors2.size(), 2u);
  EXPECT_EQ(neighbors3.size(), 1u);

  EXPECT_TRUE(llvm::is_contained(neighbors1, val2));
  EXPECT_TRUE(llvm::is_contained(neighbors2, val1));
  EXPECT_TRUE(llvm::is_contained(neighbors2, val3));
  EXPECT_TRUE(llvm::is_contained(neighbors3, val2));
}

TEST_F(GraphColoringStrategyTest, InterferenceGraphSelfEdgeIgnored) {
  InterferenceGraph graph;
  OpBuilder builder(ctx.get());

  auto tensorType = RankedTensorType::get({32, 32}, builder.getF32Type());
  auto val = builder
                 .create<arith::ConstantOp>(builder.getUnknownLoc(), tensorType,
                                            builder.getZeroAttr(tensorType))
                 .getResult();

  graph.addEdge(val, val);

  EXPECT_EQ(graph.degree(val), 0u);
}

//===----------------------------------------------------------------------===//
// Tests for ChaitinBriggsColoring
//===----------------------------------------------------------------------===//

TEST_F(GraphColoringStrategyTest, ChaitinBriggsColoringEmptyGraph) {
  ChaitinBriggsColoring coloring;
  InterferenceGraph graph;

  auto result = coloring.colorGraph(graph, 3);
  EXPECT_TRUE(result.empty());
}

TEST_F(GraphColoringStrategyTest, ChaitinBriggsColoringSimpleGraph) {
  ChaitinBriggsColoring coloring;
  InterferenceGraph graph;
  OpBuilder builder(ctx.get());

  auto tensorType = RankedTensorType::get({32, 32}, builder.getF32Type());
  auto val1 =
      builder
          .create<arith::ConstantOp>(builder.getUnknownLoc(), tensorType,
                                     builder.getZeroAttr(tensorType))
          .getResult();
  auto val2 =
      builder
          .create<arith::ConstantOp>(builder.getUnknownLoc(), tensorType,
                                     builder.getZeroAttr(tensorType))
          .getResult();
  auto val3 =
      builder
          .create<arith::ConstantOp>(builder.getUnknownLoc(), tensorType,
                                     builder.getZeroAttr(tensorType))
          .getResult();

  graph.addEdge(val1, val2);
  graph.addEdge(val2, val3);

  auto result = coloring.colorGraph(graph, 2);

  EXPECT_EQ(result.size(), 3u);
  EXPECT_TRUE(result.count(val1));
  EXPECT_TRUE(result.count(val2));
  EXPECT_TRUE(result.count(val3));

  EXPECT_NE(result[val1], result[val2]);
  EXPECT_NE(result[val2], result[val3]);
}

TEST_F(GraphColoringStrategyTest, ChaitinBriggsColoringIndexGraph) {
  ChaitinBriggsColoring coloring;

  std::vector<std::vector<size_t>> adjList = {{1}, {0, 2}, {1}};

  std::vector<unsigned> coloringResult;
  auto result = coloring.colorIndexGraph(adjList, 2, coloringResult);

  EXPECT_TRUE(succeeded(result));
  EXPECT_EQ(coloringResult.size(), 3u);

  EXPECT_NE(coloringResult[0], coloringResult[1]);
  EXPECT_NE(coloringResult[1], coloringResult[2]);
}

TEST_F(GraphColoringStrategyTest, ChaitinBriggsColoringSpill) {
  ChaitinBriggsColoring coloring;

  std::vector<std::vector<size_t>> adjList = {{1, 2}, {0, 2}, {0, 1}};

  std::vector<unsigned> coloringResult;
  auto result = coloring.colorIndexGraph(adjList, 2, coloringResult);

  EXPECT_TRUE(failed(result));
}

TEST_F(GraphColoringStrategyTest, ChaitinBriggsColoringNoColors) {
  ChaitinBriggsColoring coloring;

  std::vector<std::vector<size_t>> adjList = {{1}, {0}};
  std::vector<unsigned> coloringResult;

  auto result = coloring.colorIndexGraph(adjList, 0, coloringResult);
  EXPECT_TRUE(failed(result));
}

//===----------------------------------------------------------------------===//
// Tests for GreedyColoring
//===----------------------------------------------------------------------===//

TEST_F(GraphColoringStrategyTest, GreedyColoringEmptyGraph) {
  GreedyColoring coloring;
  InterferenceGraph graph;

  auto result = coloring.colorGraph(graph, 3);
  EXPECT_EQ(result.size(), 0u);
}

TEST_F(GraphColoringStrategyTest, GreedyColoringSimpleGraph) {
  GreedyColoring coloring;
  InterferenceGraph graph;
  OpBuilder builder(ctx.get());

  auto tensorType = RankedTensorType::get({32, 32}, builder.getF32Type());
  auto val1 =
      builder
          .create<arith::ConstantOp>(builder.getUnknownLoc(), tensorType,
                                     builder.getZeroAttr(tensorType))
          .getResult();
  auto val2 =
      builder
          .create<arith::ConstantOp>(builder.getUnknownLoc(), tensorType,
                                     builder.getZeroAttr(tensorType))
          .getResult();
  auto val3 =
      builder
          .create<arith::ConstantOp>(builder.getUnknownLoc(), tensorType,
                                     builder.getZeroAttr(tensorType))
          .getResult();

  graph.addEdge(val1, val2);
  graph.addEdge(val2, val3);

  auto result = coloring.colorGraph(graph, 2);

  EXPECT_EQ(result.size(), 3u);
  EXPECT_TRUE(result.count(val1));
  EXPECT_TRUE(result.count(val2));
  EXPECT_TRUE(result.count(val3));

  EXPECT_NE(result[val1], result[val2]);
  EXPECT_NE(result[val2], result[val3]);
}

TEST_F(GraphColoringStrategyTest, GreedyColoringIndexGraph) {
  GreedyColoring coloring;

  std::vector<std::vector<size_t>> adjList = {{1}, {0, 2}, {1}};

  std::vector<unsigned> coloringResult;
  auto result = coloring.colorIndexGraph(adjList, 2, coloringResult);

  EXPECT_TRUE(succeeded(result));
  EXPECT_EQ(coloringResult.size(), 3u);

  EXPECT_NE(coloringResult[0], coloringResult[1]);
  EXPECT_NE(coloringResult[1], coloringResult[2]);
}

TEST_F(GraphColoringStrategyTest, GreedyColoringSpill) {
  GreedyColoring coloring;

  std::vector<std::vector<size_t>> adjList = {{1, 2}, {0, 2}, {0, 1}};

  std::vector<unsigned> coloringResult;
  auto result = coloring.colorIndexGraph(adjList, 2, coloringResult);
  (void)result;

  EXPECT_EQ(coloringResult.size(), 3u);
}

} // namespace mlir::tt::d2m
