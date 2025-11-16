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
// Tests for ChaitinBriggsColoring
//===----------------------------------------------------------------------===//

TEST_F(GraphColoringStrategyTest, ChaitinBriggsColoringEmptyGraph) {
  ChaitinBriggsColoring coloring;
  std::vector<std::vector<size_t>> adjacencyList;
  std::vector<unsigned> result;

  EXPECT_TRUE(succeeded(coloring.colorGraph(adjacencyList, 3, result)));
  EXPECT_TRUE(result.empty());
}

TEST_F(GraphColoringStrategyTest, ChaitinBriggsColoringSimpleGraph) {
  ChaitinBriggsColoring coloring;
  // Graph: 0 -- 1 -- 2 (linear chain)
  std::vector<std::vector<size_t>> adjacencyList = {{1}, {0, 2}, {1}};
  std::vector<unsigned> result;

  EXPECT_TRUE(succeeded(coloring.colorGraph(adjacencyList, 2, result)));
  EXPECT_EQ(result.size(), 3u);

  // Nodes 0 and 1 should have different colors.
  EXPECT_NE(result[0], result[1]);
  // Nodes 1 and 2 should have different colors.
  EXPECT_NE(result[1], result[2]);
  // Nodes 0 and 2 can have the same color (not adjacent).
}

TEST_F(GraphColoringStrategyTest, ChaitinBriggsColoringIndexGraph) {
  ChaitinBriggsColoring coloring;

  std::vector<std::vector<size_t>> adjList = {{1}, {0, 2}, {1}};

  std::vector<unsigned> coloringResult;
  auto result = coloring.colorGraph(adjList, 2, coloringResult);

  EXPECT_TRUE(succeeded(result));
  EXPECT_EQ(coloringResult.size(), 3u);

  EXPECT_NE(coloringResult[0], coloringResult[1]);
  EXPECT_NE(coloringResult[1], coloringResult[2]);
}

TEST_F(GraphColoringStrategyTest, ChaitinBriggsColoringSpill) {
  ChaitinBriggsColoring coloring;

  std::vector<std::vector<size_t>> adjList = {{1, 2}, {0, 2}, {0, 1}};

  std::vector<unsigned> coloringResult;
  auto result = coloring.colorGraph(adjList, 2, coloringResult);

  EXPECT_TRUE(failed(result));
}

TEST_F(GraphColoringStrategyTest, ChaitinBriggsColoringNoColors) {
  ChaitinBriggsColoring coloring;

  std::vector<std::vector<size_t>> adjList = {{1}, {0}};
  std::vector<unsigned> coloringResult;

  auto result = coloring.colorGraph(adjList, 0, coloringResult);
  EXPECT_TRUE(failed(result));
}

//===----------------------------------------------------------------------===//
// Tests for GreedyColoring
//===----------------------------------------------------------------------===//

TEST_F(GraphColoringStrategyTest, GreedyColoringEmptyGraph) {
  GreedyColoring coloring;
  std::vector<std::vector<size_t>> adjacencyList;
  std::vector<unsigned> result;

  EXPECT_TRUE(succeeded(coloring.colorGraph(adjacencyList, 3, result)));
  EXPECT_EQ(result.size(), 0u);
}

TEST_F(GraphColoringStrategyTest, GreedyColoringSimpleGraph) {
  GreedyColoring coloring;
  // Graph: 0 -- 1 -- 2 (linear chain)
  std::vector<std::vector<size_t>> adjacencyList = {{1}, {0, 2}, {1}};
  std::vector<unsigned> result;

  EXPECT_TRUE(succeeded(coloring.colorGraph(adjacencyList, 2, result)));
  EXPECT_EQ(result.size(), 3u);

  // Nodes 0 and 1 should have different colors.
  EXPECT_NE(result[0], result[1]);
  // Nodes 1 and 2 should have different colors.
  EXPECT_NE(result[1], result[2]);
  // Nodes 0 and 2 can have the same color (not adjacent).
}

TEST_F(GraphColoringStrategyTest, GreedyColoringIndexGraph) {
  GreedyColoring coloring;

  std::vector<std::vector<size_t>> adjList = {{1}, {0, 2}, {1}};

  std::vector<unsigned> coloringResult;
  auto result = coloring.colorGraph(adjList, 2, coloringResult);

  EXPECT_TRUE(succeeded(result));
  EXPECT_EQ(coloringResult.size(), 3u);

  EXPECT_NE(coloringResult[0], coloringResult[1]);
  EXPECT_NE(coloringResult[1], coloringResult[2]);
}

TEST_F(GraphColoringStrategyTest, GreedyColoringSpill) {
  GreedyColoring coloring;

  std::vector<std::vector<size_t>> adjList = {{1, 2}, {0, 2}, {0, 1}};

  std::vector<unsigned> coloringResult;
  auto result = coloring.colorGraph(adjList, 2, coloringResult);

  EXPECT_TRUE(failed(result));
}

} // namespace mlir::tt::d2m
