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

//===----------------------------------------------------------------------===//
// Larger Graph Coloring Tests
//===----------------------------------------------------------------------===//

TEST_F(GraphColoringStrategyTest, ChaitinBriggsColoringSquareGraph) {
  ChaitinBriggsColoring coloring;

  // Square graph (4-cycle):
  //     A --- B
  //     |     |
  //     B --- A
  //
  // 2-colorable (A=color0, B=color1):
  std::vector<std::vector<size_t>> adjList = {
      {1, 3}, // 0 connects to 1, 3
      {0, 2}, // 1 connects to 0, 2
      {1, 3}, // 2 connects to 1, 3
      {0, 2}  // 3 connects to 0, 2
  };

  std::vector<unsigned> result;
  EXPECT_TRUE(succeeded(coloring.colorGraph(adjList, 2, result)));
  EXPECT_EQ(result.size(), 4u);

  // Verify proper coloring: adjacent nodes have different colors.
  EXPECT_NE(result[0], result[1]);
  EXPECT_NE(result[0], result[3]);
  EXPECT_NE(result[1], result[2]);
  EXPECT_NE(result[2], result[3]);

  EXPECT_EQ(result[0], result[2]);
  EXPECT_EQ(result[1], result[3]);

  // Verify it's 2-colorable (all colors should be 0 or 1).
  for (auto color : result) {
    EXPECT_LT(color, 2u);
  }
}

TEST_F(GraphColoringStrategyTest, GreedyColoringSquareGraph) {
  GreedyColoring coloring;

  // Square graph (4-cycle):
  //     A --- B
  //     |     |
  //     B --- A
  //
  // 2-colorable (A=color0, B=color1):
  std::vector<std::vector<size_t>> adjList = {{1, 3}, {0, 2}, {1, 3}, {0, 2}};

  std::vector<unsigned> result;
  EXPECT_TRUE(succeeded(coloring.colorGraph(adjList, 2, result)));
  EXPECT_EQ(result.size(), 4u);

  EXPECT_NE(result[0], result[1]);
  EXPECT_NE(result[0], result[3]);
  EXPECT_NE(result[1], result[2]);
  EXPECT_NE(result[2], result[3]);

  EXPECT_EQ(result[0], result[2]);
  EXPECT_EQ(result[1], result[3]);

  for (auto color : result) {
    EXPECT_LT(color, 2u);
  }
}

TEST_F(GraphColoringStrategyTest, ChaitinBriggsColoringPentagon) {
  ChaitinBriggsColoring coloring;

  // Pentagon (5-cycle):
  //       0
  //      / \
  //     4   1
  //     |   |
  //     3---2
  //
  // Requires 3 colors (odd cycle).
  // Multiple valid colorings exist, so we only verify it's a valid coloring.
  std::vector<std::vector<size_t>> adjList = {
      {1, 4}, // 0 connects to 1, 4
      {0, 2}, // 1 connects to 0, 2
      {1, 3}, // 2 connects to 1, 3
      {2, 4}, // 3 connects to 2, 4
      {0, 3}  // 4 connects to 0, 3
  };

  std::vector<unsigned> result;
  EXPECT_TRUE(succeeded(coloring.colorGraph(adjList, 3, result)));
  EXPECT_EQ(result.size(), 5u);

  // Verify proper coloring (adjacent nodes have different colors).
  EXPECT_NE(result[0], result[1]);
  EXPECT_NE(result[0], result[4]);
  EXPECT_NE(result[1], result[2]);
  EXPECT_NE(result[2], result[3]);
  EXPECT_NE(result[3], result[4]);

  // Should fail with only 2 colors.
  std::vector<unsigned> result2;
  EXPECT_TRUE(failed(coloring.colorGraph(adjList, 2, result2)));
}

TEST_F(GraphColoringStrategyTest, GreedyColoringPentagon) {
  GreedyColoring coloring;

  // Pentagon (5-cycle):
  //       0
  //      / \
  //     4   1
  //     |   |
  //     3---2
  //
  // Requires 3 colors (odd cycle).
  // Multiple valid colorings exist, so we only verify it's a valid coloring.
  std::vector<std::vector<size_t>> adjList = {
      {1, 4}, {0, 2}, {1, 3}, {2, 4}, {0, 3}};

  std::vector<unsigned> result;
  EXPECT_TRUE(succeeded(coloring.colorGraph(adjList, 3, result)));
  EXPECT_EQ(result.size(), 5u);

  // Verify proper coloring (adjacent nodes have different colors).
  EXPECT_NE(result[0], result[1]);
  EXPECT_NE(result[0], result[4]);
  EXPECT_NE(result[1], result[2]);
  EXPECT_NE(result[2], result[3]);
  EXPECT_NE(result[3], result[4]);

  // Should fail with only 2 colors.
  std::vector<unsigned> result2;
  EXPECT_TRUE(failed(coloring.colorGraph(adjList, 2, result2)));
}

TEST_F(GraphColoringStrategyTest, ChaitinBriggsColoringStar) {
  ChaitinBriggsColoring coloring;

  // Star graph (center node connected to all others):
  //       A
  //       |
  //   A---B---A
  //       |
  //       A
  //
  // 2-colorable: A=color0 (outer), B=color1 (center)
  std::vector<std::vector<size_t>> adjList = {
      {1, 2, 3, 4}, // 0 (center) connects to all
      {0},          // 1 connects only to center
      {0},          // 2 connects only to center
      {0},          // 3 connects only to center
      {0}           // 4 connects only to center
  };

  std::vector<unsigned> result;
  EXPECT_TRUE(succeeded(coloring.colorGraph(adjList, 2, result)));
  EXPECT_EQ(result.size(), 5u);

  // Center must have different color from all outer nodes.
  EXPECT_NE(result[0], result[1]);
  EXPECT_NE(result[0], result[2]);
  EXPECT_NE(result[0], result[3]);
  EXPECT_NE(result[0], result[4]);

  // All outer nodes can have the same color.
  EXPECT_EQ(result[1], result[2]);
  EXPECT_EQ(result[2], result[3]);
  EXPECT_EQ(result[3], result[4]);
}

TEST_F(GraphColoringStrategyTest, GreedyColoringStar) {
  GreedyColoring coloring;

  // Star graph (center node connected to all others):
  //       A
  //       |
  //   A---B---A
  //       |
  //       A
  //
  // 2-colorable: A=color0 (outer), B=color1 (center)
  std::vector<std::vector<size_t>> adjList = {{1, 2, 3, 4}, {0}, {0}, {0}, {0}};

  std::vector<unsigned> result;
  EXPECT_TRUE(succeeded(coloring.colorGraph(adjList, 2, result)));
  EXPECT_EQ(result.size(), 5u);

  EXPECT_NE(result[0], result[1]);
  EXPECT_NE(result[0], result[2]);
  EXPECT_NE(result[0], result[3]);
  EXPECT_NE(result[0], result[4]);

  EXPECT_EQ(result[1], result[2]);
  EXPECT_EQ(result[2], result[3]);
  EXPECT_EQ(result[3], result[4]);
}

TEST_F(GraphColoringStrategyTest, ChaitinBriggsColoringBinaryTree) {
  ChaitinBriggsColoring coloring;

  // Binary tree (depth 2):
  //         A
  //        / \
  //       B   B
  //      / \ / \
  //     A  A A  A
  //
  // 2-colorable (bipartite): A=color0, B=color1
  std::vector<std::vector<size_t>> adjList = {
      {1, 2},    // 0 connects to 1, 2
      {0, 3, 4}, // 1 connects to 0, 3, 4
      {0, 5, 6}, // 2 connects to 0, 5, 6
      {1},       // 3 connects to 1
      {1},       // 4 connects to 1
      {2},       // 5 connects to 2
      {2}        // 6 connects to 2
  };

  std::vector<unsigned> result;
  EXPECT_TRUE(succeeded(coloring.colorGraph(adjList, 2, result)));
  EXPECT_EQ(result.size(), 7u);

  // Verify proper coloring.
  EXPECT_NE(result[0], result[1]);
  EXPECT_NE(result[0], result[2]);
  EXPECT_NE(result[1], result[3]);
  EXPECT_NE(result[1], result[4]);
  EXPECT_NE(result[2], result[5]);
  EXPECT_NE(result[2], result[6]);

  for (auto idx = 3; idx < 7; ++idx) {
    EXPECT_EQ(result[idx], result[0]);
  }

  // Verify it's 2-colorable.
  for (auto color : result) {
    EXPECT_LT(color, 2u);
  }
}

TEST_F(GraphColoringStrategyTest, GreedyColoringBinaryTree) {
  GreedyColoring coloring;

  // Binary tree (depth 2):
  //         A
  //        / \
  //       B   B
  //      / \ / \
  //     A  A A  A
  //
  // 2-colorable (bipartite): A=color0, B=color1
  std::vector<std::vector<size_t>> adjList = {{1, 2}, {0, 3, 4}, {0, 5, 6}, {1},
                                              {1},    {2},       {2}};

  std::vector<unsigned> result;
  EXPECT_TRUE(succeeded(coloring.colorGraph(adjList, 2, result)));
  EXPECT_EQ(result.size(), 7u);

  EXPECT_NE(result[0], result[1]);
  EXPECT_NE(result[0], result[2]);
  EXPECT_NE(result[1], result[3]);
  EXPECT_NE(result[1], result[4]);
  EXPECT_NE(result[2], result[5]);
  EXPECT_NE(result[2], result[6]);

  for (auto idx = 3; idx < 7; ++idx) {
    EXPECT_EQ(result[idx], result[0]);
  }

  for (auto color : result) {
    EXPECT_LT(color, 2u);
  }
}

TEST_F(GraphColoringStrategyTest, ChaitinBriggsColoringCompleteGraph) {
  ChaitinBriggsColoring coloring;

  // Complete graph K5 (every node connected to every other):
  //     A---B
  //     |\ /|
  //     | X |
  //     |/ \|
  //     D---C
  //      \ /
  //       E
  //
  // Requires 5 colors: A=color0, B=color1, C=color2, D=color3, E=color4
  std::vector<std::vector<size_t>> adjList = {
      {1, 2, 3, 4}, // 0 connects to all others
      {0, 2, 3, 4}, // 1 connects to all others
      {0, 1, 3, 4}, // 2 connects to all others
      {0, 1, 2, 4}, // 3 connects to all others
      {0, 1, 2, 3}  // 4 connects to all others
  };

  std::vector<unsigned> result;
  EXPECT_TRUE(succeeded(coloring.colorGraph(adjList, 5, result)));
  EXPECT_EQ(result.size(), 5u);

  // Every node must have a different color.
  for (size_t i = 0; i < result.size(); ++i) {
    for (size_t j = i + 1; j < result.size(); ++j) {
      EXPECT_NE(result[i], result[j]);
    }
  }

  // Should fail with fewer colors.
  std::vector<unsigned> result2;
  EXPECT_TRUE(failed(coloring.colorGraph(adjList, 4, result2)));
}

TEST_F(GraphColoringStrategyTest, GreedyColoringCompleteGraph) {
  GreedyColoring coloring;

  // Complete graph K5 (every node connected to every other):
  //     A---B
  //     |\ /|
  //     | X |
  //     |/ \|
  //     D---C
  //      \ /
  //       E
  //
  // Requires 5 colors: A=color0, B=color1, C=color2, D=color3, E=color4
  std::vector<std::vector<size_t>> adjList = {
      {1, 2, 3, 4}, {0, 2, 3, 4}, {0, 1, 3, 4}, {0, 1, 2, 4}, {0, 1, 2, 3}};

  std::vector<unsigned> result;
  EXPECT_TRUE(succeeded(coloring.colorGraph(adjList, 5, result)));
  EXPECT_EQ(result.size(), 5u);

  for (size_t i = 0; i < result.size(); ++i) {
    for (size_t j = i + 1; j < result.size(); ++j) {
      EXPECT_NE(result[i], result[j]);
    }
  }

  std::vector<unsigned> result2;
  EXPECT_TRUE(failed(coloring.colorGraph(adjList, 4, result2)));
}

TEST_F(GraphColoringStrategyTest, ChaitinBriggsColoringLargerPath) {
  ChaitinBriggsColoring coloring;

  // Long path graph (10 nodes):
  // A---B---A---B---A---B---A---B---A---B
  //
  // 2-colorable: A=color0, B=color1 (alternating)
  std::vector<std::vector<size_t>> adjList = {
      {1},    // 0
      {0, 2}, // 1
      {1, 3}, // 2
      {2, 4}, // 3
      {3, 5}, // 4
      {4, 6}, // 5
      {5, 7}, // 6
      {6, 8}, // 7
      {7, 9}, // 8
      {8}     // 9
  };

  std::vector<unsigned> result;
  EXPECT_TRUE(succeeded(coloring.colorGraph(adjList, 2, result)));
  EXPECT_EQ(result.size(), 10u);

  // Verify proper coloring (adjacent nodes have different colors).
  for (size_t i = 0; i < 9; ++i) {
    EXPECT_NE(result[i], result[i + 1]);
    EXPECT_EQ(result[i], i % 2 == 0 ? result[0] : result[1]);
  }

  // Verify it's 2-colorable.
  for (auto color : result) {
    EXPECT_LT(color, 2u);
  }
}

TEST_F(GraphColoringStrategyTest, GreedyColoringLargerPath) {
  GreedyColoring coloring;

  // Long path graph (10 nodes):
  // A---B---A---B---A---B---A---B---A---B
  //
  // 2-colorable: A=color0, B=color1 (alternating)
  std::vector<std::vector<size_t>> adjList = {
      {1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}, {6, 8}, {7, 9}, {8}};

  std::vector<unsigned> result;
  EXPECT_TRUE(succeeded(coloring.colorGraph(adjList, 2, result)));
  EXPECT_EQ(result.size(), 10u);

  for (size_t i = 0; i < 9; ++i) {
    EXPECT_NE(result[i], result[i + 1]);
    EXPECT_EQ(result[i], i % 2 == 0 ? result[0] : result[1]);
  }

  for (auto color : result) {
    EXPECT_LT(color, 2u);
  }
}

} // namespace mlir::tt::d2m
