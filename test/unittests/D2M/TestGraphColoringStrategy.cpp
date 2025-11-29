// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/DstAnalysisGraphColoring.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/Transforms/GraphColoringStrategy.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

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
// Parameterized Tests for Graph Coloring Strategies
//===----------------------------------------------------------------------===//

enum class ColoringStrategyType { ChaitinBriggs, Greedy };

std::string
strategyTypeName(const gtest::TestParamInfo<ColoringStrategyType> &info) {
  switch (info.param) {
  case ColoringStrategyType::ChaitinBriggs:
    return "ChaitinBriggs";
  case ColoringStrategyType::Greedy:
    return "Greedy";
  }
  llvm_unreachable("Unknown strategy type");
}

struct GraphColoringParamTest
    : public gtest::TestWithParam<ColoringStrategyType> {
  std::unique_ptr<ColoringStrategy> createStrategy() {
    switch (GetParam()) {
    case ColoringStrategyType::ChaitinBriggs:
      return std::make_unique<ChaitinBriggsColoring>();
    case ColoringStrategyType::Greedy:
      return std::make_unique<GreedyColoring>();
    }
    llvm_unreachable("Unknown strategy type");
  }
};

INSTANTIATE_TEST_SUITE_P(AllStrategies, GraphColoringParamTest,
                         gtest::Values(ColoringStrategyType::ChaitinBriggs,
                                       ColoringStrategyType::Greedy),
                         strategyTypeName);

TEST_P(GraphColoringParamTest, EmptyGraph) {
  auto coloring = createStrategy();
  std::vector<std::vector<size_t>> adjacencyList;
  std::vector<unsigned> result;

  EXPECT_TRUE(succeeded(coloring->colorGraph(adjacencyList, 3, result)));
  EXPECT_TRUE(result.empty());
}

TEST_P(GraphColoringParamTest, SimpleLinearGraph) {
  auto coloring = createStrategy();
  // Graph: 0 -- 1 -- 2 (linear chain)
  std::vector<std::vector<size_t>> adjacencyList = {{1}, {0, 2}, {1}};
  std::vector<unsigned> result;

  EXPECT_TRUE(succeeded(coloring->colorGraph(adjacencyList, 2, result)));
  EXPECT_EQ(result.size(), 3u);

  // Nodes 0 and 1 should have different colors.
  EXPECT_NE(result[0], result[1]);
  // Nodes 1 and 2 should have different colors.
  EXPECT_NE(result[1], result[2]);
  // Nodes 0 and 2 can have the same color (not adjacent).
}

TEST_P(GraphColoringParamTest, TriangleSpill) {
  auto coloring = createStrategy();
  // Triangle graph (K3): requires 3 colors, fails with 2.
  std::vector<std::vector<size_t>> adjList = {{1, 2}, {0, 2}, {0, 1}};

  std::vector<unsigned> coloringResult;
  auto result = coloring->colorGraph(adjList, 2, coloringResult);

  EXPECT_TRUE(failed(result));
}

TEST_P(GraphColoringParamTest, SquareGraph) {
  auto coloring = createStrategy();

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
  EXPECT_TRUE(succeeded(coloring->colorGraph(adjList, 2, result)));
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

TEST_P(GraphColoringParamTest, Pentagon) {
  auto coloring = createStrategy();

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
  EXPECT_TRUE(succeeded(coloring->colorGraph(adjList, 3, result)));
  EXPECT_EQ(result.size(), 5u);

  // Verify proper coloring (adjacent nodes have different colors).
  EXPECT_NE(result[0], result[1]);
  EXPECT_NE(result[0], result[4]);
  EXPECT_NE(result[1], result[2]);
  EXPECT_NE(result[2], result[3]);
  EXPECT_NE(result[3], result[4]);

  // Should fail with only 2 colors.
  std::vector<unsigned> result2;
  EXPECT_TRUE(failed(coloring->colorGraph(adjList, 2, result2)));
}

TEST_P(GraphColoringParamTest, Star) {
  auto coloring = createStrategy();

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
  EXPECT_TRUE(succeeded(coloring->colorGraph(adjList, 2, result)));
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

TEST_P(GraphColoringParamTest, BinaryTree) {
  auto coloring = createStrategy();

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
  EXPECT_TRUE(succeeded(coloring->colorGraph(adjList, 2, result)));
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

TEST_P(GraphColoringParamTest, CompleteGraph) {
  auto coloring = createStrategy();

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
  EXPECT_TRUE(succeeded(coloring->colorGraph(adjList, 5, result)));
  EXPECT_EQ(result.size(), 5u);

  // Every node must have a different color.
  for (size_t i = 0; i < result.size(); ++i) {
    for (size_t j = i + 1; j < result.size(); ++j) {
      EXPECT_NE(result[i], result[j]);
    }
  }

  // Should fail with fewer colors.
  std::vector<unsigned> result2;
  EXPECT_TRUE(failed(coloring->colorGraph(adjList, 4, result2)));
}

TEST_P(GraphColoringParamTest, LongPath) {
  auto coloring = createStrategy();

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
  EXPECT_TRUE(succeeded(coloring->colorGraph(adjList, 2, result)));
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

TEST_P(GraphColoringParamTest, ColoringWithZeroColorsAlwaysFails) {
  auto coloring = createStrategy();

  std::vector<std::vector<size_t>> adjList = {{1}, {0}};
  std::vector<unsigned> coloringResult;

  auto result = coloring->colorGraph(adjList, 0, coloringResult);
  EXPECT_TRUE(failed(result));
}

//===----------------------------------------------------------------------===//
// Tests for DstAnalysisGraphColoring failure propagation
//===----------------------------------------------------------------------===//

struct DstAnalysisGraphColoringTest : public gtest::Test {
  void SetUp() override {
    ctx = std::make_unique<MLIRContext>();
    ctx->loadDialect<func::FuncDialect, d2m::D2MDialect, ttcore::TTCoreDialect,
                     affine::AffineDialect, arith::ArithDialect>();
  }

  std::unique_ptr<MLIRContext> ctx;
};

TEST_F(DstAnalysisGraphColoringTest, ReportsFailureWhenColoringFails) {
  // Verify that when maxSlices constraint is exceeded, analysis reports the
  // failure. Provides a lower bound estimate for DST slices required.
  const char *moduleStr = R"(
    #l1_ = #ttcore.memory_space<l1>
    #device = #ttcore.device<workerGrid = #ttcore.grid<1x1, (d0, d1) -> (0, d0, d1)>,
      l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0),
      dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0),
      meshShape = , chipIds = [0]>
    module attributes {ttcore.device = #device} {
      func.func @has_generic(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>,
                             %in1: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>,
                             %out: memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>) {
        d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<1x1>,
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<compute>]
        } ins(%in0, %in1 :
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>)
          outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096, 1>, #l1_>) {
        ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb_out: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>):
          %c0 = arith.constant 0 : index
          %mem0 = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>
                                   -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %mem1 = d2m.wait %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>
                                   -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %mem_out = d2m.reserve %cb_out : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>
                                           -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %val0 = affine.load %mem0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %val1 = affine.load %mem1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %result = "d2m.tile_add"(%val0, %val1)
              : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          affine.store %result, %mem_out[%c0, %c0]
              : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
        }
        return
      }
    }
  )";

  ParserConfig config(ctx.get(), /*verifyAfterParse=*/false);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);
  ASSERT_TRUE(module);

  auto funcOp = module->lookupSymbol<func::FuncOp>("has_generic");
  ASSERT_TRUE(funcOp);

  // Use GreedyColoring with maxSlices=1 to force analysis to fail.
  // The interference graph requires 3 colors, so it will exceed the limit.
  auto analysis = createGraphColoringDstAnalysis(
      std::make_unique<GreedyColoring>(), /*maxSlices=*/1);
  auto result = analysis->analyze(funcOp);
  EXPECT_FALSE(result.isValid);
  ASSERT_TRUE(result.failureReason.has_value());
  // Verify failure message includes both required and available slices.
  EXPECT_EQ(*result.failureReason,
            "Graph coloring failed: requires 3 slices but only 1 available");
  EXPECT_EQ(result.numSlicesRequired, 3u);
}

//===----------------------------------------------------------------------===//
// Tests for EquivalenceClasses and Coalescing
//===----------------------------------------------------------------------===//

TEST_F(GraphColoringStrategyTest, EquivalenceClassesBasic) {
  // Test that equivalence classes properly merge coalescing pairs.
  llvm::EquivalenceClasses<size_t> ec;

  // Insert nodes 0-4.
  for (size_t i = 0; i < 5; ++i) {
    ec.insert(i);
  }

  // Create coalescing pairs: (0,1), (2,3).
  ec.unionSets(0, 1);
  ec.unionSets(2, 3);

  // Verify equivalence relationships.
  EXPECT_TRUE(ec.isEquivalent(0, 1));
  EXPECT_TRUE(ec.isEquivalent(2, 3));
  EXPECT_FALSE(ec.isEquivalent(0, 2));
  EXPECT_FALSE(ec.isEquivalent(1, 3));
  EXPECT_FALSE(ec.isEquivalent(0, 4));

  // Node 4 should be in its own class.
  EXPECT_EQ(ec.getLeaderValue(4), 4u);
}

TEST_F(GraphColoringStrategyTest, EquivalenceClassesTransitive) {
  // Test transitive closure: if (0,1) and (1,2) are coalesced, then (0,2).
  llvm::EquivalenceClasses<size_t> ec;

  for (size_t i = 0; i < 4; ++i) {
    ec.insert(i);
  }

  ec.unionSets(0, 1);
  ec.unionSets(1, 2);

  // All three should be in the same equivalence class.
  EXPECT_TRUE(ec.isEquivalent(0, 1));
  EXPECT_TRUE(ec.isEquivalent(1, 2));
  EXPECT_TRUE(ec.isEquivalent(0, 2));
  EXPECT_FALSE(ec.isEquivalent(0, 3));
}

TEST_P(GraphColoringParamTest, GraphContractionWithCoalescing) {
  // Test that graph contraction properly handles coalesced nodes.
  // Original graph: 0 -- 1 -- 2 -- 3
  // Coalescing: (0,1) and (2,3)
  // Contracted graph: A -- B where A = {0,1} and B = {2,3}
  // This should be 2-colorable.

  auto coloring = createStrategy();

  // Build original adjacency list.
  std::vector<std::vector<size_t>> origAdjList = {
      {1},    // 0 connects to 1
      {0, 2}, // 1 connects to 0, 2
      {1, 3}, // 2 connects to 1, 3
      {2}     // 3 connects to 2
  };

  // Build equivalence classes.
  llvm::EquivalenceClasses<size_t> ec;
  for (size_t i = 0; i < 4; ++i) {
    ec.insert(i);
  }
  ec.unionSets(0, 1); // Coalesce 0 and 1.
  ec.unionSets(2, 3); // Coalesce 2 and 3.

  // Remove edges between coalesced nodes.
  for (size_t node = 0; node < origAdjList.size(); ++node) {
    auto &neighbors = origAdjList[node];
    neighbors.erase(std::remove_if(neighbors.begin(), neighbors.end(),
                                   [&](size_t neighbor) {
                                     return ec.isEquivalent(node, neighbor);
                                   }),
                    neighbors.end());
  }

  // Build contracted graph.
  llvm::DenseMap<size_t, size_t> nodeToLeader;
  llvm::DenseMap<size_t, size_t> leaderToContracted;
  std::vector<size_t> contractedToLeader;

  for (size_t i = 0; i < 4; ++i) {
    size_t leader = ec.getLeaderValue(i);
    nodeToLeader[i] = leader;
    if (leaderToContracted.find(leader) == leaderToContracted.end()) {
      size_t idx = contractedToLeader.size();
      leaderToContracted[leader] = idx;
      contractedToLeader.push_back(leader);
    }
  }

  // Should have 2 contracted nodes.
  EXPECT_EQ(contractedToLeader.size(), 2u);

  // Build contracted adjacency list.
  std::vector<std::vector<size_t>> contractedAdjList(contractedToLeader.size());
  for (size_t node = 0; node < 4; ++node) {
    size_t leader = nodeToLeader[node];
    size_t contractedNode = leaderToContracted[leader];
    for (size_t neighbor : origAdjList[node]) {
      size_t neighborLeader = nodeToLeader[neighbor];
      if (leader == neighborLeader) {
        continue;
      }
      size_t contractedNeighbor = leaderToContracted[neighborLeader];
      if (std::find(contractedAdjList[contractedNode].begin(),
                    contractedAdjList[contractedNode].end(),
                    contractedNeighbor) ==
          contractedAdjList[contractedNode].end()) {
        contractedAdjList[contractedNode].push_back(contractedNeighbor);
      }
    }
  }

  // Contracted graph should be 2 nodes connected by 1 edge.
  EXPECT_EQ(contractedAdjList.size(), 2u);
  EXPECT_EQ(contractedAdjList[0].size(), 1u);
  EXPECT_EQ(contractedAdjList[1].size(), 1u);

  // Color the contracted graph.
  std::vector<unsigned> contractedColoring;
  EXPECT_TRUE(succeeded(
      coloring->colorGraph(contractedAdjList, 2, contractedColoring)));
  EXPECT_EQ(contractedColoring.size(), 2u);
  EXPECT_NE(contractedColoring[0], contractedColoring[1]);

  // Propagate colors to original nodes.
  std::vector<unsigned> coloring_result(4);
  for (size_t i = 0; i < 4; ++i) {
    size_t leader = nodeToLeader[i];
    size_t contractedIdx = leaderToContracted[leader];
    coloring_result[i] = contractedColoring[contractedIdx];
  }

  // Coalesced nodes should have the same color.
  EXPECT_EQ(coloring_result[0], coloring_result[1]);
  EXPECT_EQ(coloring_result[2], coloring_result[3]);
  // Different equivalence classes should have different colors.
  EXPECT_NE(coloring_result[0], coloring_result[2]);
}

TEST_P(GraphColoringParamTest, CoalescingReducesChromaticNumber) {
  // Test that coalescing can reduce the chromatic number.
  // Triangle graph (K3): requires 3 colors without coalescing.
  // With coalescing (0,2): reduces to 2 nodes, requires 2 colors.

  auto coloring = createStrategy();

  // Original triangle: 0 -- 1 -- 2 -- 0.
  std::vector<std::vector<size_t>> origAdjList = {
      {1, 2}, // 0 connects to 1, 2
      {0, 2}, // 1 connects to 0, 2
      {0, 1}  // 2 connects to 0, 1
  };

  // Without coalescing, requires 3 colors.
  std::vector<unsigned> result3;
  EXPECT_TRUE(succeeded(coloring->colorGraph(origAdjList, 3, result3)));
  std::vector<unsigned> result2;
  EXPECT_TRUE(failed(coloring->colorGraph(origAdjList, 2, result2)));

  // With coalescing (0,2): nodes 0 and 2 must have same color.
  llvm::EquivalenceClasses<size_t> ec;
  for (size_t i = 0; i < 3; ++i) {
    ec.insert(i);
  }
  ec.unionSets(0, 2);

  // Remove edge between coalesced nodes (0,2).
  std::vector<std::vector<size_t>> coalescedAdjList = origAdjList;
  for (size_t node = 0; node < coalescedAdjList.size(); ++node) {
    auto &neighbors = coalescedAdjList[node];
    neighbors.erase(std::remove_if(neighbors.begin(), neighbors.end(),
                                   [&](size_t neighbor) {
                                     return ec.isEquivalent(node, neighbor);
                                   }),
                    neighbors.end());
  }

  // After removing (0,2) edge: 0 -- 1 -- 2 (linear).
  EXPECT_EQ(coalescedAdjList[0].size(), 1u); // 0 connects only to 1.
  EXPECT_EQ(coalescedAdjList[2].size(), 1u); // 2 connects only to 1.

  // Build contracted graph: 2 nodes (class {0,2} and class {1}).
  llvm::DenseMap<size_t, size_t> nodeToLeader;
  llvm::DenseMap<size_t, size_t> leaderToContracted;
  std::vector<size_t> contractedToLeader;

  for (size_t i = 0; i < 3; ++i) {
    size_t leader = ec.getLeaderValue(i);
    nodeToLeader[i] = leader;
    if (leaderToContracted.find(leader) == leaderToContracted.end()) {
      size_t idx = contractedToLeader.size();
      leaderToContracted[leader] = idx;
      contractedToLeader.push_back(leader);
    }
  }

  EXPECT_EQ(contractedToLeader.size(), 2u);

  std::vector<std::vector<size_t>> contractedAdjList(2);
  for (size_t node = 0; node < 3; ++node) {
    size_t leader = nodeToLeader[node];
    size_t contractedNode = leaderToContracted[leader];
    for (size_t neighbor : coalescedAdjList[node]) {
      size_t neighborLeader = nodeToLeader[neighbor];
      if (leader == neighborLeader) {
        continue;
      }
      size_t contractedNeighbor = leaderToContracted[neighborLeader];
      if (std::find(contractedAdjList[contractedNode].begin(),
                    contractedAdjList[contractedNode].end(),
                    contractedNeighbor) ==
          contractedAdjList[contractedNode].end()) {
        contractedAdjList[contractedNode].push_back(contractedNeighbor);
      }
    }
  }

  // Contracted graph should be 2-colorable.
  std::vector<unsigned> contractedColoring;
  EXPECT_TRUE(succeeded(
      coloring->colorGraph(contractedAdjList, 2, contractedColoring)));
}

// Parameterized test fixture for DstAnalysis tests that need MLIR context.
struct DstAnalysisParamTest
    : public gtest::TestWithParam<ColoringStrategyType> {
  void SetUp() override {
    ctx = std::make_unique<MLIRContext>();
    ctx->loadDialect<func::FuncDialect, d2m::D2MDialect, ttcore::TTCoreDialect,
                     affine::AffineDialect, arith::ArithDialect>();
  }

  std::unique_ptr<DstAnalysis> createAnalysis() {
    switch (GetParam()) {
    case ColoringStrategyType::ChaitinBriggs:
      return createChaitinBriggsDstAnalysis();
    case ColoringStrategyType::Greedy:
      return createGreedyDstAnalysis();
    }
    llvm_unreachable("Unknown strategy type");
  }

  std::unique_ptr<MLIRContext> ctx;
};

INSTANTIATE_TEST_SUITE_P(AllStrategies, DstAnalysisParamTest,
                         gtest::Values(ColoringStrategyType::ChaitinBriggs,
                                       ColoringStrategyType::Greedy),
                         strategyTypeName);

// Test that the interference graph correctly identifies interference between
// two loads that both feed into tile_add, where one goes through tile_bcast.
// This is the exact scenario from the failing pytest:
//   %load0 = affine.load %subview[...]        // Load for bcast
//   %bcast = "d2m.tile_bcast"(%load0)         // Intermediate compute op
//   %load1 = affine.load %subview_41[...]     // Load second operand (direct)
//   %add = "d2m.tile_add"(%bcast, %load1)     // Both operands must be in DST
// The loads %load0 and %load1 must interfere because tile_add needs both
// operands simultaneously in DST.
TEST_P(DstAnalysisParamTest, TileBcastIntermediateOpInterference) {
  const char *moduleStr = R"(
    #l1_ = #ttcore.memory_space<l1>
    #device = #ttcore.device<workerGrid = #ttcore.grid<1x1, (d0, d1) -> (0, d0, d1)>,
      l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0),
      dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0),
      meshShape = , chipIds = [0]>
    module attributes {ttcore.device = #device} {
      func.func @bcast_add(%in0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                           %in1: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                           %out: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
        d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<1x1>,
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<compute>]
        } ins(%in0, %in1 :
              memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>)
          outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
        ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>,
                  %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>,
                  %cb_out: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>):
          %c0 = arith.constant 0 : index
          %mem0 = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
                                   -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
          %mem1 = d2m.wait %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
                                   -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
          %mem_out = d2m.reserve %cb_out : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
                                           -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
          // Load for bcast - this load's value flows through tile_bcast to tile_add
          %val0 = affine.load %mem0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
          // Intermediate compute op: tile_bcast
          %bcast = "d2m.tile_bcast"(%val0) <{bcast_type = #d2m<tile_bcast_type col>}>
              : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          // Direct load for second operand
          %val1 = affine.load %mem1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
          // tile_add consumes both: bcast result (from val0) and val1 directly
          // Both %val0 and %val1 must be in DST when tile_add executes
          %result = "d2m.tile_add"(%bcast, %val1)
              : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %result, %mem_out[%c0, %c0]
              : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
        }
        return
      }
    }
  )";

  ParserConfig config(ctx.get(), /*verifyAfterParse=*/false);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);
  ASSERT_TRUE(module);

  auto funcOp = module->lookupSymbol<func::FuncOp>("bcast_add");
  ASSERT_TRUE(funcOp);

  auto analysis = createAnalysis();
  auto result = analysis->analyze(funcOp);
  EXPECT_TRUE(result.isValid) << "Analysis should succeed";

  // The two loads (val0 and val1) must have different DST slots because
  // they both need to be in DST when tile_add executes.
  // Find the load operations and verify they have different assigned slots.
  mlir::Operation *load0 = nullptr;
  mlir::Operation *load1 = nullptr;
  funcOp->walk([&](mlir::affine::AffineLoadOp loadOp) {
    if (!load0) {
      load0 = loadOp;
    } else {
      load1 = loadOp;
    }
  });

  ASSERT_TRUE(load0 != nullptr) << "Should find first load";
  ASSERT_TRUE(load1 != nullptr) << "Should find second load";

  auto *it0 = result.operationSlices.find(load0);
  auto *it1 = result.operationSlices.find(load1);

  ASSERT_TRUE(it0 != result.operationSlices.end())
      << "First load should have assigned slot";
  ASSERT_TRUE(it1 != result.operationSlices.end())
      << "Second load should have assigned slot";

  // The two loads must have DIFFERENT DST slots because they interfere (both
  // needed by tile_add simultaneously).
  EXPECT_NE(it0->second, it1->second)
      << "Loads feeding tile_add (one through bcast) must have different DST "
         "slots. Got slot "
      << it0->second << " for load0 and slot " << it1->second << " for load1";
}

TEST_P(GraphColoringParamTest, CoalescingWithMultiplePairs) {
  // Test coalescing with multiple independent pairs.
  // Graph: 0 -- 1 -- 2 -- 3 -- 4 -- 5
  // Coalescing: (0,1), (2,3), (4,5)
  // Contracted: A -- B -- C (linear, 2-colorable).

  auto coloring = createStrategy();

  std::vector<std::vector<size_t>> origAdjList = {
      {1},    // 0
      {0, 2}, // 1
      {1, 3}, // 2
      {2, 4}, // 3
      {3, 5}, // 4
      {4}     // 5
  };

  llvm::EquivalenceClasses<size_t> ec;
  for (size_t i = 0; i < 6; ++i) {
    ec.insert(i);
  }
  ec.unionSets(0, 1);
  ec.unionSets(2, 3);
  ec.unionSets(4, 5);

  // Remove intra-class edges.
  for (size_t node = 0; node < origAdjList.size(); ++node) {
    auto &neighbors = origAdjList[node];
    neighbors.erase(std::remove_if(neighbors.begin(), neighbors.end(),
                                   [&](size_t neighbor) {
                                     return ec.isEquivalent(node, neighbor);
                                   }),
                    neighbors.end());
  }

  // Build contracted graph.
  llvm::DenseMap<size_t, size_t> nodeToLeader;
  llvm::DenseMap<size_t, size_t> leaderToContracted;
  std::vector<size_t> contractedToLeader;

  for (size_t i = 0; i < 6; ++i) {
    size_t leader = ec.getLeaderValue(i);
    nodeToLeader[i] = leader;
    if (leaderToContracted.find(leader) == leaderToContracted.end()) {
      size_t idx = contractedToLeader.size();
      leaderToContracted[leader] = idx;
      contractedToLeader.push_back(leader);
    }
  }

  EXPECT_EQ(contractedToLeader.size(), 3u);

  std::vector<std::vector<size_t>> contractedAdjList(3);
  for (size_t node = 0; node < 6; ++node) {
    size_t leader = nodeToLeader[node];
    size_t contractedNode = leaderToContracted[leader];
    for (size_t neighbor : origAdjList[node]) {
      size_t neighborLeader = nodeToLeader[neighbor];
      if (leader == neighborLeader) {
        continue;
      }
      size_t contractedNeighbor = leaderToContracted[neighborLeader];
      if (std::find(contractedAdjList[contractedNode].begin(),
                    contractedAdjList[contractedNode].end(),
                    contractedNeighbor) ==
          contractedAdjList[contractedNode].end()) {
        contractedAdjList[contractedNode].push_back(contractedNeighbor);
      }
    }
  }

  // Contracted graph: A -- B -- C (linear chain, 2-colorable).
  std::vector<unsigned> contractedColoring;
  EXPECT_TRUE(succeeded(
      coloring->colorGraph(contractedAdjList, 2, contractedColoring)));
  EXPECT_EQ(contractedColoring.size(), 3u);

  // Propagate colors.
  std::vector<unsigned> coloring_result(6);
  for (size_t i = 0; i < 6; ++i) {
    size_t leader = nodeToLeader[i];
    size_t contractedIdx = leaderToContracted[leader];
    coloring_result[i] = contractedColoring[contractedIdx];
  }

  // Verify coalesced nodes have same color.
  EXPECT_EQ(coloring_result[0], coloring_result[1]);
  EXPECT_EQ(coloring_result[2], coloring_result[3]);
  EXPECT_EQ(coloring_result[4], coloring_result[5]);

  // Verify adjacent contracted nodes have different colors.
  EXPECT_NE(coloring_result[1], coloring_result[2]);
  EXPECT_NE(coloring_result[3], coloring_result[4]);
}

} // namespace mlir::tt::d2m
