// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/AffineMapAnalysis.h"
#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Utils.h"

#include "testing/Utils.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>

namespace ttmlir::utils {

namespace gtest = ::testing;

namespace {

/// Helper to convert ContiguityBound variant to int64_t for test comparisons.
/// Maps: UnconstrainedBound -> -1, ConstrainedBound -> value, UnanalyzableBound
/// -> 1
int64_t contiguityBoundToInt64(ttmlir::utils::ContiguityBound bound) {
  if (std::holds_alternative<ttmlir::utils::UnconstrainedBound>(bound)) {
    return -1;
  }
  if (std::holds_alternative<ttmlir::utils::UnanalyzableBound>(bound)) {
    return 1;
  }
  return std::get<ttmlir::utils::ConstrainedBound>(bound).value;
}

/// Builds an affine expression representing: (sum of (dim_i mod modulus_i) *
/// multiplier_i) floordiv divisor. Dimension indices are generated
/// automatically based on position (0, 1, 2, ...). If multipliers is empty or
/// shorter than moduli, defaults to 1 for missing multipliers.
mlir::AffineExpr buildSumOfModsFloorDivExpr(mlir::MLIRContext *context,
                                            llvm::ArrayRef<int64_t> moduli,
                                            llvm::ArrayRef<int64_t> multipliers,
                                            int64_t divisor) {
  using namespace mlir;
  TT_assert(!moduli.empty());

  AffineExpr modExpr0 =
      getAffineDimExpr(0, context) % getAffineConstantExpr(moduli[0], context);
  int64_t mult0 = multipliers.empty() ? 1 : multipliers[0];
  AffineExpr sum =
      mult0 == 1 ? modExpr0 : modExpr0 * getAffineConstantExpr(mult0, context);

  for (size_t i = 1; i < moduli.size(); ++i) {
    AffineExpr modExpr = getAffineDimExpr(i, context) %
                         getAffineConstantExpr(moduli[i], context);
    int64_t mult = (i < multipliers.size()) ? multipliers[i] : 1;
    AffineExpr term =
        mult == 1 ? modExpr : modExpr * getAffineConstantExpr(mult, context);
    sum = sum + term;
  }

  return sum.floorDiv(getAffineConstantExpr(divisor, context));
}

inline bool verifySimplifiedExprIsConstant(mlir::AffineExpr expr,
                                           int64_t expectedValue) {
  return llvm::dyn_cast<mlir::AffineConstantExpr>(expr)
             ? llvm::cast<mlir::AffineConstantExpr>(expr).getValue() ==
                   expectedValue
             : false;
}

/// Constructs a Sum of Mods Floor Div expression, simplifies it, and verifies
/// the result matches the expected value.
bool testSumOfModsFloorDivExpr(mlir::MLIRContext *context,
                               llvm::ArrayRef<int64_t> moduli,
                               llvm::ArrayRef<int64_t> multipliers,
                               int64_t divisor) {
  using namespace ttmlir::utils;
  mlir::AffineExpr expr =
      buildSumOfModsFloorDivExpr(context, moduli, multipliers, divisor);
  // Create a single-result map, simplify it using the public API
  // simplifyZeroFloorDiv is now internal, but
  // simplifyAffineMapWithRangeAnalysis will apply simplifications. For zero
  // floor div simplification, we use simplifyAffineMapWithRangeAnalysis with
  // large bounds to avoid range-based simplifications interfering.
  mlir::AffineMap map = mlir::AffineMap::get(moduli.size(), 0, {expr}, context);
  // Use very large bounds to avoid range-based simplifications, focusing on
  // zero floor div simplification
  llvm::SmallVector<int64_t> largeBounds(moduli.size(), 1000000);
  mlir::AffineMap simplifiedMap =
      simplifyAffineMapWithRangeAnalysis(map, largeBounds, false);
  mlir::AffineExpr simplified = simplifiedMap.getResult(0);
  return verifySimplifiedExprIsConstant(simplified, 0);
}

/// Builds an affine expression representing: sum of (dim_i mod modulus_i).
/// Dimension indices are generated automatically based on position (0, 1, 2,
/// ...).
mlir::AffineExpr buildSumOfModsExpr(mlir::MLIRContext *context,
                                    llvm::ArrayRef<int64_t> moduli) {
  using namespace mlir;
  TT_assert(!moduli.empty());

  AffineExpr sum =
      getAffineDimExpr(0, context) % getAffineConstantExpr(moduli[0], context);

  for (size_t i = 1; i < moduli.size(); ++i) {
    AffineExpr modExpr = getAffineDimExpr(i, context) %
                         getAffineConstantExpr(moduli[i], context);
    sum = sum + modExpr;
  }

  return sum;
}

/// Checks if a specific dimension expression (dim_i) appears directly in the
/// expression (not inside a mod operation).
bool hasDirectDimExpr(mlir::AffineExpr expr, unsigned dimIndex) {
  if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
    return dimExpr.getPosition() == dimIndex;
  }
  if (auto binOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
    if (binOp.getKind() == mlir::AffineExprKind::Mod) {
      // If we're inside a mod, the dimension is not direct.
      return false;
    }
    // For Add/Mul, check both sides.
    return hasDirectDimExpr(binOp.getLHS(), dimIndex) ||
           hasDirectDimExpr(binOp.getRHS(), dimIndex);
  }
  return false;
}

/// Checks if a specific dimension appears in a mod operation in the expression.
bool hasModForDim(mlir::AffineExpr expr, unsigned dimIndex) {
  if (auto binOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
    if (binOp.getKind() == mlir::AffineExprKind::Mod) {
      // Check if LHS is the dimension we're looking for.
      if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(binOp.getLHS())) {
        return dimExpr.getPosition() == dimIndex;
      }
    }
    return hasModForDim(binOp.getLHS(), dimIndex) ||
           hasModForDim(binOp.getRHS(), dimIndex);
  }
  return false;
}

/// Constructs a Sum of Mods expression, simplifies it with dimension bounds,
/// and verifies that mods are simplified/kept according to expectedPattern.
/// expectedPattern[i] = true means dim_i mod should be simplified away,
/// expectedPattern[i] = false means dim_i mod should remain.
bool testRedundantModSimplification(mlir::MLIRContext *context,
                                    llvm::ArrayRef<int64_t> moduli,
                                    llvm::ArrayRef<int64_t> dimBounds,
                                    llvm::ArrayRef<bool> expectedPattern) {
  using namespace ttmlir::utils;
  TT_assert(moduli.size() == dimBounds.size());
  TT_assert(moduli.size() == expectedPattern.size());

  mlir::AffineExpr expr = buildSumOfModsExpr(context, moduli);
  // Create a single-result map, simplify it, then extract the result
  mlir::AffineMap map = mlir::AffineMap::get(moduli.size(), 0, {expr}, context);
  mlir::AffineMap simplifiedMap =
      simplifyAffineMapWithRangeAnalysis(map, dimBounds, false);
  mlir::AffineExpr simplified = simplifiedMap.getResult(0);

  // Verify each dimension matches the expected pattern.
  for (size_t i = 0; i < expectedPattern.size(); ++i) {
    bool shouldBeSimplified = expectedPattern[i];
    bool hasMod = hasModForDim(simplified, i);
    bool hasDirectDim = hasDirectDimExpr(simplified, i);

    if (shouldBeSimplified) {
      // Mod should be simplified away - dim should appear directly, not in mod.
      if (!hasDirectDim || hasMod) {
        return false;
      }
    } else {
      // Mod should remain - dim should appear in mod operation.
      if (!hasMod) {
        return false;
      }
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Common helpers for coalescing factor tests.
//===----------------------------------------------------------------------===//

/// Test result codes for coalescing factor comparisons.
enum class CoalescingTestResult { Success, Subset, Failed };

/// Compares analytical and sampling coalescing factors and returns the result.
/// Success: factors match exactly.
/// Subset: analytical < sampling and sampling is divisible by analytical.
/// Failed: factors don't match and aren't in subset relationship.
CoalescingTestResult compareCoalescingFactors(int64_t samplingFactor,
                                              int64_t analyticalFactor) {
  if (samplingFactor == analyticalFactor) {
    return CoalescingTestResult::Success;
  }
  if (analyticalFactor != 0 && analyticalFactor < samplingFactor &&
      samplingFactor % analyticalFactor == 0) {
    return CoalescingTestResult::Subset;
  }
  return CoalescingTestResult::Failed;
}

/// Generates a random shape with the given rank using the provided RNG.
/// Each dimension is in range [minDim, maxDim].
llvm::SmallVector<int64_t> generateRandomShape(std::mt19937 &rng, int rank,
                                               int64_t minDim, int64_t maxDim) {
  llvm::SmallVector<int64_t> shape(rank);
  std::uniform_int_distribution<int64_t> dimDist(minDim, maxDim);
  for (int i = 0; i < rank; ++i) {
    shape[i] = dimDist(rng);
  }
  return shape;
}

/// Generates a random valid grid shape for a given logical shape.
/// Each grid dimension is a random divisor of the corresponding logical dim.
llvm::SmallVector<int64_t>
generateRandomGridShape(llvm::ArrayRef<int64_t> logicalShape,
                        std::mt19937 &rng) {
  llvm::SmallVector<int64_t> gridShape;
  for (int64_t dim : logicalShape) {
    auto divisors = ttmlir::utils::getFactors(dim);
    std::uniform_int_distribution<size_t> dist(0, divisors.size() - 1);
    gridShape.push_back(divisors[dist(rng)]);
  }
  return gridShape;
}

/// Generates a random permutation of [0, 1, ..., n-1].
llvm::SmallVector<unsigned> generateRandomPermutation(std::mt19937 &rng,
                                                      unsigned n) {
  llvm::SmallVector<unsigned> perm(n);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);
  return perm;
}

} // namespace

TEST(AffineMapUtilsTest, CanSimplifyZeroFloorDivExpr) {
  using namespace mlir;
  MLIRContext context;

  EXPECT_TRUE(testSumOfModsFloorDivExpr(&context, {4, 5}, {}, 10));

  EXPECT_FALSE(testSumOfModsFloorDivExpr(&context, {2, 8, 9}, {}, 16));
  EXPECT_TRUE(testSumOfModsFloorDivExpr(&context, {2, 8, 9}, {}, 17));

  // Test random cases where divisor equals sum of moduli (no simplification)
  // and sum of moduli + 1 (simplifies to zero).
  auto seed = tt::testing::randomSeed();
  auto gen = tt::testing::createRNG(seed);
  std::uniform_int_distribution<int32_t> unifModulusCount(1, 5);
  std::uniform_int_distribution<int64_t> unifModulus(1, 16384);
  std::uniform_int_distribution<int32_t> unifMultiplier(2, 16384);
  std::uniform_int_distribution<int32_t> unifUseMultiplier(0, 1);

  constexpr int32_t iterations = 100;
  for (int32_t i = 0; i < iterations; ++i) {
    const int32_t modulusCount = unifModulusCount(gen);
    llvm::SmallVector<int64_t> moduli;
    llvm::SmallVector<int64_t> multipliers;
    int64_t maxModSum = 0;

    for (int32_t j = 0; j < modulusCount; ++j) {
      const int64_t modulus = unifModulus(gen);
      moduli.push_back(modulus);

      // Randomly decide whether to include multiplication for this term.
      int64_t multiplier = 1;
      if (unifUseMultiplier(gen)) {
        multiplier = unifMultiplier(gen);
        multipliers.push_back(multiplier);
      } else {
        multipliers.push_back(1);
      }

      maxModSum += (modulus - 1) * multiplier;
    }

    // When divisor equals sum of moduli, it should always simplify to 0.
    EXPECT_FALSE(
        testSumOfModsFloorDivExpr(&context, moduli, multipliers, maxModSum))
        << "Failed for " << modulusCount << " moduli with sum " << maxModSum;
    EXPECT_TRUE(
        testSumOfModsFloorDivExpr(&context, moduli, multipliers, maxModSum + 1))
        << "Failed for " << modulusCount << " moduli with sum "
        << maxModSum + 1;
  }
}

TEST(AffineMapUtilsTest, CanSimplifyRedundantModExpr) {
  using namespace mlir;
  MLIRContext context;

  EXPECT_TRUE(testRedundantModSimplification(&context, {10}, {5}, {true}));
  EXPECT_TRUE(testRedundantModSimplification(&context, {3}, {5}, {false}));
  EXPECT_TRUE(testRedundantModSimplification(&context, {10, 20}, {5, 15},
                                             {true, true}));
  EXPECT_TRUE(testRedundantModSimplification(&context, {3, 5}, {5, 10},
                                             {false, false}));
  EXPECT_TRUE(
      testRedundantModSimplification(&context, {10, 5}, {5, 3}, {true, true}));
  EXPECT_TRUE(testRedundantModSimplification(&context, {10, 3}, {5, 10},
                                             {true, false}));
  EXPECT_TRUE(testRedundantModSimplification(&context, {3, 20}, {5, 15},
                                             {false, true}));
  EXPECT_TRUE(testRedundantModSimplification(&context, {10, 5, 3}, {5, 10, 5},
                                             {true, false, false}));
  EXPECT_TRUE(testRedundantModSimplification(&context, {10, 3, 20}, {5, 10, 15},
                                             {true, false, true}));
  EXPECT_TRUE(testRedundantModSimplification(&context, {2, 10, 4}, {5, 5, 10},
                                             {false, true, false}));
}

/// Collapse leading dimensions of a shape to a target rank.
/// E.g., shape = [2, 3, 4, 5], targetRank = 2 -> [24, 5].
static llvm::SmallVector<int64_t>
collapseLeadingDims(llvm::ArrayRef<int64_t> shape, int targetRank) {
  TT_assertv(targetRank > 0, "Target rank must be positive");
  if (static_cast<int>(shape.size()) <= targetRank) {
    return llvm::SmallVector<int64_t>(shape);
  }

  llvm::SmallVector<int64_t> result;
  size_t dimsToCollapse = shape.size() - targetRank + 1;
  int64_t collapsedDim = 1;
  for (size_t i = 0; i < dimsToCollapse; ++i) {
    collapsedDim *= shape[i];
  }
  result.push_back(collapsedDim);

  for (size_t i = dimsToCollapse; i < shape.size(); ++i) {
    result.push_back(shape[i]);
  }
  return result;
}

/// Computes device shape as [grid dims..., shard dims...] for a given logical
/// shape and grid shape.
static llvm::SmallVector<int64_t>
computeDeviceShape(llvm::ArrayRef<int64_t> logicalShape,
                   llvm::ArrayRef<int64_t> gridShape) {
  TT_assertv(logicalShape.size() == gridShape.size(),
             "Logical and grid shapes must have same rank");
  llvm::SmallVector<int64_t> shape;
  size_t rank = logicalShape.size();
  shape.reserve(rank * 2);
  // Append the grid shape dims
  for (size_t i = 0; i < rank; ++i) {
    shape.push_back(gridShape[i]);
  }
  // Append the logicalShape dims divided by gridShape dims ("shard shape")
  for (size_t i = 0; i < rank; ++i) {
    TT_assertv(gridShape[i] != 0, "Grid shape dimension must not be zero");
    TT_assertv(
        logicalShape[i] % gridShape[i] == 0,
        "Logical shape dimension must be divisible by grid shape dimension");
    shape.push_back(logicalShape[i] / gridShape[i]);
  }
  return shape;
}

/// Returns an affine map and input device shape for reblocking, collapsing any
/// extra leading dimensions on the smaller rank side.
static std::tuple<mlir::AffineMap, llvm::SmallVector<int64_t>>
getReblockMapAndDeviceShapeMixedRank(mlir::ArrayRef<int64_t> logicalShape,
                                     int inputRank, int outputRank,
                                     mlir::ArrayRef<int64_t> inputGridShape,
                                     mlir::ArrayRef<int64_t> outputGridShape,
                                     mlir::MLIRContext *context) {
  int maxRank = std::max(inputRank, outputRank);
  TT_assertv(static_cast<int>(logicalShape.size()) == maxRank,
             "Logical shape must have max rank");
  TT_assertv(static_cast<int>(inputGridShape.size()) == inputRank,
             "Input grid shape must match input rank");
  TT_assertv(static_cast<int>(outputGridShape.size()) == outputRank,
             "Output grid shape must match output rank");

  // Collapse logical shape to match input and output ranks.
  auto inputLogicalShape = collapseLeadingDims(logicalShape, inputRank);
  auto outputLogicalShape = collapseLeadingDims(logicalShape, outputRank);

  llvm::SmallVector<int64_t> deviceShapeInputGrid =
      computeDeviceShape(inputLogicalShape, inputGridShape);
  llvm::SmallVector<int64_t> deviceShapeOutputGrid =
      computeDeviceShape(outputLogicalShape, outputGridShape);

  // calculateReblockMap(A, B) creates a map from B's dims to A's indices.
  // We want: input device coords (domain) → output device coords (underlying).
  // So call calculateReblockMap(outputDevice, inputDevice).
  // Result: inputDevice.size() dims → outputDevice.size() results.
  mlir::AffineMap reblockMap = ttmlir::utils::calculateReblockMap(
      deviceShapeOutputGrid, deviceShapeInputGrid, context);
  // simplifyZeroFloorDiv is now internal, so we use
  // simplifyAffineMapWithRangeAnalysis which internally applies
  // simplifyZeroFloorDiv
  reblockMap = simplifyAffineMapWithRangeAnalysis(reblockMap,
                                                  deviceShapeInputGrid, false);

  // The layout map converts output device indices to memory addresses
  // (since the underlying memory is laid out according to output device shape).
  auto strides = ttmlir::utils::calculateStrides<int64_t>(
      llvm::ArrayRef<int64_t>(deviceShapeOutputGrid)
          .take_back(deviceShapeOutputGrid.size() / 2),
      1);
  auto layout_map =
      ttmlir::utils::generateAffineMapFromShardStrides(strides, context);

  // Compose: layout_map expects outputRank*2 dims, reblockMap produces
  // outputRank*2 results.
  TT_assertv(layout_map.getNumDims() == reblockMap.getNumResults(),
             "Dimension mismatch for compose");
  auto memoryMap = layout_map.compose(reblockMap);

  // Return the input device shape since that's the iteration domain.
  return std::make_tuple(memoryMap, deviceShapeInputGrid);
}

TEST(AffineMapUtilsTest, DISABLED_CanDetermineCoalescingFactor) {
  using namespace mlir;
  MLIRContext context;

  // Test result with additional info for reblock-specific logging.
  struct ReblockTestInfo {
    CoalescingTestResult result;
    mlir::AffineMap memoryMap;
    llvm::SmallVector<int64_t> inputDeviceShape;
    llvm::SmallVector<int64_t> outputDeviceShape;
    size_t coalescingFactor;
    size_t coalescingFactorAnalytical;
  };

  // Test case for reblocking (supports same and mixed ranks).
  auto testReblock =
      [&](llvm::ArrayRef<int64_t> logicalShape, int inputRank, int outputRank,
          llvm::ArrayRef<int64_t> inputGridShape,
          llvm::ArrayRef<int64_t> outputGridShape) -> ReblockTestInfo {
    auto inputLogicalShape = collapseLeadingDims(logicalShape, inputRank);
    auto outputLogicalShape = collapseLeadingDims(logicalShape, outputRank);
    auto inputDeviceShape =
        computeDeviceShape(inputLogicalShape, inputGridShape);
    auto outputDeviceShape =
        computeDeviceShape(outputLogicalShape, outputGridShape);

    auto [memoryMap, deviceShape] = getReblockMapAndDeviceShapeMixedRank(
        logicalShape, inputRank, outputRank, inputGridShape, outputGridShape,
        &context);

    auto coalescingFactorAnalytical = computeCoalescingFactorAnalytically(
        memoryMap, deviceShape, memoryMap.getNumDims() / 2, 1);

    size_t coalescingFactor = calculateCoalescingFactor(
        memoryMap, deviceShape, 1, memoryMap.getNumDims() / 2);

    CoalescingTestResult result =
        compareCoalescingFactors(coalescingFactor, coalescingFactorAnalytical);

    return {result,           memoryMap,
            inputDeviceShape, outputDeviceShape,
            coalescingFactor, coalescingFactorAnalytical};
  };

  // Configuration.
  constexpr int64_t maxCollapsedDim = 8192;
  constexpr int numTestCasesPerRankPair = 16;

  std::mt19937 rng(42); // Deterministic seed for reproducibility.

  // Helper to generate a random logical shape with given rank.
  // Ensures the product of leading dims that would be collapsed to any
  // smaller rank doesn't exceed maxCollapsedDim.
  auto generateLogicalShape = [&](int rank) -> SmallVector<int64_t> {
    SmallVector<int64_t> shape(rank);
    std::uniform_int_distribution<int64_t> dimDist(2, 16);

    // Generate dimensions from the end (least significant) to the front.
    // This ensures we can control the collapsed product.
    int64_t remainingBudget = maxCollapsedDim;
    for (int i = rank - 1; i >= 0; --i) {
      int64_t maxDim = std::min<int64_t>(16, remainingBudget);
      if (maxDim < 2) {
        maxDim = 2;
      }
      std::uniform_int_distribution<int64_t> boundedDimDist(2, maxDim);
      shape[i] = boundedDimDist(rng);
      // Update budget for remaining (more leading) dimensions.
      // We need the product of dims [0..i-1] to not exceed maxCollapsedDim.
      if (i > 0) {
        remainingBudget = maxCollapsedDim / shape[i];
        if (remainingBudget < 2) {
          remainingBudget = 2;
        }
      }
    }
    return shape;
  };

  // Test mixed-rank reblocking: input rank from 2-6, output rank from 2-6.
  std::uniform_int_distribution<int> rankDist(2, 6);

  for (int testIdx = 0; testIdx < numTestCasesPerRankPair * 4; ++testIdx) {
    int inputRank = rankDist(rng);
    int outputRank = rankDist(rng);
    int maxRank = std::max(inputRank, outputRank);

    // Generate logical shape with max rank.
    SmallVector<int64_t> logicalShape = generateLogicalShape(maxRank);

    // Collapse to get the shapes for input and output grids.
    auto inputLogicalShape = collapseLeadingDims(logicalShape, inputRank);
    auto outputLogicalShape = collapseLeadingDims(logicalShape, outputRank);

    // Verify collapsed dim constraint.
    if (inputLogicalShape[0] > maxCollapsedDim ||
        outputLogicalShape[0] > maxCollapsedDim) {
      continue; // Skip this test case.
    }

    // Generate grid shapes for each.
    SmallVector<int64_t> inputGridShape =
        generateRandomGridShape(inputLogicalShape, rng);
    SmallVector<int64_t> outputGridShape =
        generateRandomGridShape(outputLogicalShape, rng);

    auto info = testReblock(logicalShape, inputRank, outputRank, inputGridShape,
                            outputGridShape);

    EXPECT_TRUE(info.result == CoalescingTestResult::Success)
        << "Failed for inputRank=" << inputRank
        << ", outputRank=" << outputRank;

    // Mixed-rank: currently just verifies no crashes; analytical method
    // may not handle asymmetric grid/shard structures correctly.
  }
}

// Pre-cached test cases extracted from CanDetermineCoalescingFactor.
// Each entry contains reblocking parameters that are used to construct the
// affine map directly (avoiding string parsing). This validates
// computeCoalescingFactorAnalytically() against known-good results from
// calculateCoalescingFactor() without the runtime cost of sampling.
TEST(AffineMapUtilsTest, CanDetermineCoalescingFactorCached) {
  using namespace mlir;
  MLIRContext context;

  struct CachedTestCase {
    llvm::SmallVector<int64_t> logicalShape;
    int inputRank;
    int outputRank;
    llvm::SmallVector<int64_t> inputGridShape;
    llvm::SmallVector<int64_t> outputGridShape;
    int64_t expectedCoalescingFactor;
  };

  // clang-format off
  const llvm::SmallVector<CachedTestCase> testCases = {
    // Test case 0.
    {{10, 13, 12, 4, 16}, 3, 5, {65, 1, 4}, {1, 1, 1, 2, 16}, 1},
    // Test case 1.
    {{2, 2, 11, 12, 4}, 3, 5, {44, 6, 4}, {2, 1, 1, 2, 4}, 2},
    // Test case 2.
    {{8, 2, 9, 11, 6}, 2, 5, {1, 2}, {4, 2, 3, 1, 1}, 3},
    // Test case 3.
    {{11, 13, 3, 8, 5, 7}, 3, 6, {11, 1, 7}, {11, 13, 1, 1, 5, 7}, 1},
    // Test case 4.
    {{16, 16, 2, 2, 8}, 5, 2, {16, 4, 2, 1, 2}, {1, 1}, 4},
    // Test case 5.
    {{11, 3, 12, 8, 5}, 3, 5, {18, 8, 1}, {1, 3, 3, 2, 1}, 5},
    // Test case 6.
    {{10, 5, 9, 8, 6}, 5, 5, {5, 1, 1, 8, 6}, {10, 1, 9, 2, 6}, 1},
    // Test case 7.
    {{4, 10, 3, 6, 15, 12}, 6, 4, {2, 1, 3, 2, 15, 3}, {24, 2, 5, 6}, 2},
    // Test case 8.
    {{10, 11, 6, 16}, 4, 3, {2, 1, 2, 16}, {2, 1, 1}, 1},
    // Test case 9.
    {{2, 2, 6, 4, 7, 13}, 6, 4, {2, 1, 3, 4, 7, 13}, {12, 2, 1, 13}, 1},
    // Test case 10.
    {{11, 14, 14, 15, 3}, 3, 5, {44, 3, 1}, {1, 2, 2, 5, 1}, 3},
    // Test case 11.
    {{10, 15, 6, 11, 10}, 5, 5, {2, 3, 1, 11, 5}, {10, 15, 3, 11, 1}, 2},
    // Test case 12.
    {{8, 12, 9, 2, 9}, 5, 3, {1, 1, 3, 1, 1}, {1, 2, 3}, 3},
    // Test case 13.
    {{15, 12, 9, 10}, 4, 3, {1, 2, 3, 2}, {15, 9, 1}, 5},
    // Test case 14.
    {{15, 4, 12, 6, 10, 3}, 3, 6, {1080, 5, 3}, {3, 2, 1, 6, 2, 3}, 1},
    // Test case 15.
    {{15, 15, 8}, 3, 2, {5, 3, 8}, {3, 8}, 5},
    // Test case 16.
    {{6, 3, 15}, 3, 3, {1, 3, 3}, {1, 3, 3}, 30},
    // Test case 17.
    {{9, 8, 10, 9, 4, 2}, 6, 3, {1, 4, 1, 1, 2, 1}, {2160, 1, 1}, 4},
    // Test case 18.
    {{7, 8, 12, 10}, 3, 4, {1, 12, 2}, {7, 1, 2, 5}, 1},
    // Test case 19.
    {{2, 16, 6, 4, 6}, 4, 5, {2, 3, 4, 3}, {1, 1, 1, 1, 2}, 1},
    // Test case 20.
    {{8, 9, 9, 4, 12, 5}, 6, 5, {8, 3, 1, 2, 6, 1}, {24, 3, 1, 2, 5}, 1},
    // Test case 21.
    {{5, 12, 6, 10, 14}, 5, 2, {1, 6, 3, 10, 1}, {25, 7}, 2},
    // Test case 22.
    {{12, 16}, 2, 2, {3, 2}, {4, 16}, 1},
    // Test case 23.
    {{14, 4, 11, 5, 5, 12}, 4, 6, {616, 5, 5, 4}, {2, 1, 11, 1, 5, 12}, 1},
    // Test case 24.
    {{7, 6, 7, 4, 11, 15}, 4, 6, {49, 4, 11, 15}, {7, 6, 1, 4, 11, 5}, 1},
    // Test case 25.
    {{11, 16, 3, 7}, 4, 2, {11, 1, 1, 1}, {11, 7}, 1},
    // Test case 26.
    {{4, 5, 16, 11, 8}, 4, 5, {10, 1, 1, 2}, {1, 1, 8, 1, 4}, 2},
    // Test case 27.
    {{3, 10, 10, 6, 11, 12}, 5, 6, {6, 2, 3, 1, 2}, {1, 5, 10, 6, 1, 3}, 2},
    // Test case 28.
    {{11, 9, 7, 13, 8, 11}, 6, 6, {11, 9, 1, 13, 1, 1}, {11, 3, 1, 13, 1, 11}, 1},
    // Test case 29.
    {{15, 16, 2, 13, 7, 15}, 3, 6, {30, 1, 5}, {15, 4, 2, 13, 7, 1}, 3},
    // Test case 30.
    {{14, 4, 7}, 3, 3, {2, 1, 1}, {1, 2, 7}, 1},
    // Test case 31.
    {{3, 3, 10, 15, 12, 14}, 3, 6, {9, 4, 7}, {3, 1, 1, 3, 4, 2}, 1},
    // Test case 32.
    {{12, 12, 4, 12, 14, 13}, 6, 3, {3, 4, 1, 2, 14, 1}, {768, 2, 13}, 1},
    // Test case 33.
    {{13, 11, 9, 2, 9, 3}, 5, 6, {1, 3, 1, 9, 1}, {13, 1, 9, 2, 3, 1}, 3},
    // Test case 34.
    {{12, 10, 8, 3}, 3, 4, {1, 2, 1}, {1, 5, 8, 1}, 3},
    // Test case 35.
    {{14, 2, 2, 5, 2, 14}, 6, 4, {2, 1, 2, 1, 1, 7}, {28, 5, 1, 1}, 2},
    // Test case 36.
    {{9, 4, 2, 14, 3, 8}, 6, 5, {3, 2, 2, 7, 1, 4}, {2, 2, 14, 3, 4}, 2},
    // Test case 37.
    {{6, 7, 13}, 3, 2, {2, 1, 13}, {1, 1}, 1},
    // Test case 38.
    {{3, 16}, 2, 2, {3, 4}, {3, 8}, 2},
    // Test case 39.
    {{8, 4, 15, 4}, 4, 2, {2, 1, 5, 1}, {60, 1}, 4},
    // Test case 40.
    {{11, 13, 9, 16, 11, 6}, 5, 6, {13, 3, 8, 11, 2}, {1, 1, 3, 2, 1, 6}, 1},
    // Test case 41.
    {{3, 7}, 2, 2, {1, 1}, {1, 1}, 21},
    // Test case 42.
    {{3, 2, 9, 4, 12, 16}, 3, 6, {54, 3, 16}, {1, 1, 1, 2, 3, 8}, 1},
    // Test case 43.
    {{11, 16, 11}, 3, 2, {11, 1, 1}, {8, 1}, 22},
    // Test case 44.
    {{12, 11, 10, 14, 9, 9}, 5, 6, {2, 5, 1, 9, 3}, {12, 1, 10, 7, 3, 9}, 1},
    // Test case 45.
    {{8, 3, 11, 10, 7, 12}, 6, 4, {8, 3, 11, 10, 7, 3}, {88, 10, 7, 12}, 1},
    // Test case 46.
    {{5, 3, 6}, 3, 2, {1, 1, 1}, {1, 1}, 90},
    // Test case 47.
    {{7, 6, 11, 3, 7}, 5, 5, {7, 3, 1, 1, 7}, {1, 2, 1, 1, 7}, 33},
    // Test case 48.
    {{13, 2, 15, 16, 11}, 5, 3, {1, 2, 15, 2, 11}, {3, 4, 11}, 4},
    // Test case 49.
    {{7, 3, 8, 12, 16, 10}, 4, 6, {6, 6, 4, 2}, {7, 3, 4, 12, 2, 2}, 20},
    // Test case 50.
    {{7, 3, 5, 13, 13}, 5, 5, {7, 3, 5, 1, 13}, {1, 1, 1, 13, 1}, 1},
    // Test case 51.
    {{15, 16, 2}, 3, 2, {15, 1, 1}, {8, 2}, 1},
    // Test case 52.
    {{3, 11, 9, 10, 9, 16}, 6, 5, {1, 1, 1, 5, 1, 4}, {11, 1, 10, 9, 8}, 2},
    // Test case 53.
    {{16, 9, 15, 3, 3}, 5, 5, {1, 9, 5, 1, 1}, {8, 9, 5, 1, 3}, 1},
    // Test case 54.
    {{5, 3, 9, 3, 4, 12}, 6, 6, {5, 3, 3, 3, 2, 12}, {5, 1, 9, 3, 4, 4}, 1},
    // Test case 55.
    {{13, 9, 3, 12, 7}, 5, 3, {1, 3, 3, 3, 1}, {351, 2, 1}, 14},
    // Test case 56.
    {{9, 8, 2, 5}, 4, 4, {3, 1, 2, 1}, {9, 1, 2, 5}, 1},
  };
  // clang-format on

  for (size_t i = 0; i < testCases.size(); ++i) {
    const auto &tc = testCases[i];

    // Construct affine map directly using the same infrastructure as the
    // original test.
    auto [memoryMap, deviceShape] = getReblockMapAndDeviceShapeMixedRank(
        tc.logicalShape, tc.inputRank, tc.outputRank, tc.inputGridShape,
        tc.outputGridShape, &context);

    unsigned numGridDims = deviceShape.size() / 2;

    // Use the fast analytical method.
    int64_t analyticalFactor = computeCoalescingFactorAnalytically(
        memoryMap, deviceShape, numGridDims, /*elemSizeBytes=*/1);

    EXPECT_EQ(analyticalFactor, tc.expectedCoalescingFactor)
        << "Test case " << i << " failed";
  }
}

TEST(AffineMapUtilsTest, CanDetermineCoalescingFactorForPermutations) {
  using namespace mlir;
  MLIRContext context;

  // Test a permutation applied to a device shape.
  // Creates a permutation map and composes it with a row-major layout to
  // create a memory access pattern, then verifies that analytical and sampling
  // coalescing factors match.
  //
  // @param deviceShape The device shape [grid dims..., shard dims...].
  // @param permutation The permutation to apply (must be same size as
  // deviceShape).
  // @return Tuple of (result, memoryMap, samplingFactor, analyticalFactor).
  auto testPermutation = [&](llvm::ArrayRef<int64_t> deviceShape,
                             llvm::ArrayRef<unsigned> permutation)
      -> std::tuple<CoalescingTestResult, AffineMap, int64_t, int64_t> {
    unsigned numDims = deviceShape.size();
    unsigned numGridDims = numDims / 2;

    // Create the permutation map: (d0, d1, ...) -> (d_perm[0], d_perm[1], ...).
    AffineMap permMap = AffineMap::getPermutationMap(permutation, &context);

    // Apply permutation to get the permuted device shape.
    SmallVector<int64_t> permutedDeviceShape;
    for (unsigned p : permutation) {
      permutedDeviceShape.push_back(deviceShape[p]);
    }

    // Create row-major layout strides for the permuted shard shape.
    // The shard dims are the last numGridDims elements of the permuted shape.
    auto permutedShardShape =
        llvm::ArrayRef<int64_t>(permutedDeviceShape).take_back(numGridDims);
    auto strides =
        ttmlir::utils::calculateStrides<int64_t>(permutedShardShape, 1);

    // Create layout map: (grid dims..., shard dims...) -> (grid dims...,
    // linearized offset).
    auto layoutMap =
        ttmlir::utils::generateAffineMapFromShardStrides(strides, &context);

    // Compose layout with permutation to get memory access pattern.
    // layoutMap operates on permuted coordinates, permMap converts original ->
    // permuted.
    auto memoryMap = layoutMap.compose(permMap);

    // Simplify the map using range analysis.
    // simplifyZeroFloorDiv is now internal, so we use
    // simplifyAffineMapWithRangeAnalysis which internally applies
    // simplifyZeroFloorDiv
    memoryMap =
        simplifyAffineMapWithRangeAnalysis(memoryMap, deviceShape, false);

    // Analyze coalescing factor using both methods.
    auto analyticalFactor = computeCoalescingFactorAnalytically(
        memoryMap, deviceShape, numGridDims, 1);

    int64_t samplingFactor =
        calculateCoalescingFactor(memoryMap, deviceShape, 1, numGridDims);

    CoalescingTestResult result =
        compareCoalescingFactors(samplingFactor, analyticalFactor);

    return {result, memoryMap, samplingFactor, analyticalFactor};
  };

  // Configuration.
  constexpr int numRandomTests = 64;
  std::mt19937 rng(123); // Deterministic seed.

  // Test 1: Identity permutation should be fully contiguous.
  {
    SmallVector<int64_t> deviceShape = {2, 3, 4, 5}; // 2D grid, 2D shard.
    SmallVector<unsigned> identity = {0, 1, 2, 3};

    auto [result, memoryMap, samplingFactor, analyticalFactor] =
        testPermutation(deviceShape, identity);

    // Identity should give full shard volume (4*5 = 20).
    EXPECT_EQ(samplingFactor, 20) << "Identity permutation should be fully "
                                     "contiguous within shard";
    EXPECT_TRUE(result == CoalescingTestResult::Success)
        << "Identity permutation failed";
  }

  // Test 2: Swap last two shard dims (should break contiguity).
  {
    SmallVector<int64_t> deviceShape = {2, 3, 4, 5}; // 2D grid, 2D shard.
    SmallVector<unsigned> swapLast = {0, 1, 3, 2};   // Swap d2 and d3.

    auto [result, memoryMap, samplingFactor, analyticalFactor] =
        testPermutation(deviceShape, swapLast);

    EXPECT_TRUE(result == CoalescingTestResult::Success)
        << "Swap last two dims failed";
  }

  // Test 3: Swap grid dims only (should still be contiguous within shards).
  {
    SmallVector<int64_t> deviceShape = {2, 3, 4, 5}; // 2D grid, 2D shard.
    SmallVector<unsigned> swapGrid = {1, 0, 2, 3};   // Swap d0 and d1.

    auto [result, memoryMap, samplingFactor, analyticalFactor] =
        testPermutation(deviceShape, swapGrid);

    EXPECT_TRUE(result == CoalescingTestResult::Success)
        << "Swap grid dims failed";
  }

  // Test 4: Complete reversal.
  {
    SmallVector<int64_t> deviceShape = {2, 3, 4, 5}; // 2D grid, 2D shard.
    SmallVector<unsigned> reverse = {3, 2, 1, 0};    // Complete reversal.

    auto [result, memoryMap, samplingFactor, analyticalFactor] =
        testPermutation(deviceShape, reverse);

    EXPECT_TRUE(result == CoalescingTestResult::Success)
        << "Complete reversal failed";
  }

  // Test 5: Random permutations with varying ranks.
  std::uniform_int_distribution<int> rankDist(2, 4); // Grid rank from 2-4.

  for (int testIdx = 0; testIdx < numRandomTests; ++testIdx) {
    int gridRank = rankDist(rng);
    int totalDims = gridRank * 2; // Equal grid and shard dims.

    // Generate random device shape with small dimensions for fast sampling.
    auto deviceShape = generateRandomShape(rng, totalDims, 2, 6);

    // Generate random permutation.
    auto permutation =
        generateRandomPermutation(rng, static_cast<unsigned>(totalDims));

    auto [result, memoryMap, samplingFactor, analyticalFactor] =
        testPermutation(deviceShape, permutation);

    EXPECT_TRUE(result == CoalescingTestResult::Success)
        << "Failed for random permutation #" << testIdx
        << " with gridRank=" << gridRank;
  }
}

// Isolated test case for debugging a specific FAIL case.
TEST(AffineMapUtilsTest, CanTestSingleCoalescingFactorMismatch) {
  using namespace mlir;
  MLIRContext context;

  // Failing test case:
  // affine map: (d0, d1, d2, d3, d4, d5, d6, d7) -> (
  //   0,
  //   (((((d3 * 32 + d7) mod 128) floordiv 32) mod 4) floordiv 2) mod 2,
  //   (((d3 * 32 + d7) mod 128) floordiv 32) mod 2,
  //   ((((((d6 + (d3 * 32 + d7) floordiv 128) mod 32) * 128 +
  //       (d3 * 32 + d7) mod 128) mod 2304) floordiv 128 +
  //     ((d3 * 32 + d7) mod 128) floordiv 128) mod 32) * 128 + (d7 mod 32) * 4
  // )
  // shape: (1x1x1x4x1x1x32x32)
  AffineExpr d0, d1, d2, d3, d4, d5, d6, d7;
  bindDims(&context, d0, d1, d2, d3, d4, d5, d6, d7);

  // Build (d3 * 32 + d7) - common subexpression
  AffineExpr d3_32_d7 = d3 * 32 + d7;

  // r0 = 0
  AffineExpr r0 = getAffineConstantExpr(0, &context);

  // r1 = (((((d3 * 32 + d7) mod 128) floordiv 32) mod 4) floordiv 2) mod 2
  AffineExpr r1 = (((((d3_32_d7) % 128).floorDiv(32)) % 4).floorDiv(2)) % 2;

  // r2 = (((d3 * 32 + d7) mod 128) floordiv 32) mod 2
  AffineExpr r2 = (((d3_32_d7) % 128).floorDiv(32)) % 2;

  // r3 = ((((((d6 + (d3 * 32 + d7) floordiv 128) mod 32) * 128 +
  //         (d3 * 32 + d7) mod 128) mod 2304) floordiv 128 +
  //       ((d3 * 32 + d7) mod 128) floordiv 128) mod 32) * 128 + (d7 mod 32) *
  //       4
  AffineExpr innerSum = (d6 + (d3_32_d7).floorDiv(128)) % 32;
  AffineExpr bigExpr = (innerSum * 128 + (d3_32_d7) % 128) % 2304;
  AffineExpr r3 =
      (((bigExpr.floorDiv(128) + ((d3_32_d7) % 128).floorDiv(128)) % 32) * 128 +
       (d7 % 32) * 4);

  SmallVector<AffineExpr> results{r0, r1, r2, r3};
  AffineMap memoryMap =
      AffineMap::get(/*dimCount=*/8, /*symbolCount=*/0, results, &context);

  SmallVector<int64_t> inputDeviceShape = {1, 1, 1, 4, 1, 1, 32, 32};

  constexpr int64_t elemSizeBytes = 1;
  unsigned numGridDims = inputDeviceShape.size() / 2; // 4.

  auto coalescingFactorAnalytical = computeCoalescingFactorAnalytically(
      memoryMap, inputDeviceShape, numGridDims, elemSizeBytes);

  int64_t coalescingFactorSampling = calculateCoalescingFactor(
      memoryMap, inputDeviceShape, elemSizeBytes, numGridDims);

  EXPECT_EQ(coalescingFactorAnalytical, coalescingFactorSampling);
}

TEST(AffineMapUtilsTest, AnalyzeGridResultExprForDiscontinuity) {
  using namespace mlir;
  MLIRContext context;

  // Helper to create dimension bounds map.
  auto makeDimBounds = [](std::initializer_list<std::pair<int, int64_t>> bounds)
      -> llvm::DenseMap<int, int64_t> {
    llvm::DenseMap<int, int64_t> result;
    for (auto [pos, bound] : bounds) {
      result[pos] = bound;
    }
    return result;
  };

  // Test 1: Simple dimension expression d0 -> should return 1.
  // Every step in d0 changes the output.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    auto dimBounds = makeDimBounds({{0, 8}});
    auto result = analyzeGridResultExprForDiscontinuity(d0, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), 1)
        << "d0 should require 1 step to change output";
  }

  // Test 2: d0 floorDiv N -> should return N.
  // Need N steps in d0 to change the output.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = d0.floorDiv(4);
    auto dimBounds = makeDimBounds({{0, 16}});
    auto result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), 4)
        << "d0 floorDiv 4 should require 4 steps";
  }

  // Test 3: (d0 * M) floorDiv N where N % M == 0 -> should return 4.
  // Since M*d0 changes by M each step, and N/M steps cross the boundary.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = (d0 * 2).floorDiv(8);
    auto dimBounds = makeDimBounds({{0, 16}});
    auto result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    // d0 * 2 means values 0, 2, 4, 6, 8, ... so crossing 8 takes 4 steps.
    EXPECT_EQ(contiguityBoundToInt64(result), 4)
        << "(d0 * 2) floorDiv 8 should require 4 steps";
  }

  // Test 4: Expression with unrelated dimension -> should return -1.
  // d1 floorDiv N when analyzing for d0.
  {
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = d1.floorDiv(4);
    auto dimBounds = makeDimBounds({{0, 8}, {1, 16}});
    auto result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), -1)
        << "d1 floorDiv 4 should be unconstrained for d0";
  }

  // Test 5: Constant expression -> should return -1 (unconstrained).
  {
    AffineExpr constExpr = getAffineConstantExpr(42, &context);
    auto dimBounds = makeDimBounds({{0, 8}});
    auto result =
        analyzeGridResultExprForDiscontinuity(constExpr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), -1)
        << "Constant should be unconstrained";
  }

  // Test 6: (A*d0 + B*d1) floorDiv N - multiple dimensions.
  // Testing the minimizeGap integration.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    // (7*d0 + 5*d1) floorDiv 11.
    // d0 has bound 3 (values 0, 1, 2), d1 has bound 10 (values 0-9).
    // Best alignment for d1: 5*8 = 40, plus 7*2 = 14, total 54, gap to 55 is 1.
    AffineExpr expr = (d0 * 7 + d1 * 5).floorDiv(11);
    auto dimBounds = makeDimBounds({{0, 3}, {1, 10}});
    auto result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    // The gap is minimized by d1, and we compute ceil(gap / 7).
    EXPECT_EQ(contiguityBoundToInt64(result), 1)
        << "(7*d0 + 5*d1) floorDiv 11 should return >= 1";
  }

  // Test 7: Add expression combining constrained and unconstrained.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    // d0 + d1 floorDiv 4 - d0 is direct (returns 1), d1 floorDiv 4 returns 4.
    // Combined via add should return gcd(1, 4) = 1.
    AffineExpr expr = d0 + d1.floorDiv(4);
    auto dimBounds = makeDimBounds({{0, 8}, {1, 16}});
    auto result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), 1)
        << "d0 + (d1 floorDiv 4) should return 1 for d0";
  }

  // Test 8: Mul expression with target dim (not inside floorDiv).
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = d0 * 3;
    auto dimBounds = makeDimBounds({{0, 8}});
    auto result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), 1)
        << "d0 * 3 should return 1 (any change in d0 changes "
           "output)";
  }

  // Test 9: Larger floorDiv value.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = d0.floorDiv(32);
    auto dimBounds = makeDimBounds({{0, 128}});
    auto result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), 32)
        << "d0 floorDiv 32 should require 32 steps";
  }

  // Test 10: Complex expression with constant offset.
  // (d0 * 4 + 2) floorDiv 8 -> gap is 8 - 2 = 6, ceil(6/4) = 2.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = (d0 * 4 + 2).floorDiv(8);
    auto dimBounds = makeDimBounds({{0, 16}});
    auto result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    // With offset 2 and multiplier 4, we need ceil((8-2)/4) = 2 steps.
    EXPECT_EQ(contiguityBoundToInt64(result), 2)
        << "(d0 * 4 + 2) floorDiv 8 should require 2 steps";
  }

  // Test 11: Two dimensions with multipliers - (2*d0 + 3*d1) floorDiv 10.
  // d0 is target, d1 has bound 4 (values 0,1,2,3), so 3*d1 can be 0,3,6,9.
  // Best case for d1: 3*3=9, gap to 10 is 1, so ceil(1/2)=1.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d0 * 2 + d1 * 3).floorDiv(10);
    auto dimBounds = makeDimBounds({{0, 10}, {1, 4}});
    auto result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    // d1 can achieve 3*3=9, gap to 10 is 1, ceil(1/2) = 1.
    EXPECT_EQ(contiguityBoundToInt64(result), 1)
        << "(2*d0 + 3*d1) floorDiv 10 should require 1 step when d1 can align";
  }

  // Test 12: Two dimensions where other dim can achieve exact multiple.
  // (d0 + 5*d1) floorDiv 10, d1 has bound 3 (values 0,1,2).
  // d1=2 gives 5*2=10 which is exactly 10, gap=0, but minimizeGap returns 1.
  // as best achievable gap, so ceil(1/1) = 1.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d0 + d1 * 5).floorDiv(10);
    auto dimBounds = makeDimBounds({{0, 20}, {1, 3}});
    auto result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    // d1=2 gives 10, which is exact multiple, gap=0 means any step changes.
    // But actually minimizeGap with gap=0 would return bestGap=0, not 1.
    // When gap=0, we still need at least 1 step to change (can't be 0).
    EXPECT_EQ(contiguityBoundToInt64(result), 5)
        << "(d0 + 5*d1) floorDiv 10 should return 5";
  }

  // Test 13: Three dimensions - (d0 + 2*d1 + 3*d2) floorDiv 12.
  // d1 has bound 5 (0-4), d2 has bound 4 (0-3).
  // Achievable sums from d1,d2: 0,2,3,4,5,6,7,8,9,10,11.
  // Best alignment to 12: 11 (gap=1), ceil(1/1) = 1.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr d2 = getAffineDimExpr(2, &context);
    AffineExpr expr = (d0 + d1 * 2 + d2 * 3).floorDiv(12);
    auto dimBounds = makeDimBounds({{0, 20}, {1, 5}, {2, 4}});
    auto result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), 1)
        << "(d0 + 2*d1 + 3*d2) floorDiv 12 should require 1 step";
  }

  // Test 14: Two dims with larger multiplier on target dim.
  // (3*d0 + 2*d1) floorDiv 7, d1 has bound 4 (0-3).
  // 2*d1 can be 0,2,4,6. Best alignment to 7: 6 (gap=1), ceil(1/3)=1.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d0 * 3 + d1 * 2).floorDiv(7);
    auto dimBounds = makeDimBounds({{0, 10}, {1, 4}});
    auto result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    // 2*3=6, gap to 7 is 1, ceil(1/3) = 1.
    EXPECT_EQ(contiguityBoundToInt64(result), 1)
        << "(3*d0 + 2*d1) floorDiv 7 should require 1 step";
  }

  // Test 15: Multiple dims with constant offset.
  // (2*d0 + 5*d1 + 3) floorDiv 11, d1 has bound 3 (0-2).
  // 5*d1 + 3 can be 3,8,13. Modulo 11: 3,8,2. Best to 11: 8 (gap=3).
  // ceil(3/2) = 2.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d0 * 2 + d1 * 5 + 3).floorDiv(11);
    auto dimBounds = makeDimBounds({{0, 20}, {1, 3}});
    auto result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    // 5*1+3=8, gap to 11 is 3, ceil(3/2) = 2.
    EXPECT_EQ(contiguityBoundToInt64(result), 2)
        << "(2*d0 + 5*d1 + 3) floorDiv 11 should require 2 steps";
  }

  // Test 16: Case where no alignment is possible - prime divisor.
  // (d0 + 2*d1) floorDiv 7, d1 has bound 3 (0-2).
  // 2*d1 can be 0,2,4. None divides 7 evenly. Best: 4 (gap=3), ceil(3/1)=3.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d0 + d1 * 2).floorDiv(7);
    auto dimBounds = makeDimBounds({{0, 20}, {1, 3}});
    auto result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    // 2*2=4, gap to 7 is 3, ceil(3/1) = 3.
    EXPECT_EQ(contiguityBoundToInt64(result), 3)
        << "(d0 + 2*d1) floorDiv 7 should require 3 steps";
  }

  // Test 17: Large bounds allowing perfect alignment.
  // (d0 + 11*d1) floorDiv 11, d1 has bound 2 (0-1).
  // 11*1=11 is exact multiple, gap=11 (need full cycle), result=11.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d0 + d1 * 11).floorDiv(11);
    auto dimBounds = makeDimBounds({{0, 20}, {1, 2}});
    auto result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    // d1=0: gap=11, d1=1: gap=11 (exact multiple). Min gap=11, result=11.
    EXPECT_EQ(contiguityBoundToInt64(result), 11)
        << "(d0 + 11*d1) floorDiv 11 should return 11";
  }

  // Test 18: Steps exceed target dim bounds - should return -1 (unconstrained).
  // (d0 + 2*d1) floorDiv 100, d0 has bound 5 (values 0-4), d1 has bound 3.
  // d1 can achieve 0, 2, 4. Gaps: 100, 98, 96. Min gap=96, steps=96.
  // But d0 bound is 5, and 96 >= 5, so return -1.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d0 + d1 * 2).floorDiv(100);
    auto dimBounds = makeDimBounds({{0, 5}, {1, 3}});
    auto result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), -1)
        << "(d0 + 2*d1) floorDiv 100 with small d0 bound should return -1";
  }

  // Test 19: Steps exactly equal target dim bound - should return -1.
  // d0 floorDiv 8, d0 has bound 8 (values 0-7).
  // Gap=8, steps=8, bound=8, 8 >= 8, so return -1.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = d0.floorDiv(8);
    auto dimBounds = makeDimBounds({{0, 8}});
    auto result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), -1)
        << "d0 floorDiv 8 with d0 bound 8 should return -1 (never changes)";
  }

  // Test 20: Steps just under target dim bound - should return steps.
  // d0 floorDiv 8, d0 has bound 9 (values 0-8).
  // Gap=8, steps=8, bound=9, 8 < 9, so return 8.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = d0.floorDiv(8);
    auto dimBounds = makeDimBounds({{0, 9}});
    auto result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), 8)
        << "d0 floorDiv 8 with d0 bound 9 should return 8";
  }
}

TEST(AffineMapUtilsTest, AnalyzeShardDimStrides) {
  using namespace mlir;
  MLIRContext context;
}

// DISABLED: analyzeExprForDimStride is now an internal implementation detail.
// Its functionality is tested indirectly through analyzeShardDimStrides.
TEST(AffineMapUtilsTest, DISABLED_AnalyzeExprForDimStride) {
  using namespace mlir;
  MLIRContext context;
}

TEST(AffineMapUtilsTest, AnalyzeShardResultExprForContiguity) {
  using namespace mlir;
  MLIRContext context;

  // Helper to create dimension bounds map.
  auto makeDimBounds = [](std::initializer_list<std::pair<int, int64_t>> bounds)
      -> llvm::DenseMap<int, int64_t> {
    llvm::DenseMap<int, int64_t> result;
    for (auto [pos, bound] : bounds) {
      result[pos] = bound;
    }
    return result;
  };

  // Test 1: Simple dimension expression d0 -> should return -1 (unconstrained).
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    auto dimBounds = makeDimBounds({{0, 8}});
    auto result = analyzeShardResultExprForContiguity(d0, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), -1)
        << "d0 should be unconstrained";
  }

  // Test 2: Constant expression -> should return -1 (unconstrained).
  {
    AffineExpr constExpr = getAffineConstantExpr(42, &context);
    auto dimBounds = makeDimBounds({{0, 8}});
    auto result = analyzeShardResultExprForContiguity(constExpr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), -1)
        << "Constant should be unconstrained";
  }

  // Test 3: d0 mod N -> should return N (the modulus bounds the contiguity).
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = d0 % 8;
    auto dimBounds = makeDimBounds({{0, 32}});
    auto result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), 8) << "d0 mod 8 should return 8";
  }

  // Test 4: (d0 * M) mod N where N % M == 0 -> should return N/M.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = (d0 * 4) % 16;
    auto dimBounds = makeDimBounds({{0, 32}});
    auto result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    // Every 4 steps in d0, we cross a mod 16 boundary.
    EXPECT_EQ(contiguityBoundToInt64(result), 4)
        << "(d0 * 4) mod 16 should return 4";
  }

  // Test 5: (d0 + C) mod N - gap from C to next N boundary.
  // (d0 + 3) mod 8 -> gap = 8 - 3 = 5, steps = 5.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = (d0 + 3) % 8;
    auto dimBounds = makeDimBounds({{0, 32}});
    auto result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), 5)
        << "(d0 + 3) mod 8 should return 5";
  }

  // Test 6: (d0 * M + C) mod N - combined multiplier and offset.
  // (d0 * 2 + 3) mod 10 -> gap = 10 - 3 = 7, steps = ceil(7/2) = 4.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = (d0 * 2 + 3) % 10;
    auto dimBounds = makeDimBounds({{0, 32}});
    auto result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), 4)
        << "(d0 * 2 + 3) mod 10 should return 4";
  }

  // Test 7: (A*d0 + B*d1) mod N - two dimensions, target is d0.
  // (d0 + 3*d1) mod 10, d1 has bound 4 (values 0-3).
  // 3*d1 can be 0,3,6,9. Best alignment to 10: 9 (gap=1), result=1.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d0 + d1 * 3) % 10;
    auto dimBounds = makeDimBounds({{0, 20}, {1, 4}});
    auto result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), 1)
        << "(d0 + 3*d1) mod 10 should return 1";
  }

  // Test 8: (A*d0 + B*d1 + C) mod N - two dims with constant.
  // (2*d0 + 5*d1 + 3) mod 11, d1 has bound 3 (0-2).
  // 5*d1 + 3 can be 3,8,13. Modulo 11: 3,8,2. Best to 11: 8 (gap=3).
  // ceil(3/2) = 2.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d0 * 2 + d1 * 5 + 3) % 11;
    auto dimBounds = makeDimBounds({{0, 20}, {1, 3}});
    auto result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), 2)
        << "(2*d0 + 5*d1 + 3) mod 11 should return 2";
  }

  // Test 9: Target dim not in expression -> should return -1.
  {
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d1 + 5) % 10;
    auto dimBounds = makeDimBounds({{0, 20}, {1, 10}});
    auto result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), -1)
        << "Expression without d0 should be unconstrained for d0";
  }

  // Test 10: Steps exceed dim bounds -> should return -1 (unconstrained).
  // d0 mod 100, d0 has bound 5 (values 0-4).
  // Steps needed = 100, but bound is 5, so return -1.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = d0 % 100;
    auto dimBounds = makeDimBounds({{0, 5}});
    auto result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), -1)
        << "d0 mod 100 with small d0 bound should return -1";
  }

  // Test 11: FloorDiv expression - d0 floorDiv N.
  // d0 floorDiv 4 should return 4 (4 consecutive values before output changes).
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = d0.floorDiv(4);
    auto dimBounds = makeDimBounds({{0, 16}});
    auto result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), 4)
        << "d0 floorDiv 4 should return 4";
  }

  // Test 12: (d0 * M) floorDiv N.
  // (d0 * 2) floorDiv 8 -> ceil(8/2) = 4 steps before crossing boundary.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = (d0 * 2).floorDiv(8);
    auto dimBounds = makeDimBounds({{0, 16}});
    auto result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), 4)
        << "(d0 * 2) floorDiv 8 should return 4";
  }

  // Test 13: Add expression at top level (not under mod/floordiv).
  // d0 + d1 mod 4 -> d0 is unconstrained (-1), d1 mod 4 gives 4.
  // Combined: -1 and 4 -> 4.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = d0 + (d1 % 4);
    auto dimBounds = makeDimBounds({{0, 16}, {1, 16}});
    auto result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), -1)
        << "d0 + (d1 mod 4) should be unconstrained for d0";
  }

  // Test 14: Mul with parentModulus propagation.
  // When analyzing (d0 * 4) under a mod 16 context, returns 16/4 = 4.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = d0 * 4;
    auto dimBounds = makeDimBounds({{0, 32}});
    // Call with numGridDims=0, parentModulus = 16.
    auto result =
        analyzeShardResultExprForContiguity(expr, dimBounds, 0, 0, 16);
    EXPECT_EQ(contiguityBoundToInt64(result), 4)
        << "(d0 * 4) with parentModulus 16 should return 4";
  }

  // Test 15: Three dimensions - (d0 + 2*d1 + 3*d2) mod 12.
  // d1 has bound 5 (0-4), d2 has bound 4 (0-3).
  // Achievable sums from d1,d2: 0,2,3,4,5,6,7,8,9,10,11.
  // Best alignment to 12: 11 (gap=1), result=1.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr d2 = getAffineDimExpr(2, &context);
    AffineExpr expr = (d0 + d1 * 2 + d2 * 3) % 12;
    auto dimBounds = makeDimBounds({{0, 20}, {1, 5}, {2, 4}});
    auto result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), 1)
        << "(d0 + 2*d1 + 3*d2) mod 12 should return 1";
  }

  // Test 16: Simple case - just target dim and constant, no other dims.
  // (d0 + 7) mod 10 -> gap = 10 - 7 = 3.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = (d0 + 7) % 10;
    auto dimBounds = makeDimBounds({{0, 32}});
    auto result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), 3)
        << "(d0 + 7) mod 10 should return 3";
  }

  // Test 17: Case where constant is exact multiple of modulus.
  // (d0 + 10) mod 10 -> remainder = 0, gap = 10.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = (d0 + 10) % 10;
    auto dimBounds = makeDimBounds({{0, 32}});
    auto result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), 10)
        << "(d0 + 10) mod 10 should return 10";
  }

  // Test 18: Large multiplier on target dim.
  // (5*d0 + 3*d1) mod 11, d1 has bound 4 (0-3).
  // 3*d1 can be 0,3,6,9. Best to 11: 9 (gap=2), ceil(2/5) = 1.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d0 * 5 + d1 * 3) % 11;
    auto dimBounds = makeDimBounds({{0, 20}, {1, 4}});
    auto result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), 1)
        << "(5*d0 + 3*d1) mod 11 should return 1";
  }

  // Test 19: FloorDiv with Add - (d0 + 3) floorDiv 8.
  // Gap = 8 - 3 = 5, steps = 5.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = (d0 + 3).floorDiv(8);
    auto dimBounds = makeDimBounds({{0, 32}});
    auto result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(contiguityBoundToInt64(result), 5)
        << "(d0 + 3) floorDiv 8 should return 5";
  }

  // Test 20: Other dim can achieve exact multiple.
  // (d0 + 5*d1) mod 10, d1 has bound 3 (0-2).
  // 5*d1 can be 0,5,10. 10 mod 10 = 0, gap = 10.
  // Best gap comes from 5: 10 - 5 = 5.
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d0 + d1 * 5) % 10;
    auto dimBounds = makeDimBounds({{0, 20}, {1, 3}});
    auto result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    // d1=0: gap=10, d1=1: gap=5, d1=2: gap=10. Min gap=5.
    EXPECT_EQ(contiguityBoundToInt64(result), 5)
        << "(d0 + 5*d1) mod 10 should return 5";
  }
}
} // namespace ttmlir::utils
