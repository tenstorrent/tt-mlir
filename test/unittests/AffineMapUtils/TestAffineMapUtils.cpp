// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Utils.h"

#include "testing/Utils.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>

namespace ttmlir::utils {

namespace gtest = ::testing;

namespace {

/// Builds an affine expression representing: (sum of (dim_i mod modulus_i) *
/// multiplier_i) floordiv divisor Dimension indices are generated automatically
/// based on position (0, 1, 2, ...) If multipliers is empty or shorter than
/// moduli, defaults to 1 for missing multipliers.
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
  mlir::AffineExpr simplified = simplifyZeroFloorDivExpr(expr);
  return verifySimplifiedExprIsConstant(simplified, 0);
}

/// Builds an affine expression representing: sum of (dim_i mod modulus_i)
/// Dimension indices are generated automatically based on position (0, 1, 2,
/// ...)
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
      // If we're inside a mod, the dimension is not direct
      return false;
    }
    // For Add/Mul, check both sides
    return hasDirectDimExpr(binOp.getLHS(), dimIndex) ||
           hasDirectDimExpr(binOp.getRHS(), dimIndex);
  }
  return false;
}

/// Checks if a specific dimension appears in a mod operation in the expression.
bool hasModForDim(mlir::AffineExpr expr, unsigned dimIndex) {
  if (auto binOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
    if (binOp.getKind() == mlir::AffineExprKind::Mod) {
      // Check if LHS is the dimension we're looking for
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
  mlir::AffineExpr simplified =
      simplifyAffineExprWithRangeAnalysis(expr, dimBounds);

  // Verify each dimension matches the expected pattern
  for (size_t i = 0; i < expectedPattern.size(); ++i) {
    bool shouldBeSimplified = expectedPattern[i];
    bool hasMod = hasModForDim(simplified, i);
    bool hasDirectDim = hasDirectDimExpr(simplified, i);

    if (shouldBeSimplified) {
      // Mod should be simplified away - dim should appear directly, not in mod
      if (!hasDirectDim || hasMod) {
        return false;
      }
    } else {
      // Mod should remain - dim should appear in mod operation
      if (!hasMod) {
        return false;
      }
    }
  }

  return true;
}

} // namespace

TEST(AffineMapUtilsTest, CanSimplifyZeroFloorDivExpr) {
  using namespace mlir;
  MLIRContext context;

  EXPECT_TRUE(testSumOfModsFloorDivExpr(&context, {4, 5}, {}, 10));

  EXPECT_FALSE(testSumOfModsFloorDivExpr(&context, {2, 8, 9}, {}, 16));
  EXPECT_TRUE(testSumOfModsFloorDivExpr(&context, {2, 8, 9}, {}, 17));

  // Test random cases where divisor equals sum of moduli (no simplification)
  // and sum of moduli + 1 (simplifies to zero)
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

      // Randomly decide whether to include multiplication for this term
      int64_t multiplier = 1;
      if (unifUseMultiplier(gen)) {
        multiplier = unifMultiplier(gen);
        multipliers.push_back(multiplier);
      } else {
        multipliers.push_back(1);
      }

      maxModSum += (modulus - 1) * multiplier;
    }

    // When divisor equals sum of moduli, it should always simplify to 0
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

  // Test random cases where moduli are larger than dimension bounds (should
  // simplify) and moduli are smaller than dimension bounds (should not
  // simplify)
  auto seed = tt::testing::randomSeed();
  auto gen = tt::testing::createRNG(seed);
  std::uniform_int_distribution<int32_t> unifModulusCount(1, 5);
  std::uniform_int_distribution<int32_t> unifPartialModulusCount(2, 5);
  std::uniform_int_distribution<int64_t> unifSmallModulus(1, 100);
  std::uniform_int_distribution<int64_t> unifLargeModulus(1000, 16384);
  std::uniform_int_distribution<int64_t> unifSmallDimBound(1, 100);
  std::uniform_int_distribution<int64_t> unifLargeDimBound(1000, 16384);
  std::uniform_int_distribution<int32_t> unifShouldSimplify(0, 1);

  constexpr int32_t partialIterations = 50;
  for (int32_t i = 0; i < partialIterations; ++i) {
    const int32_t modulusCount = unifPartialModulusCount(gen);
    llvm::SmallVector<int64_t> moduli;
    llvm::SmallVector<int64_t> dimBounds;
    llvm::SmallVector<bool> expectedPattern;

    for (int32_t j = 0; j < modulusCount; ++j) {
      // Randomly choose whether this mod should be simplified or remain
      bool shouldSimplify = unifShouldSimplify(gen);

      int64_t modulus, dimBound;
      if (shouldSimplify) {
        modulus = unifLargeModulus(gen);
        dimBound = unifSmallDimBound(gen);
      } else {
        modulus = unifSmallModulus(gen);
        dimBound = unifLargeDimBound(gen);
      }

      moduli.push_back(modulus);
      dimBounds.push_back(dimBound);

      bool isRedundant = (dimBound - 1) < modulus;
      expectedPattern.push_back(isRedundant);
    }

    EXPECT_TRUE(testRedundantModSimplification(&context, moduli, dimBounds,
                                               expectedPattern))
        << "Failed for partial simplification with " << modulusCount
        << " moduli";
  }
}

// Returns an affine map and the device shape for a logical tensor and two grid
// shapes. The device shape is computed by dividing each logical shape dim by
// the input grid shape dim.
static std::tuple<mlir::AffineMap, llvm::SmallVector<int64_t>>
getReblockMapAndDeviceShape(mlir::ArrayRef<int64_t> logicalShape,
                            mlir::ArrayRef<int64_t> inputGridShape,
                            mlir::ArrayRef<int64_t> outputGridShape,
                            mlir::MLIRContext *context) {
  TT_assertv(inputGridShape.size() == logicalShape.size(),
             "Input grid shape must match logical shape");
  TT_assertv(outputGridShape.size() == logicalShape.size(),
             "Output grid shape must match logical shape");

  // Lambda to compute device shape for given grid
  auto computeDeviceShape = [&](llvm::ArrayRef<int64_t> logicalShape,
                                llvm::ArrayRef<int64_t> gridShape) {
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
  };

  llvm::SmallVector<int64_t> deviceShapeInputGrid =
      computeDeviceShape(logicalShape, inputGridShape);
  llvm::SmallVector<int64_t> deviceShapeOutputGrid =
      computeDeviceShape(logicalShape, outputGridShape);

  auto strides = ttmlir::utils::calculateStrides<int64_t>(
      llvm::ArrayRef<int64_t>(deviceShapeInputGrid)
          .take_back(deviceShapeInputGrid.size() / 2),
      1);

  mlir::AffineMap reblockMap = ttmlir::utils::calculateReblockMap(
      deviceShapeOutputGrid, deviceShapeInputGrid, context);
  reblockMap = simplifyAffineMapWithRangeAnalysis(
      simplifyZeroFloorDiv(reblockMap), deviceShapeInputGrid);

  auto layout_map =
      ttmlir::utils::generateAffineMapFromShardStrides(strides, context);
  auto memoryMap = layout_map.compose(reblockMap);

  return std::make_tuple(memoryMap, deviceShapeInputGrid);
}

TEST(AffineMapUtilsTest, CanGenerateAffineMapFromShardStrides) {
  using namespace mlir;
  MLIRContext context;

  using namespace mlir;
  // Explicitly build (d0, d1, d2, d3, d4, d5) -> (0, (d0 * 64 + d1 * 32 + d3 *
  // 32 + d4) floordiv 128, d2 + d5 floordiv 32, ((d0 * 64 + d1 * 32 + d3 * 32 +
  // d4) mod 128) * 128 + (d5 mod 32) * 4)
  AffineExpr d0, d1, d2, d3, d4, d5;
  bindDims(&context, d0, d1, d2, d3, d4, d5);

  // First result: 0
  AffineExpr r0 = getAffineConstantExpr(0, &context);

  // Second result: (d0 * 64 + d1 * 32 + d3 * 32 + d4) floordiv 128
  AffineExpr r1 = (d0 * 64 + d1 * 32 + d3 * 32 + d4).floorDiv(128);

  // Third result: d2 + d5 floordiv 32
  AffineExpr r2 = d2 + d5.floorDiv(32);

  // Fourth result: ((d0 * 64 + d1 * 32 + d3 * 32 + d4) mod 128) * 128 + (d5 mod
  // 32) * 4
  AffineExpr group = d0 * 64 + d1 * 32 + d3 * 32 + d4;
  AffineExpr shard = d5;
  AffineExpr r3 = (group % 128) * 128 + (shard % 32) * 4;

  SmallVector<AffineExpr, 4> results{r0, r1, r2, r3};
  AffineMap memoryMap =
      AffineMap::get(/*dimCount=*/6, /*symbolCount=*/0, results, &context);

  SmallVector<int64_t> deviceShape = {16, 1, 4, 2, 32, 32};
  constexpr int64_t elemSizeBytes = 4;

  auto coalescingFactorAnalytical = analyzeShardDimContiguity(
      memoryMap, deviceShape, memoryMap.getNumDims() / 2,
      memoryMap.getNumResults() - 1, elemSizeBytes);

  // Also compute coalescing factor using the sampling-based approach
  int64_t coalescingFactorSampling = calculateCoalescingFactor(
      memoryMap, deviceShape, elemSizeBytes, memoryMap.getNumDims() / 2);

  GTEST_ASSERT_EQ(coalescingFactorAnalytical, coalescingFactorSampling);
}

TEST(AffineMapUtilsTest, CanDetermineCoalescingFactor) {
  using namespace mlir;
  MLIRContext context;

  // Test result codes
  enum class TestResult { Success, Subset, Failed };

  auto testSingleReblockingTestcase =
      [&](llvm::ArrayRef<int64_t> logicalShape,
          llvm::ArrayRef<int64_t> inputGridShape,
          llvm::ArrayRef<int64_t> outputGridShape) -> TestResult {
    auto [memoryMap, deviceShape] = getReblockMapAndDeviceShape(
        logicalShape, inputGridShape, outputGridShape, &context);

    auto coalescingFactorAnalytical = analyzeShardDimContiguity(
        memoryMap, deviceShape, memoryMap.getNumDims() / 2,
        memoryMap.getNumResults() - 1, 1);

    // Also compute coalescing factor using the sampling-based approach
    int64_t coalescingFactor = calculateCoalescingFactor(
        memoryMap, deviceShape, 1, memoryMap.getNumDims() / 2);

    if (coalescingFactor == coalescingFactorAnalytical) {
      return TestResult::Success;
    } else if (coalescingFactorAnalytical < coalescingFactor &&
               coalescingFactor % coalescingFactorAnalytical == 0) {
      return TestResult::Subset;
    } else {
      return TestResult::Failed;
    }
  };

  // Configuration for 3D and 4D test generation
  constexpr int64_t maxVolume = 6400;
  constexpr int numLogicalShapeExamples = 4;
  constexpr int numGridPairsPerShape = 16;

  std::mt19937 rng(42); // deterministic seed for reproducibility

  // Valid grid dimensions: only 1, 2, 4, 8 (even or 1)
  const SmallVector<int64_t> validGridDims = {1, 2, 3, 4, 5, 6, 7, 8};

  // Helper to get valid grid dimensions that divide n
  auto getValidGridDivisors = [&](int64_t n) -> SmallVector<int64_t> {
    SmallVector<int64_t> divisors;
    for (int64_t d : validGridDims) {
      if (n % d == 0) {
        divisors.push_back(d);
      }
    }
    return divisors;
  };

  // Helper to generate random logical shape with given rank and max volume
  // All dimensions are divisible by 4
  auto generateLogicalShape = [&](int rank) -> SmallVector<int64_t> {
    SmallVector<int64_t> shape(rank, 4); // Start with 16 (divisible by 16)
    int64_t volume = static_cast<int64_t>(std::pow(4, rank));

    // Build up the shape by multiplying random dimensions by factors of 2
    // to maintain divisibility by 4
    std::uniform_int_distribution<int> dimDist(0, rank - 1);
    std::uniform_int_distribution<int64_t> factorDist(1, 2); // 2^1, 2^2, 2^3

    // Keep growing until we're close to target volume
    while (volume < maxVolume / 8) {
      int dim = dimDist(rng);
      int64_t factor = 1 << factorDist(rng); // 2, 4, or 8
      if (volume * factor <= maxVolume) {
        shape[dim] *= factor;
        volume *= factor;
      }
    }

    return shape;
  };

  // Helper to generate a random valid grid shape for a given logical shape
  // Only uses grid dimensions 1, 2, 4, 8
  auto generateGridShape =
      [&](ArrayRef<int64_t> logicalShape) -> SmallVector<int64_t> {
    SmallVector<int64_t> gridShape;
    for (int64_t dim : logicalShape) {
      auto divisors = getValidGridDivisors(dim);
      std::uniform_int_distribution<size_t> dist(0, divisors.size() - 1);
      gridShape.push_back(divisors[dist(rng)]);
    }
    return gridShape;
  };

  // Generate test cases for 3D and 4D shapes
  // Use a set to skip test cases that have already been generated
  llvm::DenseSet<std::tuple<SmallVector<int64_t>, SmallVector<int64_t>,
                            SmallVector<int64_t>>>
      generatedCases;

  for (int rank = 2; rank <= 6; ++rank) {
    for (int shapeIdx = 0; shapeIdx < numLogicalShapeExamples; ++shapeIdx) {
      SmallVector<int64_t> logicalShape = generateLogicalShape(rank);

      for (int gridPairIdx = 0; gridPairIdx < numGridPairsPerShape;
           ++gridPairIdx) {
        SmallVector<int64_t> inputGridShape = generateGridShape(logicalShape);
        SmallVector<int64_t> outputGridShape = generateGridShape(logicalShape);

        // Build a tuple key for this test case
        auto key =
            std::make_tuple(logicalShape, inputGridShape, outputGridShape);

        // Skip if this test case has already been generated
        if (generatedCases.contains(key)) {
          continue;
        }
        generatedCases.insert(key);

        auto result = testSingleReblockingTestcase(logicalShape, inputGridShape,
                                                   outputGridShape);
        GTEST_EXPECT_TRUE(result == TestResult::Success);
      }
    }
  }
}

TEST(AffineMapUtilsTest, AnalyzeGridResultExprForDiscontinuity) {
  using namespace mlir;
  MLIRContext context;

  // Helper to create dimension bounds map
  auto makeDimBounds = [](std::initializer_list<std::pair<int, int64_t>> bounds)
      -> llvm::DenseMap<int, int64_t> {
    llvm::DenseMap<int, int64_t> result;
    for (auto [pos, bound] : bounds) {
      result[pos] = bound;
    }
    return result;
  };

  // Test 1: Simple dimension expression d0 -> should return 1
  // Every step in d0 changes the output
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    auto dimBounds = makeDimBounds({{0, 8}});
    int64_t result = analyzeGridResultExprForDiscontinuity(d0, dimBounds, 0);
    EXPECT_EQ(result, 1) << "d0 should require 1 step to change output";
  }

  // Test 2: d0 floorDiv N -> should return N
  // Need N steps in d0 to change the output
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = d0.floorDiv(4);
    auto dimBounds = makeDimBounds({{0, 16}});
    int64_t result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    EXPECT_EQ(result, 4) << "d0 floorDiv 4 should require 4 steps";
  }

  // Test 3: (d0 * M) floorDiv N where N % M == 0 -> should return 4
  // Since M*d0 changes by M each step, and N/M steps cross the boundary
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = (d0 * 2).floorDiv(8);
    auto dimBounds = makeDimBounds({{0, 16}});
    int64_t result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    // d0 * 2 means values 0, 2, 4, 6, 8, ... so crossing 8 takes 4 steps
    EXPECT_EQ(result, 4) << "(d0 * 2) floorDiv 8 should require 4 steps";
  }

  // Test 4: Expression with unrelated dimension -> should return -1
  // d1 floorDiv N when analyzing for d0
  {
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = d1.floorDiv(4);
    auto dimBounds = makeDimBounds({{0, 8}, {1, 16}});
    int64_t result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    EXPECT_EQ(result, -1) << "d1 floorDiv 4 should be unconstrained for d0";
  }

  // Test 5: Constant expression -> should return -1 (unconstrained)
  {
    AffineExpr constExpr = getAffineConstantExpr(42, &context);
    auto dimBounds = makeDimBounds({{0, 8}});
    int64_t result =
        analyzeGridResultExprForDiscontinuity(constExpr, dimBounds, 0);
    EXPECT_EQ(result, -1) << "Constant should be unconstrained";
  }

  // Test 6: (A*d0 + B*d1) floorDiv N - multiple dimensions
  // Testing the minimizeGap integration
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    // (7*d0 + 5*d1) floorDiv 11
    // d0 has bound 3 (values 0, 1, 2), d1 has bound 10 (values 0-9)
    // Best alignment for d1: 5*8 = 40, plus 7*2 = 14, total 54, gap to 55 is 1
    AffineExpr expr = (d0 * 7 + d1 * 5).floorDiv(11);
    auto dimBounds = makeDimBounds({{0, 3}, {1, 10}});
    int64_t result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    // The gap is minimized by d1, and we compute ceil(gap / 7)
    EXPECT_EQ(result, 1) << "(7*d0 + 5*d1) floorDiv 11 should return >= 1";
  }

  // Test 7: Add expression combining constrained and unconstrained
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    // d0 + d1 floorDiv 4 - d0 is direct (returns 1), d1 floorDiv 4 returns 4
    // Combined via add should return gcd(1, 4) = 1
    AffineExpr expr = d0 + d1.floorDiv(4);
    auto dimBounds = makeDimBounds({{0, 8}, {1, 16}});
    int64_t result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    EXPECT_EQ(result, 1) << "d0 + (d1 floorDiv 4) should return 1 for d0";
  }

  // Test 8: Mul expression with target dim (not inside floorDiv)
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = d0 * 3;
    auto dimBounds = makeDimBounds({{0, 8}});
    int64_t result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    EXPECT_EQ(result, 1) << "d0 * 3 should return 1 (any change in d0 changes "
                            "output)";
  }

  // Test 9: Larger floorDiv value
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = d0.floorDiv(32);
    auto dimBounds = makeDimBounds({{0, 128}});
    int64_t result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    EXPECT_EQ(result, 32) << "d0 floorDiv 32 should require 32 steps";
  }

  // Test 10: Complex expression with constant offset
  // (d0 * 4 + 2) floorDiv 8 -> gap is 8 - 2 = 6, ceil(6/4) = 2
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = (d0 * 4 + 2).floorDiv(8);
    auto dimBounds = makeDimBounds({{0, 16}});
    int64_t result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    // With offset 2 and multiplier 4, we need ceil((8-2)/4) = 2 steps
    EXPECT_EQ(result, 2) << "(d0 * 4 + 2) floorDiv 8 should require 2 steps";
  }

  // Test 11: Two dimensions with multipliers - (2*d0 + 3*d1) floorDiv 10
  // d0 is target, d1 has bound 4 (values 0,1,2,3), so 3*d1 can be 0,3,6,9
  // Best case for d1: 3*3=9, gap to 10 is 1, so ceil(1/2)=1
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d0 * 2 + d1 * 3).floorDiv(10);
    auto dimBounds = makeDimBounds({{0, 10}, {1, 4}});
    int64_t result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    // d1 can achieve 3*3=9, gap to 10 is 1, ceil(1/2) = 1
    EXPECT_EQ(result, 1)
        << "(2*d0 + 3*d1) floorDiv 10 should require 1 step when d1 can align";
  }

  // Test 12: Two dimensions where other dim can achieve exact multiple
  // (d0 + 5*d1) floorDiv 10, d1 has bound 3 (values 0,1,2)
  // d1=2 gives 5*2=10 which is exactly 10, gap=0, but minimizeGap returns 1
  // as best achievable gap, so ceil(1/1) = 1
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d0 + d1 * 5).floorDiv(10);
    auto dimBounds = makeDimBounds({{0, 20}, {1, 3}});
    int64_t result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    // d1=2 gives 10, which is exact multiple, gap=0 means any step changes
    // But actually minimizeGap with gap=0 would return bestGap=0, not 1
    // When gap=0, we still need at least 1 step to change (can't be 0)
    EXPECT_EQ(result, 5) << "(d0 + 5*d1) floorDiv 10 should return 5";
  }

  // Test 13: Three dimensions - (d0 + 2*d1 + 3*d2) floorDiv 12
  // d1 has bound 5 (0-4), d2 has bound 4 (0-3)
  // Achievable sums from d1,d2: 0,2,3,4,5,6,7,8,9,10,11
  // Best alignment to 12: 11 (gap=1), ceil(1/1) = 1
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr d2 = getAffineDimExpr(2, &context);
    AffineExpr expr = (d0 + d1 * 2 + d2 * 3).floorDiv(12);
    auto dimBounds = makeDimBounds({{0, 20}, {1, 5}, {2, 4}});
    int64_t result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    EXPECT_EQ(result, 1)
        << "(d0 + 2*d1 + 3*d2) floorDiv 12 should require 1 step";
  }

  // Test 14: Two dims with larger multiplier on target dim
  // (3*d0 + 2*d1) floorDiv 7, d1 has bound 4 (0-3)
  // 2*d1 can be 0,2,4,6. Best alignment to 7: 6 (gap=1), ceil(1/3)=1
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d0 * 3 + d1 * 2).floorDiv(7);
    auto dimBounds = makeDimBounds({{0, 10}, {1, 4}});
    int64_t result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    // 2*3=6, gap to 7 is 1, ceil(1/3) = 1
    EXPECT_EQ(result, 1) << "(3*d0 + 2*d1) floorDiv 7 should require 1 step";
  }

  // Test 15: Multiple dims with constant offset
  // (2*d0 + 5*d1 + 3) floorDiv 11, d1 has bound 3 (0-2)
  // 5*d1 + 3 can be 3,8,13. Modulo 11: 3,8,2. Best to 11: 8 (gap=3)
  // ceil(3/2) = 2
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d0 * 2 + d1 * 5 + 3).floorDiv(11);
    auto dimBounds = makeDimBounds({{0, 20}, {1, 3}});
    int64_t result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    // 5*1+3=8, gap to 11 is 3, ceil(3/2) = 2
    EXPECT_EQ(result, 2)
        << "(2*d0 + 5*d1 + 3) floorDiv 11 should require 2 steps";
  }

  // Test 16: Case where no alignment is possible - prime divisor
  // (d0 + 2*d1) floorDiv 7, d1 has bound 3 (0-2)
  // 2*d1 can be 0,2,4. None divides 7 evenly. Best: 4 (gap=3), ceil(3/1)=3
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d0 + d1 * 2).floorDiv(7);
    auto dimBounds = makeDimBounds({{0, 20}, {1, 3}});
    int64_t result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    // 2*2=4, gap to 7 is 3, ceil(3/1) = 3
    EXPECT_EQ(result, 3) << "(d0 + 2*d1) floorDiv 7 should require 3 steps";
  }

  // Test 17: Large bounds allowing perfect alignment
  // (d0 + 11*d1) floorDiv 11, d1 has bound 2 (0-1)
  // 11*1=11 is exact multiple, gap=11 (need full cycle), result=11
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d0 + d1 * 11).floorDiv(11);
    auto dimBounds = makeDimBounds({{0, 20}, {1, 2}});
    int64_t result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    // d1=0: gap=11, d1=1: gap=11 (exact multiple). Min gap=11, result=11
    EXPECT_EQ(result, 11) << "(d0 + 11*d1) floorDiv 11 should return 11";
  }

  // Test 18: Steps exceed target dim bounds - should return -1 (unconstrained)
  // (d0 + 2*d1) floorDiv 100, d0 has bound 5 (values 0-4), d1 has bound 3
  // d1 can achieve 0, 2, 4. Gaps: 100, 98, 96. Min gap=96, steps=96
  // But d0 bound is 5, and 96 >= 5, so return -1
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d0 + d1 * 2).floorDiv(100);
    auto dimBounds = makeDimBounds({{0, 5}, {1, 3}});
    int64_t result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    EXPECT_EQ(result, -1)
        << "(d0 + 2*d1) floorDiv 100 with small d0 bound should return -1";
  }

  // Test 19: Steps exactly equal target dim bound - should return -1
  // d0 floorDiv 8, d0 has bound 8 (values 0-7)
  // Gap=8, steps=8, bound=8, 8 >= 8, so return -1
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = d0.floorDiv(8);
    auto dimBounds = makeDimBounds({{0, 8}});
    int64_t result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    EXPECT_EQ(result, -1)
        << "d0 floorDiv 8 with d0 bound 8 should return -1 (never changes)";
  }

  // Test 20: Steps just under target dim bound - should return steps
  // d0 floorDiv 8, d0 has bound 9 (values 0-8)
  // Gap=8, steps=8, bound=9, 8 < 9, so return 8
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = d0.floorDiv(8);
    auto dimBounds = makeDimBounds({{0, 9}});
    int64_t result = analyzeGridResultExprForDiscontinuity(expr, dimBounds, 0);
    EXPECT_EQ(result, 8) << "d0 floorDiv 8 with d0 bound 9 should return 8";
  }
}

TEST(AffineMapUtilsTest, AnalyzeShardDimStrides) {
  using namespace mlir;
  MLIRContext context;
}

TEST(AffineMapUtilsTest, AnalyzeExprForDimStride) {
  using namespace mlir;
  MLIRContext context;

  // Test 1: Simple dimension expression d0 -> stride is 1
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    int64_t stride = analyzeExprForDimStride(d0, 0);
    EXPECT_EQ(stride, 1) << "d0 should have stride 1";
  }

  // Test 2: Different dimension -> stride is 0
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    int64_t stride = analyzeExprForDimStride(d0, 1);
    EXPECT_EQ(stride, 0) << "d0 should have stride 0 for d1";
  }

  // Test 3: Constant expression -> stride is 0
  {
    AffineExpr constExpr = getAffineConstantExpr(42, &context);
    int64_t stride = analyzeExprForDimStride(constExpr, 0);
    EXPECT_EQ(stride, 0) << "Constant should have stride 0";
  }

  // Test 4: Simple mul d0 * 128 -> stride is 128
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = d0 * 128;
    int64_t stride = analyzeExprForDimStride(expr, 0);
    EXPECT_EQ(stride, 128) << "d0 * 128 should have stride 128";
  }

  // Test 5: Mul as LHS of mod (d0 * 64) mod 4096 -> stride is 64
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = (d0 * 64) % 4096;
    int64_t stride = analyzeExprForDimStride(expr, 0);
    EXPECT_EQ(stride, 64) << "(d0 * 64) mod 4096 should have stride 64";
  }

  // Test 6: Multiple terms summed d0*128 + (d0*64) mod 4096 -> stride is 192
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = d0 * 128 + (d0 * 64) % 4096;
    int64_t stride = analyzeExprForDimStride(expr, 0);
    EXPECT_EQ(stride, 192)
        << "d0*128 + (d0*64) mod 4096 should have stride 192";
  }

  // Test 7: Mul as LHS of floordiv, mul > floordiv: (d0*16) floorDiv 2 ->
  // stride is 8
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = (d0 * 16).floorDiv(2);
    int64_t stride = analyzeExprForDimStride(expr, 0);
    EXPECT_EQ(stride, 8) << "(d0 * 16) floorDiv 2 should have stride 8";
  }

  // Test 8: Mul as LHS of floordiv, mul < floordiv: (d0*2) floorDiv 16 ->
  // stride is 0
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = (d0 * 2).floorDiv(16);
    int64_t stride = analyzeExprForDimStride(expr, 0);
    EXPECT_EQ(stride, 1) << "(d0 * 2) floorDiv 16 should have stride 1";
  }

  // Test 9: Plain dim with mod: d0 mod 32 -> stride is 1
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = d0 % 32;
    int64_t stride = analyzeExprForDimStride(expr, 0);
    EXPECT_EQ(stride, 1) << "d0 mod 32 should have stride 1";
  }

  // Test 10: Plain dim floordiv: d0 floorDiv 4 -> stride is 0
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = d0.floorDiv(4);
    int64_t stride = analyzeExprForDimStride(expr, 0);
    EXPECT_EQ(stride, 1) << "d0 floorDiv 4 should have stride 0";
  }

  // Test 11: Two dimensions added: d0*128 + d1*4
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = d0 * 128 + d1 * 4;

    int64_t stride0 = analyzeExprForDimStride(expr, 0);
    EXPECT_EQ(stride0, 128) << "d0*128 + d1*4 should have stride 128 for d0";

    int64_t stride1 = analyzeExprForDimStride(expr, 1);
    EXPECT_EQ(stride1, 4) << "d0*128 + d1*4 should have stride 4 for d1";
  }

  // Test 12: Complex expression from the failing test case:
  // ((d2 mod 32) * 128 + (d3 mod 32) * 4)
  {
    AffineExpr d2 = getAffineDimExpr(2, &context);
    AffineExpr d3 = getAffineDimExpr(3, &context);
    AffineExpr expr = (d2 % 32) * 128 + (d3 % 32) * 4;

    int64_t stride2 = analyzeExprForDimStride(expr, 2);
    EXPECT_EQ(stride2, 128)
        << "(d2 mod 32)*128 + (d3 mod 32)*4 should have stride 128 for d2";

    int64_t stride3 = analyzeExprForDimStride(expr, 3);
    EXPECT_EQ(stride3, 4)
        << "(d2 mod 32)*128 + (d3 mod 32)*4 should have stride 4 for d3";
  }

  // Test 13: Nested floordiv: ((d0 * 8) floorDiv 4) floorDiv 2 -> stride is 1
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = ((d0 * 8).floorDiv(4)).floorDiv(2);
    int64_t stride = analyzeExprForDimStride(expr, 0);
    // (d0 * 8) floorDiv 4 = d0 * 2 (stride 2)
    // (d0 * 2) floorDiv 2 = d0 (stride 1)
    EXPECT_EQ(stride, 1)
        << "((d0 * 8) floorDiv 4) floorDiv 2 should have stride 1";
  }

  // Test 14: CeilDiv: (d0 * 16) ceilDiv 4 -> stride is 4
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = (d0 * 16).ceilDiv(4);
    int64_t stride = analyzeExprForDimStride(expr, 0);
    EXPECT_EQ(stride, 4) << "(d0 * 16) ceilDiv 4 should have stride 4";
  }
}

TEST(AffineMapUtilsTest, AnalyzeShardResultExprForContiguity) {
  using namespace mlir;
  MLIRContext context;

  // Helper to create dimension bounds map
  auto makeDimBounds = [](std::initializer_list<std::pair<int, int64_t>> bounds)
      -> llvm::DenseMap<int, int64_t> {
    llvm::DenseMap<int, int64_t> result;
    for (auto [pos, bound] : bounds) {
      result[pos] = bound;
    }
    return result;
  };

  // Test 1: Simple dimension expression d0 -> should return -1 (unconstrained)
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    auto dimBounds = makeDimBounds({{0, 8}});
    int64_t result = analyzeShardResultExprForContiguity(d0, dimBounds, 0);
    EXPECT_EQ(result, -1) << "d0 should be unconstrained";
  }

  // Test 2: Constant expression -> should return -1 (unconstrained)
  {
    AffineExpr constExpr = getAffineConstantExpr(42, &context);
    auto dimBounds = makeDimBounds({{0, 8}});
    int64_t result =
        analyzeShardResultExprForContiguity(constExpr, dimBounds, 0);
    EXPECT_EQ(result, -1) << "Constant should be unconstrained";
  }

  // Test 3: d0 mod N -> should return N (the modulus bounds the contiguity)
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = d0 % 8;
    auto dimBounds = makeDimBounds({{0, 32}});
    int64_t result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(result, 8) << "d0 mod 8 should return 8";
  }

  // Test 4: (d0 * M) mod N where N % M == 0 -> should return N/M
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = (d0 * 4) % 16;
    auto dimBounds = makeDimBounds({{0, 32}});
    int64_t result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    // Every 4 steps in d0, we cross a mod 16 boundary
    EXPECT_EQ(result, 4) << "(d0 * 4) mod 16 should return 4";
  }

  // Test 5: (d0 + C) mod N - gap from C to next N boundary
  // (d0 + 3) mod 8 -> gap = 8 - 3 = 5, steps = 5
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = (d0 + 3) % 8;
    auto dimBounds = makeDimBounds({{0, 32}});
    int64_t result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(result, 5) << "(d0 + 3) mod 8 should return 5";
  }

  // Test 6: (d0 * M + C) mod N - combined multiplier and offset
  // (d0 * 2 + 3) mod 10 -> gap = 10 - 3 = 7, steps = ceil(7/2) = 4
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = (d0 * 2 + 3) % 10;
    auto dimBounds = makeDimBounds({{0, 32}});
    int64_t result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(result, 4) << "(d0 * 2 + 3) mod 10 should return 4";
  }

  // Test 7: (A*d0 + B*d1) mod N - two dimensions, target is d0
  // (d0 + 3*d1) mod 10, d1 has bound 4 (values 0-3)
  // 3*d1 can be 0,3,6,9. Best alignment to 10: 9 (gap=1), result=1
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d0 + d1 * 3) % 10;
    auto dimBounds = makeDimBounds({{0, 20}, {1, 4}});
    int64_t result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(result, 1) << "(d0 + 3*d1) mod 10 should return 1";
  }

  // Test 8: (A*d0 + B*d1 + C) mod N - two dims with constant
  // (2*d0 + 5*d1 + 3) mod 11, d1 has bound 3 (0-2)
  // 5*d1 + 3 can be 3,8,13. Modulo 11: 3,8,2. Best to 11: 8 (gap=3)
  // ceil(3/2) = 2
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d0 * 2 + d1 * 5 + 3) % 11;
    auto dimBounds = makeDimBounds({{0, 20}, {1, 3}});
    int64_t result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(result, 2) << "(2*d0 + 5*d1 + 3) mod 11 should return 2";
  }

  // Test 9: Target dim not in expression -> should return -1
  {
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d1 + 5) % 10;
    auto dimBounds = makeDimBounds({{0, 20}, {1, 10}});
    int64_t result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(result, -1)
        << "Expression without d0 should be unconstrained for d0";
  }

  // Test 10: Steps exceed dim bounds -> should return -1 (unconstrained)
  // d0 mod 100, d0 has bound 5 (values 0-4)
  // Steps needed = 100, but bound is 5, so return -1
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = d0 % 100;
    auto dimBounds = makeDimBounds({{0, 5}});
    int64_t result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(result, -1) << "d0 mod 100 with small d0 bound should return -1";
  }

  // Test 11: FloorDiv expression - d0 floorDiv N
  // d0 floorDiv 4 should return 4 (4 consecutive values before output changes)
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = d0.floorDiv(4);
    auto dimBounds = makeDimBounds({{0, 16}});
    int64_t result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(result, 4) << "d0 floorDiv 4 should return 4";
  }

  // Test 12: (d0 * M) floorDiv N
  // (d0 * 2) floorDiv 8 -> ceil(8/2) = 4 steps before crossing boundary
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = (d0 * 2).floorDiv(8);
    auto dimBounds = makeDimBounds({{0, 16}});
    int64_t result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(result, 4) << "(d0 * 2) floorDiv 8 should return 4";
  }

  // Test 13: Add expression at top level (not under mod/floordiv)
  // d0 + d1 mod 4 -> d0 is unconstrained (-1), d1 mod 4 gives 4
  // Combined: -1 and 4 -> 4
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = d0 + (d1 % 4);
    auto dimBounds = makeDimBounds({{0, 16}, {1, 16}});
    int64_t result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(result, -1) << "d0 + (d1 mod 4) should be unconstrained for d0";
  }

  // Test 14: Mul with parentModulus propagation
  // When analyzing (d0 * 4) under a mod 16 context, returns 16/4 = 4
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = d0 * 4;
    auto dimBounds = makeDimBounds({{0, 32}});
    // Call with parentModulus = 16
    int64_t result =
        analyzeShardResultExprForContiguity(expr, dimBounds, 0, 16);
    EXPECT_EQ(result, 4) << "(d0 * 4) with parentModulus 16 should return 4";
  }

  // Test 15: Three dimensions - (d0 + 2*d1 + 3*d2) mod 12
  // d1 has bound 5 (0-4), d2 has bound 4 (0-3)
  // Achievable sums from d1,d2: 0,2,3,4,5,6,7,8,9,10,11
  // Best alignment to 12: 11 (gap=1), result=1
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr d2 = getAffineDimExpr(2, &context);
    AffineExpr expr = (d0 + d1 * 2 + d2 * 3) % 12;
    auto dimBounds = makeDimBounds({{0, 20}, {1, 5}, {2, 4}});
    int64_t result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(result, 1) << "(d0 + 2*d1 + 3*d2) mod 12 should return 1";
  }

  // Test 16: Simple case - just target dim and constant, no other dims
  // (d0 + 7) mod 10 -> gap = 10 - 7 = 3
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = (d0 + 7) % 10;
    auto dimBounds = makeDimBounds({{0, 32}});
    int64_t result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(result, 3) << "(d0 + 7) mod 10 should return 3";
  }

  // Test 17: Case where constant is exact multiple of modulus
  // (d0 + 10) mod 10 -> remainder = 0, gap = 10
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = (d0 + 10) % 10;
    auto dimBounds = makeDimBounds({{0, 32}});
    int64_t result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(result, 10) << "(d0 + 10) mod 10 should return 10";
  }

  // Test 18: Large multiplier on target dim
  // (5*d0 + 3*d1) mod 11, d1 has bound 4 (0-3)
  // 3*d1 can be 0,3,6,9. Best to 11: 9 (gap=2), ceil(2/5) = 1
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d0 * 5 + d1 * 3) % 11;
    auto dimBounds = makeDimBounds({{0, 20}, {1, 4}});
    int64_t result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(result, 1) << "(5*d0 + 3*d1) mod 11 should return 1";
  }

  // Test 19: FloorDiv with Add - (d0 + 3) floorDiv 8
  // Gap = 8 - 3 = 5, steps = 5
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr expr = (d0 + 3).floorDiv(8);
    auto dimBounds = makeDimBounds({{0, 32}});
    int64_t result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    EXPECT_EQ(result, 5) << "(d0 + 3) floorDiv 8 should return 5";
  }

  // Test 20: Other dim can achieve exact multiple
  // (d0 + 5*d1) mod 10, d1 has bound 3 (0-2)
  // 5*d1 can be 0,5,10. 10 mod 10 = 0, gap = 10
  // Best gap comes from 5: 10 - 5 = 5
  {
    AffineExpr d0 = getAffineDimExpr(0, &context);
    AffineExpr d1 = getAffineDimExpr(1, &context);
    AffineExpr expr = (d0 + d1 * 5) % 10;
    auto dimBounds = makeDimBounds({{0, 20}, {1, 3}});
    int64_t result = analyzeShardResultExprForContiguity(expr, dimBounds, 0);
    // d1=0: gap=10, d1=1: gap=5, d1=2: gap=10. Min gap=5
    EXPECT_EQ(result, 5) << "(d0 + 5*d1) mod 10 should return 5";
  }
}
} // namespace ttmlir::utils
