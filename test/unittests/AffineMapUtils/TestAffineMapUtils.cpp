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
  llvm::dbgs() << "[testSumOfModsFloorDivExpr] expr: " << expr << "\n";
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

TEST(AffineMapUtilsTest, CanDetermineCoalescingFactor) {
  using namespace mlir;
  MLIRContext context;

  // Test result codes
  enum class TestResult { Success, Subset, Failed };

  auto printContiguityConstraints =
      [&](llvm::ArrayRef<int64_t> logicalShape,
          llvm::ArrayRef<int64_t> inputGridShape,
          llvm::ArrayRef<int64_t> outputGridShape) -> TestResult {
    auto [memoryMap, deviceShape] = getReblockMapAndDeviceShape(
        logicalShape, inputGridShape, outputGridShape, &context);

    auto simpleMap = simplifyAffineMapWithRangeAnalysis(
        simplifyZeroFloorDiv(memoryMap), deviceShape);
    llvm::dbgs() << "[TestCanDetermineCoalescingFactor] simple map: "
                 << simpleMap << "\n";

    auto start_time = std::chrono::high_resolution_clock::now();
    auto coalescingFactorAnalytical = analyzeShardDimContiguity(
        memoryMap, deviceShape, memoryMap.getNumDims() / 2);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);

    // Also compute coalescing factor using the sampling-based approach
    int64_t coalescingFactor = calculateCoalescingFactor(
        memoryMap, deviceShape, 1, memoryMap.getNumDims() / 2);

    if (coalescingFactor == coalescingFactorAnalytical.value_or(-1)) {
      llvm::dbgs() << "[TestCanDetermineCoalescingFactor] logical shape: "
                   << ttmlir::utils::formatIterable(logicalShape, "x")
                   << " input grid: "
                   << ttmlir::utils::formatIterable(inputGridShape, "x")
                   << " output grid: "
                   << ttmlir::utils::formatIterable(outputGridShape, "x")
                   << " SUCCESS (coalescing factor: "
                   << coalescingFactorAnalytical.value_or(-1) << " in "
                   << duration.count() << "us)\n\n";
      return TestResult::Success;
    } else if (coalescingFactorAnalytical.value_or(-1) < coalescingFactor &&
               coalescingFactor % coalescingFactorAnalytical.value_or(-1) ==
                   0) {
      llvm::dbgs() << "[TestCanDetermineCoalescingFactor] logical shape: "
                   << ttmlir::utils::formatIterable(logicalShape, "x")
                   << " input grid: "
                   << ttmlir::utils::formatIterable(inputGridShape, "x")
                   << " output grid: "
                   << ttmlir::utils::formatIterable(outputGridShape, "x")
                   << " SUBSET (analytical="
                   << coalescingFactorAnalytical.value_or(-1)
                   << " vs calculated=" << coalescingFactor << " in "
                   << duration.count() << "us)\n\n";
      return TestResult::Subset;
    } else {
      llvm::dbgs() << "[TestCanDetermineCoalescingFactor] logical shape: "
                   << ttmlir::utils::formatIterable(logicalShape, "x")
                   << " input grid: "
                   << ttmlir::utils::formatIterable(inputGridShape, "x")
                   << " output grid: "
                   << ttmlir::utils::formatIterable(outputGridShape, "x")
                   << " FAILED : analytical = "
                   << coalescingFactorAnalytical.value_or(-1)
                   << ", calculated = " << coalescingFactor << "\n";
      return TestResult::Failed;
    }
  };

  // Configuration for 3D and 4D test generation
  constexpr int64_t maxVolume = 10000;
  constexpr int numLogicalShapeExamples = 10;
  constexpr int numGridPairsPerShape = 10;
  constexpr int64_t maxGridDim = 8;

  std::mt19937 rng(42); // deterministic seed for reproducibility

  // Helper to get all divisors of n that are <= maxVal
  auto getDivisors = [](int64_t n, int64_t maxVal) -> SmallVector<int64_t> {
    SmallVector<int64_t> divisors;
    for (int64_t d = 1; d <= std::min(n, maxVal); ++d) {
      if (n % d == 0) {
        divisors.push_back(d);
      }
    }
    return divisors;
  };

  // Helper to generate random logical shape with given rank and max volume
  auto generateLogicalShape = [&](int rank) -> SmallVector<int64_t> {
    SmallVector<int64_t> shape(rank, 1);
    int64_t volume = 1;

    // Build up the shape by multiplying random dimensions by random factors
    std::uniform_int_distribution<int> dimDist(0, rank - 1);
    std::uniform_int_distribution<int64_t> factorDist(2, 8);

    // Keep growing until we're close to target volume
    while (volume < maxVolume / 8) {
      int dim = dimDist(rng);
      int64_t factor = factorDist(rng);
      if (volume * factor <= maxVolume) {
        shape[dim] *= factor;
        volume *= factor;
      }
    }

    return shape;
  };

  // Helper to generate a random valid grid shape for a given logical shape
  auto generateGridShape =
      [&](ArrayRef<int64_t> logicalShape) -> SmallVector<int64_t> {
    SmallVector<int64_t> gridShape;
    for (int64_t dim : logicalShape) {
      auto divisors = getDivisors(dim, maxGridDim);
      std::uniform_int_distribution<size_t> dist(0, divisors.size() - 1);
      gridShape.push_back(divisors[dist(rng)]);
    }
    return gridShape;
  };

  // Track test result counts
  int successCount = 0;
  int subsetCount = 0;
  int failedCount = 0;
  int totalTests = 0;

  // Generate test cases for 3D and 4D shapes
  for (int rank = 2; rank <= 4; ++rank) {
    for (int shapeIdx = 0; shapeIdx < numLogicalShapeExamples; ++shapeIdx) {
      SmallVector<int64_t> logicalShape = generateLogicalShape(rank);

      for (int gridPairIdx = 0; gridPairIdx < numGridPairsPerShape;
           ++gridPairIdx) {
        SmallVector<int64_t> inputGridShape = generateGridShape(logicalShape);
        SmallVector<int64_t> outputGridShape = generateGridShape(logicalShape);

        TestResult result = printContiguityConstraints(
            logicalShape, inputGridShape, outputGridShape);

        switch (result) {
        case TestResult::Success:
          ++successCount;
          break;
        case TestResult::Subset:
          ++subsetCount;
          break;
        case TestResult::Failed:
          ++failedCount;
          break;
        }

        ++totalTests;
        if (totalTests % 100 == 0) {
          llvm::dbgs() << "[TestCanDetermineCoalescingFactor] Progress: "
                       << totalTests << " tests run (" << successCount
                       << " success, " << subsetCount << " subset, "
                       << failedCount << " failed)\n";
        }
      }
    }
  }

  llvm::dbgs() << "[TestCanDetermineCoalescingFactor] Summary: " << successCount
               << " success, " << subsetCount << " subset, " << failedCount
               << " failed (total: " << totalTests << ")\n";
}
} // namespace ttmlir::utils
