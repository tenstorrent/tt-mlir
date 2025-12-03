// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/Utils.h"

#include <gtest/gtest.h>

namespace mlir::tt::d2m {

class GenericOpAnalysisTest : public ::testing::Test {
protected:
  mlir::MLIRContext context;
  mlir::AffineExpr d0, d1, d2, d3;

  GenericOpAnalysisTest() {
    d0 = mlir::getAffineDimExpr(0, &context);
    d1 = mlir::getAffineDimExpr(1, &context);
    d2 = mlir::getAffineDimExpr(2, &context);
    d3 = mlir::getAffineDimExpr(3, &context);
  }

  mlir::AffineMap makeAffineMap(unsigned dimCount,
                                mlir::ArrayRef<mlir::AffineExpr> exprs) {
    return mlir::AffineMap::get(dimCount, 0, exprs, &context);
  }

  void checkDimConstraints(
      std::vector<std::pair<mlir::AffineMap, mlir::SmallVector<int64_t>>>
          operands,
      std::optional<mlir::SmallVector<int64_t>> expected) {
    mlir::SmallVector<mlir::AffineMap> maps;
    mlir::SmallVector<mlir::SmallVector<int64_t>> shapes;
    for (const auto &op : operands) {
      maps.push_back(op.first);
      shapes.push_back(op.second);
    }
    auto result = d2m::utils::computeDimConstraints(maps, shapes);
    EXPECT_EQ(result, expected);
  }
};

TEST_F(GenericOpAnalysisTest, CanAnalyzeDimConstraintsEltwise) {
  // 2D shape
  checkDimConstraints({{makeAffineMap(2, {d0, d1}), {1, 2}},
                       {makeAffineMap(2, {d0, d1}), {1, 2}}},
                      std::make_optional(mlir::SmallVector<int64_t>{1, 2}));

  // 3D shape
  checkDimConstraints({{makeAffineMap(3, {d0, d1, d2}), {1, 2, 4}},
                       {makeAffineMap(3, {d0, d1, d2}), {1, 2, 4}}},
                      std::make_optional(mlir::SmallVector<int64_t>{1, 2, 4}));
}

TEST_F(GenericOpAnalysisTest, CanAnalyzeDimConstraintsForPermute) {
  // 2D transpose
  checkDimConstraints({{makeAffineMap(2, {d0, d1}), {1, 2}},
                       {makeAffineMap(2, {d1, d0}), {2, 1}}},
                      std::make_optional(mlir::SmallVector<int64_t>{1, 2}));

  // 3D permute
  checkDimConstraints(
      {
          {makeAffineMap(3, {d2, d1, d0}), {3, 2, 1}},
          {makeAffineMap(3, {d0, d1, d2}), {1, 2, 3}},
      },
      std::make_optional(mlir::SmallVector<int64_t>{1, 2, 3}));

  // 4D permute with multiple dim swaps
  checkDimConstraints(
      {
          {makeAffineMap(4, {d0, d1, d2, d3}), {4, 3, 2, 5}},
          {makeAffineMap(4, {d2, d3, d0, d1}), {2, 5, 4, 3}},
      },
      std::make_optional(mlir::SmallVector<int64_t>{4, 3, 2, 5}));
}

TEST_F(GenericOpAnalysisTest, CanAnalyzeDimConstraintsForMatmul) {
  // dim mapping: d0 = M, d1 = N, d2 = K
  checkDimConstraints({{makeAffineMap(3, {d0, d2}), {3, 5}},
                       {makeAffineMap(3, {d2, d1}), {5, 2}},
                       {makeAffineMap(3, {d0, d1}), {3, 2}}},
                      std::make_optional(mlir::SmallVector<int64_t>{3, 2, 5}));
}

TEST_F(GenericOpAnalysisTest, CanAnalyzePartialDimConstraintsForMatmul) {
  // only output shape is provided; d2 is a 'free dim' and not constrained.
  checkDimConstraints({{makeAffineMap(3, {d0, d1}), {3, 2}}},
                      std::make_optional(mlir::SmallVector<int64_t>{3, 2, 0}));
}

TEST_F(GenericOpAnalysisTest, CanAnalyzeDimConstraintsForIncompatibleShapes) {
  checkDimConstraints({{makeAffineMap(2, {d0, d1}), {1, 2}},
                       {makeAffineMap(2, {d0, d1}), {1, 3}}},
                      std::nullopt);
}

} // namespace mlir::tt::d2m
