// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/GenericInterchangeAnalysis.h"

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

namespace mlir::tt::d2m {

static std::optional<SmallVector<int64_t>>
matchAndCalculateMatmulInterchange(ArrayRef<AffineMap> maps,
                                   ArrayRef<ttcore::IteratorType> iters,
                                   ArrayRef<int64_t> desiredInterchange) {
  if (desiredInterchange.empty()) {
    return std::nullopt;
  }

  if (iters.size() < 3) {
    return std::nullopt;
  }

  if (iters.back() != ttcore::IteratorType::Reduction) {
    return std::nullopt;
  }

  if (!llvm::all_of(iters.take_front(iters.size() - 1),
                    [](ttcore::IteratorType t) {
                      return t == ttcore::IteratorType::Parallel;
                    })) {
    return std::nullopt;
  }

  if (maps.size() != 3) {
    return std::nullopt;
  }

  auto lhs = maps[0];
  auto rhs = maps[1];
  auto out = maps[2];
  auto numResults = lhs.getNumResults();
  assert(numResults == rhs.getNumResults());
  assert(numResults == out.getNumResults());

  auto lhsM = lhs.getResult(numResults - 2);
  auto lhsK = lhs.getResult(numResults - 1);
  auto rhsK = rhs.getResult(numResults - 2);
  auto rhsN = rhs.getResult(numResults - 1);
  auto outM = out.getResult(numResults - 2);
  auto outN = out.getResult(numResults - 1);

  if (lhsM != outM || lhsK != rhsK || rhsN != outN) {
    return std::nullopt;
  }

  SmallVector<int64_t> interchange = llvm::map_to_vector(
      llvm::seq<int64_t>(0, iters.size()), [](int64_t i) { return i; });

  assert(desiredInterchange.size() == 3);

  auto offset = iters.size() - 3;
  interchange[offset + 0] = desiredInterchange[0];
  interchange[offset + 1] = desiredInterchange[1];
  interchange[offset + 2] = desiredInterchange[2];

  return interchange;
}

std::optional<SmallVector<int64_t>>
calculateInterchange(GenericOp op, const InterchangeOptions &options) {
  return matchAndCalculateMatmulInterchange(op.getIndexingMapsValue(),
                                            op.getIteratorTypesValue(),
                                            options.matmulInterchange);
}

} // namespace mlir::tt::d2m
