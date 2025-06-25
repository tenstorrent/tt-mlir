// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/GenericInterchangeAnalysis.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::tt::ttir {

static std::optional<SmallVector<int64_t>>
matchAndCalculateMatmulInterchange(ArrayRef<AffineMap> maps,
                                   ArrayRef<IteratorType> iters,
                                   ArrayRef<int64_t> desiredInterchange) {
  if (desiredInterchange.empty()) {
    return std::nullopt;
  }

  if (iters.size() < 3) {
    return std::nullopt;
  }

  if (iters.back() != IteratorType::Reduction) {
    return std::nullopt;
  }

  if (!llvm::all_of(iters.take_front(iters.size() - 1), [](IteratorType t) {
        return t == IteratorType::Parallel;
      })) {
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

static std::optional<SmallVector<int64_t>>
calculateInterchange(ArrayRef<AffineMap> maps, ArrayRef<IteratorType> iters,
                     const InterchangeOptions &options) {
  std::optional<SmallVector<int64_t>> interchange;

  interchange = matchAndCalculateMatmulInterchange(maps, iters,
                                                   options.matmulInterchange);
  if (interchange) {
    return interchange;
  }

  return interchange;
}

std::optional<SmallVector<int64_t>>
calculateInterchange(GenericOp op, const InterchangeOptions &options) {
  return calculateInterchange(op.getIndexingMapsValue(),
                              op.getIteratorTypesValue(), options);
}

} // namespace mlir::tt::ttir
