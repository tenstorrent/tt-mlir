// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDECOUPLELOADSTOREOPSFROMCOMPUTE
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MDecoupleLoadStoreOpsFromCompute
    : public impl::D2MDecoupleLoadStoreOpsFromComputeBase<
          D2MDecoupleLoadStoreOpsFromCompute> {
public:
  using impl::D2MDecoupleLoadStoreOpsFromComputeBase<
      D2MDecoupleLoadStoreOpsFromCompute>::
      D2MDecoupleLoadStoreOpsFromComputeBase;

  void runOnOperation() final {
    // Stub implementation - pass does nothing
  }
};
} // namespace

} // namespace mlir::tt::d2m
