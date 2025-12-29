// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERDMAOPS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MLowerDMAOps : public impl::D2MLowerDMAOpsBase<D2MLowerDMAOps> {
public:
  using impl::D2MLowerDMAOpsBase<D2MLowerDMAOps>::D2MLowerDMAOpsBase;

  void runOnOperation() final {
    // TODO: Implement D2M DMA lowering logic
  }
};
} // namespace

} // namespace mlir::tt::d2m
