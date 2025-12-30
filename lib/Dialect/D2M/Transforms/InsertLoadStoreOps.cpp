// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTLOADSTOREOPS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MInsertLoadStoreOps
    : public impl::D2MInsertLoadStoreOpsBase<D2MInsertLoadStoreOps> {
public:
  using impl::D2MInsertLoadStoreOpsBase<
      D2MInsertLoadStoreOps>::D2MInsertLoadStoreOpsBase;

  void runOnOperation() final {}
};
} // namespace

} // namespace mlir::tt::d2m
