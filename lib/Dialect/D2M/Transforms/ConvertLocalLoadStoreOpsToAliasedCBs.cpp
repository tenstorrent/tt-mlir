// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MCONVERTLOCALLOADSTOREOPSTOALIASEDCBS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MConvertLocalLoadStoreOpsToAliasedCBs
    : public impl::D2MConvertLocalLoadStoreOpsToAliasedCBsBase<
          D2MConvertLocalLoadStoreOpsToAliasedCBs> {
public:
  using impl::D2MConvertLocalLoadStoreOpsToAliasedCBsBase<
      D2MConvertLocalLoadStoreOpsToAliasedCBs>::
      D2MConvertLocalLoadStoreOpsToAliasedCBsBase;

  void runOnOperation() final {
    // Stub implementation - pass does nothing for now
  }
};
} // namespace

} // namespace mlir::tt::d2m
