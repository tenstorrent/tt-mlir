// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/Utils/Utils.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MANNOTATECOREINDEXMAPS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

class D2MAnnotateCoreIndexMaps
    : public impl::D2MAnnotateCoreIndexMapsBase<D2MAnnotateCoreIndexMaps> {
public:
  using impl::D2MAnnotateCoreIndexMapsBase<
      D2MAnnotateCoreIndexMaps>::D2MAnnotateCoreIndexMapsBase;

  void runOnOperation() final {
    utils::annotateCoreIndexOpsWithPhysicalToVirtualMaps(getOperation());
  }
};

} // namespace

} // namespace mlir::tt::d2m
