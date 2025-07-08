// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_IR_TTNNOPSRESOURCES_H
#define TTMLIR_DIALECT_TTNN_IR_TTNNOPSRESOURCES_H

#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::tt::ttnn {
struct TraceResource
    : public ::mlir::SideEffects::Resource::Base<TraceResource> {
  static TraceResource *get() {
    static TraceResource instance;
    return &instance;
  }

  llvm::StringRef getName() override { return "TraceResource"; }
};
} // namespace mlir::tt::ttnn

#endif
