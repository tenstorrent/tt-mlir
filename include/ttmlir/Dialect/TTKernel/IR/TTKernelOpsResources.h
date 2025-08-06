// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTKERNEL_IR_TTKERNELOPSRESOURCES_H
#define TTMLIR_DIALECT_TTKERNEL_IR_TTKERNELOPSRESOURCES_H

#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::tt::ttkernel {
struct HWInitState
    : public ::mlir::SideEffects::Resource::Base<HWInitState> {
  static HWInitState *get() {
    static HWInitState instance;
    return &instance;
  }

  llvm::StringRef getName() override { return "HWInitState"; }
};
} // namespace mlir::tt::ttkernel

#endif // TTMLIR_DIALECT_TTKERNEL_IR_TTKERNELOPSRESOURCES_H 