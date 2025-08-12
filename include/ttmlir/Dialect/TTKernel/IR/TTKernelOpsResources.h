// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTKERNEL_IR_TTKERNELOPSRESOURCES_H
#define TTMLIR_DIALECT_TTKERNEL_IR_TTKERNELOPSRESOURCES_H

#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// Singular Memory Resource Used for 'Init' Hoisting with LICM
//===----------------------------------------------------------------------===//

namespace mlir::tt::ttkernel {
struct HWInitState : public ::mlir::SideEffects::Resource::Base<HWInitState> {
  static HWInitState *get() {
    static HWInitState instance;
    return &instance;
  }

  llvm::StringRef getName() override { return "HWInitState"; }
};

//===----------------------------------------------------------------------===//
// Placeholder Resources for Verifying Multi-Resource Memory Inits
//===----------------------------------------------------------------------===//

struct RegAInitState
    : public ::mlir::SideEffects::Resource::Base<RegAInitState> {
  static RegAInitState *get() {
    static RegAInitState instance;
    return &instance;
  }

  llvm::StringRef getName() override { return "RegAInitState"; }
};

struct RegBInitState
    : public ::mlir::SideEffects::Resource::Base<RegBInitState> {
  static RegBInitState *get() {
    static RegBInitState instance;
    return &instance;
  }

  llvm::StringRef getName() override { return "RegBInitState"; }
};

struct RegCInitState
    : public ::mlir::SideEffects::Resource::Base<RegCInitState> {
  static RegCInitState *get() {
    static RegCInitState instance;
    return &instance;
  }

  llvm::StringRef getName() override { return "RegCInitState"; }
};

struct RegDInitState
    : public ::mlir::SideEffects::Resource::Base<RegDInitState> {
  static RegDInitState *get() {
    static RegDInitState instance;
    return &instance;
  }

  llvm::StringRef getName() override { return "RegDInitState"; }
};

struct RegEInitState
    : public ::mlir::SideEffects::Resource::Base<RegEInitState> {
  static RegEInitState *get() {
    static RegEInitState instance;
    return &instance;
  }

  llvm::StringRef getName() override { return "RegEInitState"; }
};

struct RegFInitState
    : public ::mlir::SideEffects::Resource::Base<RegFInitState> {
  static RegFInitState *get() {
    static RegFInitState instance;
    return &instance;
  }

  llvm::StringRef getName() override { return "RegFInitState"; }
};

} // namespace mlir::tt::ttkernel

#endif // TTMLIR_DIALECT_TTKERNEL_IR_TTKERNELOPSRESOURCES_H
