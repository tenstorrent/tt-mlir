// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_OPMODELDEVICEGUARD_H
#define TTMLIR_OPMODEL_TTNN_OPMODELDEVICEGUARD_H
#ifdef TTMLIR_ENABLE_OPMODEL

#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"

namespace mlir::tt::ttnn::op_model {

// RAII guard for OpModel device lifecycle management.
// This guard ensures that if a pass opens a device, it will be properly closed
// when the guard goes out of scope, even if exceptions or early returns occur.
// Useful for standalone passes that need to ensure device is available.
//
// Usage:
//   void runOnOperation() override {
//     OpModelDeviceGuard deviceGuard;
//     // ... do work with device ...
//   }  // Device automatically closed if opened by this guard
//
class OpModelDeviceGuard {
public:
  OpModelDeviceGuard() {
    if (!SingletonDeviceContext::getInstance().isDeviceInitialized()) {
      SingletonDeviceContext::getInstance().openDevice();
      m_deviceOpenedByGuard = true;
    }
  }

  ~OpModelDeviceGuard() {
    if (m_deviceOpenedByGuard &&
        SingletonDeviceContext::getInstance().isDeviceInitialized()) {
      SingletonDeviceContext::closeInstance();
    }
  }

  // Non-copyable
  OpModelDeviceGuard(const OpModelDeviceGuard &) = delete;
  OpModelDeviceGuard &operator=(const OpModelDeviceGuard &) = delete;

  // Non-movable
  OpModelDeviceGuard(OpModelDeviceGuard &&) = delete;
  OpModelDeviceGuard &operator=(OpModelDeviceGuard &&) = delete;

private:
  bool m_deviceOpenedByGuard = false;
};

} // namespace mlir::tt::ttnn::op_model

#else // TTMLIR_ENABLE_OPMODEL

namespace mlir::tt::ttnn::op_model {

// No-op guard when OpModel is disabled
class OpModelDeviceGuard {
public:
  OpModelDeviceGuard() = default;
  ~OpModelDeviceGuard() = default;

  // Non-copyable
  OpModelDeviceGuard(const OpModelDeviceGuard &) = delete;
  OpModelDeviceGuard &operator=(const OpModelDeviceGuard &) = delete;

  // Non-movable
  OpModelDeviceGuard(OpModelDeviceGuard &&) = delete;
  OpModelDeviceGuard &operator=(OpModelDeviceGuard &&) = delete;
};

} // namespace mlir::tt::ttnn::op_model

#endif // TTMLIR_ENABLE_OPMODEL
#endif // TTMLIR_OPMODEL_TTNN_OPMODELDEVICEGUARD_H
