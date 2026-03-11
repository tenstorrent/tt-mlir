// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_SINGLETONDEVICECONTEXT_H
#define TTMLIR_OPMODEL_TTNN_SINGLETONDEVICECONTEXT_H
#ifdef TTMLIR_ENABLE_OPMODEL

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "Constants.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

namespace mlir {
class Operation;
} // namespace mlir

namespace tt::tt_metal::distributed {
class MeshDevice;
} // namespace tt::tt_metal::distributed

namespace mlir::tt::ttnn::op_model {

// Singleton class to provide access to the global device context.
//
// The singleton exposes methods to open device, provide already opened device
// and close the current device.
//
// Because we need to support both externally provided device
// and to open device on our own (if needed), the device is not managed in RAII
// fashion.
//
// Due to the above, it is essential that the users of this class ensure that
// they have a valid device in the context when they need it.
//
// Because the lifetime of the device needs to be managed manually, there are
// asserts in place to verify that the singleton instance is used correctly.
//
// NOTE: If we would close active device in the destructor, we would crash. The
// reason being that the `tt-metal` singletons (needed for managing devices)
// would be tore down before ours.
//
// TODO (mbezulj): enforce mockup/simulation device when it's enabled in
// tt-metal.
class SingletonDeviceContext {
public:
  static SingletonDeviceContext &getInstance();

  // Resets the instance by closing and reopening the device (this is used in
  // testing). The device needs to be open before calling this function. Asserts
  // that the device is not externally managed (since we cannot close it).
  static void resetInstance();

  // Clears the device context and closes the device if needed; when the device
  // is externally managed we will just reset the `shared_ptr` without closing
  // the device.
  //
  // After this function is called we can safely call
  // `openDevice()` or `setExternalDevice()` again and initialize the device
  // context with a new device.
  static void closeInstance();

  // Sets an externally managed device - this is used when the device is passed
  // to us by the frontend.
  static void setExternalDevice(
      std::shared_ptr<::tt::tt_metal::distributed::MeshDevice> device);

  // Sets the system descriptor attribute. This is used extract architecture
  // info for mock mode.
  static void setSystemDesc(ttcore::SystemDescAttr systemDesc);

  // Opens a new (owned) device. When isMock is true, configures mock mode
  // using the system descriptor (setSystemDesc must be called first).
  // When isMock is false (default), opens a real device connected to hardware.
  // Users need to ensure that we don't have an active device in the current
  // context, otherwise this method will assert.
  // meshShape overrides the default {1, 1} mesh shape when provided.
  void openDevice(
      const size_t traceRegionSize =
          ::tt::constants::opModelDefaultTraceRegionSize,
      bool isMock = false,
      const std::optional<std::pair<size_t, size_t>> &meshShape = std::nullopt);

  // Convenience method that opens a mock device.
  // TODO(#7384) TraceRegionSize might be irrelevant for mock devices,
  // but we set it to a default value just in case for now.
  void openMockDevice(
      const size_t traceRegionSize =
          ::tt::constants::opModelDefaultTraceRegionSize,
      const std::optional<std::pair<size_t, size_t>> &meshShape = std::nullopt);

  // Destroys the current MeshDevice and creates a new one with a different
  // mesh shape. Does NOT re-configure or disable mock mode, so mock mode
  // must already be active.
  // This exists because Metal's configure_mock_mode/disable_mock_mode
  // cannot be reliably cycled within the same process.
  void reshapeMeshDevice(const std::pair<size_t, size_t> &meshShape,
                         size_t traceRegionSize = 0);

  // Returns a pointer to the device. Asserts that we have an active device in
  // our context.
  ::tt::tt_metal::distributed::MeshDevice *getDevice() {
    assert(m_device != nullptr && "Device is not initialized.");
    return m_device.get();
  }

  // Returns true if the device is initialized.
  bool isDeviceInitialized() const { return m_device != nullptr; }

  // Returns true if the device was opened via the mock path
  // (no real HW).
  bool isMockDevice() const { return m_isMockDevice; }

private:
  SingletonDeviceContext() = default;
  ~SingletonDeviceContext();

  SingletonDeviceContext(const SingletonDeviceContext &) = delete;
  SingletonDeviceContext &operator=(const SingletonDeviceContext &) = delete;

  std::shared_ptr<::tt::tt_metal::distributed::MeshDevice> m_device;
  ttcore::SystemDescAttr m_systemDesc;

  // This field is technically not needed, but is there to assert that
  // `resetInstance()` is legal. If we are using an external device, we cannot
  // reset the instance.
  bool m_isExternalDevice = false;
  bool m_isMockDevice = false;
};

// RAII guard for OpModel device lifecycle management.
// This guard ensures that if a pass opens a device, it will be properly closed
// when the guard goes out of scope, even if exceptions or early returns occur.
// Useful for standalone passes that need to ensure device is available.
//
// Usage:
//   void runOnOperation() override {
//     ScopedSingletonDeviceGuard deviceGuard;
//     // ... do work with device ...
//   }  // Device automatically closed if opened by this guard
//
class ScopedSingletonDeviceGuard {
public:
  explicit ScopedSingletonDeviceGuard(mlir::Operation *op) {
    SingletonDeviceContext::setSystemDesc(
        ttcore::getCurrentScopeSystemDesc(op));
    if (!SingletonDeviceContext::getInstance().isDeviceInitialized()) {
      SingletonDeviceContext::getInstance().openMockDevice();
      m_deviceOpenedByGuard = true;
    }
  }

  ~ScopedSingletonDeviceGuard() {
    if (m_deviceOpenedByGuard &&
        SingletonDeviceContext::getInstance().isDeviceInitialized()) {
      SingletonDeviceContext::closeInstance();
    }
  }

  // Non-copyable
  ScopedSingletonDeviceGuard(const ScopedSingletonDeviceGuard &) = delete;
  ScopedSingletonDeviceGuard &
  operator=(const ScopedSingletonDeviceGuard &) = delete;

  // Non-movable
  ScopedSingletonDeviceGuard(ScopedSingletonDeviceGuard &&) = delete;
  ScopedSingletonDeviceGuard &operator=(ScopedSingletonDeviceGuard &&) = delete;

private:
  bool m_deviceOpenedByGuard = false;
};

} // namespace mlir::tt::ttnn::op_model

#endif // TTMLIR_ENABLE_OPMODEL
#endif // TTMLIR_OPMODEL_TTNN_SINGLETONDEVICECONTEXT_H
