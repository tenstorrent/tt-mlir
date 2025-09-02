// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_SINGLETONDEVICECONTEXT_H
#define TTMLIR_OPMODEL_TTNN_SINGLETONDEVICECONTEXT_H
#ifdef TTMLIR_ENABLE_OPMODEL

#include "hostdevcommon/common_values.hpp"
#include <cassert>
#include <cstddef>
#include <memory>

namespace tt::tt_metal::distributed {
class MeshDevice;
} // namespace tt::tt_metal::distributed

namespace mlir::tt::ttnn::op_model {

// Singleton class to provide access to the global device context.
//
// Since we do not always own the device (it can be provided by frontends) and
// there currently isn't a clearly defined scope of the device (within a single
// function), the lifetime of the device is not managed automatically. The users
// of this class should ensure that the device lifetime fully contains their
// scope of usage. For now, if the device is not provided externally, we open
// our device once we start executing the optimizer and we close it after the
// `TTNNPrepareConv2dWeightsAndBias` pass - this covers the whole span where we
// might need the device (for now).
//
// Because the lifetime needs to be managed manually, there are asserts in place
// to verify that the instance is used correctly.
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

  // Opens a new (owned) device. Users need to ensure that we don't have an
  // active device in the current context, otherwise this method will assert.
  void
  openDevice(const size_t trace_region_size = opModelDefaultTraceRegionSize);

  // Returns a pointer to the device. Asserts that we have an active device in
  // our context.
  ::tt::tt_metal::distributed::MeshDevice *getDevice() {
    assert(m_device != nullptr && "Device is not initialized.");
    return m_device.get();
  }

private:
  SingletonDeviceContext() = default;
  ~SingletonDeviceContext();

  SingletonDeviceContext(const SingletonDeviceContext &) = delete;
  SingletonDeviceContext &operator=(const SingletonDeviceContext &) = delete;

  std::shared_ptr<::tt::tt_metal::distributed::MeshDevice> m_device;

  // This field is technically not needed, but is there to assert that
  // `resetInstance()` is legal. If we are using an external device, we cannot
  // reset the instance.
  bool m_isExternalDevice = false;

  // todo(arminaleTT): look into dynamically adjusting this
  // getOpRuntime() uses trace capture to run and measure the runtime of an op.
  // This requires the device to be opened with sufficient trace region size.
  // This number is currently set based on manual testing of supported ops to
  // accommodate the highest required trace buffer size (2004992B)
  static constexpr size_t opModelDefaultTraceRegionSize = 5000000;
};
} // namespace mlir::tt::ttnn::op_model

#endif // TTMLIR_ENABLE_OPMODEL
#endif // TTMLIR_OPMODEL_TTNN_SINGLETONDEVICECONTEXT_H
