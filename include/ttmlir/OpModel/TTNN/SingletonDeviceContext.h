// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_SINGLETONDEVICECONTEXT_H
#define TTMLIR_OPMODEL_TTNN_SINGLETONDEVICECONTEXT_H
#ifdef TTMLIR_ENABLE_OPMODEL

#include "hostdevcommon/common_values.hpp"
#include <cstddef>
#include <memory>

namespace tt {
namespace tt_metal {
namespace distributed {
class MeshDevice;
} // namespace distributed
} // namespace tt_metal
} // namespace tt

namespace mlir::tt::ttnn::op_model {

// Singleton class to manage the device context, ensuring the device remains
// active while compiler is running multiple graph traces without real
// allocations and op dispatching.

// TODO (mbezulj): enforce mockup/simulation device when it's enabled in
// tt-metal.

class SingletonDeviceContext {
public:
  static SingletonDeviceContext &getInstance();
  static void resetInstance();
  static void closeInstance();

  ::tt::tt_metal::distributed::MeshDevice *getDevice() {
    return m_device.get();
  }

private:
  SingletonDeviceContext(
      const size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE);
  ~SingletonDeviceContext();

  SingletonDeviceContext(const SingletonDeviceContext &) = delete;
  SingletonDeviceContext &operator=(const SingletonDeviceContext &) = delete;

  std::shared_ptr<::tt::tt_metal::distributed::MeshDevice> m_device;

  void openDevice(const size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE);
  void closeDevice();
};
} // namespace mlir::tt::ttnn::op_model

#endif // TTMLIR_ENABLE_OPMODEL
#endif // TTMLIR_OPMODEL_TTNN_SINGLETONDEVICECONTEXT_H
