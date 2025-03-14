// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_SINGLETONDEVICECONTEXT_H
#define TTMLIR_OPMODEL_TTNN_SINGLETONDEVICECONTEXT_H
#ifdef TTMLIR_ENABLE_OPMODEL

#include "hostdevcommon/common_values.hpp"
#include <cstddef>

namespace tt {
namespace tt_metal {
class IDevice;
} // namespace tt_metal
} // namespace tt

namespace mlir::tt::op_model::ttnn {

// Singleton class to manage the device context, ensuring the device remains
// active while compiler is running multiple graph traces without real
// allocations and op dispatching.

// TODO (mbezulj): enforce mockup/simulation device when it's enabled in
// tt-metal.

class SingletonDeviceContext {
public:
  static SingletonDeviceContext &getInstance();
  static void resetInstance();

  ::tt::tt_metal::IDevice *getDevice() { return m_device; }

private:
  SingletonDeviceContext(
      const size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE);
  ~SingletonDeviceContext();

  SingletonDeviceContext(const SingletonDeviceContext &) = delete;
  SingletonDeviceContext &operator=(const SingletonDeviceContext &) = delete;

  ::tt::tt_metal::IDevice *m_device;

  void resetDevice(const size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE);
};
} // namespace mlir::tt::op_model::ttnn

#endif // TTMLIR_ENABLE_OPMODEL
#endif // TTMLIR_OPMODEL_TTNN_SINGLETONDEVICECONTEXT_H
