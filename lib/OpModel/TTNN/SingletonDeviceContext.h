// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_SINGLETONDEVICECONTEXT_H
#define TTMLIR_OPMODEL_TTNN_SINGLETONDEVICECONTEXT_H

#include <cstddef>

namespace tt {
namespace tt_metal {
inline namespace v0 {
class Device;
} // namespace v0
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

  ::tt::tt_metal::v0::Device *getDevice() { return m_device; }

private:
  SingletonDeviceContext();
  ~SingletonDeviceContext();

  SingletonDeviceContext(const SingletonDeviceContext &) = delete;
  SingletonDeviceContext &operator=(const SingletonDeviceContext &) = delete;

  ::tt::tt_metal::v0::Device *m_device;
};
} // namespace mlir::tt::op_model::ttnn

#endif // TTMLIR_OPMODEL_TTNN_SINGLETONDEVICECONTEXT_H
