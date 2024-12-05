// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_SINGLETONDEVICECONTEXT_H
#define TTMLIR_OPMODEL_TTNN_SINGLETONDEVICECONTEXT_H

#include "TTNNOpModelLib_Impl.h"
#include "impl/buffers/buffer_constants.hpp"

namespace mlir::tt::op_model::ttnn {

// Singleton class to manage the device context, ensuring the device remains
// active while compiler is running multiple graph traces without real
// allocations and op dispatching.

// TODO (mbezulj): enforce mockup/simulation device when it's enabled in
// tt-metal.

class SingletonDeviceContext {
public:
  static SingletonDeviceContext &getInstance() {
    static SingletonDeviceContext instance;
    return instance;
  }

  ::tt::tt_metal::Device *getDevice() { return m_device; }

  size_t getNumL1Banks() const {
    return m_device->num_banks(::tt::tt_metal::BufferType::L1);
  }

private:
  SingletonDeviceContext() : m_device(::tt::tt_metal::CreateDevice(0)) {}
  ~SingletonDeviceContext() { ::tt::tt_metal::CloseDevice(m_device); }

  SingletonDeviceContext(const SingletonDeviceContext &) = delete;
  SingletonDeviceContext &operator=(const SingletonDeviceContext &) = delete;

  ::tt::tt_metal::Device *m_device;
};
} // namespace mlir::tt::op_model::ttnn

#endif // TTMLIR_OPMODEL_TTNN_SINGLETONDEVICECONTEXT_H
