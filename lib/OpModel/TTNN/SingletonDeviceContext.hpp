// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "TTNNOpModelLib_Impl.hpp"

namespace mlir::tt::op_model::ttnn {

// Singleton class to manage the device context, ensuring the device remains
// active while compiler is running multiple graph traces without dispatching.
// TODO (mbezulj): enforce mockup/simulation device when it's enabled in main.

class SingletonDeviceContext {
public:
  static SingletonDeviceContext &get_instance() {
    static SingletonDeviceContext instance;
    return instance;
  }

  ::tt::tt_metal::Device *get_device() { return m_device; }

  size_t get_compute_with_storage_grid_size() const {
    return m_device->compute_with_storage_grid_size().x *
           m_device->compute_with_storage_grid_size().y;
  }

private:
  SingletonDeviceContext() : m_device(::tt::tt_metal::CreateDevice(0)) {}
  ~SingletonDeviceContext() { ::tt::tt_metal::CloseDevice(m_device); }

  SingletonDeviceContext(const SingletonDeviceContext &) = delete;
  SingletonDeviceContext &operator=(const SingletonDeviceContext &) = delete;

  ::tt::tt_metal::Device *m_device;
};
} // namespace mlir::tt::op_model::ttnn
