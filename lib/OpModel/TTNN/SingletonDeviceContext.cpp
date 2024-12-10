// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "SingletonDeviceContext.h"

#include "TTNNOpModelLib_Impl.h"
#include "impl/buffers/buffer_constants.hpp"

#include <iostream>

namespace mlir::tt::op_model::ttnn {

size_t SingletonDeviceContext::getNumL1Banks() const {
  return m_device->num_banks(::tt::tt_metal::BufferType::L1);
}
SingletonDeviceContext::SingletonDeviceContext() {
  m_device = ::tt::tt_metal::CreateDevice(0);
}
SingletonDeviceContext::~SingletonDeviceContext() {
  ::tt::tt_metal::CloseDevice(m_device);
}

SingletonDeviceContext &SingletonDeviceContext::getInstance() {
  static SingletonDeviceContext instance;
  return instance;
}

} // namespace mlir::tt::op_model::ttnn