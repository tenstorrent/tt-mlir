// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "SingletonDeviceContext.h"

#include "MetalHeaders.h"

namespace mlir::tt::op_model::ttnn {

SingletonDeviceContext::SingletonDeviceContext() {
  m_device = ::tt::tt_metal::v0::CreateDevice(0);
}

SingletonDeviceContext::~SingletonDeviceContext() {
  ::tt::tt_metal::v0::CloseDevice(m_device);
}

SingletonDeviceContext &SingletonDeviceContext::getInstance() {
  static SingletonDeviceContext instance;
  return instance;
}

} // namespace mlir::tt::op_model::ttnn