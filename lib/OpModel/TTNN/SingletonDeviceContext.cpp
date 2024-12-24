// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifdef TTMLIR_ENABLE_OPMODEL
#include "SingletonDeviceContext.h"

#include "MetalHeaders.h"

namespace mlir::tt::op_model::ttnn {

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
#endif // TTMLIR_ENABLE_OPMODEL
