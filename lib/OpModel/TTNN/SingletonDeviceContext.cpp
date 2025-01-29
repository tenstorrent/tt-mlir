// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifdef TTMLIR_ENABLE_OPMODEL
#include "SingletonDeviceContext.h"

#include "MetalHeaders.h"

namespace mlir::tt::op_model::ttnn {

// todo(arminaleTT): look into dynamically adjusting this
static constexpr size_t OP_MODEL_DEFAULT_TRACE_REGION_SIZE = 200000;

SingletonDeviceContext::SingletonDeviceContext(const size_t trace_region_size) {

  // todo: this replicates logic in runtime/include/tt/runtime/detail/common.h,
  // move to shared location
  size_t numDevices = ::tt::tt_metal::GetNumAvailableDevices();
  size_t numPCIeDevices = ::tt::tt_metal::GetNumPCIeDevices();
  ::tt::tt_metal::DispatchCoreType dispatchCoreType =
      numDevices == numPCIeDevices ? ::tt::tt_metal::DispatchCoreType::WORKER
                                   : ::tt::tt_metal::DispatchCoreType::ETH;
  m_device = ::tt::tt_metal::CreateDevice(
      0, /* num_hw_cqs = */ 1, /* l1_small_size = */ DEFAULT_L1_SMALL_SIZE,
      /* trace_region_size = */ trace_region_size, dispatchCoreType);
}

SingletonDeviceContext::~SingletonDeviceContext() {
  ::tt::tt_metal::CloseDevice(m_device);
}

SingletonDeviceContext &SingletonDeviceContext::getInstance() {
  static SingletonDeviceContext instance =
      SingletonDeviceContext(OP_MODEL_DEFAULT_TRACE_REGION_SIZE);
  return instance;
}

} // namespace mlir::tt::op_model::ttnn
#endif // TTMLIR_ENABLE_OPMODEL
