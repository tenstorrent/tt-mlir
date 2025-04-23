// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifdef TTMLIR_ENABLE_OPMODEL
#include "SingletonDeviceContext.h"

#include "Constants.h"
#include "MetalHeaders.h"
#include <tt-metallium/distributed.hpp>

namespace mlir::tt::op_model::ttnn {

// todo(arminaleTT): look into dynamically adjusting this
// getOpRuntime() uses trace capture to run and measure the runtime of an op.
// This requires the device to be opened with sufficient trace region size. This
// number is currently set based on manual testing of supported ops
static constexpr size_t opModelDefaultTraceRegionSize = 200000;

SingletonDeviceContext::SingletonDeviceContext(const size_t traceRegionSize) {
  resetDevice(traceRegionSize);
}

SingletonDeviceContext::~SingletonDeviceContext() {
  ::tt::tt_metal::CloseDevice(m_device);
}

SingletonDeviceContext &SingletonDeviceContext::getInstance() {
  static SingletonDeviceContext instance =
      SingletonDeviceContext(opModelDefaultTraceRegionSize);
  return instance;
}

void SingletonDeviceContext::resetInstance() {
  SingletonDeviceContext &instance = getInstance();
  instance.resetDevice(opModelDefaultTraceRegionSize);
}

void SingletonDeviceContext::resetDevice(const size_t traceRegionSize) {
  if (m_device) {
    ::tt::tt_metal::CloseDevice(m_device);
  }

  // todo: this replicates logic in runtime/include/tt/runtime/detail/common.h,
  // move to shared location
  size_t numDevices = ::tt::tt_metal::GetNumAvailableDevices();
  size_t numPCIeDevices = ::tt::tt_metal::GetNumPCIeDevices();
  ::tt::tt_metal::DispatchCoreType dispatchCoreType =
      numDevices == numPCIeDevices ? ::tt::tt_metal::DispatchCoreType::WORKER
                                   : ::tt::tt_metal::DispatchCoreType::ETH;
  m_device = ::tt::tt_metal::distributed::MeshDevice::create_unit_mesh(
      /*device_id*/ 0,
      /*l1_small_size*/ ::tt::constants::L1_SMALL_SIZE,
      /*trace_region_size*/ traceRegionSize,
      /*num_command_queues*/ 1,
      /*dispatch_core_config*/ DispatchCoreConfig(dispatchCoreType));
}

} // namespace mlir::tt::op_model::ttnn
#endif // TTMLIR_ENABLE_OPMODEL
