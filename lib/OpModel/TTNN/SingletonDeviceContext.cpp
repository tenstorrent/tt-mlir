// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifdef TTMLIR_ENABLE_OPMODEL

#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "Constants.h"
#include "ttmlir/OpModel/TTNN/MetalHeaders.h"

namespace mlir::tt::ttnn::op_model {

// todo(arminaleTT): look into dynamically adjusting this
// getOpRuntime() uses trace capture to run and measure the runtime of an op.
// This requires the device to be opened with sufficient trace region size. This
// number is currently set based on manual testing of supported ops to
// accommodate the highest required trace buffer size (2004992B)
static constexpr size_t opModelDefaultTraceRegionSize = 5000000;

SingletonDeviceContext::SingletonDeviceContext(const size_t traceRegionSize) {
  openDevice(traceRegionSize);
}

SingletonDeviceContext::~SingletonDeviceContext() { closeDevice(); }

SingletonDeviceContext &SingletonDeviceContext::getInstance() {
  static SingletonDeviceContext instance =
      SingletonDeviceContext(opModelDefaultTraceRegionSize);
  assert(instance.m_device != nullptr);
  return instance;
}

void SingletonDeviceContext::resetInstance() {
  SingletonDeviceContext &instance = getInstance();
  instance.closeDevice();
  instance.openDevice(opModelDefaultTraceRegionSize);
}

void SingletonDeviceContext::closeInstance() {
  SingletonDeviceContext &instance = getInstance();
  instance.closeDevice();
}

void SingletonDeviceContext::openDevice(const size_t traceRegionSize) {
  assert(m_device == nullptr);
  // todo: this replicates logic in
  // runtime/include/tt/runtime/detail/common/common.h, move to shared location
  size_t numDevices = ::tt::tt_metal::GetNumAvailableDevices();
  size_t numPCIeDevices = ::tt::tt_metal::GetNumPCIeDevices();
  ::tt::tt_metal::DispatchCoreType dispatchCoreType =
      numDevices == numPCIeDevices ? ::tt::tt_metal::DispatchCoreType::WORKER
                                   : ::tt::tt_metal::DispatchCoreType::ETH;

  ::tt::tt_metal::distributed::MeshShape shape{1, 1};
  m_device = ::tt::tt_metal::distributed::MeshDevice::create(
      ::tt::tt_metal::distributed::MeshDeviceConfig{shape},
      /* l1_small_size = */ ::tt::constants::L1_SMALL_SIZE,
      /* trace_region_size = */ traceRegionSize,
      /* num_hw_cqs = */ 1, dispatchCoreType);

  m_device->disable_and_clear_program_cache();
}

void SingletonDeviceContext::closeDevice() {
  if (m_device) {
    m_device->close();
    m_device.reset();
  }
}

} // namespace mlir::tt::ttnn::op_model
#endif // TTMLIR_ENABLE_OPMODEL
