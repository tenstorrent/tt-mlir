// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifdef TTMLIR_ENABLE_OPMODEL

#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "Constants.h"
#include "ttmlir/OpModel/TTNN/MetalHeaders.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::ttnn::op_model {

SingletonDeviceContext::~SingletonDeviceContext() {
  assert(
      m_device == nullptr &&
      "Device should be null when SingletonDeviceContext is destructed. Call "
      "closeInstance() once you are done with the device.");
}

SingletonDeviceContext &SingletonDeviceContext::getInstance() {
  static SingletonDeviceContext instance = SingletonDeviceContext();

  return instance;
}

void SingletonDeviceContext::resetInstance() {
  SingletonDeviceContext &instance = getInstance();
  assert(!instance.m_isExternalDevice &&
         "Cannot reset instance when using an external device.");
  instance.closeInstance();
  instance.openDevice(opModelDefaultTraceRegionSize);
}

void SingletonDeviceContext::closeInstance() {
  SingletonDeviceContext &instance = getInstance();
  assert(instance.m_device != nullptr && "No device to close");
  instance.m_device.reset();
}

void SingletonDeviceContext::setExternalDevice(
    std::shared_ptr<::tt::tt_metal::distributed::MeshDevice> device) {
  SingletonDeviceContext &instance = getInstance();
  assert(device != nullptr && "External device pointer cannot be null");
  assert(instance.m_device == nullptr &&
         "Device is already initialized. Cannot set external device.");
  instance.m_device = std::move(device);
  instance.m_isExternalDevice = true;
}

void SingletonDeviceContext::openDevice(const size_t traceRegionSize) {
  llvm::errs() << "[SingletonDeviceContext] openDevice() called with traceRegionSize=" 
               << traceRegionSize << "\n";
  
  assert(m_device == nullptr &&
         "Device is already initialized. Cannot open device again.");
  // todo: this replicates logic in
  // runtime/include/tt/runtime/detail/common/common.h, move to shared location
  size_t numDevices = ::tt::tt_metal::GetNumAvailableDevices();
  size_t numPCIeDevices = ::tt::tt_metal::GetNumPCIeDevices();
  ::tt::tt_metal::DispatchCoreType dispatchCoreType =
      numDevices == numPCIeDevices ? ::tt::tt_metal::DispatchCoreType::WORKER
                                   : ::tt::tt_metal::DispatchCoreType::ETH;

  llvm::errs() << "[SingletonDeviceContext] Device config: numDevices=" << numDevices
               << ", numPCIeDevices=" << numPCIeDevices
               << ", dispatchCoreType=" << (dispatchCoreType == ::tt::tt_metal::DispatchCoreType::WORKER ? "WORKER" : "ETH")
               << "\n";

  ::tt::tt_metal::distributed::MeshShape shape{1, 1};
  llvm::errs() << "[SingletonDeviceContext] Creating MeshDevice with shape={1,1}, "
               << "l1_small_size=" << ::tt::constants::L1_SMALL_SIZE
               << ", trace_region_size=" << traceRegionSize << "\n";
  
  m_device = ::tt::tt_metal::distributed::MeshDevice::create(
      ::tt::tt_metal::distributed::MeshDeviceConfig{shape},
      /* l1_small_size = */ ::tt::constants::L1_SMALL_SIZE,
      /* trace_region_size = */ traceRegionSize,
      /* num_hw_cqs = */ 1, dispatchCoreType);

  llvm::errs() << "[SingletonDeviceContext] Device created successfully, disabling program cache\n";
  m_device->disable_and_clear_program_cache();
  llvm::errs() << "[SingletonDeviceContext] openDevice() completed\n";
}

} // namespace mlir::tt::ttnn::op_model
#endif // TTMLIR_ENABLE_OPMODEL
