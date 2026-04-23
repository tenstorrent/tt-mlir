// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifdef TTMLIR_ENABLE_OPMODEL

#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"

#include "impl/context/metal_context.hpp"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/OpModel/TTNN/MetalHeaders.h"

#include <cstdlib>
#include <string>

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
  bool wasMock = instance.m_isMockDevice;
  instance.closeInstance();
  instance.openDevice(::tt::constants::opModelDefaultTraceRegionSize, wasMock);
}

void SingletonDeviceContext::closeInstance() {
  SingletonDeviceContext &instance = getInstance();
  assert(instance.m_device != nullptr && "No device to close");
  bool wasExternalDevice = instance.m_isExternalDevice;
  bool wasMockDevice = instance.m_isMockDevice;
  if (!wasExternalDevice) {
    disableFabric();
  }
  instance.m_device.reset();
  instance.m_isMockDevice = false;
  if (!wasExternalDevice && wasMockDevice) {
    ::tt::tt_metal::experimental::disable_mock_mode();
  }
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

void SingletonDeviceContext::setSystemDesc(ttcore::SystemDescAttr systemDesc) {
  SingletonDeviceContext &instance = getInstance();
  instance.m_systemDesc = systemDesc;
}

void SingletonDeviceContext::openMockDevice(
    const size_t traceRegionSize,
    const std::optional<std::pair<size_t, size_t>> &meshShape) {
#ifdef TTMLIR_DISABLE_MOCK_DEVICE
  bool disableMock = true;
#else
  bool disableMock = false;
#endif
  if (const char *env = std::getenv("TTMLIR_DISABLE_MOCK_DEVICE")) {
    disableMock = std::string(env) != "0";
  }
  openDevice(traceRegionSize, /*isMock=*/!disableMock, meshShape);
}

void SingletonDeviceContext::openDevice(
    const size_t traceRegionSize, bool isMock,
    const std::optional<std::pair<size_t, size_t>> &meshShape) {
  assert(m_device == nullptr &&
         "Device is already initialized. Cannot open device again.");

  m_isMockDevice = isMock;

  if (isMock) {
    assert(m_systemDesc && "System desc must be set for mock device mode");
    auto arch = m_systemDesc.getChipDesc(0).getArch().getValue();
    uint32_t numChips = m_systemDesc.getChipDescIndices().size();
    ::tt::ARCH metalArch;
    switch (arch) {
    case ttcore::Arch::WormholeB0:
      metalArch = ::tt::ARCH::WORMHOLE_B0;
      break;
    case ttcore::Arch::Blackhole:
      metalArch = ::tt::ARCH::BLACKHOLE;
      break;
    }
    ::tt::tt_metal::experimental::configure_mock_mode(metalArch, numChips);
  }

  // todo: this replicates logic in
  // runtime/include/tt/runtime/detail/common/common.h, move to shared location
  size_t numDevices = ::tt::tt_metal::GetNumAvailableDevices();
  size_t numPCIeDevices = ::tt::tt_metal::GetNumPCIeDevices();
  ::tt::tt_metal::DispatchCoreType dispatchCoreType =
      numDevices == numPCIeDevices ? ::tt::tt_metal::DispatchCoreType::WORKER
                                   : ::tt::tt_metal::DispatchCoreType::ETH;

  ::tt::tt_metal::distributed::MeshShape shape{
      meshShape ? static_cast<unsigned int>(meshShape->first) : 1,
      meshShape ? static_cast<unsigned int>(meshShape->second) : 1};

  // For real multi-chip devices, set fabric config before MeshDevice::create()
  // so device_manager initializes routing tables with the correct config.
  if (!isMock && shape.mesh_size() > 1) {
    ::tt::tt_fabric::SetFabricConfig(::tt::tt_fabric::FabricConfig::FABRIC_1D,
                                     ::tt::tt_fabric::FabricReliabilityMode::
                                         STRICT_SYSTEM_HEALTH_SETUP_MODE);
  }

  m_device = ::tt::tt_metal::distributed::MeshDevice::create(
      ::tt::tt_metal::distributed::MeshDeviceConfig{shape},
      ::tt::constants::L1_SMALL_SIZE, traceRegionSize,
      /* num_hw_cqs = */ 1, dispatchCoreType);

  m_device->disable_and_clear_program_cache();

  // For mock multi-chip devices, device_manager skips fabric init, so do it
  // manually after device creation.
  if (isMock && shape.mesh_size() > 1) {
    initializeFabricForMockDevice();
  }
}

void SingletonDeviceContext::reshapeMeshDevice(
    const std::pair<size_t, size_t> &meshShape, size_t traceRegionSize) {
  assert(m_device != nullptr && "Device must be initialized to reshape");
  assert(m_isMockDevice && "Can only reshape mock devices");

  disableFabric();
  m_device.reset();

  size_t numDevices = ::tt::tt_metal::GetNumAvailableDevices();
  size_t numPCIeDevices = ::tt::tt_metal::GetNumPCIeDevices();
  ::tt::tt_metal::DispatchCoreType dispatchCoreType =
      numDevices == numPCIeDevices ? ::tt::tt_metal::DispatchCoreType::WORKER
                                   : ::tt::tt_metal::DispatchCoreType::ETH;

  ::tt::tt_metal::distributed::MeshShape shape{
      static_cast<unsigned int>(meshShape.first),
      static_cast<unsigned int>(meshShape.second)};
  m_device = ::tt::tt_metal::distributed::MeshDevice::create(
      ::tt::tt_metal::distributed::MeshDeviceConfig{shape},
      ::tt::constants::L1_SMALL_SIZE, traceRegionSize,
      /* num_hw_cqs = */ 1, dispatchCoreType);

  m_device->disable_and_clear_program_cache();

  if (shape.mesh_size() > 1) {
    initializeFabricForMockDevice();
  }
}

llvm::SmallVector<int64_t>
SingletonDeviceContext::getComputeGridShape() const {
  assert(m_device != nullptr && "Device is not initialized.");
  auto grid = m_device->compute_with_storage_grid_size();
  // CoreCoord holds {x=cols, y=rows}; return as {rows, cols} to match GridAttr
  return {static_cast<int64_t>(grid.y), static_cast<int64_t>(grid.x)};
}

void SingletonDeviceContext::initializeFabricForMockDevice() {
  ::tt::tt_fabric::SetFabricConfig(
      ::tt::tt_fabric::FabricConfig::FABRIC_1D,
      ::tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
  ::tt::tt_metal::MetalContext::instance().initialize_fabric_config();
}

void SingletonDeviceContext::disableFabric() {
  ::tt::tt_fabric::SetFabricConfig(::tt::tt_fabric::FabricConfig::DISABLED);
}

} // namespace mlir::tt::ttnn::op_model
#endif // TTMLIR_ENABLE_OPMODEL
