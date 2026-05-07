// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifdef TTMLIR_ENABLE_OPMODEL

#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"

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
  instance.m_device.reset();
  instance.m_isMockDevice = false;
  // m_metalEnv owns the cluster environment (mock or silicon); destroy it after
  // the device so teardown order is device → context → env.
  if (!instance.m_isExternalDevice) {
    instance.m_metalEnv.reset();
  }
  instance.m_isExternalDevice = false;
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
  assert(m_metalEnv == nullptr &&
         "MetalEnv is already initialized. Cannot open device again.");

  m_isMockDevice = isMock;

  fprintf(stderr, "[SingletonDeviceContext] openDevice isMock=%d\n", (int)isMock);
  ::tt::tt_metal::MetalEnvDescriptor descriptor;
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
    auto clusterDescName =
        ::tt::tt_metal::experimental::get_mock_cluster_desc_name(metalArch,
                                                                  numChips);
    assert(clusterDescName.has_value() &&
           "No mock cluster descriptor for the requested arch/num_chips");
    descriptor = ::tt::tt_metal::MetalEnvDescriptor(*clusterDescName);
  }
  m_metalEnv = std::make_unique<::tt::tt_metal::MetalEnv>(descriptor);

  auto dispatchCoreType =
      m_metalEnv->get_num_available_devices() ==
              m_metalEnv->get_num_pcie_devices()
          ? ::tt::tt_metal::DispatchCoreType::WORKER
          : ::tt::tt_metal::DispatchCoreType::ETH;

  ::tt::tt_metal::distributed::MeshShape shape{
      meshShape ? static_cast<unsigned int>(meshShape->first) : 1,
      meshShape ? static_cast<unsigned int>(meshShape->second) : 1};
  m_device = m_metalEnv->create_mesh_device(
      ::tt::tt_metal::distributed::MeshDeviceConfig{shape},
      ::tt::constants::L1_SMALL_SIZE, traceRegionSize,
      /* num_hw_cqs = */ 1,
      ::tt::tt_metal::DispatchCoreConfig{dispatchCoreType});

  m_device->disable_and_clear_program_cache();
}

void SingletonDeviceContext::reshapeMeshDevice(
    const std::pair<size_t, size_t> &meshShape, size_t traceRegionSize) {
  assert(m_device != nullptr && "Device must be initialized to reshape");
  assert(m_isMockDevice && "Can only reshape mock devices");
  assert(m_metalEnv != nullptr && "MetalEnv must exist to reshape");

  m_device.reset();

  auto dispatchCoreType =
      m_metalEnv->get_num_available_devices() ==
              m_metalEnv->get_num_pcie_devices()
          ? ::tt::tt_metal::DispatchCoreType::WORKER
          : ::tt::tt_metal::DispatchCoreType::ETH;

  ::tt::tt_metal::distributed::MeshShape shape{
      static_cast<unsigned int>(meshShape.first),
      static_cast<unsigned int>(meshShape.second)};
  m_device = m_metalEnv->create_mesh_device(
      ::tt::tt_metal::distributed::MeshDeviceConfig{shape},
      ::tt::constants::L1_SMALL_SIZE, traceRegionSize,
      /* num_hw_cqs = */ 1,
      ::tt::tt_metal::DispatchCoreConfig{dispatchCoreType});

  m_device->disable_and_clear_program_cache();
}

} // namespace mlir::tt::ttnn::op_model
#endif // TTMLIR_ENABLE_OPMODEL
