// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifdef TTMLIR_ENABLE_OPMODEL

#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/OpModel/TTNN/MetalHeaders.h"

#include "impl/context/metal_env_accessor.hpp"
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/mock_device.hpp>

#include <cstdlib>
#include <string>

namespace mlir::tt::ttnn::op_model {

namespace {

::tt::ARCH toMetalArch(ttcore::Arch arch) {
  switch (arch) {
  case ttcore::Arch::WormholeB0:
    return ::tt::ARCH::WORMHOLE_B0;
  case ttcore::Arch::Blackhole:
    return ::tt::ARCH::BLACKHOLE;
  }
  llvm_unreachable("Unknown arch");
}

// Build a MetalEnvDescriptor with fabric DISABLED. For multi-chip mock
// topologies we enable fabric AFTER device creation via
// MetalEnvAccessor(env).impl().set_fabric_config(...), matching the legacy
// order (configure_mock_mode -> create device -> SetFabricConfig ->
// initialize_fabric_config). Enabling fabric at env construction time fires
// FabricBuilder before mock devices are fully registered and crashes in
// fabric_builder_context.cpp:173.
::tt::tt_metal::MetalEnvDescriptor makeEnvDescriptor(bool isMock,
                                                     ttcore::Arch arch,
                                                     uint32_t numChips) {
  if (!isMock) {
    return ::tt::tt_metal::MetalEnvDescriptor{};
  }
  auto mockClusterPath = ::tt::tt_metal::experimental::get_mock_cluster_desc_name(
      toMetalArch(arch), numChips);
  assert(mockClusterPath &&
         "Unsupported (arch, numChips) for mock cluster descriptor");
  return ::tt::tt_metal::MetalEnvDescriptor{*mockClusterPath};
}

// Enable FABRIC_1D in RELAXED mode for multi-chip mock. Called AFTER device
// creation so the FabricBuilder has the chip routing tables it needs.
// nullopt num_routing_planes resolves to numeric_limits<uint8_t>::max() inside
// MetalEnvImpl::set_fabric_config (via value_or) — "reserve all available".
void enableFabricForMockMultiChip(::tt::tt_metal::MetalEnv &env) {
  ::tt::tt_metal::MetalEnvAccessor(env).impl().set_fabric_config(
      ::tt::tt_fabric::FabricConfig::FABRIC_1D,
      ::tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
}

} // namespace

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
  instance.m_device.reset();
  if (!wasExternalDevice) {
    // RAII teardown: destroying the env tears down fabric + mock cluster.
    instance.m_env.reset();
  }
  instance.m_isMockDevice = false;
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
  assert(m_env == nullptr && "MetalEnv must be torn down before opening.");

  m_isMockDevice = isMock;

  ttcore::Arch arch = ttcore::Arch::WormholeB0;
  uint32_t numChips = 1;
  if (isMock) {
    assert(m_systemDesc && "System desc must be set for mock device mode");
    arch = m_systemDesc.getChipDesc(0).getArch().getValue();
    numChips = m_systemDesc.getChipDescIndices().size();
  }

  m_env = std::make_unique<::tt::tt_metal::MetalEnv>(
      makeEnvDescriptor(isMock, arch, numChips));

  ::tt::tt_metal::DispatchCoreType dispatchCoreType =
      m_env->get_num_available_devices() == m_env->get_num_pcie_devices()
          ? ::tt::tt_metal::DispatchCoreType::WORKER
          : ::tt::tt_metal::DispatchCoreType::ETH;

  ::tt::tt_metal::distributed::MeshShape shape{
      meshShape ? static_cast<unsigned int>(meshShape->first) : 1,
      meshShape ? static_cast<unsigned int>(meshShape->second) : 1};

  m_device = m_env->create_mesh_device(
      ::tt::tt_metal::distributed::MeshDeviceConfig{shape},
      ::tt::constants::L1_SMALL_SIZE, traceRegionSize,
      /*num_command_queues=*/1,
      ::tt::tt_metal::DispatchCoreConfig{dispatchCoreType});

  m_device->disable_and_clear_program_cache();

  // For mock multi-chip, enable fabric AFTER device creation (matching the
  // legacy lifecycle). Doing this at env construction time crashes in
  // FabricBuilder because mock chips aren't fully registered yet.
  if (isMock && shape.mesh_size() > 1) {
    enableFabricForMockMultiChip(*m_env);
  }
}

void SingletonDeviceContext::reshapeMeshDevice(
    const std::pair<size_t, size_t> &meshShape, size_t traceRegionSize) {
  assert(m_device != nullptr && "Device must be initialized to reshape");
  assert(m_isMockDevice && "Can only reshape mock devices");
  assert(m_env != nullptr && "MetalEnv must be alive for reshape");

  // MetalEnv pins the cluster descriptor at construction, so reshaping to a
  // different mesh size means a fresh env with a different cluster_desc_name.
  // Derive arch from the env that owns the live device (its descriptor
  // remembers which mock cluster YAML it loaded); numChips comes from the new
  // mesh shape. We can't trust m_systemDesc here because the test fixture
  // calls reshape BEFORE re-setting m_systemDesc for the new shape.
  ::tt::ARCH metalArch = m_env->get_arch();
  ttcore::Arch arch = (metalArch == ::tt::ARCH::BLACKHOLE)
                          ? ttcore::Arch::Blackhole
                          : ttcore::Arch::WormholeB0;
  uint32_t numChips =
      static_cast<uint32_t>(meshShape.first * meshShape.second);

  m_device.reset();
  m_env.reset();

  ::tt::tt_metal::distributed::MeshShape shape{
      static_cast<unsigned int>(meshShape.first),
      static_cast<unsigned int>(meshShape.second)};

  m_env = std::make_unique<::tt::tt_metal::MetalEnv>(
      makeEnvDescriptor(/*isMock=*/true, arch, numChips));

  ::tt::tt_metal::DispatchCoreType dispatchCoreType =
      m_env->get_num_available_devices() == m_env->get_num_pcie_devices()
          ? ::tt::tt_metal::DispatchCoreType::WORKER
          : ::tt::tt_metal::DispatchCoreType::ETH;

  m_device = m_env->create_mesh_device(
      ::tt::tt_metal::distributed::MeshDeviceConfig{shape},
      ::tt::constants::L1_SMALL_SIZE, traceRegionSize,
      /*num_command_queues=*/1,
      ::tt::tt_metal::DispatchCoreConfig{dispatchCoreType});

  m_device->disable_and_clear_program_cache();

  if (shape.mesh_size() > 1) {
    enableFabricForMockMultiChip(*m_env);
  }
}

llvm::SmallVector<int64_t> SingletonDeviceContext::getComputeGridShape() const {
  assert(m_device != nullptr && "Device is not initialized.");
  auto grid = m_device->compute_with_storage_grid_size();
  // CoreCoord holds {x=cols, y=rows}; return as {rows, cols} to match GridAttr
  return {static_cast<int64_t>(grid.y), static_cast<int64_t>(grid.x)};
}

} // namespace mlir::tt::ttnn::op_model
#endif // TTMLIR_ENABLE_OPMODEL
