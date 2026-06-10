// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifdef TTMLIR_ENABLE_OPMODEL

#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/OpModel/TTNN/MetalHeaders.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
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
  case ttcore::Arch::Quasar:
    return ::tt::ARCH::QUASAR;
  }
  llvm_unreachable("Unknown arch");
}

// Inverse of toMetalArch. Used by reshapeMeshDevice to recover the ttcore arch
// from the live env. report_fatal_error (not llvm_unreachable) so an arch we
// can't map fails loudly in release builds instead of silently picking a wrong
// mock cluster descriptor.
ttcore::Arch fromMetalArch(::tt::ARCH arch) {
  switch (arch) {
  case ::tt::ARCH::WORMHOLE_B0:
    return ttcore::Arch::WormholeB0;
  case ::tt::ARCH::BLACKHOLE:
    return ttcore::Arch::Blackhole;
  case ::tt::ARCH::QUASAR:
    return ttcore::Arch::Quasar;
  default:
    llvm::report_fatal_error("Unsupported tt-metal arch for OpModel device");
  }
}

// Build a MetalEnvDescriptor for the requested device. Mock mode loads the
// per-(arch, numChips) mock cluster descriptor; silicon uses the default.
// Fabric is left DISABLED: multi-chip mock CCL fabric setup is blocked on
// tt-metal-side work (https://github.com/tenstorrent/tt-metal/issues/44748).
::tt::tt_metal::MetalEnvDescriptor
makeEnvDescriptor(bool isMock, ttcore::Arch arch, uint32_t numChips) {
  if (!isMock) {
    return ::tt::tt_metal::MetalEnvDescriptor{};
  }
  auto mockClusterPath =
      ::tt::tt_metal::experimental::get_mock_cluster_desc_name(
          toMetalArch(arch), numChips);
  if (!mockClusterPath) {
    llvm::report_fatal_error(
        "Unsupported (arch, numChips) for mock cluster descriptor");
  }
  return ::tt::tt_metal::MetalEnvDescriptor{*mockClusterPath};
}

// Pick the dispatch core type for a device opened on this env. When every
// available device is PCIe-attached there are no spare Ethernet-connected
// chips to dispatch from, so fall back to WORKER cores; otherwise prefer ETH.
::tt::tt_metal::DispatchCoreType
selectDispatchCoreType(const ::tt::tt_metal::MetalEnv &env) {
  return env.get_num_available_devices() == env.get_num_pcie_devices()
             ? ::tt::tt_metal::DispatchCoreType::WORKER
             : ::tt::tt_metal::DispatchCoreType::ETH;
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

void SingletonDeviceContext::refreshComputeGridShape() {
  assert(m_device != nullptr && "Device must be initialized to query grid");
  const ::tt::tt_metal::CoreCoord grid =
      m_device->compute_with_storage_grid_size();
  // CoreCoord holds (x, y); GridAttr/layout convention is {y, x}.
  llvm::SmallVector<int64_t, 2> newGrid = {static_cast<int64_t>(grid.y),
                                           static_cast<int64_t>(grid.x)};
  if (newGrid != m_computeGridShape) {
    m_computeGridShape = std::move(newGrid);
    ++m_deviceGeneration;
  }
  validateComputeGridAgainstSystemDesc();
}

void SingletonDeviceContext::validateComputeGridAgainstSystemDesc() const {
  if (!m_systemDesc) {
    return;
  }
  llvm::ArrayRef<ttcore::ChipDescAttr> chipDescs = m_systemDesc.getChipDescs();
  if (chipDescs.empty()) {
    return;
  }

  // All chips on a given (multi-chip) system are assumed identical, so the
  // first chip's grid is representative. Both the chip grid and
  // m_computeGridShape use the {y, x} convention.
  llvm::ArrayRef<int64_t> expected = chipDescs.front().getGrid();
  // A non-2D chip grid is a malformed system descriptor (the rest of the stack
  // assumes 2D worker grids); fail fast rather than index out of bounds below
  // in release builds, where the equivalent assert is compiled out.
  if (expected.size() != 2) {
    llvm::report_fatal_error(
        llvm::Twine("OpModel system descriptor has a malformed chip grid: "
                    "expected a 2D grid, got rank ") +
        llvm::Twine(expected.size()) + ".");
  }

  if (expected != llvm::ArrayRef<int64_t>(m_computeGridShape)) {
    llvm::report_fatal_error(
        llvm::Twine(
            "OpModel device worker grid does not match the registered system "
            "descriptor: device compute-with-storage grid {y=") +
        llvm::Twine(m_computeGridShape[0]) +
        ", x=" + llvm::Twine(m_computeGridShape[1]) +
        "}, system desc grid {y=" + llvm::Twine(expected[0]) +
        ", x=" + llvm::Twine(expected[1]) + "}.");
  }
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
  if (!wasExternalDevice) {
    // Flush the program cache while the device's MetalContext is still alive.
    // An env-backed device destroys its MetalContext on close but does not
    // clear the cache first, so cached programs would otherwise be freed by
    // ~MeshDeviceImpl against a dead context and crash in
    // deallocate_circular_buffers.
    instance.m_device->disable_and_clear_program_cache();
    instance.m_device.reset();
    // RAII teardown: destroying the env tears down fabric + mock cluster.
    instance.m_env.reset();
  } else {
    instance.m_device.reset();
  }
  // m_computeGridShape is intentionally retained across close so that a reset
  // to the same grid does not bump m_deviceGeneration (and needlessly drop the
  // still-valid op-model caches). It is only read while a device is active.
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
  instance.refreshComputeGridShape();
}

void SingletonDeviceContext::setSystemDesc(ttcore::SystemDescAttr systemDesc) {
  SingletonDeviceContext &instance = getInstance();
  instance.m_systemDesc = systemDesc;
  // If a device is already open, validate the new descriptor against it now:
  // the grid check otherwise only runs on device open/reshape, so a desc set
  // on an already-active device (e.g. via ScopedSingletonDeviceGuard reusing a
  // device) would never be checked.
  if (instance.m_device != nullptr) {
    instance.validateComputeGridAgainstSystemDesc();
  }
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

  ttcore::Arch arch = ttcore::Arch::WormholeB0;
  uint32_t numChips = 1;
  if (isMock) {
    assert(m_systemDesc && "System desc must be set for mock device mode");
    arch = m_systemDesc.getChipDesc(0).getArch().getValue();
    numChips = m_systemDesc.getChipDescIndices().size();
  }

  // Build into locals; commit to members only on success so a throw leaves
  // m_env/m_device null and the context reusable.
  auto env = std::make_unique<::tt::tt_metal::MetalEnv>(
      makeEnvDescriptor(isMock, arch, numChips));

  ::tt::tt_metal::distributed::MeshShape shape{
      meshShape ? static_cast<unsigned int>(meshShape->first) : 1,
      meshShape ? static_cast<unsigned int>(meshShape->second) : 1};

  // create_mesh_device registers a process-global MetalContext before it
  // validates the mesh size and does not unwind it if the size check fails,
  // wedging the next open. Reject an over-subscribed mesh up front instead.
  if (shape.mesh_size() > env->get_num_available_devices()) {
    throw std::runtime_error("Requested mesh of " +
                             std::to_string(shape.mesh_size()) +
                             " devices exceeds the " +
                             std::to_string(env->get_num_available_devices()) +
                             " available in the system.");
  }

  auto device = env->create_mesh_device(
      ::tt::tt_metal::distributed::MeshDeviceConfig{shape},
      ::tt::constants::L1_SMALL_SIZE, traceRegionSize,
      /*num_command_queues=*/1,
      ::tt::tt_metal::DispatchCoreConfig{selectDispatchCoreType(*env)});

  device->disable_and_clear_program_cache();

  m_env = std::move(env);
  m_device = std::move(device);
  m_isMockDevice = isMock;

  refreshComputeGridShape();
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
  // mesh shape.
  ttcore::Arch arch = fromMetalArch(m_env->get_arch());
  uint32_t numChips = static_cast<uint32_t>(meshShape.first * meshShape.second);

  // Flush the program cache before destroying the env-backed device; see the
  // matching note in closeInstance() for why this must happen while the
  // device's MetalContext is still alive.
  m_device->disable_and_clear_program_cache();
  m_device.reset();
  m_env.reset();

  ::tt::tt_metal::distributed::MeshShape shape{
      static_cast<unsigned int>(meshShape.first),
      static_cast<unsigned int>(meshShape.second)};

  m_env = std::make_unique<::tt::tt_metal::MetalEnv>(
      makeEnvDescriptor(/*isMock=*/true, arch, numChips));

  ::tt::tt_metal::DispatchCoreType dispatchCoreType =
      selectDispatchCoreType(*m_env);

  m_device = m_env->create_mesh_device(
      ::tt::tt_metal::distributed::MeshDeviceConfig{shape},
      ::tt::constants::L1_SMALL_SIZE, traceRegionSize,
      /*num_command_queues=*/1,
      ::tt::tt_metal::DispatchCoreConfig{dispatchCoreType});

  m_device->disable_and_clear_program_cache();

  refreshComputeGridShape();
}

} // namespace mlir::tt::ttnn::op_model
#endif // TTMLIR_ENABLE_OPMODEL
