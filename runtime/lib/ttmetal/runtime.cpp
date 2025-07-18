// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <variant>

#include "tracy/Tracy.hpp"
#include "tt-metalium/fabric.hpp"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/dylib.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttmetal/ttmetal.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTMetal/Target.h"
#include "ttmlir/Version.h"

#include "executor.h"

namespace tt::runtime::ttmetal {

namespace target = ::tt::target;
namespace tt_metal = ::tt::tt_metal;

static const target::metal::TTMetalBinary *getBinary(Flatbuffer binary) {
  bool isTTMetal = target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
      binary.handle.get());
  if (!isTTMetal) {
    LOG_FATAL("Unsupported binary format");
  }
  return target::metal::GetSizePrefixedTTMetalBinary(binary.handle.get());
}

Layout getLayout(Binary executableHandle, std::uint32_t programIndex,
                 std::uint32_t inputIndex) {
  // TODO(#3126): Implement device copy toLayout for metal runtime
  return Layout(nullptr, DeviceRuntime::TTMetal);
}

static Tensor alignUpTensor(Tensor tensor, const TensorDesc &desc) {
  std::int64_t alignment = utils::tileAlignment(desc.dataType);
  if (desc.alignment % alignment == 0) {
    return tensor;
  }

  TensorDesc alignedDesc(desc.shape, desc.stride, desc.itemsize, desc.dataType,
                         alignment);

  size_t alignedSize = alignedDesc.sizeBytes();
  auto alignedData = std::shared_ptr<void>(std::malloc(alignedSize), std::free);
  if (!alignedData) {
    LOG_FATAL("toLayout: Failed to allocate host memory.");
  }
  assert(tensor.data.get() != nullptr);
  std::memcpy(alignedData.get(), tensor.data.get(), desc.sizeBytes());
  // Zero fill the rest
  std::memset(static_cast<char *>(alignedData.get()) + desc.sizeBytes(), 0,
              alignedSize - desc.sizeBytes());
  return Tensor(
      static_pointer_cast<void>(std::make_shared<MetalTensor>(alignedDesc)),
      alignedData, DeviceRuntime::TTMetal);
}

Tensor toLayout(Tensor tensor, Device, Layout layout, std::optional<bool>) {
  return std::visit(
      utils::overloaded{
          [&](const TensorDesc &desc) { return alignUpTensor(tensor, desc); },
          [&](const DeviceBuffer &buffer) {
            // TODO(#3126): Implement device copy toLayout for
            // metal runtime
            LOG_FATAL("getTensorDesc from DeviceBuffer not supported.");
            return tensor;
          },
      },
      tensor.as<MetalTensor>(DeviceRuntime::TTMetal));
}

static Tensor createNullTensor() {
  return Tensor(nullptr, nullptr, DeviceRuntime::TTMetal);
}

static MemoryView
createMemoryView(const tt_metal::detail::MemoryView &memoryView) {
  return MemoryView{
      .numBanks = memoryView.num_banks,
      .totalBytesPerBank = memoryView.total_bytes_per_bank,
      .totalBytesAllocatedPerBank = memoryView.total_bytes_allocated_per_bank,
      .totalBytesFreePerBank = memoryView.total_bytes_free_per_bank,
      .largestContiguousBytesFreePerBank =
          memoryView.largest_contiguous_bytes_free_per_bank,
      .blockTable = memoryView.block_table,
  };
}

Tensor createBorrowedHostTensor(std::shared_ptr<void> data,
                                const TensorDesc &desc) {
  std::shared_ptr<MetalTensor> tensor = std::make_shared<MetalTensor>(desc);
  return Tensor(static_pointer_cast<void>(tensor), data,
                DeviceRuntime::TTMetal);
}

bool isTensorAllocated(Tensor tensor) {
  LOG_FATAL("isTensorAllocated not implemented for metal runtime");
}

target::DataType getTensorDataType(Tensor tensor) {
  const MetalTensor &metalTensor =
      tensor.as<MetalTensor>(DeviceRuntime::TTMetal);
  return std::visit(
      utils::overloaded{
          [&](const TensorDesc &desc) { return desc.dataType; },
          [&](const DeviceBuffer &buffer) {
            LOG_FATAL("Datatype mapping from buffer not supported yet.");
            return target::DataType::Float32;
          },
      },
      metalTensor);
}

bool getTensorRetain(Tensor tensor) {
  LOG_FATAL("getTensorRetain not implemented for metal runtime");
}

void setTensorRetain(Tensor tensor, bool retain) {
  LOG_FATAL("setTensorRetain not implemented for metal runtime");
}

Arch getArch() {
  return ::tt::runtime::common::toRuntimeArch(::tt::tt_metal::hal::get_arch());
}

void enablePersistentKernelCache() {
  ::tt::tt_metal::detail::EnablePersistentKernelCache();
}

void disablePersistentKernelCache() {
  ::tt::tt_metal::detail::DisablePersistentKernelCache();
}

size_t getNumAvailableDevices() { return tt_metal::GetNumAvailableDevices(); }

Device openMeshDevice(const std::vector<uint32_t> &meshShape,
                      const MeshDeviceOptions &options) {
  LOG_ASSERT(meshShape.size() == 2, "Mesh shape must be 2D for now");
  tt_metal::distributed::MeshShape shape(meshShape);

  LOG_ASSERT(options.meshOffset.size() == 2, "Mesh offset must be 2D for now");
  tt_metal::distributed::MeshCoordinate offset(options.meshOffset);

  size_t l1SmallSize = options.l1SmallSize.value_or(DEFAULT_L1_SMALL_SIZE);
  size_t traceRegionSize =
      options.traceRegionSize.value_or(DEFAULT_TRACE_REGION_SIZE);
  tt_metal::DispatchCoreType dispatchCoreType =
      common::getDispatchCoreType(options.dispatchCoreType);

  tt_metal::distributed::MeshDeviceConfig meshConfig(shape, offset,
                                                     options.deviceIds);
  bool isFabricEnabled = false;
  if (options.fabricConfig.has_value()) {
    tt::tt_fabric::FabricConfig fabricConfig =
        common::getFabricConfig(options.fabricConfig.value());
    tt::tt_fabric::SetFabricConfig(fabricConfig);
    isFabricEnabled = (fabricConfig != tt::tt_fabric::FabricConfig::DISABLED);
  }

  std::shared_ptr<tt_metal::distributed::MeshDevice> meshDevice =
      tt_metal::distributed::MeshDevice::create(
          meshConfig, l1SmallSize, traceRegionSize, options.numHWCQs,
          dispatchCoreType);

  LOG_DEBUG("Device grid size = { ",
            meshDevice->compute_with_storage_grid_size().x, ", ",
            meshDevice->compute_with_storage_grid_size().y, " }");

  return Device(std::static_pointer_cast<void>(meshDevice),
                /*traceCache=*/nullptr, DeviceRuntime::TTMetal,
                isFabricEnabled);
}

void closeMeshDevice(Device parentMesh) {
  tt_metal::distributed::MeshDevice &metalMeshDevice =
      parentMesh.as<tt_metal::distributed::MeshDevice>(DeviceRuntime::TTMetal);

  LOG_ASSERT(metalMeshDevice.is_parent_mesh(),
             "Mesh device must be a parent mesh");

  if (uint32_t numSubMeshes = metalMeshDevice.get_submeshes().size()) {
    LOG_WARNING("Calling close on parent mesh device ", metalMeshDevice,
                " that has ", numSubMeshes, " unreleased submeshes.");
  }

#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1
  ::tt::tt_metal::DumpMeshDeviceProfileResults(metalMeshDevice);
#endif

  metalMeshDevice.close();
  tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
}

Device createSubMeshDevice(
    Device parentMesh, const std::vector<uint32_t> &meshShape,
    const std::optional<const std::vector<uint32_t>> &meshOffset) {
  tt_metal::distributed::MeshDevice &parentMeshDevice =
      parentMesh.as<tt_metal::distributed::MeshDevice>(DeviceRuntime::TTMetal);
  LOG_ASSERT(parentMeshDevice.is_parent_mesh(),
             "Mesh device must be a parent mesh");
  LOG_ASSERT(meshShape.size() == 2, "Mesh shape must be 2D for now");
  tt_metal::distributed::MeshShape shape{meshShape[0], meshShape[1]};

  std::optional<tt_metal::distributed::MeshCoordinate> offset = std::nullopt;
  if (meshOffset.has_value()) {
    LOG_ASSERT(meshOffset.value().size() == 2,
               "Mesh offset must be 2D for now");
    offset = tt_metal::distributed::MeshCoordinate{meshOffset.value()[0],
                                                   meshOffset.value()[1]};
  }

  std::shared_ptr<tt_metal::distributed::MeshDevice> subMeshDevice =
      parentMeshDevice.create_submesh(shape, offset);

  return Device(std::static_pointer_cast<void>(subMeshDevice),
                /*traceCache=*/nullptr, DeviceRuntime::TTMetal);
}

void releaseSubMeshDevice(Device subMesh) {
  tt_metal::distributed::MeshDevice &metalMeshDevice =
      subMesh.as<tt_metal::distributed::MeshDevice>(DeviceRuntime::TTMetal);

  LOG_ASSERT(!metalMeshDevice.is_parent_mesh(),
             "Mesh device must be a submesh");

  metalMeshDevice.close();
  subMesh.handle.reset();
}

void reshapeMeshDevice(Device meshDevice,
                       const std::vector<uint32_t> &meshShape) {
  ::tt::tt_metal::distributed::MeshDevice &metalMeshDevice =
      meshDevice.as<::tt::tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);

  metalMeshDevice.reshape(
      ::tt::tt_metal::distributed::MeshShape(meshShape[0], meshShape[1]));
}

std::vector<uint32_t> getMeshShape(Device meshDevice) {
  ::tt::tt_metal::distributed::MeshDevice &metalMeshDevice =
      meshDevice.as<::tt::tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);
  std::vector<uint32_t> shape(metalMeshDevice.shape().view().begin(),
                              metalMeshDevice.shape().view().end());
  return shape;
}

std::vector<int> getDeviceIds(Device meshDevice) {
  ::tt::tt_metal::distributed::MeshDevice &metalMeshDevice =
      meshDevice.as<::tt::tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);
  return metalMeshDevice.get_device_ids();
}

size_t getNumHwCqs(Device meshDevice) {
  ::tt::tt_metal::distributed::MeshDevice &metalMeshDevice =
      meshDevice.as<::tt::tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);
  return static_cast<size_t>(metalMeshDevice.num_hw_cqs());
}

bool isProgramCacheEnabled(Device meshDevice) {
  ::tt::tt_metal::distributed::MeshDevice &metalMeshDevice =
      meshDevice.as<::tt::tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);
  return metalMeshDevice.get_program_cache().is_enabled();
}

size_t getL1SmallSize(Device meshDevice) {
  ::tt::tt_metal::distributed::MeshDevice &metalMeshDevice =
      meshDevice.as<::tt::tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);
  return metalMeshDevice.allocator()->get_config().l1_small_size;
}

size_t getTraceRegionSize(Device meshDevice) {
  ::tt::tt_metal::distributed::MeshDevice &metalMeshDevice =
      meshDevice.as<::tt::tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);
  return metalMeshDevice.allocator()->get_config().trace_region_size;
}

size_t getNumDramChannels(Device meshDevice) {
  ::tt::tt_metal::distributed::MeshDevice &metalMeshDevice =
      meshDevice.as<::tt::tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);
  return metalMeshDevice.num_dram_channels();
}

size_t getDramSizePerChannel(Device meshDevice) {
  ::tt::tt_metal::distributed::MeshDevice &metalMeshDevice =
      meshDevice.as<::tt::tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);
  return metalMeshDevice.dram_size_per_channel();
}

size_t getL1SizePerCore(Device meshDevice) {
  ::tt::tt_metal::distributed::MeshDevice &metalMeshDevice =
      meshDevice.as<::tt::tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);
  return metalMeshDevice.l1_size_per_core();
}

void deallocateBuffers(Device deviceHandle) {
  tt_metal::distributed::MeshDevice &meshDevice =
      deviceHandle.as<tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);

  for (tt_metal::IDevice *device : meshDevice.get_devices()) {
    device->allocator()->deallocate_buffers();
  }
}

void dumpMemoryReport(Device deviceHandle) {
  tt_metal::distributed::MeshDevice &meshDevice =
      deviceHandle.as<tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);

  for (tt_metal::IDevice *device : meshDevice.get_devices()) {
    tt_metal::detail::DumpDeviceMemoryState(device);
  }
}

void dumpDeviceProfileResults(Device deviceHandle) {
  tt_metal::distributed::MeshDevice &metalMeshDevice =
      deviceHandle.as<tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);

  LOG_ASSERT(metalMeshDevice.is_parent_mesh(),
             "Mesh device must be a parent mesh");

#if defined(TT_RUNTIME_ENABLE_PERF_TRACE)
  ::tt::tt_metal::DumpMeshDeviceProfileResults(metalMeshDevice);
#endif
}

std::unordered_map<MemoryBufferType, MemoryView>
getMemoryView(Device deviceHandle) {
  std::unordered_map<MemoryBufferType, MemoryView> memoryMap;
  tt_metal::distributed::MeshDevice &meshDevice =
      deviceHandle.as<tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);

  auto dramMemoryView =
      tt_metal::detail::GetMemoryView(&meshDevice, tt_metal::BufferType::DRAM);
  auto l1MemoryView =
      tt_metal::detail::GetMemoryView(&meshDevice, tt_metal::BufferType::L1);
  auto l1SmallMemoryView = tt_metal::detail::GetMemoryView(
      &meshDevice, tt_metal::BufferType::L1_SMALL);
  auto traceMemoryView =
      tt_metal::detail::GetMemoryView(&meshDevice, tt_metal::BufferType::TRACE);

  memoryMap[MemoryBufferType::DRAM] = createMemoryView(dramMemoryView);
  memoryMap[MemoryBufferType::L1] = createMemoryView(l1MemoryView);
  memoryMap[MemoryBufferType::L1_SMALL] = createMemoryView(l1SmallMemoryView);
  memoryMap[MemoryBufferType::TRACE] = createMemoryView(traceMemoryView);

  return memoryMap;
}

void setFabricConfig(FabricConfig config) {
  ::tt::tt_fabric::SetFabricConfig(common::getFabricConfig(config));
}

void wait(Event event) {
  std::shared_ptr<tt_metal::Event> eventPtr =
      event.asSharedPtr<tt_metal::Event>(DeviceRuntime::TTMetal);
  if (eventPtr) {
    tt_metal::EventSynchronize(eventPtr);
  }
}

void wait(Tensor tensor, std::optional<uint8_t> cqId) {
  ::tt::runtime::ttmetal::wait(tensor.event);
}

void wait(const std::vector<Tensor> &tensors, std::optional<uint8_t> cqId) {
  for (Tensor tensor : tensors) {
    ::tt::runtime::ttmetal::wait(tensor);
  }
}

std::vector<Tensor> toHost(Tensor tensor, bool untilize, bool blocking) {
  ::tt::runtime::ttmetal::wait(tensor);
  std::visit(utils::overloaded{
                 [&](const TensorDesc &) { /* no-op */ },
                 [&](const DeviceBuffer &) {
                   LOG_FATAL("toHost not yet implemented for device buffer");
                 },
             },
             tensor.as<MetalTensor>(DeviceRuntime::TTMetal));
  return {tensor};
}

void memcpy(void *dst, Tensor src,
            std::optional<tt::target::DataType> dstDataType) {
  if (dstDataType.has_value()) {
    LOG_ASSERT(
        ::tt::runtime::utils::isSupportedDataType(dstDataType.value()),
        "dstDataType must be a supported data type if using TTMetal runtime");
  }
  const auto &metalSrc = src.as<MetalTensor>(DeviceRuntime::TTMetal);
  LOG_ASSERT(std::holds_alternative<TensorDesc>(metalSrc),
             "Only TensorDesc supported for now");
  const auto &hostSrc = std::get<TensorDesc>(metalSrc);
  std::memcpy(dst, src.data.get(), hostSrc.sizeBytes());
}

void memcpy(Tensor dst, Tensor src) {
  auto &metalDst = dst.as<MetalTensor>(DeviceRuntime::TTMetal);
  const auto &metalSrc = src.as<MetalTensor>(DeviceRuntime::TTMetal);
  LOG_ASSERT(std::holds_alternative<TensorDesc>(metalDst),
             "Only TensorDesc supported for now");
  LOG_ASSERT(std::holds_alternative<TensorDesc>(metalSrc),
             "Only TensorDesc supported for now");
  auto &hostDst = std::get<TensorDesc>(metalDst);
  const auto &hostSrc = std::get<TensorDesc>(metalSrc);
  std::int64_t maxAlignment = std::max(hostDst.alignment, hostSrc.alignment);
  LOG_ASSERT(utils::alignUp(hostDst.sizeBytes(), maxAlignment) ==
                 utils::alignUp(hostSrc.sizeBytes(), maxAlignment),
             "Tensor size mismatch");
  LOG_ASSERT(hostDst.dataType == hostSrc.dataType, "Tensor data type mismatch");
  std::int64_t copySize = std::min(hostDst.sizeBytes(), hostSrc.sizeBytes());
  std::memcpy(dst.data.get(), src.data.get(), copySize);
}

void deallocateTensor(Tensor &tensor, bool) {
  std::visit(
      utils::overloaded{
          [&](const TensorDesc &) { /* no-op */ },
          [&](const DeviceBuffer &) {
            LOG_FATAL("deallocateTensor not yet implemented for device buffer");
          },
      },
      tensor.as<MetalTensor>(DeviceRuntime::TTMetal));
}

std::vector<Tensor> submit(Device deviceHandle, Binary executableHandle,
                           std::uint32_t programIndex,
                           std::vector<Tensor> &inputs) {
  const target::metal::TTMetalBinary &fbb = *getBinary(executableHandle);
  const target::metal::Program *program = fbb.programs()->Get(programIndex);
  tt_metal::distributed::MeshDevice &meshDevice =
      deviceHandle.as<tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);
  std::vector<tt_metal::IDevice *> allDevices = meshDevice.get_devices();
  LOG_ASSERT(allDevices.size() > 0, "Unexpected empty device mesh");
  std::vector<tt_metal::IDevice *> deviceList = {allDevices[0]};
  LOG_ASSERT(deviceList.size() == 1, "Only one device is supported for now");
  LOG_ASSERT(program->device_programs()->size() == deviceList.size(),
             "Device programs size mismatch");

  std::vector<Tensor> outputs;
  for (std::size_t i = 0; i < program->device_programs()->size(); ++i) {
    tt_metal::IDevice *device = deviceList[i];

    ZoneScoped;
    std::string zoneName = "submit_" + std::string(program->name()->c_str()) +
                           "_device_" + std::to_string(device->id());
    ZoneName(zoneName.c_str(), zoneName.size());

    LOG_ASSERT(outputs.empty(), "Multi-device outputs not supported");
    outputs = executeDeviceProgram(device, program->device_programs()->Get(i),
                                   inputs, common::DylibManager(fbb.dylibs()));
    LOG_ASSERT(outputs.size() == program->outputs()->size(),
               "Outputs size mismatch");
  }

  return outputs;
}

std::string getOpDebugString(OpContext opContextHandle) {
  // Not implemented
  LOG_WARNING("obtaining op debug string for metal runtime not implemented");
  return "";
}

std::string getOpLocInfo(OpContext opContextHandle) {
  // Not implemented
  LOG_WARNING("obtaining op location info for metal runtime not implemented");
  return "";
}

Tensor getOpOutputTensor(OpContext opContextHandle,
                         CallbackContext programContextHandle) {
  // Not implemented
  LOG_WARNING("obtaining op output tensor for metal runtime not implemented");
  return createNullTensor();
}

std::optional<tt::runtime::TensorRef>
getOpOutputRef(OpContext opContextHandle,
               CallbackContext programContextHandle) {
  // Not implemented
  LOG_FATAL("Obtaining op output ref for metal runtime is not implemented");
  return std::nullopt;
}

std::vector<tt::runtime::TensorRef>
getOpInputRefs(OpContext opContextHandle,
               CallbackContext programContextHandle) {
  // Not implemented
  LOG_FATAL(
      "Obtaining op input references for metal runtime is not implemented");
  return {};
}

std::optional<Tensor>
retrieveTensorFromPool(CallbackContext programContextHandle,
                       TensorRef tensorRef, bool untilize) {
  // Not implemented
  LOG_FATAL(
      "Obtaining tensor from device for metal runtime is not implemented");
  return std::nullopt;
}

void updateTensorInPool(CallbackContext programContextHandle,
                        TensorRef tensorRef, Tensor srcTensor) {
  // Not implemented
  LOG_FATAL("Updating tensor from device for metal runtime is not implemented");
}

std::vector<std::byte> getTensorDataBuffer(Tensor tensor) {
  return std::visit(
      utils::overloaded{
          [&](const TensorDesc &desc) {
            const std::byte *data =
                static_cast<const std::byte *>(tensor.data.get());
            assert(data);
            return std::vector<std::byte>(data, data + desc.sizeBytes());
          },
          [&](const DeviceBuffer &buffer) {
            LOG_FATAL("getTensorDataBuffer from DeviceBuffer not supported.");
            return std::vector<std::byte>{};
          },
      },
      tensor.as<MetalTensor>(DeviceRuntime::TTMetal));
}

std::vector<std::uint32_t> getTensorShape(Tensor tensor) {
  return ttmetal::getTensorDesc(tensor).shape;
}

std::vector<std::uint32_t> getTensorStride(Tensor tensor) {
  return ttmetal::getTensorDesc(tensor).stride;
}

std::uint32_t getTensorElementSize(Tensor tensor) {
  return ttmetal::getTensorDesc(tensor).itemsize;
}

std::uint32_t getTensorVolume(Tensor tensor) {
  return ttmetal::getTensorDesc(tensor).volume();
}

TensorDesc getTensorDesc(Tensor tensor) {
  return std::visit(utils::overloaded{
                        [&](const TensorDesc &desc) { return desc; },
                        [&](const DeviceBuffer &buffer) {
                          LOG_FATAL(
                              "getTensorDesc from DeviceBuffer not supported.");
                          return TensorDesc{};
                        },
                    },
                    tensor.as<MetalTensor>(DeviceRuntime::TTMetal));
}

} // namespace tt::runtime::ttmetal
