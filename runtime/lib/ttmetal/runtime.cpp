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
#include "tt/runtime/detail/common/runtime_context.h"
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
          [&](const HostBuffer &buffer) {
            LOG_FATAL("toLayout not yet implemented for HostBuffer");
            return tensor;
          },
          [&](const DistributedHostBuffer &buffer) {
            LOG_FATAL("toLayout not yet implemented for DistributedHostBuffer");
            return tensor;
          },
          [&](const MeshBuffer &buffer) {
            // TODO(#3126): Implement device copy toLayout for
            // metal runtime
            LOG_FATAL("getTensorDesc from MeshBuffer not supported.");
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

std::shared_ptr<tt_metal::HostBuffer>
createMetalHostBuffer(const void *data, const std::vector<std::uint32_t> &shape,
                      const ::tt::target::DataType dataType) {

  const std::int64_t volume =
      std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1),
                      std::multiplies<int64_t>());

  auto createTypedHostBuffer = [&]<typename T>() {
    auto owned = std::make_shared<std::vector<T>>(volume);
    std::memcpy(owned->data(), data, volume * sizeof(T));
    return std::make_shared<tt_metal::HostBuffer>(owned);
  };

  std::shared_ptr<tt_metal::HostBuffer> hostBuffer;
  switch (dataType) {
  case ::tt::target::DataType::Float32:
    hostBuffer = createTypedHostBuffer.template operator()<float>();
    break;
  case ::tt::target::DataType::BFloat16:
    hostBuffer = createTypedHostBuffer.template operator()<bfloat16>();
    break;
  case ::tt::target::DataType::UInt32:
    hostBuffer = createTypedHostBuffer.template operator()<uint32_t>();
    break;
  case ::tt::target::DataType::UInt16:
    hostBuffer = createTypedHostBuffer.template operator()<uint16_t>();
    break;
  case ::tt::target::DataType::UInt8:
    hostBuffer = createTypedHostBuffer.template operator()<uint8_t>();
    break;
  case ::tt::target::DataType::Int32:
    hostBuffer = createTypedHostBuffer.template operator()<int32_t>();
    break;
  default:
    LOG_FATAL("Unsupported data type");
  }

  return hostBuffer;
}

Tensor createOwnedHostTensor(const void *data,
                             const std::vector<std::uint32_t> &shape,
                             const std::vector<std::uint32_t> &stride,
                             std::uint32_t itemsize,
                             ::tt::target::DataType dataType) {
  LOG_ASSERT(utils::isSupportedDataType(dataType),
             "Creating owned tensor with unsupported data type: " +
                 std::string(target::EnumNameDataType(dataType)) +
                 "is not implemented for the TTMetal runtime");

  auto hostBuffer = createMetalHostBuffer(data, shape, dataType);
  return Tensor(std::static_pointer_cast<void>(hostBuffer), nullptr,
                DeviceRuntime::TTMetal);
}

Tensor createMultiDeviceHostTensor(
    const std::vector<Tensor> &tensorShards,
    const std::unordered_map<std::string, std::string> &strategy,
    const std::vector<uint32_t> &meshShape) {
  std::vector<tt_metal::HostBuffer> hostBuffers;
  hostBuffers.reserve(tensorShards.size());
  std::transform(
      tensorShards.begin(), tensorShards.end(), std::back_inserter(hostBuffers),
      [&](Tensor tensorShard) -> tt_metal::HostBuffer {
        return tensorShard.as<tt_metal::HostBuffer>(DeviceRuntime::TTMetal);
      });
  auto metalMeshShape = tt_metal::distributed::MeshShape(meshShape);

  // For now, we only focus on a single host.
  auto distributedHostBuffer =
      std::make_shared<tt_metal::DistributedHostBuffer>(
          tt_metal::DistributedHostBuffer::create(metalMeshShape));

  auto meshCoordRange =
      tt_metal::distributed::MeshCoordinateRange(metalMeshShape);
  auto meshCoord = meshCoordRange.begin();
  for (auto hostBuffer : hostBuffers) {
    // HostBuffer is to be placed in row-major order.
    distributedHostBuffer->emplace_shard(
        *meshCoord, [&hostBuffer]() { return hostBuffer; });
    meshCoord++;
  }

  return Tensor(std::static_pointer_cast<void>(distributedHostBuffer), nullptr,
                DeviceRuntime::TTMetal);
}

Tensor createMultiDeviceHostTensor(
    const std::vector<const void *> &data,
    const std::vector<std::uint32_t> &shape,
    const std::vector<std::uint32_t> &stride, std::uint32_t itemsize,
    ::tt::target::DataType dataType,
    const std::unordered_map<std::string, std::string> &strategy,
    const std::vector<uint32_t> &meshShape) {
  std::vector<Tensor> tensorShards;
  tensorShards.reserve(data.size());
  std::transform(data.begin(), data.end(), std::back_inserter(tensorShards),
                 [&](const void *dataShard) -> Tensor {
                   return createOwnedHostTensor(dataShard, shape, stride,
                                                itemsize, dataType);
                 });
  return ttmetal::createMultiDeviceHostTensor(tensorShards, strategy,
                                              meshShape);
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
          [&](const HostBuffer &buffer) {
            LOG_FATAL("Datatype mapping from HostBuffer not supported yet.");
            return target::DataType::Float32;
          },
          [&](const DistributedHostBuffer &buffer) {
            LOG_FATAL("Datatype mapping from DistributedHostBuffer not "
                      "supported yet.");
            return target::DataType::Float32;
          },
          [&](const MeshBuffer &buffer) {
            LOG_FATAL("Datatype mapping from MeshBuffer not supported yet.");
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

Device openMeshDevice(const MeshDeviceOptions &options) {
  std::optional<tt_metal::distributed::MeshShape> meshShape = std::nullopt;
  if (options.meshShape.has_value()) {
    LOG_ASSERT(options.meshShape.value().size() == 2,
               "Mesh shape must be 2D for now");
    meshShape = tt_metal::distributed::MeshShape(options.meshShape.value());
  }

  LOG_ASSERT(options.meshOffset.size() == 2, "Mesh offset must be 2D for now");
  tt_metal::distributed::MeshCoordinate offset(options.meshOffset);

  size_t l1SmallSize = options.l1SmallSize.value_or(DEFAULT_L1_SMALL_SIZE);
  size_t traceRegionSize =
      options.traceRegionSize.value_or(DEFAULT_TRACE_REGION_SIZE);
  tt_metal::DispatchCoreType dispatchCoreType =
      common::getDispatchCoreType(options.dispatchCoreType);

  tt_metal::distributed::MeshDeviceConfig meshConfig(meshShape, offset,
                                                     options.deviceIds);

  std::shared_ptr<tt_metal::distributed::MeshDevice> meshDevice =
      tt_metal::distributed::MeshDevice::create(
          meshConfig, l1SmallSize, traceRegionSize, options.numHWCQs,
          dispatchCoreType);

  LOG_DEBUG("Device grid size = { ",
            meshDevice->compute_with_storage_grid_size().x, ", ",
            meshDevice->compute_with_storage_grid_size().y, " }");

  return Device(std::static_pointer_cast<void>(meshDevice),
                /*traceCache=*/nullptr, DeviceRuntime::TTMetal);
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
  ::tt::tt_metal::ReadMeshDeviceProfilerResults(metalMeshDevice);
#endif

  metalMeshDevice.close();
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

void readDeviceProfilerResults(Device deviceHandle) {
  tt_metal::distributed::MeshDevice &metalMeshDevice =
      deviceHandle.as<tt_metal::distributed::MeshDevice>(
          DeviceRuntime::TTMetal);

  LOG_ASSERT(metalMeshDevice.is_parent_mesh(),
             "Mesh device must be a parent mesh");

#if defined(TT_RUNTIME_ENABLE_PERF_TRACE)
  ::tt::tt_metal::ReadMeshDeviceProfilerResults(metalMeshDevice);
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
  ::tt::tt_fabric::SetFabricConfig(common::toTTFabricConfig(config));
  RuntimeContext::instance().setCurrentFabricConfig(config);
}

void wait(Event event) {
  std::shared_ptr<tt_metal::distributed::MeshEvent> eventPtr =
      event.asSharedPtr<tt_metal::distributed::MeshEvent>(
          DeviceRuntime::TTMetal);
  if (eventPtr) {
    tt_metal::distributed::EventSynchronize(*eventPtr);
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
                 [&](const HostBuffer &) { /* no-op */ },
                 [&](const DistributedHostBuffer &) {
                   LOG_FATAL(
                       "toHost not yet implemented for DistributedHostBuffer");
                 },
                 [&](const MeshBuffer &) {
                   LOG_FATAL("toHost not yet implemented for MeshBuffer");
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
  std::visit(utils::overloaded{
                 [&](const TensorDesc &tensorDesc) {
                   std::cout << "0 output data ptr c++ 0x" << dst << std::endl;
                   std::memcpy(dst, src.data.get(), tensorDesc.sizeBytes());
                 },
                 [&](const HostBuffer &hostBuffer) {
                   auto span = hostBuffer->view_bytes();
                   std::cout << "1 output data ptr c++ 0x" << dst << std::endl;
                   std::memcpy(dst, span.data(), span.size_bytes());
                 },
                 [&](const DistributedHostBuffer &) {
                   LOG_FATAL(
                       "memcpy not yet implemented for DistributedHostBuffer");
                 },
                 [&](const MeshBuffer &) {
                   LOG_FATAL("memcpy not yet implemented for MeshBuffer");
                 },
             },
             src.as<MetalTensor>(DeviceRuntime::TTMetal));
}

void memcpy(Tensor dst, Tensor src) {
  auto &metalDst = dst.as<MetalTensor>(DeviceRuntime::TTMetal);
  auto &hostDst = std::get<TensorDesc>(metalDst);
  std::visit(
      utils::overloaded{
          [&](const TensorDesc &tensorDesc) {
            std::int64_t maxAlignment =
                std::max(hostDst.alignment, tensorDesc.alignment);
            LOG_ASSERT(utils::alignUp(hostDst.sizeBytes(), maxAlignment) ==
                           utils::alignUp(tensorDesc.sizeBytes(), maxAlignment),
                       "Tensor size mismatch");
            LOG_ASSERT(hostDst.dataType == tensorDesc.dataType,
                       "Tensor data type mismatch");
            std::int64_t copySize =
                std::min(hostDst.sizeBytes(), tensorDesc.sizeBytes());
            std::cout << "2 output data ptr c++ " << dst.data.get() << " "
                      << src.data.get() << " " << *((float *)src.data.get())
                      << " " << copySize << std::endl;
            std::memcpy(dst.data.get(), src.data.get(), copySize);
          },
          [&](const HostBuffer &hostBuffer) {
            auto span = hostBuffer->view_bytes();
            std::int64_t copyByteSize =
                std::min(hostDst.sizeBytes(),
                         static_cast<std::int64_t>(span.size_bytes()));
            std::cout << "3 output data ptr c++ 0x" << dst.data.get()
                      << std::endl;
            std::memcpy(dst.data.get(), span.data(), copyByteSize);
          },
          [&](const DistributedHostBuffer &) {
            LOG_FATAL("memcpy not yet implemented for DistributedHostBuffer");
          },
          [&](const MeshBuffer &) {
            LOG_FATAL("memcpy not yet implemented for MeshBuffer");
          },
      },
      src.as<MetalTensor>(DeviceRuntime::TTMetal));
}

void deallocateTensor(Tensor &tensor, bool) {
  std::visit(utils::overloaded{
                 [&](const TensorDesc &) { /* no-op */ },
                 [&](const HostBuffer &) { /* no-op */ },
                 [&](const DistributedHostBuffer &) { /* no-op */ },
                 [&](const MeshBuffer &) {
                   LOG_FATAL(
                       "deallocateTensor not yet implemented for MeshBuffer");
                 },
             },
             tensor.as<MetalTensor>(DeviceRuntime::TTMetal));
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
          [&](const HostBuffer &buffer) {
            auto span = buffer->view_bytes();
            return std::vector<std::byte>(span.begin(), span.end());
          },
          [&](const DistributedHostBuffer &buffer) {
            LOG_FATAL("getTensorDataBuffer from DistributedHostBuffer not "
                      "supported.");
            return std::vector<std::byte>{};
          },
          [&](const MeshBuffer &buffer) {
            LOG_FATAL("getTensorDataBuffer from MeshBuffer not supported.");
            return std::vector<std::byte>{};
          },
      },
      tensor.as<MetalTensor>(DeviceRuntime::TTMetal));
}

std::vector<std::uint32_t> getTensorShape(Tensor tensor) {
  return std::visit(
      utils::overloaded{
          [&](const TensorDesc &desc) {
            return ttmetal::getTensorDesc(tensor).shape;
          },
          [&](const HostBuffer &buffer) {
            LOG_FATAL("getTensorShape from HostBuffer not supported.");
            return std::vector<std::uint32_t>{};
          },
          [&](const DistributedHostBuffer &buffer) {
            LOG_FATAL(
                "getTensorShape from DistributedHostBuffer not supported.");
            return std::vector<std::uint32_t>{};
          },
          [&](const MeshBuffer &buffer) {
            LOG_FATAL("getTensorShape from MeshBuffer not supported.");
            return std::vector<std::uint32_t>{};
          },
      },
      tensor.as<MetalTensor>(DeviceRuntime::TTMetal));
}

std::vector<std::uint32_t> getTensorStride(Tensor tensor) {
  return std::visit(
      utils::overloaded{
          [&](const TensorDesc &desc) {
            return ttmetal::getTensorDesc(tensor).stride;
          },
          [&](const HostBuffer &buffer) {
            LOG_FATAL("getTensorStride from HostBuffer not supported.");
            return std::vector<std::uint32_t>{};
          },
          [&](const DistributedHostBuffer &buffer) {
            LOG_FATAL(
                "getTensorStride from DistributedHostBuffer not supported.");
            return std::vector<std::uint32_t>{};
          },
          [&](const MeshBuffer &buffer) {
            LOG_FATAL("getTensorStride from MeshBuffer not supported.");
            return std::vector<std::uint32_t>{};
          },
      },
      tensor.as<MetalTensor>(DeviceRuntime::TTMetal));
}

std::uint32_t getTensorElementSize(Tensor tensor) {
  return std::visit(
      utils::overloaded{
          [&](const TensorDesc &desc) {
            return static_cast<std::uint32_t>(
                ttmetal::getTensorDesc(tensor).itemsize);
          },
          [&](const HostBuffer &buffer) {
            auto span = buffer->view_bytes();
            using ElementType = decltype(span)::element_type;
            return static_cast<std::uint32_t>(sizeof(ElementType));
          },
          [&](const DistributedHostBuffer &buffer) {
            LOG_FATAL("getTensorElementSize from DistributedHostBuffer not "
                      "supported.");
            return static_cast<std::uint32_t>(0);
          },
          [&](const MeshBuffer &buffer) {
            LOG_FATAL("getTensorElementSize from MeshBuffer not supported.");
            return static_cast<std::uint32_t>(0);
          },
      },
      tensor.as<MetalTensor>(DeviceRuntime::TTMetal));
}

std::uint32_t getTensorVolume(Tensor tensor) {
  return std::visit(
      utils::overloaded{
          [&](const TensorDesc &desc) {
            return static_cast<std::uint32_t>(
                ttmetal::getTensorDesc(tensor).volume());
          },
          [&](const HostBuffer &buffer) {
            return static_cast<std::uint32_t>(
                buffer->view_bytes().size_bytes());
          },
          [&](const DistributedHostBuffer &buffer) {
            LOG_FATAL(
                "getTensorVolume from DistributedHostBuffer not supported.");
            return static_cast<std::uint32_t>(0);
          },
          [&](const MeshBuffer &buffer) {
            LOG_FATAL("getTensorVolume from MeshBuffer not supported.");
            return static_cast<std::uint32_t>(0);
          },
      },
      tensor.as<MetalTensor>(DeviceRuntime::TTMetal));
}

TensorDesc getTensorDesc(Tensor tensor) {
  return std::visit(
      utils::overloaded{
          [&](const TensorDesc &desc) { return desc; },
          [&](const HostBuffer &buffer) {
            LOG_FATAL("getTensorDesc from HostBuffer not supported.");
            return TensorDesc{};
          },
          [&](const DistributedHostBuffer &buffer) {
            LOG_FATAL(
                "getTensorDesc from DistributedHostBuffer not supported.");
            return TensorDesc{};
          },
          [&](const MeshBuffer &buffer) {
            LOG_FATAL("getTensorDesc from MeshBuffer not supported.");
            return TensorDesc{};
          },
      },
      tensor.as<MetalTensor>(DeviceRuntime::TTMetal));
}

HostBuffer getHostBuffer(Tensor tensor) {
  return std::visit(
      utils::overloaded{
          [&](const TensorDesc &desc) {
            LOG_FATAL("getHostBuffer from TensorDesc not supported.");
            return HostBuffer{};
          },
          [&](const HostBuffer &buffer) { return buffer; },
          [&](const DistributedHostBuffer &buffer) {
            LOG_FATAL("getHostBuffer from DistributedHostBuffer "
                      "not supported.");
            return HostBuffer{};
          },
          [&](const MeshBuffer &buffer) {
            LOG_FATAL("getHostBuffer from MeshBuffer not supported.");
            return HostBuffer{};
          },
      },
      tensor.as<MetalTensor>(DeviceRuntime::TTMetal));
}

DistributedHostBuffer getDistributedHostBuffer(Tensor tensor) {
  return std::visit(
      utils::overloaded{
          [&](const TensorDesc &desc) {
            LOG_FATAL("getDistributedHostBufferFromMetalTensor from TensorDesc "
                      "not supported.");
            return DistributedHostBuffer{};
          },
          [&](const HostBuffer &buffer) {
            LOG_FATAL("getDistributedHostBufferFromMetalTensor from HostBuffer "
                      "not supported.");
            return DistributedHostBuffer{};
          },
          [&](const DistributedHostBuffer &buffer) { return buffer; },
          [&](const MeshBuffer &buffer) {
            LOG_FATAL("getDistributedHostBufferFromMetalTensor from MeshBuffer "
                      "not supported.");
            return DistributedHostBuffer{};
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
  LOG_ASSERT(meshDevice.num_rows() == 1 && meshDevice.num_cols() == 1,
             "Currently we only support 1x1 mesh.");

  std::vector<Tensor> outputs;
  for (std::size_t i = 0; i < program->device_programs()->size(); ++i) {
    ZoneScoped;
    std::string zoneName = "submit_" + std::string(program->name()->c_str()) +
                           "_device_" + std::to_string(meshDevice.id());
    ZoneName(zoneName.c_str(), zoneName.size());

    outputs = executeMeshDeviceProgram(
        &meshDevice, program->device_programs()->Get(i), inputs,
        common::DylibManager(fbb.dylibs()));

    LOG_ASSERT(outputs.size() == program->outputs()->size(),
               "Outputs size mismatch");
  }

  return outputs;
}

} // namespace tt::runtime::ttmetal
