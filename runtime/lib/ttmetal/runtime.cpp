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

Tensor toLayout(Tensor tensor, Device, Layout layout, std::optional<bool>) {
  return std::visit(
      utils::overloaded{
          // TODO(#3126): Implement device copy toLayout for metal runtime.
          [&](const TensorDesc &desc) { return tensor; },
          [&](const MeshBuffer &buffer) {
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
  std::shared_ptr<MetalTensor> handle = std::make_shared<MetalTensor>(desc);
  return Tensor(static_pointer_cast<void>(handle), data,
                DeviceRuntime::TTMetal);
}

std::array<size_t, 2> computePhysicalShape2D(const TensorDesc &desc) {
  std::array<size_t, 2> physicalShape2D = {1, 1};
  for (int i = -1; i >= -static_cast<int>(desc.shape.size()); i--) {
    size_t &dim = i == -1 ? physicalShape2D[1] : physicalShape2D[0];
    dim *= desc.shape[i + desc.shape.size()];

    if (i >= -static_cast<int>(desc.alignments.size())) {
      dim = utils::roundUp(dim, desc.alignments[i + desc.alignments.size()]);
    }
  }
  return physicalShape2D;
}

// This function calculates how to copy a row-major ND tensor whose data is
// tightly packed according to its logical shape, into a 2D physical storage
// that is a row-major grid of 32x32 row-major tiles.
//
// Since we work with 32x32 tiles, the last two dimensions are special:
// 1. The logical shape is seen as many WxH slices (its last two dimensions).
// 2. The 2D physical storage is seen as many Px1 blocks of 32x32 tiles, each
// block has enough space to store the WxH slice (Px1 because it's a
// pre-tilization row-major Metalium 'tensor').
//
// The task is then: copy entire rows of the HxW slice into the (P*32)x32
// row-major storage, with proper padding if needed.
//
// Return of this function is all the pairs of: the start index of the logical
// row, and the start index of the physical row. The number of elements in the
// row is known (shape[-1]).
//
// Padding occurs at two places:
// 1. At the end of the slice, if H % 32 != 0. This is done by incrementing the
// physical block start index with the stride of P tiles.
// 2. At the end of each logical row, if W % 32 != 0. This is done by
// incrementing the physical row start index with the stride of roundUp(W, 32).
static std::vector<std::pair<size_t, size_t>>
computeLogicalToPhysicalMapping(const TensorDesc &desc) {
  LOG_ASSERT(desc.logicalShape2D[0] <= desc.physicalShape2D[0] &&
                 desc.logicalShape2D[1] <= desc.physicalShape2D[1],
             "Incompatible 2D logical and physical shapes.");
  LOG_ASSERT(desc.physicalShape2D[0] % 32 == 0 &&
                 desc.physicalShape2D[1] % 32 == 0,
             "2D physical shape isn't tile-aligned.");
  const uint32_t rank = desc.shape.size();
  const size_t nSlices =
      rank <= 2
          ? 1
          : std::accumulate(desc.shape.cbegin(), desc.shape.cend() - 2,
                            static_cast<size_t>(1), std::multiplies<size_t>());
  const uint32_t sliceW = rank > 0 ? desc.shape[rank - 1] : 1;
  const uint32_t sliceH = rank > 1 ? desc.shape[rank - 2] : 1;
  const uint32_t tilesPerSliceW = utils::roundUp(sliceW, 32u) / 32;
  const uint32_t tilesPerSliceH = utils::roundUp(sliceH, 32u) / 32;
  const uint32_t tilesPerBlock = tilesPerSliceH * tilesPerSliceW;

  LOG_ASSERT((desc.physicalVolume() / (32 * 32)) == nSlices * tilesPerBlock,
             "Unmatching number of logical and physical 2D slices.");

  std::vector<std::pair<size_t, size_t>> mapping;
  mapping.reserve(nSlices * sliceH);

  const size_t physicalStride = desc.physicalShape2D[1];

  for (size_t i = 0; i < nSlices; i++) {
    const size_t logicalSliceStart = i * sliceW * sliceH;
    const size_t physicalBlockStart = i * tilesPerBlock * 32 * 32;
    for (uint32_t r = 0; r < sliceH; r++) {
      const size_t logicalRowStart = logicalSliceStart + r * sliceW;
      const size_t physicalRowStart = physicalBlockStart + r * physicalStride;
      mapping.emplace_back(logicalRowStart, physicalRowStart);
    }
  }

  return mapping;
}

static void alignAndPadPhysical2DTensorData(void *physicalData,
                                            const void *logicalData,
                                            const TensorDesc &desc) {
  const auto copyIndices = computeLogicalToPhysicalMapping(desc);
  const std::byte *logicalPtr = static_cast<const std::byte *>(logicalData);
  std::byte *physicalPtr = static_cast<std::byte *>(physicalData);
  const size_t rowSize = desc.shape[desc.shape.size() - 1] * desc.itemsize;
  for (const auto &[logicalIdxStart, physicalIdxStart] : copyIndices) {
    std::memcpy(physicalPtr + physicalIdxStart * desc.itemsize,
                logicalPtr + logicalIdxStart * desc.itemsize, rowSize);
  }
}

Tensor createOwnedHostTensor(const void *data, const TensorDesc &desc,
                             const bool alignToTiles) {
  LOG_ASSERT(utils::isSupportedDataType(desc.dataType),
             "Creating owned tensor with unsupported data type: " +
                 std::string(target::EnumNameDataType(desc.dataType)) +
                 "is not implemented for the TTMetal runtime");

  std::shared_ptr<void> owned = nullptr;

  // Even when aligning & padding is requested, skip it if the physical shape
  // isn't in the default flatterned shape anymore, since even logical inputs
  // like 1xN needs to be padded to 32xN in the current design.
  if (!alignToTiles || desc.physicalShape2D[0] != 1) {
    owned = utils::mallocShared(desc.sizeBytes());
    std::memcpy(owned.get(), data, desc.sizeBytes());
    return ttmetal::createBorrowedHostTensor(owned, desc);
  }

  const auto physicalShape2D = computePhysicalShape2D(desc);
  TensorDesc alignedDesc(desc.shape, desc.stride, desc.itemsize, desc.dataType,
                         physicalShape2D);
  owned = utils::callocShared(alignedDesc.sizeBytes()); // Default zero-fill
  alignAndPadPhysical2DTensorData(owned.get(), data, alignedDesc);
  return ttmetal::createBorrowedHostTensor(owned, alignedDesc);
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
          [&](const MeshBuffer &buffer) {
            LOG_FATAL("Datatype mapping from mesh buffer not supported yet.");
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

static void unpadPhysical2DTensorData(void *logicalData,
                                      const void *physicalData,
                                      const TensorDesc &desc) {
  const auto copyIndices = computeLogicalToPhysicalMapping(desc);
  const std::byte *physicalPtr = static_cast<const std::byte *>(physicalData);
  std::byte *logicalPtr = static_cast<std::byte *>(logicalData);
  const size_t rowSize = desc.shape[desc.shape.size() - 1] * desc.itemsize;
  for (const auto &[logicalIdxStart, physicalIdxStart] : copyIndices) {
    std::memcpy(logicalPtr + logicalIdxStart * desc.itemsize,
                physicalPtr + physicalIdxStart * desc.itemsize, rowSize);
  }
}

std::vector<Tensor> toHost(Tensor tensor, bool untilize, bool blocking,
                           bool unalignToTiles) {
  using RetType = std::vector<Tensor>;
  ::tt::runtime::ttmetal::wait(tensor);
  return std::visit(
      utils::overloaded{
          [&](const TensorDesc &desc) -> RetType {
            if (unalignToTiles && desc.volume() != desc.physicalVolume()) {
              auto unpaddedDesc = TensorDesc(desc.shape, desc.dataType);
              auto unpaddedData = utils::mallocShared(unpaddedDesc.sizeBytes());
              unpadPhysical2DTensorData(unpaddedData.get(), tensor.data.get(),
                                        desc);
              return {ttmetal::createBorrowedHostTensor(unpaddedData,
                                                        unpaddedDesc)};
            }
            return {tensor};
          },
          [&](const MeshBuffer &) -> RetType {
            LOG_FATAL("toHost not yet implemented for mesh buffer");
            return {tensor};
          },
      },
      tensor.as<MetalTensor>(DeviceRuntime::TTMetal));
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
  LOG_ASSERT(hostDst.dataType == hostSrc.dataType, "Tensor data type mismatch");
  size_t copySize = std::min(hostDst.sizeBytes(), hostSrc.sizeBytes());
  std::memcpy(dst.data.get(), src.data.get(), copySize);
}

void deallocateTensor(Tensor &tensor, bool) {
  std::visit(utils::overloaded{
                 [&](const TensorDesc &) { /* no-op */ },
                 [&](const MeshBuffer &) {
                   LOG_FATAL(
                       "deallocateTensor not yet implemented for mesh buffer");
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
          [&](const MeshBuffer &buffer) {
            LOG_FATAL("getTensorDataBuffer from MeshBuffer not supported.");
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
                        [&](const MeshBuffer &buffer) {
                          LOG_FATAL(
                              "getTensorDesc from MeshBuffer not supported.");
                          return TensorDesc{};
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
