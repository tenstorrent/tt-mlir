// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/cuda/program_executor.h"
#include "tt/runtime/detail/cuda/ttcuda.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/Common/types_generated.h"

namespace tt::runtime::cuda {

std::vector<::tt::runtime::Tensor>
submit(Device deviceHandle, Binary executableHandle, std::uint32_t programIndex,
       std::vector<::tt::runtime::Tensor> &inputs) {
  ProgramExecutor executor(deviceHandle, executableHandle, inputs);
  tt::runtime::Tensor result = executor.execute();
  return {result};
}

::tt::runtime::Tensor
createBorrowedHostTensor(void *data, const std::vector<std::uint32_t> &shape,
                         const std::vector<std::uint32_t> &stride,
                         std::uint32_t itemsize,
                         ::tt::target::DataType dataType) {
  LOG_ASSERT(data != nullptr, "Cannot create borrowed tensor with null data");
  LOG_ASSERT(::tt::runtime::utils::isSupportedDataType(dataType),
             "Cannot create borrowed tensor with unsupported data type");
  LOG_ASSERT(itemsize > 0, "Item size must be greater than 0");

  auto cudaHandle = std::make_shared<CudaTensorHandle>();
  cudaHandle->shape = shape;
  cudaHandle->stride = stride;
  cudaHandle->dataType = dataType;
  cudaHandle->itemsize = itemsize;

  std::shared_ptr<void> borrowedData = utils::unsafe_borrow_shared(data);

  return Tensor(std::static_pointer_cast<void>(cudaHandle), borrowedData,
                DeviceRuntime::CUDA);
}

::tt::runtime::Tensor
createOwnedHostTensor(const void *data, const std::vector<std::uint32_t> &shape,
                      const std::vector<std::uint32_t> &stride,
                      std::uint32_t itemsize, ::tt::target::DataType dataType) {
  LOG_ASSERT(data != nullptr, "Cannot create owned tensor with null data");
  LOG_ASSERT(::tt::runtime::utils::isSupportedDataType(dataType),
             "Cannot create owned tensor with unsupported data type");
  LOG_ASSERT(itemsize > 0, "Item size must be greater than 0");

  auto cudaHandle = std::make_shared<CudaTensorHandle>();
  cudaHandle->shape = shape;
  cudaHandle->stride = stride;
  cudaHandle->dataType = dataType;
  cudaHandle->itemsize = itemsize;

  // Calculate total data size
  size_t dataSize = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    dataSize *= shape[i];
  }
  dataSize *= itemsize;

  auto ownedData = std::shared_ptr<void>(std::malloc(dataSize), std::free);
  LOG_ASSERT(ownedData != nullptr,
             "Failed to allocate memory for owned tensor");
  std::memcpy(ownedData.get(), data, dataSize);

  return Tensor(std::static_pointer_cast<void>(cudaHandle), ownedData,
                DeviceRuntime::CUDA);
}

std::vector<::tt::runtime::Tensor> toHost(::tt::runtime::Tensor tensor,
                                          bool untilize, bool blocking) {
  return {tensor};
}

std::vector<std::uint32_t> getTensorShape(Tensor tensor) {
  auto cudaHandle = std::static_pointer_cast<CudaTensorHandle>(tensor.handle);
  return cudaHandle->shape;
}

std::vector<std::uint32_t> getTensorStride(Tensor tensor) {
  auto cudaHandle = std::static_pointer_cast<CudaTensorHandle>(tensor.handle);
  return cudaHandle->stride;
}

size_t getNumAvailableDevices() {
  int deviceCount = 0;
  cudaError_t error = cudaGetDeviceCount(&deviceCount);
  if (error != cudaSuccess) {
    LOG_WARNING("Failed to get CUDA device count: " +
                std::string(cudaGetErrorString(error)));
    return 0;
  }
  return static_cast<size_t>(deviceCount);
}

Device openMeshDevice(const MeshDeviceOptions &options) {

  auto error = cuInit(0);

  if (error != CUDA_SUCCESS) {
    LOG_FATAL("Failed to initialize CUDA: " + std::to_string(error));
  }

  CUdevice device;
  error = cuDeviceGet(&device, 0);
  if (error != CUDA_SUCCESS) {
    LOG_FATAL("Failed to get CUDA device: " + std::to_string(error));
  }

  CUcontext context;
  error = cuCtxCreate(&context, 0, device);
  if (error != CUDA_SUCCESS) {
    LOG_FATAL("Failed to create CUDA context: " + std::to_string(error));
  }

  auto deviceHandle = std::make_shared<CudaDeviceHandle>();
  deviceHandle->device = device;
  deviceHandle->context = context;

  return Device(std::static_pointer_cast<void>(deviceHandle), nullptr,
                DeviceRuntime::CUDA);
}

void closeMeshDevice(Device parentMesh) {
  LOG_ASSERT(parentMesh.matchesRuntime(DeviceRuntime::CUDA),
             "Device must be a CUDA device");
  auto cudaHandle =
      std::static_pointer_cast<CudaDeviceHandle>(parentMesh.handle);
  cuCtxDestroy(cudaHandle->context);
}

void memcpy(void *dst, Tensor src,
            std::optional<tt::target::DataType> targetDataType) {
  LOG_ASSERT(src.matchesRuntime(DeviceRuntime::CUDA),
             "Source tensor must be a CUDA tensor");
  LOG_ASSERT(dst != nullptr, "Destination pointer cannot be null");
  LOG_ASSERT(src.data.get() != nullptr, "Source tensor data cannot be null");

  auto srcCudaHandle = std::static_pointer_cast<CudaTensorHandle>(src.handle);
  size_t dataSize = 1;
  for (size_t i = 0; i < srcCudaHandle->shape.size(); i++) {
    dataSize *= srcCudaHandle->shape[i];
  }
  dataSize *= srcCudaHandle->itemsize;
  std::memcpy(dst, src.data.get(), dataSize);
}

void memcpy(Tensor dst, Tensor src) {
  LOG_ASSERT(dst.matchesRuntime(DeviceRuntime::CUDA),
             "Destination tensor must be a CUDA tensor");
  LOG_ASSERT(src.matchesRuntime(DeviceRuntime::CUDA),
             "Source tensor must be a CUDA tensor");
  LOG_ASSERT(dst.data.get() != nullptr,
             "Destination tensor data cannot be null");
  LOG_ASSERT(src.data.get() != nullptr, "Source tensor data cannot be null");
  auto srcCudaHandle = std::static_pointer_cast<CudaTensorHandle>(src.handle);
  auto dstCudaHandle = std::static_pointer_cast<CudaTensorHandle>(dst.handle);
  LOG_ASSERT(srcCudaHandle->dataType == dstCudaHandle->dataType,
             "Source and destination tensor data types must match");
  LOG_ASSERT(srcCudaHandle->shape.size() == dstCudaHandle->shape.size(),
             "Source and destination tensor shapes must match");
  size_t dataSize = 1;
  for (size_t i = 0; i < srcCudaHandle->shape.size(); i++) {
    LOG_ASSERT(srcCudaHandle->shape[i] == dstCudaHandle->shape[i],
               "Source and destination tensor shapes must match");
    dataSize *= srcCudaHandle->shape[i];
  }
  dataSize *= srcCudaHandle->itemsize;
  std::memcpy(dst.data.get(), src.data.get(), dataSize);
}

void deallocateTensor(Tensor &tensor, bool force) {
  // All tensors are owned by the runtime or created and destroyed in program
  // execution.
}

Tensor toLayout(Tensor tensor, Device device, Layout layout,
                std::optional<bool> retain) {
  // CUDA program executor handles all memory transfers.
  return tensor;
}

Layout getLayout(Binary executableHandle, std::uint32_t programIndex,
                 std::uint32_t inputIndex) {
  // CUDA program executor handles all memory transfers.
  std::shared_ptr<CudaLayoutDesc> layoutDesc = std::make_shared<CudaLayoutDesc>(
      StorageType::HOST, Layout::ROW_MAJOR, DataType::Float32);

  return Layout(layoutDesc, DeviceRuntime::CUDA);
}

SystemDesc
getCurrentSystemDesc(std::optional<DispatchCoreType> dispatchCoreType,
                     std::optional<Device> meshDevice) {
  ::flatbuffers::FlatBufferBuilder fbb;

  std::vector<::flatbuffers::Offset<tt::target::CPUDesc>> cpuDescs;
  cpuDescs.emplace_back(
      ::tt::target::CreateCPUDesc(fbb, ::tt::target::CPURole::Host,
                                  fbb.CreateString("x86_64-pc-linux-gnu")));

  std::vector<::flatbuffers::Offset<tt::target::ChipDesc>> chipDescs;
  std::vector<uint32_t> chipDescIndices = {0};
  std::vector<::tt::target::ChipCapability> chipCapabilities = {
      ::tt::target::ChipCapability::HostMMIO};
  std::vector<::tt::target::ChipCoord> chipCoords = {
      ::tt::target::ChipCoord(0, 0, 0, 0)};
  std::vector<::tt::target::ChipChannel> allConnections;

  auto systemDesc = ::tt::target::CreateSystemDescDirect(
      fbb, &cpuDescs, &chipDescs, &chipDescIndices, &chipCapabilities,
      &chipCoords, &allConnections);

  ::ttmlir::Version ttmlirVersion = ::ttmlir::getVersion();
  ::tt::target::Version version(ttmlirVersion.major, ttmlirVersion.minor,
                                ttmlirVersion.patch);

  auto root = ::tt::target::CreateSystemDescRootDirect(
      fbb, &version, "cuda_hash", "cuda_git", "CUDA", systemDesc);

  fbb.Finish(::tt::target::CreateSizePrefixedSystemDescRoot(fbb, root));

  return ::tt::runtime::SystemDesc(std::shared_ptr<void>(
      fbb.GetBufferPointer(), [fbb = std::move(fbb)](void *) mutable {}));
}
} // namespace tt::runtime::cuda
