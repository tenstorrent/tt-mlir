// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/cuda/program_executor.h"
#include "tt/runtime/detail/cuda/ttcuda.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"

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

} // namespace tt::runtime::cuda
