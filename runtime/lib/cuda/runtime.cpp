// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/cuda/program_executor.h"
#include "tt/runtime/detail/cuda/ttcuda.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Version.h"
#include "types_generated.h"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/CUDA/program_generated.h"
#pragma clang diagnostic pop

#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/Error.h"
namespace tt::runtime::cuda {

// Helper function to convert CUDA DataType enum to common DataType enum
static ::tt::target::DataType
cudaDataTypeToCommon(::tt::target::cuda::DataType cudaType) {
  switch (cudaType) {
  case ::tt::target::cuda::DataType::Float64:
    return ::tt::target::DataType::Float64;
  case ::tt::target::cuda::DataType::UInt64:
    return ::tt::target::DataType::UInt64;
  case ::tt::target::cuda::DataType::Int64:
    return ::tt::target::DataType::Int64;
  case ::tt::target::cuda::DataType::Float32:
    return ::tt::target::DataType::Float32;
  case ::tt::target::cuda::DataType::UInt32:
    return ::tt::target::DataType::UInt32;
  case ::tt::target::cuda::DataType::Int32:
    return ::tt::target::DataType::Int32;
  case ::tt::target::cuda::DataType::Float16:
    return ::tt::target::DataType::Float16;
  case ::tt::target::cuda::DataType::BFloat16:
    return ::tt::target::DataType::BFloat16;
  case ::tt::target::cuda::DataType::UInt16:
    return ::tt::target::DataType::UInt16;
  case ::tt::target::cuda::DataType::Int16:
    return ::tt::target::DataType::Int16;
  }
}

// Helper function to convert common DataType enum to CUDA DataType enum
static ::tt::target::cuda::DataType
commonDataTypeToCuda(::tt::target::DataType commonType) {
  switch (commonType) {
  case ::tt::target::DataType::Float64:
    return ::tt::target::cuda::DataType::Float64;
  case ::tt::target::DataType::UInt64:
    return ::tt::target::cuda::DataType::UInt64;
  case ::tt::target::DataType::Int64:
    return ::tt::target::cuda::DataType::Int64;
  case ::tt::target::DataType::Float32:
    return ::tt::target::cuda::DataType::Float32;
  case ::tt::target::DataType::UInt32:
    return ::tt::target::cuda::DataType::UInt32;
  case ::tt::target::DataType::Int32:
    return ::tt::target::cuda::DataType::Int32;
  case ::tt::target::DataType::Float16:
    return ::tt::target::cuda::DataType::Float16;
  case ::tt::target::DataType::BFloat16:
    return ::tt::target::cuda::DataType::BFloat16;
  case ::tt::target::DataType::UInt16:
    return ::tt::target::cuda::DataType::UInt16;
  case ::tt::target::DataType::Int16:
    return ::tt::target::cuda::DataType::Int16;
  default:
    return ::tt::target::cuda::DataType::Float32; // fallback
  }
}

// Helper function to get CUDA binary from handle
static const ::tt::target::cuda::Program *getBinary(Binary binary) {
  bool isCUDA = ::tt::target::cuda::SizePrefixedProgramBufferHasIdentifier(
      binary.handle.get());
  LOG_ASSERT(isCUDA, "Unsupported binary format");
  return ::tt::target::cuda::GetSizePrefixedProgram(binary.handle.get());
}

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
  cudaHandle->dataType = commonDataTypeToCuda(dataType);
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
  cudaHandle->dataType = commonDataTypeToCuda(dataType);
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
  // Get the CUDA binary and access the program
  const target::cuda::Program *program = getBinary(executableHandle);
  LOG_ASSERT(inputIndex < program->memrefs()->size(), "Invalid input index");

  // Get the input memref and its data type
  const target::cuda::MemRefDesc *inputMemref =
      program->memrefs()->Get(inputIndex);
  // Use CUDA enum directly, convert to common enum for CudaLayoutDesc
  ::tt::target::DataType inputDataType =
      cudaDataTypeToCommon(inputMemref->type()->data_type());

  std::shared_ptr<CudaLayoutDesc> layoutDesc = std::make_shared<CudaLayoutDesc>(
      CudaLayoutDesc::StorageType::HOST, CudaLayoutDesc::Layout::ROW_MAJOR,
      inputDataType);

  return Layout(layoutDesc, DeviceRuntime::CUDA);
}

::tt::target::DataType getTensorDataType(Tensor tensor) {
  // Extract data type from CUDA tensor handle and convert to common enum
  auto cudaHandle = std::static_pointer_cast<CudaTensorHandle>(tensor.handle);
  return cudaDataTypeToCommon(cudaHandle->dataType);
}

bool isTensorAllocated(Tensor tensor) {
  // CUDA tensors are considered allocated if they have valid data
  return tensor.data != nullptr;
}

void wait(Event event) {
  // CUDA execution is synchronous, so no-op
}

void wait(Tensor tensor, std::optional<uint8_t> cqId) {
  // CUDA execution is synchronous, so no-op
}

void wait(const std::vector<Tensor> &tensors, std::optional<uint8_t> cqId) {
  // CUDA execution is synchronous, so no-op
}

std::vector<std::byte> getTensorDataBuffer(Tensor tensor) {
  auto cudaHandle = std::static_pointer_cast<CudaTensorHandle>(tensor.handle);

  if (!tensor.data || !tensor.data.get()) {
    LOG_WARNING(
        "getTensorDataBuffer: Tensor has null data; returning empty buffer");
    return {};
  }

  size_t totalElements = 1;
  for (size_t i = 0; i < cudaHandle->shape.size(); i++) {
    totalElements *= cudaHandle->shape[i];
  }
  size_t totalBytes = totalElements * cudaHandle->itemsize;

  if (totalBytes == 0) {
    LOG_WARNING(
        "getTensorDataBuffer: Tensor has zero volume; returning empty buffer");
    return {};
  }

  const std::byte *data = static_cast<const std::byte *>(tensor.data.get());

  if (!data) {
    LOG_WARNING(
        "getTensorDataBuffer: data pointer is NULL; returning empty buffer");
    return {};
  }

  std::vector<std::byte> result;
  result.reserve(totalBytes);

  result.resize(totalBytes);
  std::memcpy(result.data(), data, totalBytes);

  return result;
}

std::uint32_t getTensorElementSize(Tensor tensor) {
  auto cudaHandle = std::static_pointer_cast<CudaTensorHandle>(tensor.handle);
  return cudaHandle->itemsize;
}

std::uint32_t getTensorVolume(Tensor tensor) {
  auto cudaHandle = std::static_pointer_cast<CudaTensorHandle>(tensor.handle);
  std::uint32_t volume = 1;
  for (size_t i = 0; i < cudaHandle->shape.size(); i++) {
    volume *= cudaHandle->shape[i];
  }
  return volume;
}

namespace detail {

void readDeviceProfilerResults(Device device) {
  // CUDA doesn't have device profiler results - no-op.
}

} // namespace detail

SystemDesc
getCurrentSystemDesc(std::optional<DispatchCoreType> dispatchCoreType,
                     std::optional<Device> meshDevice) {
  ::flatbuffers::FlatBufferBuilder fbb;

  std::vector<::flatbuffers::Offset<::tt::target::CPUDesc>> cpuDescs;
  cpuDescs.emplace_back(
      ::tt::target::CreateCPUDesc(fbb, ::tt::target::CPURole::Host,
                                  fbb.CreateString("x86_64-pc-linux-gnu")));

  std::vector<::flatbuffers::Offset<::tt::target::ChipDesc>> chipDescs;
  std::vector<uint32_t> chipDescIndices;
  std::vector<::tt::target::ChipCapability> chipCapabilities;
  std::vector<::tt::target::ChipCoord> chipCoords;
  std::vector<::tt::target::ChipChannel> allConnections;

  auto systemDesc = ::tt::target::CreateSystemDescDirect(
      fbb, &cpuDescs, &chipDescs, &chipDescIndices, &chipCapabilities,
      &chipCoords, &allConnections);

  ::ttmlir::Version ttmlirVersion = ::ttmlir::getVersion();
  ::tt::target::Version version(ttmlirVersion.major, ttmlirVersion.minor,
                                ttmlirVersion.patch);

  auto root = ::tt::target::CreateSystemDescRootDirect(
      fbb, &version, "cuda_hash", "cuda_git", "CUDA", systemDesc);

  // Use the correct finish function like the working implementation
  ::tt::target::FinishSizePrefixedSystemDescRootBuffer(fbb, root);

  // Copy buffer data like the working implementation
  uint8_t *buf = fbb.GetBufferPointer();
  auto size = fbb.GetSize();
  auto handle = ::tt::runtime::utils::mallocShared(size);
  std::memcpy(handle.get(), buf, size);

  return ::tt::runtime::SystemDesc(handle);
}
} // namespace tt::runtime::cuda
