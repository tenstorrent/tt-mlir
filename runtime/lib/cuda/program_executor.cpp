// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/cuda/program_executor.h"
#include "tt/runtime/detail/cuda/ttcuda.h"
#include "tt/runtime/types.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/Error.h"
#include <cassert>
#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>

namespace tt::runtime::cuda {

ProgramExecutor::ProgramExecutor(
    ::tt::runtime::Device deviceHandle, ::tt::runtime::Binary &executableHandle,
    std::vector<::tt::runtime::Tensor> &programInputs)
    : executableHandle(executableHandle), deviceHandle(deviceHandle),
      programInputs(programInputs) {
  program =
      ::tt::target::cuda::GetSizePrefixedProgram(executableHandle.handle.get());
  auto cudaHandle =
      std::static_pointer_cast<CudaDeviceHandle>(deviceHandle.handle);
  context = cudaHandle->context;
  device = cudaHandle->device;
}

static int64_t getDim(::tt::target::cuda::DataType dataType) {
  switch (dataType) {
  case ::tt::target::cuda::DataType::Int64:
  case ::tt::target::cuda::DataType::UInt64:
  case ::tt::target::cuda::DataType::Float64:
    return 8;
  case ::tt::target::cuda::DataType::Int32:
  case ::tt::target::cuda::DataType::UInt32:
  case ::tt::target::cuda::DataType::Float32:
    return 4;
  case ::tt::target::cuda::DataType::Float16:
  case ::tt::target::cuda::DataType::BFloat16:
  case ::tt::target::cuda::DataType::UInt16:
  case ::tt::target::cuda::DataType::Int16:
    return 2;
  }
}

void ProgramExecutor::finishing() {
  for (const auto &pair : tensorMap) {
    cuMemFree(pair.second);
  }
  tensorMap.clear();
  memrefDescMap.clear();
}

::tt::runtime::Tensor ProgramExecutor::execute() {

  if (!program) {
    llvm::errs() << "No program found\n";
    return ::tt::runtime::Tensor(nullptr, nullptr,
                                 ::tt::runtime::DeviceRuntime::CUDA);
  }

  if (!program->memrefs()) {
    llvm::errs() << "No memrefs found in program\n";
    return ::tt::runtime::Tensor(nullptr, nullptr,
                                 ::tt::runtime::DeviceRuntime::CUDA);
    ;
  }

  if (!program->constants()) {
    llvm::errs() << "No constants found in program\n";
    return ::tt::runtime::Tensor(nullptr, nullptr,
                                 ::tt::runtime::DeviceRuntime::CUDA);
    ;
  }

  if (!program->actions()) {
    llvm::errs() << "No actions found in program\n";
    return ::tt::runtime::Tensor(nullptr, nullptr,
                                 ::tt::runtime::DeviceRuntime::CUDA);
    ;
  }

  if (program->return_variable()->str().size() == 0) {
    llvm::errs() << "No return variable found in program\n";
    return ::tt::runtime::Tensor(nullptr, nullptr,
                                 ::tt::runtime::DeviceRuntime::CUDA);
    ;
  }
  // Declare tensors.
  for (auto *memref : *program->memrefs()) {
    memrefDescMap.insert({memref->id()->str(), memref});
  }

  for (auto *constant : *program->constants()) {
    uint64_t totalElements = 1;
    for (size_t dim = 0; dim < constant->type()->shape()->size(); ++dim) {
      totalElements *= constant->type()->shape()->Get(dim);
    }
    if (totalElements == 1) {
      constantMap.insert({constant->id()->str(), constant});
      continue;
    }
    const uint8_t *byteData = constant->value()->data();

    size_t size = totalElements * getDim(constant->type()->data_type());
    CUdeviceptr devicePtr;
    auto cudaStatus = cuMemAlloc(&devicePtr, size);
    if (cudaStatus != 0) {
      llvm::errs() << "cudaMalloc failed for tensor " << constant->id()->str()
                   << " with error code " << cudaStatus << "\n";
    }
    tensorMap.insert({constant->id()->str(), devicePtr});

    size_t originalSize = constant->value()->size();
    std::vector<uint8_t> repeatedData;

    for (size_t i = 0; i < totalElements; ++i) {
      for (size_t j = 0; j < originalSize; ++j) {
        repeatedData.push_back(*(byteData + j));
      }
    }
    cuMemcpyHtoD(devicePtr, repeatedData.data(), size);
  }

  // Process actions.
  for (size_t i = 0; i < program->actions()->size(); ++i) {
    ::tt::target::cuda::Action actionType = program->actions_type()->Get(i);
    const void *actionObj = program->actions()->Get(i);
    switch (actionType) {
    case ::tt::target::cuda::Action::Kernel: {
      const ::tt::target::cuda::Kernel *kernel =
          static_cast<const ::tt::target::cuda::Kernel *>(actionObj);
      for (auto name : *kernel->input_names()) {

        if (!tensorMap.contains(name->str()) &&
            !constantMap.contains(name->str())) {

          int64_t dim = 1;
          // Get size of tensor.
          for (auto shapeDim : *memrefDescMap[name->str()]->type()->shape()) {
            dim *= shapeDim;
          }
          int64_t size =
              getDim(memrefDescMap[name->str()]->type()->data_type()) * dim;
          if (size < 0) {
            llvm::errs() << "Failed to get size of tensor: " << name->str()
                         << "\n";
            finishing();
            return ::tt::runtime::Tensor(nullptr, nullptr,
                                         ::tt::runtime::DeviceRuntime::CUDA);
          }
          CUdeviceptr devicePtr;

          auto cudaStatus = cuMemAlloc(&devicePtr, size);
          if (cudaStatus != 0) {
            llvm::errs() << "cudaMalloc failed for tensor " << name->str()
                         << " with error code " << cudaStatus << "\n";
          }
          tensorMap.insert({name->str(), devicePtr});
          if (name->str().find("%arg") != std::string::npos) {
            // Copy input tensors to device.
            size_t i = std::stoi(name->str().substr(4));
            if (programInputs.size() <= i) {
              llvm::errs() << "Not enough arguments provided\n";
              finishing();
              return ::tt::runtime::Tensor(nullptr, nullptr,
                                           ::tt::runtime::DeviceRuntime::CUDA);
              ;
            }
            cuMemcpyHtoD(devicePtr, programInputs[i].data.get(), size);
          }
        }
      }
      runKernel(kernel);

      for (auto name : *kernel->input_names()) {
        if (memrefDescMap.count(name->str()) &&
            memrefDescMap[name->str()]->last() == i &&
            program->return_variable()->str() != name->str()) {
          cuMemFree(tensorMap[name->str()]);
          tensorMap.erase(name->str());
        }
      }

      break;
    }
    case ::tt::target::cuda::Action::CopyFunction: {
      const ::tt::target::cuda::CopyFunction *copyFunc =
          static_cast<const ::tt::target::cuda::CopyFunction *>(actionObj);
      if (!tensorMap.contains(copyFunc->src()->str()) &&
          !constantMap.contains(copyFunc->src()->str())) {

        int64_t dim = 1;
        // Get size of tensor.
        for (auto shapeDim :
             *memrefDescMap[copyFunc->src()->str()]->type()->shape()) {
          dim *= shapeDim;
        }
        int64_t size =
            getDim(memrefDescMap[copyFunc->src()->str()]->type()->data_type()) *
            dim;
        if (size < 0) {
          llvm::errs() << "Failed to get size of tensor: "
                       << copyFunc->src()->str() << "\n";
          finishing();
          return ::tt::runtime::Tensor(nullptr, nullptr,
                                       ::tt::runtime::DeviceRuntime::CUDA);
        }
        CUdeviceptr devicePtr;

        auto cudaStatus = cuMemAlloc(&devicePtr, size);

        if (cudaStatus != 0) {
          llvm::errs() << "cudaMalloc failed for tensor "
                       << copyFunc->src()->str() << " with error code "
                       << cudaStatus << "\n";
        }
        tensorMap.insert({copyFunc->src()->str(), devicePtr});
        if (copyFunc->src()->str().find("%arg") != std::string::npos) {
          // Copy input tensors to device.
          size_t i = std::stoi(copyFunc->src()->str().substr(4));
          if (programInputs.size() <= i) {
            llvm::errs() << "Not enough arguments provided\n";
            finishing();
            return ::tt::runtime::Tensor(nullptr, nullptr,
                                         ::tt::runtime::DeviceRuntime::CUDA);
            ;
          }
          cuMemcpyHtoD(devicePtr, programInputs[i].data.get(), size);
        }
      }
      if (!tensorMap.contains(copyFunc->dst()->str()) &&
          !constantMap.contains(copyFunc->dst()->str())) {

        int64_t dim = 1;
        // Get size of tensor.
        for (auto shapeDim :
             *memrefDescMap[copyFunc->dst()->str()]->type()->shape()) {
          dim *= shapeDim;
        }
        int64_t size =
            getDim(memrefDescMap[copyFunc->dst()->str()]->type()->data_type()) *
            dim;
        if (size < 0) {
          llvm::errs() << "Failed to get size of tensor: "
                       << copyFunc->dst()->str() << "\n";
          finishing();
          return ::tt::runtime::Tensor(nullptr, nullptr,
                                       ::tt::runtime::DeviceRuntime::CUDA);
        }
        CUdeviceptr devicePtr;

        auto cudaStatus = cuMemAlloc(&devicePtr, size);
        if (cudaStatus != 0) {
          llvm::errs() << "cudaMalloc failed for tensor "
                       << copyFunc->dst()->str() << " with error code "
                       << cudaStatus << "\n";
        }
        tensorMap.insert({copyFunc->dst()->str(), devicePtr});
        if (copyFunc->dst()->str().find("%arg") != std::string::npos) {
          // Copy input tensors to device.
          size_t i = std::stoi(copyFunc->dst()->str().substr(4));
          if (programInputs.size() <= i) {
            llvm::errs() << "Not enough arguments provided\n";
            finishing();
            return ::tt::runtime::Tensor(nullptr, nullptr,
                                         ::tt::runtime::DeviceRuntime::CUDA);
            ;
          }
          cuMemcpyHtoD(devicePtr, programInputs[i].data.get(), size);
        }
      }
      runCopyFunction(copyFunc);

      if (memrefDescMap.count(copyFunc->src()->str()) &&
          memrefDescMap[copyFunc->src()->str()]->last() == i &&
          program->return_variable()->str() != copyFunc->src()->str()) {
        cuMemFree(tensorMap[copyFunc->src()->str()]);
        tensorMap.erase(copyFunc->src()->str());
      }

      if (memrefDescMap.count(copyFunc->dst()->str()) &&
          memrefDescMap[copyFunc->dst()->str()]->last() == i &&
          program->return_variable()->str() != copyFunc->dst()->str()) {
        cuMemFree(tensorMap[copyFunc->dst()->str()]);
        tensorMap.erase(copyFunc->dst()->str());
      }
      break;
    }
    case ::tt::target::cuda::Action::NONE:
      break;
    }
  }

  // Copy return value to host.
  CUdeviceptr returnVariable = tensorMap[program->return_variable()->str()];
  int64_t returnDim = 1;
  for (auto shapeDim :
       *memrefDescMap[program->return_variable()->str()]->type()->shape()) {
    returnDim *= shapeDim;
  }
  size_t returnSize = getDim(memrefDescMap[program->return_variable()->str()]
                                 ->type()
                                 ->data_type()) *
                      returnDim;
  auto returnPtr = std::shared_ptr<void>(std::malloc(returnSize), std::free);
  cuMemcpyDtoH(returnPtr.get(), returnVariable, returnSize);

  finishing();
  ::tt::runtime::Tensor returnTensor(nullptr, returnPtr,
                                     ::tt::runtime::DeviceRuntime::CUDA);

  return returnTensor;
}

void ProgramExecutor::runCopyFunction(
    const ::tt::target::cuda::CopyFunction *copyFunc) {
  auto source = copyFunc->src();
  auto dest = copyFunc->dst();
  assert(memrefDescMap.contains(source->str()) &&
         memrefDescMap.contains(dest->str()));

  auto sourceType = memrefDescMap[source->str()]->type()->data_type();
  auto destType = memrefDescMap[dest->str()]->type()->data_type();
  assert(sourceType == destType);
  int64_t elementSize = getDim(sourceType);

  auto sourceDim = memrefDescMap[source->str()]->type()->shape();
  auto destDim = memrefDescMap[dest->str()]->type()->shape();

  // Copy operations occur during conv2d ops and maxpool2d ops.
  // These ops use padding, so the source and destination tensors may have
  // different shapes. Tosa ops have a strict padding requirement, so slicing is
  // done after the conv2d or maxpool2d to maintain the correct shape if needed.
  // These ops use (N, H, W, C) layout. N and C are the same for source and
  // destination. Therefore, the first two dimensions can be seen as the height
  // and second two dimensions can be seen as the width. This is done to avoid
  // complex logic for copying operations and use Cuda's 2D copy operations.
  int sourceHeight = sourceDim->Get(0) * sourceDim->Get(1);
  int sourceWidth = sourceDim->Get(2) * sourceDim->Get(3);

  int destHeight = destDim->Get(0) * destDim->Get(1);
  int destWidth = destDim->Get(2) * destDim->Get(3);

  auto offset = copyFunc->offset();

  CUDA_MEMCPY2D copyParams = {};

  // Source parameters (contiguous layout)
  copyParams.srcXInBytes = 0;
  copyParams.srcY = 0;
  copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copyParams.srcDevice = tensorMap[source->str()];
  copyParams.srcPitch = sourceWidth * elementSize;

  // Destination parameters (strided layout with offset)
  copyParams.dstXInBytes = 0;
  copyParams.dstY = 0;
  copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  copyParams.dstDevice = tensorMap[dest->str()] + (offset * elementSize);
  copyParams.dstPitch = destWidth * elementSize;

  // Copy dimensions - copy the smaller of source/dest. This is done because
  // copying is called for both padding and slicing. If the source is smaller,
  // then it is padded. If the destination is smaller, then it is sliced.
  copyParams.WidthInBytes = std::min(sourceWidth, destWidth) * elementSize;
  copyParams.Height = std::min(sourceHeight, destHeight);

  auto result = cuMemcpy2D(&copyParams);
  if (result != CUDA_SUCCESS) {
    llvm::errs() << "cuMemcpy2D failed with error: " << result << "\n";
  }
}
void ProgramExecutor::runKernel(const ::tt::target::cuda::Kernel *kernel) {

  auto kernelArgs = std::make_unique<void *[]>(kernel->input_names()->size());
  size_t i = 0;
  for (const auto *arg : *kernel->input_names()) {
    if (!tensorMap.contains(arg->str()) && !constantMap.contains(arg->str())) {
      llvm::errs() << "Tensor not found: " << arg->str() << "\n";
      return;
    }
    if (constantMap.contains(arg->str())) {
      const uint8_t *byteData = constantMap[arg->str()]->value()->data();
      kernelArgs[i] = const_cast<uint8_t *>(byteData);
    } else {
      kernelArgs[i] = &tensorMap[arg->str()];
    }
    i++;
  }
  CUmodule module;
  CUfunction function;
  if (cuModuleLoadData(&module, kernel->ptx()->c_str()) != CUDA_SUCCESS) {
    llvm::errs() << "Failed to load module!\n";
    return;
  }
  if (cuModuleGetFunction(&function, module, kernel->name()->c_str()) !=
      CUDA_SUCCESS) {
    llvm::errs() << "Failed to load func!\n";
    cuModuleUnload(module);
    return;
  }

  dim3 gridSize(kernel->grid_size_x(), kernel->grid_size_y(),
                kernel->grid_size_z());
  dim3 blockSize(kernel->block_size_x(), kernel->block_size_y(),
                 kernel->block_size_z());
  auto result =
      cuLaunchKernel(function, gridSize.x, gridSize.y, gridSize.z, blockSize.x,
                     blockSize.y, blockSize.z, 0, 0, kernelArgs.get(), nullptr);
  if (result != CUDA_SUCCESS) {
    llvm::errs() << kernel->name()->c_str() << " failed with code " << result
                 << "\n";
    for (const auto *arg : *kernel->input_names()) {
      llvm::errs() << arg->str() << " ";
    }
    llvm::errs() << "\n";
    llvm::errs() << gridSize.x << " " << gridSize.y << " " << gridSize.z
                 << "\n";
    llvm::errs() << blockSize.x << " " << blockSize.y << " " << blockSize.z
                 << "\n\n";
  }

  cuModuleUnload(module);
}

} // namespace tt::runtime::cuda
