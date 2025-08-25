// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/cuda/program_executor.h"

#include "tt/runtime/types.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/Error.h"
#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>

namespace tt::runtime::cuda {

ProgramExecutor::ProgramExecutor(
    ::tt::runtime::Binary &executableHandle,
    std::vector<::tt::runtime::Tensor> &programInputs)
    : executableHandle(executableHandle), programInputs(programInputs) {
  program = ::cuda::GetSizePrefixedProgram(executableHandle.handle.get());
  cuInit(0);
  cuDeviceGet(&device, 0);
  cuCtxCreate(&context, 0, device);
}

static int64_t getDim(::cuda::DataType dataType) {
  switch (dataType) {
  case ::cuda::DataType::Int64:
  case ::cuda::DataType::UInt64:
  case ::cuda::DataType::Float64:
    return 8;
  case ::cuda::DataType::Int32:
  case ::cuda::DataType::UInt32:
  case ::cuda::DataType::Float32:
    return 4;
  case ::cuda::DataType::Float16:
  case ::cuda::DataType::BFloat16:
  case ::cuda::DataType::UInt16:
  case ::cuda::DataType::Int16:
    return 2;
  }
}

void ProgramExecutor::finishing() {
  for (const auto &pair : tensorMap) {
    cuMemFree(pair.second);
  }
  tensorMap.clear();
  memrefDescMap.clear();

  cuCtxDestroy(context);
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

  if (!program->kernels()) {
    llvm::errs() << "No kernels found in program\n";
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
  // Allocate memory for tensors.
  for (auto *memref : *program->memrefs()) {
    int64_t dim = 1;
    // Get size of tensor.
    for (auto shapeDim : *memref->type()->shape()) {
      dim *= shapeDim;
    }
    int64_t size = getDim(memref->type()->data_type()) * dim;
    if (size < 0) {
      llvm::errs() << "Failed to get size of tensor: " << memref->id()->str()
                   << "\n";
      finishing();
      return ::tt::runtime::Tensor(nullptr, nullptr,
                                   ::tt::runtime::DeviceRuntime::CUDA);
      ;
    }
    CUdeviceptr devicePtr;
    auto cudaStatus = cuMemAlloc(&devicePtr, size);
    if (cudaStatus != 0) {
      llvm::errs() << "cudaMalloc failed for tensor " << memref->id()->str()
                   << " with error code " << cudaStatus << "\n";
    }
    tensorMap.insert({memref->id()->str(), devicePtr});
    memrefDescMap.insert({memref->id()->str(), memref});

    if (memref->id()->str().find("%arg") != std::string::npos) {
      // Copy input tensors to device.
      size_t i = std::stoi(memref->id()->str().substr(4));
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

  for (auto *constant : *program->constants()) {
    constantMap.insert({constant->id()->str(), constant});
  }

  // Run kernels.
  for (const auto *kernel : *program->kernels()) {
    runKernel(kernel);
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

void ProgramExecutor::runKernel(const ::cuda::Kernel *kernel) {

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
  cuModuleLoadData(&module, kernel->ptx()->c_str());
  cuModuleGetFunction(&function, module, kernel->name()->c_str());

  dim3 gridSize(kernel->grid_size_x(), kernel->grid_size_y(),
                kernel->grid_size_z());
  dim3 blockSize(kernel->block_size_x(), kernel->block_size_y(),
                 kernel->block_size_z());
  // llvm::outs() << gridSize.x << " " <<gridSize.y << " " <<gridSize.z << "\n";
  // llvm::outs() << blockSize.x << " " <<blockSize.y << " " <<blockSize.z <<
  // "\n\n";
  cuLaunchKernel(function, gridSize.x, gridSize.y, gridSize.z, blockSize.x,
                 blockSize.y, blockSize.z, 0, 0, kernelArgs.get(), nullptr);
  cudaDeviceSynchronize();
  cuModuleUnload(module);
}

} // namespace tt::runtime::cuda
