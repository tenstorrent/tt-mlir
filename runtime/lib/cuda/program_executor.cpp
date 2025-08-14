// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>

#include "tt/runtime/detail/cuda/program_executor.h"
#include "tt/runtime/types.h"

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

static int64_t getDim(std::string typeStr) {
  if (typeStr.find("f32") != std::string::npos) {
    return sizeof(float);
  }
  if (typeStr.find("i32") != std::string::npos) {
    return sizeof(int32_t);
  }
  if (typeStr.find("i64") != std::string::npos) {
    return sizeof(int64_t);
  }
  return -1;
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
    std::string typeStr = memref->type()->str();
    while (typeStr.find("x") != std::string::npos) {
      dim *= std::stoi(typeStr.substr(0, typeStr.find("x")));
      typeStr = typeStr.substr(typeStr.find("x") + 1);
    }
    int64_t size = getDim(typeStr) * dim;
    if (size < 0) {
      llvm::errs() << "Failed to get size of tensor: " << memref->name()->str()
                   << "\n";
      finishing();
      return ::tt::runtime::Tensor(nullptr, nullptr,
                                   ::tt::runtime::DeviceRuntime::CUDA);
      ;
    }
    CUdeviceptr devicePtr;
    auto cudaStatus = cuMemAlloc(&devicePtr, size);
    if (cudaStatus != 0) {
      llvm::errs() << "cudaMalloc failed for tensor " << memref->name()->str()
                   << " with error code " << cudaStatus << "\n";
    }
    tensorMap.insert({memref->name()->str(), devicePtr});
    memrefDescMap.insert({memref->name()->str(), memref});

    if (memref->name()->str().find("%arg") != std::string::npos) {
      // Copy input tensors to device.
      size_t i = std::stoi(memref->name()->str().substr(4));
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

  // Run kernels.
  for (const auto *kernel : *program->kernels()) {
    runKernel(kernel);
  }

  // Copy return value to host.
  CUdeviceptr returnVariable = tensorMap[program->return_variable()->str()];
  std::string returnTypeStr =
      memrefDescMap[program->return_variable()->str()]->type()->str();
  int64_t returnDim = 1;
  while (returnTypeStr.find("x") != std::string::npos) {
    returnDim *= std::stoi(returnTypeStr.substr(0, returnTypeStr.find("x")));
    returnTypeStr = returnTypeStr.substr(returnTypeStr.find("x") + 1);
  }
  size_t returnSize = getDim(returnTypeStr) * returnDim;
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
    if (tensorMap.count(arg->str()) == 0) {
      llvm::errs() << "Tensor not found: " << arg->str() << "\n";
      return;
    }
    if (memrefDescMap[arg->str()]->value()->str().size() > 0) {
      if (memrefDescMap[arg->str()]->type()->str().find("f32") !=
          std::string::npos) {
        float value = std::stof(memrefDescMap[arg->str()]->value()->str());
        kernelArgs[i] = &value;
      } else if (memrefDescMap[arg->str()]->type()->str().find("i32") !=
                 std::string::npos) {
        int32_t value = std::stoi(memrefDescMap[arg->str()]->value()->str());
        kernelArgs[i] = &value;
      } else if (memrefDescMap[arg->str()]->type()->str().find("i64") !=
                 std::string::npos) {
        int64_t value = std::stoll(memrefDescMap[arg->str()]->value()->str());
        kernelArgs[i] = &value;
      } else {
        llvm::errs() << "Unsupported tensor type: "
                     << memrefDescMap[arg->str()]->type()->str() << "\n";
      }
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
