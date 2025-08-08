// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include "tt/runtime/detail/cuda/program_executor.h"
#include "tt/runtime/types.h"

namespace tt::runtime::cuda {

ProgramExecutor::ProgramExecutor(
    ::tt::runtime::Binary &executableHandle,
    std::vector<::tt::runtime::Tensor> &programInputs)
    : executableHandle(executableHandle), programInputs(programInputs) {
  program = ::gpu::GetSizePrefixedProgram(executableHandle.handle.get());
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

void ProgramExecutor::execute() {

  if (!program) {
    llvm::errs() << "No program found\n";
    return;
  }

  if (!program->memrefs()) {
    llvm::errs() << "No memrefs found in program\n";
    return;
  }

  if (!program->kernels()) {
    llvm::errs() << "No kernels found in program\n";
    return;
  }

  for (auto *memref : *program->memrefs()) {
    int64_t dim = 1;
    void *devicePtr = nullptr;
    std::string typeStr = memref->type()->str();
    while (typeStr.find("x") != std::string::npos) {
      dim *= std::stoi(typeStr.substr(0, typeStr.find("x")));
      typeStr = typeStr.substr(typeStr.find("x") + 1);
    }
    int64_t size = getDim(typeStr) * dim;
    if (size < 0) {
      llvm::errs() << "Failed to get size of tensor: " << memref->name()->str()
                   << "\n";
      return;
    }
    auto cudaStatus = cudaMalloc(reinterpret_cast<void **>(&devicePtr), size);
    if (cudaStatus != 0) {
      llvm::errs() << "cudaMalloc failed for tensor " << memref->name()->str()
                   << " with error code " << cudaStatus << "\n";
    }
    tensorMap.insert({memref->name()->str(), devicePtr});
    memrefDescMap.insert({memref->name()->str(), memref});

    if (memref->name()->str().find("%arg") != std::string::npos) {

      size_t i = std::stoi(memref->name()->str().substr(4));
      cudaMemcpy(devicePtr, programInputs[i].data.get(), dim * sizeof(float),
                 cudaMemcpyHostToDevice);
    }

    if (memref->value()->str().size() > 0) {
      if (memref->type()->str().find("f32") != std::string::npos) {
        float value = std::stof(memref->value()->str());
        cudaMemcpy(devicePtr, &value, sizeof(float), cudaMemcpyHostToDevice);
      } else if (memref->type()->str().find("i32") != std::string::npos) {
        int32_t value = std::stoi(memref->value()->str());
        cudaMemcpy(devicePtr, &value, sizeof(int32_t), cudaMemcpyHostToDevice);
      } else if (memref->type()->str().find("i64") != std::string::npos) {
        int64_t value = std::stoll(memref->value()->str());
        cudaMemcpy(devicePtr, &value, sizeof(int64_t), cudaMemcpyHostToDevice);
      } else {
        llvm::errs() << "Unsupported tensor type: " << memref->type()->str()
                     << "\n";
      }
    }
  }

  for (const auto *kernel : *program->kernels()) {
    runKernel(kernel);
  }
  void *returnPtr = nullptr;
  void *returnTensor = tensorMap[program->return_variable()->str()];
  cudaMemcpy(returnPtr, returnTensor, sizeof(float), cudaMemcpyDeviceToHost);
}

void ProgramExecutor::runKernel(const ::gpu::Kernel *kernel) {

  auto kernelArgs = std::make_unique<void *[]>(kernel->input_names()->size());
  size_t i = 0;
  for (const auto *arg : *kernel->input_names()) {
    if (tensorMap.count(arg->str()) == 0) {
      llvm::errs() << "Tensor not found: " << arg->str() << "\n";
      return;
    }
    void *devicePtr = tensorMap[arg->str()];
    if (memrefDescMap[arg->str()]->value()->str().size() > 0) {
      kernelArgs[i] = devicePtr;
    } else {
      kernelArgs[i] = &devicePtr;
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

  cuLaunchKernel(function, gridSize.x, gridSize.y, gridSize.z, blockSize.x,
                 blockSize.y, blockSize.z, 0, 0, kernelArgs.get(), nullptr);
  cudaDeviceSynchronize();
}

} // namespace tt::runtime::cuda
