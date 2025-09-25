// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/cuda/ttcuda.h"
#include "types_generated.h"
#include <cstdint>
#ifdef TTMLIR_ENABLE_CUDA

#include "tt/runtime/detail/cuda/program_executor.h"
#include "tt/runtime/types.h"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/CUDA/program_generated.h"
#pragma clang diagnostic pop

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <flatbuffers/flatbuffers.h>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace tt::runtime::cuda {

class CudaProgramExecutorTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Check if CUDA is available
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "CUDA not available, skipping CUDA tests";
    }
    // tt::runtime::setCurrentRuntime(DeviceRuntime::CUDA);
  }

  // Helper to compile MLIR file to CUDA flatbuffer using compiler pipeline.
  std::shared_ptr<void>
  compileMlirToCudaFlatbuffer(const std::string &mlirFilePath) {
    std::string testDir = std::string(__FILE__);
    testDir = testDir.substr(0, testDir.find_last_of("/"));
    std::string fullMlirPath = testDir + "/" + mlirFilePath;

    std::string projectRoot = testDir;
    // Go up 4 levels: gtest -> cuda -> test -> runtime -> root
    for (int i = 0; i < 4; ++i) {
      size_t pos = projectRoot.find_last_of("/");
      if (pos != std::string::npos) {
        projectRoot = projectRoot.substr(0, pos);
      }
    }

    // Step 1: Convert TTIR to NVVM using ttmlir-opt:
    std::string ttmlirOptPath = projectRoot + "/build/bin/ttmlir-opt";
    std::string nvvmCommand =
        ttmlirOptPath + " --convert-ttir-to-nvvm " + fullMlirPath;

    FILE *nvvmPipe = popen(nvvmCommand.c_str(), "r");
    if (!nvvmPipe) {
      return nullptr;
    }
    std::string nvvmOutput;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), nvvmPipe) != nullptr) {
      nvvmOutput += buffer;
    }
    pclose(nvvmPipe);

    if (nvvmOutput.empty()) {
      return nullptr;
    }

    // Step 2: Convert NVVM to flatbuffer using ttmlir-translate:
    std::string tempNvvmFile = "/tmp/test_nvvm.mlir";
    std::ofstream nvvmFile(tempNvvmFile);
    nvvmFile << nvvmOutput;
    nvvmFile.close();

    std::string ttmlirTranslatePath =
        projectRoot + "/build/bin/ttmlir-translate";
    std::string flatbufferCommand =
        ttmlirTranslatePath + " --ptx-to-flatbuffer " + tempNvvmFile;

    FILE *fbPipe = popen(flatbufferCommand.c_str(), "r");
    if (!fbPipe) {
      return nullptr;
    }

    std::vector<uint8_t> fbData;
    int byte;
    while ((byte = fgetc(fbPipe)) != EOF) {
      fbData.push_back(static_cast<uint8_t>(byte));
    }
    pclose(fbPipe);

    std::remove(tempNvvmFile.c_str());

    if (fbData.empty()) {
      return nullptr;
    }
    std::shared_ptr<void> bufferPtr =
        std::shared_ptr<void>(std::malloc(fbData.size()), std::free);
    std::memcpy(bufferPtr.get(), fbData.data(), fbData.size());
    return bufferPtr;
  }

  // Helper to create a simple mock flatbuffer for tests that don't need real
  // compilation.
  std::shared_ptr<void> createMockEmptyProgram() {
    flatbuffers::FlatBufferBuilder fbb;
    auto program = ::tt::target::cuda::CreateProgram(fbb);
    fbb.FinishSizePrefixed(program);

    uint8_t *buf = fbb.GetBufferPointer();
    std::size_t size = fbb.GetSize();
    std::shared_ptr<void> bufferPtr =
        std::shared_ptr<void>(std::malloc(size), std::free);
    std::memcpy(bufferPtr.get(), buf, size);

    return bufferPtr;
  }

  // Helper to create test input tensors.
  std::vector<::tt::runtime::Tensor>
  createTestInputs(std::vector<uint32_t> dimensions) {
    std::vector<::tt::runtime::Tensor> inputs;

    for (uint32_t i = 0; i < dimensions.size(); i++) {
      std::vector<float> data;
      float ctr = dimensions[i] / 2.0f;
      ctr *= -1.0f;
      for (uint32_t j = 0; j < dimensions[i]; j++) {
        float element = std::max(-10.0f, (ctr / 100.0f));
        data.push_back(i + std::min(element, 10.0f));
        ctr += 1.0f;
      }
      ::tt::runtime::Tensor input = ::tt::runtime::cuda::createOwnedHostTensor(
          data.data(), dimensions, dimensions, sizeof(float),
          ::tt::target::DataType::Float32);
      inputs.push_back(input);
    }

    return inputs;
  }
};

TEST_F(CudaProgramExecutorTest, ConstructorInitialization) {
  // Test that constructor doesn't crash and initializes properly.
  auto programBuffer = createMockEmptyProgram();
  ::tt::runtime::Binary binary(programBuffer);
  ::tt::runtime::Device device = ::tt::runtime::cuda::openMeshDevice({});
  std::vector<::tt::runtime::Tensor> inputs;
  EXPECT_NO_THROW({ ProgramExecutor executor(device, binary, inputs); });
  ::tt::runtime::cuda::closeMeshDevice(device);
}

TEST_F(CudaProgramExecutorTest, CompileAndExecuteRealMlir) {
  // Test: [1.0, -2.0] -> ReLU -> [1.0, 0.0].
  auto programBuffer = compileMlirToCudaFlatbuffer("test_simple.mlir");
  if (!programBuffer) {
    GTEST_SKIP() << "MLIR compilation failed, skipping real compilation test";
  }

  ::tt::runtime::Binary binary(programBuffer);
  std::vector<::tt::runtime::Tensor> inputs;

  std::vector<float> data1 = {1.0f, -2.0f};
  ::tt::runtime::Tensor input = ::tt::runtime::cuda::createOwnedHostTensor(
      data1.data(), {2}, {0}, sizeof(float), ::tt::target::DataType::Float32);

  inputs.push_back(input);

  ::tt::runtime::Device device = ::tt::runtime::cuda::openMeshDevice({});
  ProgramExecutor executor(device, binary, inputs);
  ::tt::runtime::Tensor resultTensor = executor.execute();
  float *result = reinterpret_cast<float *>(resultTensor.data.get());
  EXPECT_FLOAT_EQ(result[0], 1.0f);
  EXPECT_FLOAT_EQ(result[1], 0.0f);
  ::tt::runtime::cuda::closeMeshDevice(device);
}

TEST_F(CudaProgramExecutorTest, CompileAndExecuteVectorAdd) {
  // Test operation on multidimensional tensors.
  auto programBuffer = compileMlirToCudaFlatbuffer("test.mlir");
  if (!programBuffer) {
    GTEST_SKIP() << "MLIR compilation failed, skipping vector add test";
  }

  ::tt::runtime::Binary binary(programBuffer);
  std::vector<uint32_t> inputSizes;
  inputSizes.push_back(40);
  inputSizes.push_back(40);
  auto inputs = createTestInputs(inputSizes);

  ::tt::runtime::Device device = ::tt::runtime::cuda::openMeshDevice({});
  ProgramExecutor executor(device, binary, inputs);
  ::tt::runtime::Tensor resultTensor = executor.execute();
  float *result = reinterpret_cast<float *>(resultTensor.data.get());
  for (int i = 0; i < 40; i++) {
    float element = (-0.2f + i / 100.f) * 2.0f + 1.0f;
    EXPECT_FLOAT_EQ(result[i], element);
  }
  ::tt::runtime::cuda::closeMeshDevice(device);
}

TEST_F(CudaProgramExecutorTest, CompileAndExecuteMultipleOperations) {
  // Test program with multiple operations.
  auto programBuffer = compileMlirToCudaFlatbuffer("test_complex.mlir");
  if (!programBuffer) {
    GTEST_SKIP() << "MLIR compilation failed, skipping vector add test";
  }
  ::tt::runtime::Binary binary(programBuffer);
  std::vector<uint32_t> inputSizes;
  inputSizes.push_back(784);
  inputSizes.push_back(784 * 512);
  inputSizes.push_back(512);
  inputSizes.push_back(512 * 512);
  inputSizes.push_back(512);
  inputSizes.push_back(5120);
  inputSizes.push_back(10);
  auto inputs = createTestInputs(inputSizes);
  EXPECT_NO_THROW({
    ::tt::runtime::Device device = ::tt::runtime::cuda::openMeshDevice({});
    ProgramExecutor executor(device, binary, inputs);
    executor.execute();
    ::tt::runtime::cuda::closeMeshDevice(device);
  });
}

TEST_F(CudaProgramExecutorTest, ExecuteEmptyProgram) {
  // Test empty program.
  auto programBuffer = createMockEmptyProgram();

  ::tt::runtime::Binary binary(programBuffer);
  std::vector<::tt::runtime::Tensor> inputs;

  ::tt::runtime::Device device = ::tt::runtime::cuda::openMeshDevice({});
  ProgramExecutor executor(device, binary, inputs);
  EXPECT_NO_THROW({ executor.execute(); });
  ::tt::runtime::cuda::closeMeshDevice(device);
}

TEST_F(CudaProgramExecutorTest, HandleEmptyFlatbuffer) {
  // Test with empty flatbuffer data.
  std::shared_ptr<void> emptyBuffer =
      std::shared_ptr<void>(std::malloc(1), std::free);
  EXPECT_NO_THROW({
    ::tt::runtime::Binary binary(emptyBuffer);
    std::vector<::tt::runtime::Tensor> inputs;
    ::tt::runtime::Device device = ::tt::runtime::cuda::openMeshDevice({});
    ProgramExecutor executor(device, binary, inputs);
    ::tt::runtime::cuda::closeMeshDevice(device);
  });
}

TEST_F(CudaProgramExecutorTest, HandleInvalidInput) {
  // Test program insufficient input.
  auto programBuffer = compileMlirToCudaFlatbuffer("test.mlir");
  if (!programBuffer) {
    GTEST_SKIP() << "MLIR compilation failed, skipping invalid input test";
  }

  ::tt::runtime::Binary binary(programBuffer);
  std::vector<::tt::runtime::Tensor> inputs;

  ::tt::runtime::Device device = ::tt::runtime::cuda::openMeshDevice({});
  ProgramExecutor executor(device, binary, inputs);
  EXPECT_NO_THROW({ executor.execute(); });
  ::tt::runtime::cuda::closeMeshDevice(device);
}

} // namespace tt::runtime::cuda

#endif // TTMLIR_ENABLE_CUDA
