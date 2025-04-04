// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>
#include <string>

#include <gtest/gtest.h>

#include "tt/runtime/runtime.h"
#include "tt/runtime/utils.h"

#ifndef TT_RUNTIME_ENABLE_TTNN
#error "TT_RUNTIME_ENABLE_TTNN must be defined"
#endif

TEST(TTNNSubtract, Equal) {
  const char *fbPath = std::getenv("TTMLIR_SUBTRACT_FB_PATH");
  assert(fbPath && "Path to subtract flatbuffer must be provided");
  ::tt::runtime::Binary fbb = ::tt::runtime::Binary::loadFromPath(fbPath);
  ASSERT_EQ(fbb.getFileIdentifier(), "TTNN");
  ::tt::runtime::setCompatibleRuntime(fbb);
  std::vector<::tt::runtime::TensorDesc> inputDescs = fbb.getProgramInputs(0);
  assert(inputDescs.size() == 2);
  std::vector<::tt::runtime::TensorDesc> outputDescs = fbb.getProgramOutputs(0);
  assert(outputDescs.size() == 1);
  std::vector<::tt::runtime::Tensor> inputTensors;
  std::uint32_t tensorSize = inputDescs[0].itemsize;
  for (const int dim : inputDescs[0].shape) {
    tensorSize *= dim;
  }

  for (const auto &desc : inputDescs) {
    std::shared_ptr<void> data =
        ::tt::runtime::utils::malloc_shared(tensorSize);
    std::memset(data.get(), 1, tensorSize);
    inputTensors.emplace_back(::tt::runtime::createTensor(data, desc));
  }

  std::shared_ptr<void> outputDataPtr =
      ::tt::runtime::utils::malloc_shared(tensorSize);
  // Set to wrong value on purpose here
  std::memset(outputDataPtr.get(), 1, tensorSize);
  ::tt::runtime::Tensor outputTensor =
      ::tt::runtime::createTensor(outputDataPtr, outputDescs[0]);

  size_t numDevices = ::tt::runtime::getNumAvailableDevices();
  std::vector<int> deviceIds(numDevices);
  std::iota(deviceIds.begin(), deviceIds.end(), 0);
  auto device = ::tt::runtime::openDevice({deviceIds[0]});
  std::vector<::tt::runtime::Tensor> output =
      ::tt::runtime::submit(device, fbb, 0, inputTensors);
  ::tt::runtime::closeDevice(device);
  assert(output.size() == 1);
  std::shared_ptr<void> expected =
      ::tt::runtime::utils::malloc_shared(tensorSize);
  std::memset(expected.get(), 0, tensorSize);
  ::tt::runtime::Tensor submitOutput = output[0];
  ASSERT_NE(std::memcmp(outputDataPtr.get(), expected.get(), tensorSize), 0);
  ::tt::runtime::memcpy(outputTensor, submitOutput);
  ASSERT_EQ(std::memcmp(outputDataPtr.get(), expected.get(), tensorSize), 0);
}
