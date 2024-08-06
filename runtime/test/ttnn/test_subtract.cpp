// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt/runtime/runtime.h"
#include "tt/runtime/utils.h"
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <string>
TEST(TTNNSubtract, Equal) {
  const char *fbPath = std::getenv("TTMLIR_SUBTRACT_FB_PATH");
  assert(fbPath && "Path to subtract flatbuffer must be provided");
  ::tt::runtime::Binary fbb = ::tt::runtime::Binary::loadFromPath(fbPath);
  EXPECT_EQ(fbb.getFileIdentifier(), "TTNN");
  ::tt::runtime::setCompatibleRuntime(fbb);
  std::vector<::tt::runtime::TensorDesc> inputDescs = fbb.getProgramInputs(0);
  std::vector<::tt::runtime::TensorDesc> outputDescs = fbb.getProgramOutputs(0);
  std::vector<::tt::runtime::Tensor> inputTensors, outputTensors;

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
  for (const auto &desc : outputDescs) {
    std::shared_ptr<void> data =
        ::tt::runtime::utils::malloc_shared(tensorSize);
    // Set to wrong value on purpose here
    std::memset(data.get(), 1, tensorSize);
    outputTensors.emplace_back(::tt::runtime::createTensor(data, desc));
  }

  auto device = ::tt::runtime::openDevice();
  auto ev = ::tt::runtime::submit(device, fbb, 0, inputTensors, outputTensors);
  ::tt::runtime::closeDevice(device);

  std::shared_ptr<void> expected =
      ::tt::runtime::utils::malloc_shared(tensorSize);
  std::memset(expected.get(), 0, tensorSize);
  for (const auto &outputTensor : outputTensors) {
    EXPECT_EQ(std::memcmp(outputTensor.data.get(), expected.get(), tensorSize),
              0);
  }
}
