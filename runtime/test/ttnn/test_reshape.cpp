// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <filesystem>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/utils.h"

#ifndef TT_RUNTIME_ENABLE_TTNN
#error "TT_RUNTIME_ENABLE_TTNN must be defined"
#endif

TEST(TTNNReshape, IdentityData) {
  const char *fbPath = std::getenv("TTMLIR_RESHAPE_FB_PATH");
  assert(fbPath && "Path to reshape flatbuffer must be provided");
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

  /* Reshape has single input */
  std::shared_ptr<void> inData = ::tt::runtime::utils::malloc_shared(tensorSize);
  /* Fill with sequential pattern */
  for (std::uint32_t i = 0; i < tensorSize; ++i) {
    static_cast<uint8_t *>(inData.get())[i] = static_cast<uint8_t>(i % 255);
  }
  inputTensors.emplace_back(::tt::runtime::createTensor(inData, inputDescs[0]));

  for (const auto &desc : outputDescs) {
    std::shared_ptr<void> data = ::tt::runtime::utils::malloc_shared(tensorSize);
    // Initialize with zeros so we can verify it gets overwritten.
    std::memset(data.get(), 0, tensorSize);
    outputTensors.emplace_back(::tt::runtime::createTensor(data, desc));
  }

  size_t numDevices = ::tt::runtime::getNumAvailableDevices();
  std::vector<int> deviceIds(numDevices);
  std::iota(deviceIds.begin(), deviceIds.end(), 0);
  auto device = ::tt::runtime::openDevice(deviceIds);
  auto ev = ::tt::runtime::submit(device, fbb, 0, inputTensors, outputTensors);
  ::tt::runtime::closeDevice(device);

  // Verify that output data matches input data byte-for-byte.
  for (const auto &outputTensor : outputTensors) {
    EXPECT_EQ(std::memcmp(outputTensor.data.get(), inData.get(), tensorSize), 0);
  }
} 