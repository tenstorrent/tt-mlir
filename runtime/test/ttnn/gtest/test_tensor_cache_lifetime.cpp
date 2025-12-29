// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/types/global_tensor_cache.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using tt::runtime::CacheKey;

class TensorCacheLifetimeTest : public ::testing::Test {
protected:
  void SetUp() override {
    tt::runtime::setCurrentDeviceRuntime(tt::runtime::DeviceRuntime::TTNN);
    tt::runtime::GlobalTensorCache::getInstance().clear();
  }

  void TearDown() override {
    tt::runtime::GlobalTensorCache::getInstance().clear();
  }

  tt::runtime::Tensor createTestTensor() {
    std::vector<uint32_t> shape = {2, 2};
    std::vector<uint32_t> stride = tt::runtime::utils::calculateStride(shape);
    tt::target::DataType dataType = tt::target::DataType::Float32;
    uint32_t itemSize = sizeof(float);

    std::vector<float> testData = {1.0f, 2.0f, 3.0f, 4.0f};

    return tt::runtime::createOwnedHostTensor(testData.data(), shape, stride,
                                              itemSize, dataType);
  }

  std::vector<tt::runtime::Tensor> createDummyOutputs() {
    return {createTestTensor()};
  }
};

TEST_F(TensorCacheLifetimeTest, CacheEntryRemovedOnTensorDestruction) {
  auto &cache = tt::runtime::GlobalTensorCache::getInstance();

  CacheKey key{/*deviceId=*/0, "test_func_hash", {}};

  // Scope to limit input tensor lifetime.
  {
    tt::runtime::Tensor tensor = createTestTensor();
    auto &wrapper = tensor.as<tt::runtime::ttnn::TTNNTensorWrapper>(
        tt::runtime::DeviceRuntime::TTNN);

    key.inputVersions = {wrapper.getVersion()};

    std::vector<tt::runtime::Tensor> inputs = {tensor};
    cache.store(key, inputs, createDummyOutputs());

    // Verify entry exists
    const std::vector<tt::runtime::Tensor> *cachedOutputs = cache.getAll(key);
    EXPECT_NE(cachedOutputs, nullptr) << "Cache entry should exist after store";

    // tensor goes out of scope here - callbacks registered by the `store()`
    // method should trigger to remove the cache entry.
  }

  // Verify that the cache entry was removed after tensor destruction.
  const std::vector<tt::runtime::Tensor> *cachedOutputs = cache.getAll(key);
  EXPECT_EQ(cachedOutputs, nullptr)
      << "Cache entry should be removed after tensor destruction";
}

TEST_F(TensorCacheLifetimeTest, CacheSanitySameProgramDifferentTensors) {
  auto &cache = tt::runtime::GlobalTensorCache::getInstance();

  // Create tensor A and store cache entry associated with it.
  tt::runtime::Tensor tensorA = createTestTensor();
  auto &wrapperA = tensorA.as<tt::runtime::ttnn::TTNNTensorWrapper>(
      tt::runtime::DeviceRuntime::TTNN);
  uint64_t versionA = wrapperA.getVersion();

  CacheKey keyA{/*deviceId=*/0, "test_func_hash", {versionA}};
  std::vector<tt::runtime::Tensor> inputsA = {tensorA};
  cache.store(keyA, inputsA, createDummyOutputs());

  CacheKey keyB{/*deviceId=*/0, "test_func_hash", {}};

  // Verify entry exists.
  const std::vector<tt::runtime::Tensor> *cachedOutputs = cache.getAll(keyA);
  EXPECT_NE(cachedOutputs, nullptr) << "Cache entry should exist after store";

  uint64_t versionB;
  // Create a new tensor B - which should have a different version than tensor
  // A.
  {
    tt::runtime::Tensor tensorB = createTestTensor();
    auto &wrapperB = tensorB.as<tt::runtime::ttnn::TTNNTensorWrapper>(
        tt::runtime::DeviceRuntime::TTNN);
    versionB = wrapperB.getVersion();

    EXPECT_NE(versionA, versionB) << "Tensor versions should be different";

    // Store with a different key (different version)
    keyB.inputVersions = {versionB};
    std::vector<tt::runtime::Tensor> inputsB = {tensorB};
    cache.store(keyB, inputsB, createDummyOutputs());

    EXPECT_NE(cache.getAll(keyA), nullptr);
    EXPECT_NE(cache.getAll(keyB), nullptr)
        << "Cache entry with keyB should exist after store";
    // tensorB goes out of scope here - should remove keyB, not keyA.
  }

  EXPECT_EQ(cache.getAll(keyB), nullptr)
      << "Cache entry with keyB should be removed when tensorB is destroyed";

  // Verify original cache entry still exists (different key).
  cachedOutputs = cache.getAll(keyA);
  EXPECT_NE(cachedOutputs, nullptr)
      << "Cache entry with keyA should still exist when tensorB is destroyed";
}

TEST_F(TensorCacheLifetimeTest, MultipleInputTensorsRemoveCacheEntry) {
  auto &cache = tt::runtime::GlobalTensorCache::getInstance();

  tt::runtime::Tensor tensor1 = createTestTensor();
  tt::runtime::Tensor tensor2 = createTestTensor();

  auto &wrapper1 = tensor1.as<tt::runtime::ttnn::TTNNTensorWrapper>(
      tt::runtime::DeviceRuntime::TTNN);
  auto &wrapper2 = tensor2.as<tt::runtime::ttnn::TTNNTensorWrapper>(
      tt::runtime::DeviceRuntime::TTNN);

  uint64_t version1 = wrapper1.getVersion();
  uint64_t version2 = wrapper2.getVersion();

  CacheKey key{/*deviceId=*/0, "test_func_hash", {version1, version2}};

  // Store entry with both tensors as inputs.
  {
    std::vector<tt::runtime::Tensor> inputs = {tensor1, tensor2};
    cache.store(key, inputs, createDummyOutputs());
  }

  // Verify entry exists.
  const std::vector<tt::runtime::Tensor> *cachedOutputs = cache.getAll(key);
  EXPECT_NE(cachedOutputs, nullptr) << "Cache entry should exist after store";

  // Destroy only the first tensor.
  {
    tt::runtime::Tensor temp = std::move(tensor1);
    // `temp` goes out of scope hence destroying the original tensor.
    // This should trigger the callback and remove the cache entry.
  }

  // Verify cache entry was removed (one of the input tensors was destroyed).
  cachedOutputs = cache.getAll(key);
  EXPECT_EQ(cachedOutputs, nullptr)
      << "Cache entry should be removed when any input tensor is destroyed";
}
