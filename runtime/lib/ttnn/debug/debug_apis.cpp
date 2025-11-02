// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#if !defined(TT_RUNTIME_DEBUG) || !TT_RUNTIME_DEBUG
#error "TT_RUNTIME_DEBUG must be defined and set"
#endif

#include "tt/runtime/detail/ttnn/debug_apis.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::debug {

void checkTensorRefMatchesTTNNTensor(
    const ::tt::target::ttnn::TensorRef *tensorRef,
    const ::ttnn::Tensor &ttnnTensor) {
  ::ttnn::Layout expectedLayout =
      ::tt::runtime::ttnn::utils::inferLayoutFromTileShape(tensorRef);
  ::ttnn::Layout actualLayout = ttnnTensor.layout();
  DEBUG_ASSERT(expectedLayout == actualLayout, "Layout mismatch, expected ",
               toString(expectedLayout), ", got ", toString(actualLayout));

  //   ::ttnn::DataType expectedDataType =
  //       ::tt::runtime::ttnn::utils::toTTNNDataType(
  //           tensorRef->desc()->layout()->memory_desc()->data_type());
  //   ::ttnn::DataType actualDataType = ttnnTensor.dtype();
  //   DEBUG_ASSERT(expectedDataType == actualDataType,
  //                "DataType mismatch, expected ", toString(expectedDataType),
  //                ", got ", toString(actualDataType));

  ::ttnn::StorageType expectedStorageType =
      ::tt::runtime::ttnn::utils::toTTNNStorageType(
          tensorRef->desc()->layout()->memory_desc()->storage_type());
  ::ttnn::StorageType actualStorageType = ttnnTensor.storage_type();

  // With TT-Mesh backend, single device tensors are also
  // MULTI_DEVICE_HOST_STORAGE Therefore, for host tensors we loosen the
  // constraint a bit and just check that the storage type is on host
  if (utils::isOnHost(expectedStorageType)) {
    DEBUG_ASSERT(
        utils::isOnHost(actualStorageType), "Storage type mismatch, expected ",
        toString(expectedStorageType), ", got ", toString(actualStorageType));
  } else {
    DEBUG_ASSERT(expectedStorageType == actualStorageType,
                 "Storage type mismatch, expected ",
                 toString(expectedStorageType), ", got ",
                 toString(actualStorageType));
  }

  if (!::tt::runtime::ttnn::utils::inSystemMemory(tensorRef)) {
    const ::tt::target::ttnn::MemoryConfig *memcfg =
        ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(tensorRef);
    DEBUG_ASSERT(memcfg, "Device tensor must have memory config");
    ::ttnn::MemoryConfig expectedMemoryConfig =
        ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(memcfg).value();
    ::ttnn::MemoryConfig actualMemoryConfig = ttnnTensor.memory_config();
    DEBUG_ASSERT(expectedMemoryConfig == actualMemoryConfig,
                 "Memory config mismatch, expected ", expectedMemoryConfig,
                 ", got ", actualMemoryConfig);
  }
}

} // namespace tt::runtime::ttnn::debug
