// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#if !defined(TT_RUNTIME_DEBUG) || !TT_RUNTIME_DEBUG
#error "TT_RUNTIME_DEBUG must be defined and set"
#endif

#include "tt/runtime/ttnn/debug_apis.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::debug {

static std::string toString(::ttnn::Layout layout) {
  switch (layout) {
  case ::ttnn::Layout::ROW_MAJOR:
    return "ROW_MAJOR";
  case ::ttnn::Layout::TILE:
    return "TILE";
  case ::ttnn::Layout::INVALID:
    return "INVALID";
  }
}

static std::string toString(::ttnn::DataType dtype) {
  switch (dtype) {
  case ::ttnn::DataType::FLOAT32:
    return "FLOAT32";
  case ::ttnn::DataType::BFLOAT16:
    return "BFLOAT16";
  case ::ttnn::DataType::BFLOAT8_B:
    return "BFLOAT8_B";
  case ::ttnn::DataType::BFLOAT4_B:
    return "BFLOAT4_B";
  case ::ttnn::DataType::UINT32:
    return "UINT32";
  case ::ttnn::DataType::UINT16:
    return "UINT16";
  case ::ttnn::DataType::UINT8:
    return "UINT8";
  case ::ttnn::DataType::INT32:
    return "INT32";
  case ::ttnn::DataType::INVALID:
    return "INVALID";
  }
}

void checkTensorRefMatchesTTNNTensor(
    const ::tt::target::ttnn::TensorRef *tensorRef,
    const ::ttnn::Tensor &ttnnTensor) {
  ::ttnn::Layout expectedLayout =
      ::tt::runtime::ttnn::utils::inferLayoutFromTileShape(tensorRef);
  ::ttnn::Layout actualLayout = ttnnTensor.get_layout();
  DEBUG_ASSERT(expectedLayout == actualLayout, "Layout mismatch, expected ",
               toString(expectedLayout), ", got ", toString(actualLayout));

  ::ttnn::DataType expectedDataType =
      ::tt::runtime::ttnn::utils::toTTNNDataType(
          tensorRef->desc()->layout()->memory_desc()->data_type());
  ::ttnn::DataType actualDataType = ttnnTensor.get_dtype();
  DEBUG_ASSERT(expectedDataType == actualDataType,
               "DataType mismatch, expected ", toString(expectedDataType),
               ", got ", toString(actualDataType));

  // TODO (jnie): Compare storage once we correctly determine it in the
  // flatbuffer. This requires compiler support which is missing.
  //
  // ::ttnn::StorageType expectedStorageType =
  //     ::tt::runtime::ttnn::utils::toTTNNStorageType(
  //         tensorRef->desc()->layout()->memory_desc()->storage_type());
  // ::ttnn::StorageType actualStorageType =
  //     ttnnTensor.storage_type();
  // DEBUG_ASSERT(expectedStorageType == actualStorageType, "Storage type
  // mismatch, expected ", static_cast<int>(expectedStorageType), ", got ",
  // static_cast<int>(actualStorageType));

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
