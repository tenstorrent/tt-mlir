// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "to_memory_config.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::layout {

static ::ttnn::Tensor tilize(::ttnn::Tensor const &input) {
  // NOLINTNEXTLINE
  return ::ttnn::to_layout(input, ::ttnn::TILE_LAYOUT, std::nullopt,
                           std::nullopt,
                           static_cast<::ttnn::Device *>(nullptr));
}

static ::ttnn::Tensor untilize(::ttnn::Tensor const &input) {
  return ::ttnn::to_layout(input, ::ttnn::ROW_MAJOR_LAYOUT, std::nullopt,
                           std::nullopt,
                           static_cast<::ttnn::Device *>(nullptr));
}

static ::ttnn::Tensor convertDataType(const ::ttnn::Tensor &input,
                                      const ::ttnn::DataType &targetDataType) {
  if (utils::isOnHost(input)) {
    return ::ttnn::to_dtype(input, targetDataType);
  }

  if (utils::isOnDevice(input)) {
    // TODO (bug #272): right now hardcoding tilize/untilize
    // should determine layout with tile shape instead of calling a getter
    // function
    if (input.get_layout() != ::ttnn::TILE_LAYOUT) {
      // typecast op requires tilized tensor
      ::ttnn::Tensor converted =
          ::ttnn::typecast(tilize(input), targetDataType);
      // untilize and return
      return untilize(converted);
    }
    return ::ttnn::typecast(input, targetDataType);
  }

  throw std::runtime_error("Unsupported storage type");
}

/* TODO(bug #272), ideal flow is to determine tilize/untilize with
 * tile_shape */
static ::ttnn::Tensor
updateLayoutAndDataType(const ::ttnn::Tensor &inputTensor,
                        const ::ttnn::DataType &inputDataType,
                        const ::ttnn::DataType &targetDataType,
                        const bool shouldTilize, const bool shouldUntilize) {

  ::ttnn::Tensor outputTensor = inputTensor;
  const bool shouldConvertDataType = (inputDataType != targetDataType);
  // const int targetTileX = targetTileShape->x();
  // const int targetTileY = targetTileShape->y();
  // const bool shouldTilize =
  //     targetTileX == 32 and targetTileY == 32;
  // const bool shouldUntilize = (targetTileX != 32 or targetTileY != 32);
  assert(not(shouldTilize and shouldUntilize) &&
         "Cannot tilize and untilize tensor at the same time");
  if (shouldTilize) {
    outputTensor = tilize(outputTensor);
  } else if (shouldUntilize) {
    outputTensor = untilize(outputTensor);
  }
  if (shouldConvertDataType) {
    outputTensor = convertDataType(outputTensor, targetDataType);
  }
  return outputTensor;
}

static void
handleToHostMemoryConfigOp(const ::tt::target::TensorRef *inputTensorRef,
                           const ::tt::target::TensorRef *outputTensorRef,
                           ProgramTensorPool &tensorPool) {
  ::ttnn::Tensor result;
  const ::ttnn::Tensor &inputTensor =
      tensorPool.at(inputTensorRef->global_id());
  ::ttnn::DataType inputDataTypeTTNN = utils::getDataType(inputTensorRef);
  ::ttnn::DataType targetDataTypeTTNN = utils::getDataType(outputTensorRef);
  bool shouldTilize, shouldUntilize;
  if (utils::isOnHost(inputTensor)) {
    shouldTilize = false;
    shouldUntilize = true;
    result = updateLayoutAndDataType(inputTensor, inputDataTypeTTNN,
                                     targetDataTypeTTNN, shouldTilize,
                                     shouldUntilize);
  } else if (utils::isOnDevice(inputTensor)) {
    shouldTilize = false;
    shouldUntilize = true;
    result = updateLayoutAndDataType(inputTensor.cpu(), inputDataTypeTTNN,
                                     targetDataTypeTTNN, shouldTilize,
                                     shouldUntilize);
  }
  // copy the output to the output tensor if it exists
  if (tensorPool.contains(outputTensorRef->global_id())) {
    ::ttnn::Tensor &outputTensor = tensorPool.at(outputTensorRef->global_id());
    void *src = ::tt::tt_metal::get_raw_host_data_ptr(result);
    void *dst = ::tt::tt_metal::get_raw_host_data_ptr(outputTensor);
    std::uint32_t size = result.volume() * result.element_size();
    std::memcpy(dst, src, size);
  } else {
    tensorPool.insert_or_assign(outputTensorRef->global_id(), result);
  }
}

static void
handleToDramMemoryConfigOp(::ttnn::Device &device,
                           const ::tt::target::TensorRef *inputTensorRef,
                           const ::tt::target::TensorRef *outputTensorRef,
                           ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &inputTensor =
      tensorPool.at(inputTensorRef->global_id());
  ::ttnn::DataType inputDataTypeTTNN = utils::getDataType(inputTensorRef);
  ::ttnn::DataType targetDataTypeTTNN = utils::getDataType(outputTensorRef);
  ::tt::tt_metal::MemoryConfig targetMemoryConfig =
      utils::createMemoryConfig(outputTensorRef);
  bool shouldTilize, shouldUntilize;
  if (utils::isOnHost(inputTensor)) {
    ::ttnn::Tensor result = inputTensor;
    shouldTilize = true;
    shouldUntilize = false;
    // device tilize requires BFLOAT16, if not then tilize on host
    if (inputDataTypeTTNN != ::ttnn::DataType::BFLOAT16) {
      result = tilize(result);
      shouldTilize = false;
    }
    result = ::ttnn::to_device(result, &device, targetMemoryConfig);
    result =
        updateLayoutAndDataType(result, inputDataTypeTTNN, targetDataTypeTTNN,
                                shouldTilize, shouldUntilize);
    tensorPool.insert_or_assign(outputTensorRef->global_id(), result);
  } else if (utils::isOnDevice(inputTensor)) {
    shouldTilize = false;
    shouldUntilize = false;
    ::ttnn::Tensor result = updateLayoutAndDataType(
        inputTensor, inputDataTypeTTNN, targetDataTypeTTNN, shouldTilize,
        shouldUntilize);
    result = ::ttnn::to_memory_config(result, targetMemoryConfig, std::nullopt);
    tensorPool.insert_or_assign(outputTensorRef->global_id(), result);
  }
}

static void
handleToL1MemoryConfigOp(::ttnn::Device &device,
                         const ::tt::target::TensorRef *inputTensorRef,
                         const ::tt::target::TensorRef *outputTensorRef,
                         ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &inputTensor =
      tensorPool.at(inputTensorRef->global_id());
  ::ttnn::DataType inputDataTypeTTNN = utils::getDataType(inputTensorRef);
  ::ttnn::DataType targetDataTypeTTNN = utils::getDataType(outputTensorRef);
  ::tt::tt_metal::MemoryConfig targetMemoryConfig =
      utils::createMemoryConfig(outputTensorRef);
  bool shouldTilize, shouldUntilize;
  if (utils::isOnHost(inputTensor)) {
    ::ttnn::Tensor result = inputTensor;
    // device tilize requires BFLOAT16, if not then tilize on host
    if (inputDataTypeTTNN != ::ttnn::DataType::BFLOAT16) {
      result = tilize(result);
      result = ::ttnn::to_device(result, &device, targetMemoryConfig);
      shouldTilize = false;
      shouldUntilize = false;
      result =
          updateLayoutAndDataType(result, inputDataTypeTTNN, targetDataTypeTTNN,
                                  shouldTilize, shouldUntilize);
    } else {
      shouldTilize = true;
      shouldUntilize = false;
      // device tilize op requires height sharded or interleaved tensors
      // thus tilize first with default mem config, then convert memory config
      result = ::ttnn::to_device(result, &device, std::nullopt);
      result =
          updateLayoutAndDataType(result, inputDataTypeTTNN, targetDataTypeTTNN,
                                  shouldTilize, shouldUntilize);
      result =
          ::ttnn::to_memory_config(result, targetMemoryConfig, std::nullopt);
    }
    tensorPool.insert_or_assign(outputTensorRef->global_id(), result);
  } else if (utils::isOnDevice(inputTensor)) {
    shouldTilize = false;
    shouldUntilize = false;
    ::ttnn::Tensor result = updateLayoutAndDataType(
        inputTensor, inputDataTypeTTNN, targetDataTypeTTNN, shouldTilize,
        shouldUntilize);
    result = ::ttnn::to_memory_config(result, targetMemoryConfig, std::nullopt);
    tensorPool.insert_or_assign(outputTensorRef->global_id(), result);
  }
}

// TODO(bug #272): right now hardcoding tilize/untilize, should determine with
// tile shape blocked by issue #272
void run(const ::tt::target::ttnn::ToMemoryConfigOp *op,
         ProgramContext &context) {

  ProgramTensorPool &tensorPool = context.tensorPool;
  DeviceMap &devicePool = context.devicePool;
  const ::ttnn::Tensor &inputTensor = tensorPool.at(op->in0()->global_id());
  assert(utils::isOnHost(inputTensor) or
         utils::isOnDevice(inputTensor) && "Unsupported storage type");

  const ::tt::target::Dim2d *targetTileShape =
      op->out()->desc()->layout()->memory_desc()->tile_shape();
  assert(::tt::runtime::ttnn::utils::isValidTileShape(targetTileShape) &&
         "Invalid tile shape");

  const ::tt::target::MemorySpace targetMemorySpace =
      op->out()->desc()->layout()->memory_desc()->memory_space();

  switch (targetMemorySpace) {
  // This case should only be used when gathering outputs at the end of the
  // program
  case ::tt::target::MemorySpace::System:
  case ::tt::target::MemorySpace::SystemMMIO: {
    handleToHostMemoryConfigOp(op->in0(), op->out(), tensorPool);
    break;
  }
  case ::tt::target::MemorySpace::DeviceDRAM: {
    ::ttnn::Device &device = utils::getDevice(op->device(), devicePool);
    handleToDramMemoryConfigOp(device, op->in0(), op->out(), tensorPool);
    break;
  }
  case ::tt::target::MemorySpace::DeviceL1: {
    ::ttnn::Device &device = utils::getDevice(op->device(), devicePool);
    handleToL1MemoryConfigOp(device, op->in0(), op->out(), tensorPool);
    break;
  }
  }
}
} // namespace tt::runtime::ttnn::operations::layout
