// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <optional>
#include <string>
#include <unordered_map>

#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/runtime.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/device.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/pool/maxpool/max_pool2d.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "types_generated.h"
#include "utils.h"

#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Version.h"

namespace tt::runtime::ttnn {

class ProgramTensorPool {
public:
  ProgramTensorPool(
      std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &&liveTensors)
      : liveTensors(liveTensors) {}

  auto try_emplace(std::uint32_t global_id, ::ttnn::Tensor &&tensor) {
    auto it = liveTensors.find(global_id);
    if (it != liveTensors.end()) {
      return std::make_pair(it, false);
    }
    assert(!intermedTensors.contains(global_id));
    intermedTensors.try_emplace(global_id, tensor);
    return liveTensors.try_emplace(global_id, &intermedTensors.at(global_id));
  }

  auto insert_or_assign(std::uint32_t global_id, ::ttnn::Tensor &&tensor) {
    intermedTensors.insert_or_assign(global_id, tensor);
    return liveTensors.insert_or_assign(global_id,
                                        &intermedTensors.at(global_id));
  }

  ::ttnn::Tensor &at(std::uint32_t global_id) {
    assert(liveTensors.contains(global_id));
    return *liveTensors.at(global_id);
  }

  size_t erase(std::uint32_t global_id) {
    assert(liveTensors.contains(global_id) &&
           intermedTensors.contains(global_id));
    intermedTensors.erase(global_id);
    return liveTensors.erase(global_id);
  }

  bool contains(std::uint32_t global_id) const {
    return liveTensors.contains(global_id);
  }

private:
  // A superset of intermedTensors, containing all tensors created by the
  // program and the input/output tensors passed in by the user
  std::unordered_map<std::uint32_t, ::ttnn::Tensor *> liveTensors;

  // A subset of liveTensors, containing any intermediate tensors created by the
  // program
  std::unordered_map<std::uint32_t, ::ttnn::Tensor> intermedTensors;
};

static bool isOnHost(const ::ttnn::Tensor &tensor) {
  // Currently only supports borrowed or owned host storage
  return tensor.storage_type() == ::tt::tt_metal::StorageType::BORROWED or
         tensor.storage_type() == ::tt::tt_metal::StorageType::OWNED;
}

static bool isOnDevice(const ::ttnn::Tensor &tensor) {
  // Currently only supports single device storage
  return tensor.storage_type() == ::tt::tt_metal::StorageType::DEVICE;
}

static ::ttnn::DataType getDataType(const ::tt::target::TensorRef *tensorRef) {
  return utils::toTTNNDataType(
      tensorRef->desc()->layout()->memory_desc()->data_type());
}

static ::ttnn::Device &
getDevice(const ::tt::target::DeviceRef *deviceRef,
          std::unordered_map<uint32_t, ::ttnn::Device *> &devicePool) {
  uint32_t deviceId = deviceRef->global_id();
  assert(devicePool.contains(deviceId) && "Device not found in device pool");
  return *devicePool.at(deviceId);
}

static CoreRangeSet toCoreRangeSet(
    const ::flatbuffers::Vector<const tt::target::Dim2dRange *> *coreRangeSet) {
  std::set<CoreRange> coreRanges;
  for (::tt::target::Dim2dRange const *coreRange : *coreRangeSet) {
    CoreCoord start(coreRange->loc().x(), coreRange->loc().y());
    // End is inclusive
    CoreCoord end(coreRange->loc().x() + coreRange->size().x() - 1,
                  coreRange->loc().y() + coreRange->size().y() - 1);

    coreRanges.emplace(start, end);
  }
  return CoreRangeSet(coreRanges);
}

static ::tt::tt_metal::MemoryConfig
createMemoryConfig(const ::tt::target::TensorRef *tensorRef) {
  const ::tt::target::LayoutDesc *layout = tensorRef->desc()->layout();
  const ::tt::target::TensorMemoryLayout targetMemoryLayout =
      layout->memory_desc()->memory_layout();
  const ::tt::target::MemorySpace targetMemorySpace =
      layout->memory_desc()->memory_space();
  const ::flatbuffers::Vector<const tt::target::Dim2dRange *>
      *targetCoreRangeSet = layout->core_range_set();
  const ::flatbuffers::Vector<int32_t> *targetShardShape =
      layout->memory_desc()->shape();

  // TODO (jnie): Hardcoding to interleaved and block sharded for now
  // Add support for other types once compiler supports it
  assert(targetMemoryLayout == ::tt::target::TensorMemoryLayout::Interleaved ||
         targetMemoryLayout == ::tt::target::TensorMemoryLayout::BlockSharded);
  assert(targetMemoryLayout != target::TensorMemoryLayout::BlockSharded ||
         targetMemorySpace == target::MemorySpace::DeviceL1 &&
             "Only L1 memory space supports sharded memory layout");
  assert(targetCoreRangeSet->size() == 1 &&
         "Currently only single core range/grid is supported");
  assert(targetShardShape->size() == 2 &&
         "Only 2D shard shape is supported in TTNN backend");

  CoreRangeSet ttnnCoreRangeSet = toCoreRangeSet(targetCoreRangeSet);
  std::array<uint32_t, 2> ttnnShardShape;
  std::copy(targetShardShape->begin(), targetShardShape->end(),
            ttnnShardShape.begin());

  if (targetMemoryLayout == ::tt::target::TensorMemoryLayout::BlockSharded) {
    assert(ttnnShardShape[0] % ::tt::constants::TILE_HEIGHT == 0 &&
           ttnnShardShape[1] % ::tt::constants::TILE_WIDTH == 0 &&
           "Shard shape must divide tile shape (32, 32) evenly");
  }

  ::tt::tt_metal::ShardSpec shardSpec(
      ttnnCoreRangeSet, ttnnShardShape,
      ::tt::tt_metal::ShardOrientation::ROW_MAJOR, false);

  ::tt::tt_metal::TensorMemoryLayout ttnnMemLayout =
      utils::toTTNNTensorMemoryLayout(targetMemoryLayout);

  ::tt::tt_metal::BufferType ttnnBufferType =
      utils::toTTNNBufferType(targetMemorySpace);

  return {ttnnMemLayout, ttnnBufferType, shardSpec};
}

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
  if (isOnHost(input)) {
    return ::ttnn::to_dtype(input, targetDataType);
  }

  if (isOnDevice(input)) {
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
                        const ::ttnn::DataType targetDataType,
                        const bool shouldTilize, const bool shouldUntilize) {

  ::ttnn::Tensor outputTensor = inputTensor;
  const bool shouldConvertDataType = inputTensor.get_dtype() != targetDataType;
  // const int targetTileX = targetTileShape->x();
  // const int targetTileY = targetTileShape->y();
  // const bool shouldTilize =
  //     targetTileX == 32 and targetTileY == 32 and
  //     inputTensor.get_layout() == ::ttnn::ROW_MAJOR_LAYOUT;
  // const bool shouldUntilize = (targetTileX != 32 or targetTileY != 32) and
  //                             inputTensor.get_layout() ==
  //                             ::ttnn::TILE_LAYOUT;
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
handleToHostMemoryConfigOp(const ::ttnn::Tensor &inputTensor,
                           const ::tt::target::TensorRef *outputTensorRef,
                           ProgramTensorPool &tensorPool) {
  ::ttnn::Tensor result;
  ::ttnn::DataType targetDataTypeTTNN = getDataType(outputTensorRef);
  bool shouldTilize, shouldUntilize;
  if (isOnHost(inputTensor)) {
    shouldTilize = false;
    shouldUntilize = true;
    result = updateLayoutAndDataType(inputTensor, targetDataTypeTTNN,
                                     shouldTilize, shouldUntilize);
  } else if (isOnDevice(inputTensor)) {
    shouldTilize = false;
    shouldUntilize = true;
    result = updateLayoutAndDataType(inputTensor.cpu(), targetDataTypeTTNN,
                                     shouldTilize, shouldUntilize);
  }
  // copy the output to the output tensor if it exists
  if (tensorPool.contains(outputTensorRef->global_id())) {
    ::ttnn::Tensor &outputTensor = tensorPool.at(outputTensorRef->global_id());
    void *src = ::tt::tt_metal::get_raw_host_data_ptr(result);
    void *dst = ::tt::tt_metal::get_raw_host_data_ptr(outputTensor);
    std::uint32_t size = result.volume() * result.element_size();
    std::memcpy(dst, src, size);
  } else {
    tensorPool.insert_or_assign(outputTensorRef->global_id(),
                                std::move(result));
  }
}

static void
handleToDramMemoryConfigOp(::ttnn::Device &device,
                           const ::ttnn::Tensor &inputTensor,
                           const ::tt::target::TensorRef *outputTensorRef,
                           ProgramTensorPool &tensorPool) {
  ::ttnn::DataType targetDataTypeTTNN = getDataType(outputTensorRef);
  ::tt::tt_metal::MemoryConfig targetMemoryConfig =
      createMemoryConfig(outputTensorRef);
  bool shouldTilize, shouldUntilize;
  if (isOnHost(inputTensor)) {
    ::ttnn::Tensor result = inputTensor;
    shouldTilize = true;
    shouldUntilize = false;
    // device tilize requires BFLOAT16, if not then tilize on host
    if (result.get_dtype() != ::ttnn::DataType::BFLOAT16) {
      result = tilize(result);
      shouldTilize = false;
    }
    result = ::ttnn::to_device(result, &device, targetMemoryConfig);
    result = updateLayoutAndDataType(result, targetDataTypeTTNN, shouldTilize,
                                     shouldUntilize);
    tensorPool.insert_or_assign(outputTensorRef->global_id(),
                                std::move(result));
  } else if (isOnDevice(inputTensor)) {
    shouldTilize = false;
    shouldUntilize = false;
    ::ttnn::Tensor result = updateLayoutAndDataType(
        inputTensor, targetDataTypeTTNN, shouldTilize, shouldUntilize);
    result = ::ttnn::to_memory_config(result, targetMemoryConfig, std::nullopt);
    tensorPool.insert_or_assign(outputTensorRef->global_id(),
                                std::move(result));
  }
}

static void
handleToL1MemoryConfigOp(::ttnn::Device &device,
                         const ::ttnn::Tensor &inputTensor,
                         const ::tt::target::TensorRef *outputTensorRef,
                         ProgramTensorPool &tensorPool) {
  ::ttnn::DataType targetDataTypeTTNN = getDataType(outputTensorRef);
  ::tt::tt_metal::MemoryConfig targetMemoryConfig =
      createMemoryConfig(outputTensorRef);
  bool shouldTilize, shouldUntilize;
  if (isOnHost(inputTensor)) {
    ::ttnn::Tensor result = inputTensor;
    // device tilize requires BFLOAT16, if not then tilize on host
    if (result.get_dtype() != ::ttnn::DataType::BFLOAT16) {
      result = tilize(result);
      result = ::ttnn::to_device(result, &device, targetMemoryConfig);
      shouldTilize = false;
      shouldUntilize = false;
      result = updateLayoutAndDataType(result, targetDataTypeTTNN, shouldTilize,
                                       shouldUntilize);
    } else {
      shouldTilize = true;
      shouldUntilize = false;
      // device tilize op requires height sharded or interleaved tensors
      // thus tilize first with default mem config, then convert memory config
      result = ::ttnn::to_device(result, &device, std::nullopt);
      result = updateLayoutAndDataType(result, targetDataTypeTTNN, shouldTilize,
                                       shouldUntilize);
      result =
          ::ttnn::to_memory_config(result, targetMemoryConfig, std::nullopt);
    }
    tensorPool.insert_or_assign(outputTensorRef->global_id(),
                                std::move(result));
  } else if (isOnDevice(inputTensor)) {
    shouldTilize = false;
    shouldUntilize = false;
    ::ttnn::Tensor result = updateLayoutAndDataType(
        inputTensor, targetDataTypeTTNN, shouldTilize, shouldUntilize);
    result = ::ttnn::to_memory_config(result, targetMemoryConfig, std::nullopt);
    tensorPool.insert_or_assign(outputTensorRef->global_id(),
                                std::move(result));
  }
}

// TODO(bug #272): right now hardcoding tilize/untilize, should determine with
// tile shape blocked by issue #272
static void run(::tt::target::ttnn::ToMemoryConfigOp const *op,
                std::unordered_map<uint32_t, ::ttnn::Device *> &devicePool,
                ProgramTensorPool &tensorPool) {

  const ::ttnn::Tensor &inputTensor = tensorPool.at(op->in0()->global_id());
  assert(isOnHost(inputTensor) or
         isOnDevice(inputTensor) && "Unsupported storage type");

  const ::tt::target::Dim2d *targetTileShape =
      op->out()->desc()->layout()->memory_desc()->tile_shape();
  assert(utils::isValidTileShape(targetTileShape) && "Invalid tile shape");

  const ::tt::target::MemorySpace targetMemorySpace =
      op->out()->desc()->layout()->memory_desc()->memory_space();

  switch (targetMemorySpace) {
  // This case should only be used when gathering outputs at the end of the
  // program
  case ::tt::target::MemorySpace::System:
  case ::tt::target::MemorySpace::SystemMMIO: {
    handleToHostMemoryConfigOp(inputTensor, op->out(), tensorPool);
    break;
  }
  case ::tt::target::MemorySpace::DeviceDRAM: {
    ::ttnn::Device &device = getDevice(op->device(), devicePool);
    handleToDramMemoryConfigOp(device, inputTensor, op->out(), tensorPool);
    break;
  }
  case ::tt::target::MemorySpace::DeviceL1: {
    ::ttnn::Device &device = getDevice(op->device(), devicePool);
    handleToL1MemoryConfigOp(device, inputTensor, op->out(), tensorPool);
    break;
  }
  }
}

static void run(::tt::target::ttnn::ToLayoutOp const *op,
                std::unordered_map<uint32_t, ::ttnn::Device *> &devicePool,
                ProgramTensorPool &tensorPool) {

  const ::ttnn::Tensor &inputTensor = tensorPool.at(op->in()->global_id());
  assert((isOnHost(inputTensor) or isOnDevice(inputTensor)) &&
         "Unsupported storage type");

  ::ttnn::Layout layout;
  switch (op->layout()) {
  case ::tt::target::TensorLayout::RowMajor:
    layout = ::ttnn::Layout::ROW_MAJOR;
    break;
  case ::tt::target::TensorLayout::Tile:
    layout = ::ttnn::Layout::TILE;
    break;
  case ::tt::target::TensorLayout::Invalid:
    layout = ::ttnn::Layout::INVALID;
    break;
  }

  ::ttnn::Device &device = getDevice(op->device(), devicePool);
  ::ttnn::Tensor out = ::ttnn::to_layout(inputTensor, layout, std::nullopt,
                                         std::nullopt, &device);

  tensorPool.try_emplace(op->out()->global_id(), std::move(out));
}

static void run(::tt::target::ttnn::ToDeviceOp const *op,
                std::unordered_map<uint32_t, ::ttnn::Device *> &devicePool,
                ProgramTensorPool &tensorPool) {

  const ::ttnn::Tensor &inputTensor = tensorPool.at(op->in()->global_id());
  assert((isOnHost(inputTensor) or isOnDevice(inputTensor)) &&
         "Unsupported storage type");

  op->memcfg()->tensor_memory_layout();
  op->memcfg()->buffer_type();

  ::ttnn::TensorMemoryLayout tensorMemoryLayout;
  switch (op->memcfg()->tensor_memory_layout()) {
  case ::tt::target::TensorMemoryLayout::Interleaved:
    tensorMemoryLayout = ::ttnn::TensorMemoryLayout::INTERLEAVED;
    break;
  case ::tt::target::TensorMemoryLayout::SingleBank:
    tensorMemoryLayout = ::ttnn::TensorMemoryLayout::SINGLE_BANK;
    break;
  case ::tt::target::TensorMemoryLayout::HeightSharded:
    tensorMemoryLayout = ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED;
    break;
  case ::tt::target::TensorMemoryLayout::WidthSharded:
    tensorMemoryLayout = ::ttnn::TensorMemoryLayout::WIDTH_SHARDED;
    break;
  case ::tt::target::TensorMemoryLayout::BlockSharded:
    tensorMemoryLayout = ::ttnn::TensorMemoryLayout::BLOCK_SHARDED;
    break;
  case ::tt::target::TensorMemoryLayout::None:
    assert(false &&
           "Unsupported tensor memory layout TensorMemoryLayout::None");
    break;
  }

  ::ttnn::BufferType bufferType;
  switch (op->memcfg()->buffer_type()) {
  case ::tt::target::BufferType::DRAM:
    bufferType = ::ttnn::BufferType::DRAM;
    break;
  case ::tt::target::BufferType::L1:
    bufferType = ::ttnn::BufferType::L1;
    break;
  case ::tt::target::BufferType::SystemMemory:
    bufferType = ::ttnn::BufferType::SYSTEM_MEMORY;
    break;
  case ::tt::target::BufferType::L1Small:
    bufferType = ::ttnn::BufferType::L1_SMALL;
    break;
  case ::tt::target::BufferType::Trace:
    bufferType = ::ttnn::BufferType::TRACE;
    break;
  }

  // TODO(bug #620):
  // Until ShardSpec support is added in TTNN, read it from the output tensor.
  // If ShardSpec is not supplied, an error will be thrown in ttnn lib.
  //
  const ::tt::target::LayoutDesc *layout = op->out()->desc()->layout();
  const ::flatbuffers::Vector<const tt::target::Dim2dRange *>
      *targetCoreRangeSet = layout->core_range_set();
  const ::flatbuffers::Vector<int32_t> *targetShardShape =
      layout->memory_desc()->shape();
  CoreRangeSet ttnnCoreRangeSet = toCoreRangeSet(targetCoreRangeSet);
  std::array<uint32_t, 2> ttnnShardShape;
  std::copy(targetShardShape->begin(), targetShardShape->end(),
            ttnnShardShape.begin());
  ::tt::tt_metal::ShardSpec shardSpec(
      ttnnCoreRangeSet, ttnnShardShape,
      ::tt::tt_metal::ShardOrientation::ROW_MAJOR, false);

  ::ttnn::MemoryConfig memoryConfig = {tensorMemoryLayout, bufferType,
                                       shardSpec};
  ::ttnn::Device &device = getDevice(op->device(), devicePool);
  ::ttnn::Tensor out = ::ttnn::to_device(inputTensor, &device, memoryConfig);

  tensorPool.try_emplace(op->out()->global_id(), std::move(out));
}

static void run(::tt::target::ttnn::EmptyOp const *op,
                std::unordered_map<uint32_t, ::ttnn::Device *> &devicePool,
                ProgramTensorPool &tensorPool) {
  ::ttnn::DataType targetDataTypeTTNN = getDataType(op->out());
  // TODO(bug #582): ttnn::empty doesn't work properly with tile layout,
  // using ROW_MAJOR until we fix it
  auto desiredLayout = ::ttnn::Layout::ROW_MAJOR;
  auto shape = ::ttnn::Shape(::tt::tt_metal::Shape(
      utils::toShapeFromFBShape(*op->out()->desc()->shape())));

  // Create output memory config for the op
  //
  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      createMemoryConfig(op->out());

  // ::ttnn::Device &device = getDevice(op->device(), devicePool);
  // ::ttnn::Tensor out = ::ttnn::empty(shape, targetDataTypeTTNN,
  // desiredLayout,
  //                                    device, outputMemoryConfig);
  ::ttnn::Tensor out = ::ttnn::empty(shape);
  (void)targetDataTypeTTNN;
  (void)desiredLayout;

  // use try emplace here so the program output tensor doesn't get overwritten
  tensorPool.try_emplace(op->out()->global_id(), std::move(out));
}

static void
getEltwiseBinaryOPInputTensors(::tt::target::ttnn::EltwiseOp const *op,
                               ProgramTensorPool &tensorPool,
                               ::ttnn::Tensor **lhs, ::ttnn::Tensor **rhs) {
  assert(op->ins()->size() == 2 && "Expected 2 inputs");
  *lhs = &(tensorPool.at(op->ins()->Get(0)->global_id()));
  *rhs = &(tensorPool.at(op->ins()->Get(1)->global_id()));
}

static void runEltwiseBinaryOP(
    ::tt::target::ttnn::EltwiseOp const *op, ProgramTensorPool &tensorPool,
    std::function<::ttnn::Tensor(
        const ::ttnn::Tensor &, const ::ttnn::Tensor &,
        const std::optional<const ::ttnn::DataType> &,
        const std::optional<::tt::tt_metal::MemoryConfig> &,
        std::optional<::ttnn::Tensor>,
        std::optional<::ttnn::operations::unary::FusedActivations>,
        std::optional<::ttnn::operations::unary::UnaryWithParam>)>
        ttnnOp) {

  ::ttnn::Tensor *lhs = nullptr;
  ::ttnn::Tensor *rhs = nullptr;
  getEltwiseBinaryOPInputTensors(op, tensorPool, &lhs, &rhs);

  ::ttnn::DataType outputDataType = getDataType(op->out());
  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      createMemoryConfig(op->out());

  ::ttnn::Tensor out = ttnnOp(*lhs, *rhs, outputDataType, outputMemoryConfig,
                              std::nullopt, std::nullopt, std::nullopt);
  tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
}

static void runEltwiseBinaryCompositeOP(
    ::tt::target::ttnn::EltwiseOp const *op, ProgramTensorPool &tensorPool,
    std::function<
        ::ttnn::Tensor(const ::ttnn::Tensor &, const ::ttnn::Tensor &,
                       const std::optional<::tt::tt_metal::MemoryConfig> &)>
        ttnnOp) {

  ::ttnn::Tensor *lhs = nullptr;
  ::ttnn::Tensor *rhs = nullptr;
  getEltwiseBinaryOPInputTensors(op, tensorPool, &lhs, &rhs);

  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      createMemoryConfig(op->out());

  ::ttnn::Tensor out = ttnnOp(*lhs, *rhs, outputMemoryConfig);
  tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
}

static void
getEltwiseUnaryOPInputTensor(::tt::target::ttnn::EltwiseOp const *op,
                             ProgramTensorPool &tensorPool,
                             ::ttnn::Tensor **in) {
  assert(op->ins()->size() == 1 && "Expected 1 input");
  *in = &(tensorPool.at(op->ins()->Get(0)->global_id()));
}

static void runEltwiseUnaryOP(
    ::tt::target::ttnn::EltwiseOp const *op, ProgramTensorPool &tensorPool,
    std::function<
        ::ttnn::Tensor(const ::ttnn::Tensor &,
                       const std::optional<::tt::tt_metal::MemoryConfig> &,
                       const std::optional<::ttnn::Tensor> &)>
        ttnnOp) {

  ::ttnn::Tensor *in = nullptr;
  getEltwiseUnaryOPInputTensor(op, tensorPool, &in);

  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      createMemoryConfig(op->out());

  ::ttnn::Tensor out = ttnnOp(*in, outputMemoryConfig, std::nullopt);
  tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
}

static void runEltwiseUnaryWithFastAndApproximateModeOP(
    ::tt::target::ttnn::EltwiseOp const *op, ProgramTensorPool &tensorPool,
    std::function<
        ::ttnn::Tensor(const ::ttnn::Tensor &, const bool,
                       const std::optional<::tt::tt_metal::MemoryConfig> &,
                       const std::optional<::ttnn::Tensor> &)>
        ttnnOp) {

  ::ttnn::Tensor *in = nullptr;
  getEltwiseUnaryOPInputTensor(op, tensorPool, &in);

  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      createMemoryConfig(op->out());

  ::ttnn::Tensor out =
      ttnnOp(*in, false /* parameter */, outputMemoryConfig, std::nullopt);
  tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
}

static void run(::tt::target::ttnn::EltwiseOp const *op,
                std::unordered_map<uint32_t, ::ttnn::Device *> &devicePool,
                ProgramTensorPool &tensorPool) {
  switch (op->type()) {
  /* Eltwise Binary */
  case ::tt::target::ttnn::EltwiseOpType::Add: {
    runEltwiseBinaryOP(op, tensorPool, ::ttnn::add);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Multiply: {
    runEltwiseBinaryOP(op, tensorPool, ::ttnn::multiply);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Subtract: {
    runEltwiseBinaryOP(op, tensorPool, ::ttnn::subtract);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::GreaterEqual: {
    runEltwiseBinaryOP(op, tensorPool, ::ttnn::ge);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Div: {
    runEltwiseBinaryOP(op, tensorPool, ::ttnn::divide);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Maximum: {
    runEltwiseBinaryCompositeOP(op, tensorPool, ::ttnn::maximum);
    break;
  }
  /* Eltwise Unary */
  case ::tt::target::ttnn::EltwiseOpType::Abs: {
    runEltwiseUnaryOP(op, tensorPool, ::ttnn::abs);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Relu: {
    runEltwiseUnaryOP(op, tensorPool, ::ttnn::relu);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Sqrt: {
    runEltwiseUnaryOP(op, tensorPool, ::ttnn::sqrt);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Sigmoid: {
    runEltwiseUnaryOP(op, tensorPool, ::ttnn::sigmoid);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Reciprocal: {
    runEltwiseUnaryOP(op, tensorPool, ::ttnn::reciprocal);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Exp: {
    runEltwiseUnaryWithFastAndApproximateModeOP(op, tensorPool, ::ttnn::exp);
    break;
  }
  }
}

static void runReductionOp(
    ::tt::target::ttnn::ReductionOp const *op, ProgramTensorPool &tensorPool,
    std::function<::ttnn::Tensor(
        const ::ttnn::Tensor &,
        const std::optional<std::variant<int, std::vector<int>>> &, const bool,
        const std::optional<::tt::tt_metal::MemoryConfig> &,
        const std::optional<::ttnn::DeviceComputeKernelConfig> &, float)>
        ttnnOp) {
  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      createMemoryConfig(op->out());
  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());

  const auto *fbDimArg = op->dim_arg();
  std::optional<vector<int>> dimArg =
      fbDimArg ? std::make_optional(
                     std::vector<int>(fbDimArg->begin(), fbDimArg->end()))
               : std::nullopt;

  ::ttnn::Tensor out = ttnnOp(
      in, dimArg, op->keep_dim(), outputMemoryConfig /* memory_config_arg */,
      std::nullopt /* compute_kernel_config */, 1.0f /* scalar */);

  tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
}

static void run(::tt::target::ttnn::ReductionOp const *op,
                std::unordered_map<uint32_t, ::ttnn::Device *> &devicePool,
                ProgramTensorPool &tensorPool) {
  switch (op->type()) {
  case ::tt::target::ttnn::ReductionOpType::Sum: {
    runReductionOp(op, tensorPool, ::ttnn::sum);
    break;
  }
  case ::tt::target::ttnn::ReductionOpType::Mean: {
    runReductionOp(op, tensorPool, ::ttnn::mean);
    break;
  }
  case ::tt::target::ttnn::ReductionOpType::Max: {
    runReductionOp(op, tensorPool, ::ttnn::max);
    break;
  }
  }
}

template <int32_t Rank>
static std::array<int32_t, Rank>
vectorToArray(const std::vector<int32_t> &vec) {
  if (vec.size() != Rank) {
    throw std::invalid_argument("Vector size does not match array size");
  }
  std::array<int32_t, Rank> arr;
  std::copy(vec.begin(), vec.end(), arr.begin());
  return arr;
}

template <int32_t Rank>
static ::ttnn::Tensor invoke_reshape(const ::ttnn::Tensor &tensor,
                                     const std::vector<int32_t> &shape) {
  return ::ttnn::reshape(tensor, vectorToArray<Rank>(shape));
}

static void run(::tt::target::ttnn::ReshapeOp const *op,
                std::unordered_map<uint32_t, ::ttnn::Device *> &devicePool,
                ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());
  const auto *fbShape = op->shape();
  std::vector<int32_t> shape(fbShape->begin(), fbShape->end());
  constexpr int32_t Rank1 = 1;
  constexpr int32_t Rank2 = 2;
  constexpr int32_t Rank3 = 3;
  constexpr int32_t Rank4 = 4;
  constexpr int32_t Rank5 = 5;

  ::ttnn::Tensor out;
  switch (fbShape->size()) {
  case Rank1:
    out = invoke_reshape<Rank1>(in, shape);
    break;
  case Rank2:
    out = invoke_reshape<Rank2>(in, shape);
    break;
  case Rank3:
    out = invoke_reshape<Rank3>(in, shape);
    break;
  case Rank4:
    out = invoke_reshape<Rank4>(in, shape);
    break;
  case Rank5:
    out = invoke_reshape<Rank5>(in, shape);
    break;
  default:
    throw std::invalid_argument("Unsupported rank for reshape");
  }

  tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
}

static void run(::tt::target::ttnn::EmbeddingOp const *op,
                std::unordered_map<uint32_t, ::ttnn::Device *> &devicePool,
                ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &input = tensorPool.at(op->input()->global_id());
  const ::ttnn::Tensor &weight = tensorPool.at(op->weight()->global_id());
  // default params for embedding op
  std::optional<int> padToken = std::nullopt;
  ::tt::tt_metal::Layout layout = ::ttnn::ROW_MAJOR_LAYOUT;
  auto embeddingsType = ::ttnn::operations::embedding::EmbeddingsType::GENERIC;
  ::ttnn::DataType outputDataType = getDataType(op->output());
  ::ttnn::MemoryConfig outputMemoryConfig = createMemoryConfig(op->output());
  ::ttnn::Tensor out =
      ::ttnn::embedding(input, weight, padToken, layout, embeddingsType,
                        outputDataType, outputMemoryConfig);
  tensorPool.insert_or_assign(op->output()->global_id(), std::move(out));
}

static void run(::tt::target::ttnn::SoftmaxOp const *op,
                std::unordered_map<uint32_t, ::ttnn::Device *> &devicePool,
                ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());
  int32_t dimension = op->dimension();
  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      createMemoryConfig(op->out());
  ::ttnn::Tensor out = ::ttnn::softmax(in, dimension, outputMemoryConfig);
  tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
}

static void run(::tt::target::ttnn::TransposeOp const *op,
                std::unordered_map<uint32_t, ::ttnn::Device *> &devicePool,
                ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());
  int32_t dim0 = op->dim0();
  int32_t dim1 = op->dim1();
  auto inputRank = in.get_shape().rank();
  // for the current version of permute, we need to work in 4D, so we add
  // leading dimensions of size 1
  std::vector<std::int64_t> dimensionOrder(4);
  std::iota(dimensionOrder.begin(), dimensionOrder.end(), 0);
  if (dim0 < 0) {
    dim0 += 4;
  } else {
    dim0 = dim0 + 4 - inputRank;
  }
  if (dim1 < 0) {
    dim1 += 4;
  } else {
    dim1 = dim1 + 4 - inputRank;
  }
  std::swap(dimensionOrder[dim0], dimensionOrder[dim1]);
  // Ideally this would use ttnn::transpose, but since ttnn::transpose doesn't
  // work at the moment, we use this temporary solution.
  ::ttnn::Tensor unsqueezedInput = ::ttnn::unsqueeze_to_4D(in);
  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      createMemoryConfig(op->out());
  ::ttnn::Tensor out =
      ::ttnn::permute(unsqueezedInput, dimensionOrder, outputMemoryConfig);
  tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
}

static void run(::tt::target::ttnn::ConcatOp const *op,
                std::unordered_map<uint32_t, ::ttnn::Device *> &devicePool,
                ProgramTensorPool &tensorPool) {
  std::vector<::ttnn::Tensor> inputs;
  for (const auto &input : *op->inputs()) {
    inputs.push_back(tensorPool.at(input->global_id()));
  }
  int32_t dim = op->dim();
  ::ttnn::Tensor out = ::ttnn::concat(inputs, dim);
  tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
}

// ANCHOR: adding_an_op_matmul_runtime
static void run(::tt::target::ttnn::MatmulOp const *op,
                std::unordered_map<uint32_t, ::ttnn::Device *> &devicePool,
                ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &lhs = tensorPool.at(op->in0()->global_id());
  const ::ttnn::Tensor &rhs = tensorPool.at(op->in1()->global_id());
  ::ttnn::DataType outputDataType = getDataType(op->out());
  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      createMemoryConfig(op->out());
  ::ttnn::Tensor out = ::ttnn::operations::matmul::matmul(
      lhs, rhs, /*bias=*/std::nullopt,
      ::ttnn::operations::matmul::Matmul{/*program_config=*/std::nullopt,
                                         /*bcast_batch=*/std::nullopt,
                                         outputMemoryConfig, outputDataType});
  tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
}
// ANCHOR_END: adding_an_op_matmul_runtime

static void run(::tt::target::ttnn::Conv2dOp const *op,
                std::unordered_map<uint32_t, ::ttnn::Device *> &devicePool,
                ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &input = tensorPool.at(op->input()->global_id());
  const ::ttnn::Tensor &weight = tensorPool.at(op->weight()->global_id());
  std::optional<::ttnn::Tensor> bias =
      op->bias() ? std::make_optional(tensorPool.at(op->bias()->global_id()))
                 : std::nullopt;
  auto config = ::ttnn::operations::conv::conv2d::Conv2dConfig();
  config.dtype = input.dtype();
  config.weights_dtype = weight.dtype();
  ::ttnn::Device &device = getDevice(op->device(), devicePool);
  ::ttnn::Tensor out =
      std::get<0>(::ttnn::operations::conv::conv2d::conv2d<::ttnn::Device>(
          input, weight, &device, op->in_channels(), op->out_channels(),
          op->batch_size(), op->input_height(), op->input_width(),
          {op->kernel_height(), op->kernel_width()},
          {op->stride_height(), op->stride_width()},
          {op->padding_height(), op->padding_width()},
          {op->dilation_height(), op->dilation_width()}, op->groups(), bias,
          config));

  tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
  return;
}

static void run(::tt::target::ttnn::MaxPool2dOp const *op,
                std::unordered_map<uint32_t, ::ttnn::Device *> &devicePool,
                ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &input = tensorPool.at(op->in()->global_id());
  const ::ttnn::operations::pool::MaxPool2DOp operation =
      ::ttnn::operations::pool::MaxPool2DOp();

  ::ttnn::Device &device = getDevice(op->device(), devicePool);
  ::ttnn::Tensor out = operation.invoke(
      0, input, op->batch_size(), op->input_height(), op->input_width(),
      op->channels(), {op->kernel_height(), op->kernel_width()},
      {op->stride_height(), op->stride_width()},
      {op->padding_height(), op->padding_width()},
      {op->dilation_height(), op->dilation_width()}, &device);

  tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
  return;
}

static void run(::tt::target::ttnn::DeallocOp const *op,
                std::unordered_map<uint32_t, ::ttnn::Device *> &devicePool,
                ProgramTensorPool &tensorPool) {
  bool force = true;
  ::ttnn::Tensor &tensor = tensorPool.at(op->in()->global_id());
  tensor.deallocate(force);
  tensorPool.erase(op->in()->global_id());
}

static void
run(::tt::target::ttnn::GetDeviceOp const *op,
    const std::unordered_map<uint32_t, ::ttnn::Device *> &allDevices,
    std::unordered_map<uint32_t, ::ttnn::Device *> &devicePool,
    ProgramTensorPool &tensorPool) {
  const flatbuffers::Vector<uint32_t> *chipIds = op->chip_ids();
  assert(chipIds->size() == 1 && "Expected 1 chip id");
  for (const uint32_t chipId : *chipIds) {
    assert(allDevices.contains(chipId) && "Device not found");
    auto [iter, inserted] =
        devicePool.try_emplace(chipId, allDevices.at(chipId));
    assert(inserted && "Duplicate device");
  }
}

static void run(::tt::target::ttnn::FullOp const *op,
                std::unordered_map<uint32_t, ::ttnn::Device *> &devicePool,
                ProgramTensorPool &tensorPool) {
  ::ttnn::Device &device = getDevice(op->device(), devicePool);
  ::ttnn::DataType outputDataType = getDataType(op->out());
  auto shape = ::ttnn::Shape(::tt::tt_metal::Shape(
      utils::toShapeFromFBShape(*op->out()->desc()->shape())));
  float fillValue = op->fill_value();
  // TODO(bug #272), determine correct layout by tile shape in the future
  ::ttnn::Layout outputLayout = ::ttnn::Layout::ROW_MAJOR;
  std::optional<std::reference_wrapper<::ttnn::Device>> outputDevice =
      std::make_optional(std::ref(device));
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      std::make_optional(createMemoryConfig(op->out()));

  ::ttnn::Tensor out =
      ::ttnn::full(shape, fillValue, outputDataType, outputLayout, outputDevice,
                   outputMemoryConfig);

  tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
}

static void
run(::tt::target::ttnn::Operation const *op,
    const std::unordered_map<uint32_t, ::ttnn::Device *> &allDevices,
    std::unordered_map<uint32_t, ::ttnn::Device *> &devicePool,
    ProgramTensorPool &tensorPool) {

  switch (op->type_type()) {
  case ::tt::target::ttnn::OpType::GetDeviceOp: {
    return run(op->type_as_GetDeviceOp(), allDevices, devicePool, tensorPool);
    break;
  }
  case ::tt::target::ttnn::OpType::ToMemoryConfigOp: {
    return run(op->type_as_ToMemoryConfigOp(), devicePool, tensorPool);
  }
  case ::tt::target::ttnn::OpType::ToLayoutOp: {
    return run(op->type_as_ToLayoutOp(), devicePool, tensorPool);
  }
  case ::tt::target::ttnn::OpType::ToDeviceOp: {
    return run(op->type_as_ToDeviceOp(), devicePool, tensorPool);
  }
  case ::tt::target::ttnn::OpType::EmptyOp: {
    return run(op->type_as_EmptyOp(), devicePool, tensorPool);
  }
  case ::tt::target::ttnn::OpType::FullOp: {
    return run(op->type_as_FullOp(), devicePool, tensorPool);
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseOp: {
    return run(op->type_as_EltwiseOp(), devicePool, tensorPool);
  }
  case ::tt::target::ttnn::OpType::MatmulOp: {
    return run(op->type_as_MatmulOp(), devicePool, tensorPool);
  }
  case ::tt::target::ttnn::OpType::ReductionOp: {
    return run(op->type_as_ReductionOp(), devicePool, tensorPool);
  }
  case ::tt::target::ttnn::OpType::EmbeddingOp: {
    return run(op->type_as_EmbeddingOp(), devicePool, tensorPool);
  }
  case ::tt::target::ttnn::OpType::SoftmaxOp: {
    return run(op->type_as_SoftmaxOp(), devicePool, tensorPool);
  }
  case ::tt::target::ttnn::OpType::TransposeOp: {
    return run(op->type_as_TransposeOp(), devicePool, tensorPool);
  }
  case ::tt::target::ttnn::OpType::Conv2dOp: {
    return run(op->type_as_Conv2dOp(), devicePool, tensorPool);
  }
  case ::tt::target::ttnn::OpType::ConcatOp: {
    return run(op->type_as_ConcatOp(), devicePool, tensorPool);
  }
  case ::tt::target::ttnn::OpType::ReshapeOp: {
    return run(op->type_as_ReshapeOp(), devicePool, tensorPool);
  }
  case ::tt::target::ttnn::OpType::DeallocOp: {
    return run(op->type_as_DeallocOp(), devicePool, tensorPool);
  }
  case ::tt::target::ttnn::OpType::MaxPool2dOp: {
    return run(op->type_as_MaxPool2dOp(), devicePool, tensorPool);
  }
  default: {
    throw std::runtime_error("Unsupported operation type");
  }
  }
}

// Nop is single input, output tensor where input is returned as output.
bool handleNopProgram(::tt::target::ttnn::Program const *program,
                      std::vector<::ttnn::Tensor *> const &inputs,
                      std::vector<::ttnn::Tensor *> const &outputs) {

  bool isNop = program->inputs()->size() == 1 &&
               program->outputs()->size() == 1 &&
               program->inputs()->Get(0)->global_id() ==
                   program->outputs()->Get(0)->global_id();

  if (isNop) {
    void *src = ::tt::tt_metal::get_raw_host_data_ptr(*inputs.at(0));
    void *dst = ::tt::tt_metal::get_raw_host_data_ptr(*outputs.at(0));
    std::uint32_t size = outputs[0]->volume() * outputs[0]->element_size();
    std::memcpy(dst, src, size);
  }
  return isNop;
}

void runProgram(::ttnn::Device &device,
                ::tt::target::ttnn::Program const *program,
                std::vector<::ttnn::Tensor *> const &inputs,
                std::vector<::ttnn::Tensor *> const &outputs) {
  if (handleNopProgram(program, inputs, outputs)) {
    return;
  }
  std::unordered_map<std::uint32_t, ::ttnn::Tensor *> liveTensors;
  std::unordered_map<std::uint32_t, ::ttnn::Device *> allDevices;
  std::unordered_map<std::uint32_t, ::ttnn::Device *> devicePool;
  int inputIndex = 0;
  assert(program->inputs()->size() == inputs.size());
  // Assuming single device for now until we support multichip
  allDevices.try_emplace(device.id(), &device);
  for (::tt::target::TensorRef const *input : *program->inputs()) {
    auto [iter, inserted] =
        liveTensors.try_emplace(input->global_id(), inputs[inputIndex++]);
    assert(inserted && "Duplicate input tensor");
  }

  int outputIndex = 0;
  assert(program->outputs()->size() == outputs.size());
  for (::tt::target::TensorRef const *output : *program->outputs()) {
    auto [iter, inserted] =
        liveTensors.try_emplace(output->global_id(), outputs[outputIndex++]);
    assert(inserted && "Duplicate output tensor");
  }
  ProgramTensorPool tensorPool(std::move(liveTensors));
  for (::tt::target::ttnn::Operation const *op : *program->operations()) {
    run(op, allDevices, devicePool, tensorPool);
  }
}
} // namespace tt::runtime::ttnn
