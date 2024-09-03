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
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "types_generated.h"
#include "utils.h"

#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Version.h"

// It seems like `ttnn::to_layout` cannot be called inside of the
// `tt::runtime::ttnn` namespace.  TTNN uses a lot of metaprogramming and for
// some reason a static_assert fails when this is called from within our
// namespace.
ttnn::Tensor tilize(ttnn::Tensor const &input) {
  // NOLINTNEXTLINE
  return ttnn::to_layout(input, ::ttnn::TILE_LAYOUT, std::nullopt, std::nullopt,
                         static_cast<Device *>(nullptr));
}

ttnn::Tensor untilize(ttnn::Tensor const &input) {
  return ttnn::to_layout(input, ::ttnn::ROW_MAJOR_LAYOUT, std::nullopt,
                         std::nullopt, static_cast<Device *>(nullptr));
}

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
createShardedMemoryConfig(const ::tt::target::TensorMemoryLayout memLayout,
                          const CoreRangeSet &coreRangeSet,
                          const std::array<uint32_t, 2> &shardShape) {
  ::tt::tt_metal::ShardSpec shardSpec(
      coreRangeSet, shardShape, ::tt::tt_metal::ShardOrientation::ROW_MAJOR,
      false);
  ::tt::tt_metal::TensorMemoryLayout ttnnMemLayout =
      utils::toTTNNTensorMemoryLayout(memLayout);
  // TODO (jnie): Hardcoding to block sharded for now
  // Add support for other types once compiler supports it
  assert(ttnnMemLayout == ::tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED &&
         "Only block sharded supported for now");
  return {ttnnMemLayout, ::tt::tt_metal::BufferType::L1, shardSpec};
}

static ::tt::tt_metal::MemoryConfig
createL1MemoryConfig(const ::tt::target::TensorRef *tensorRef) {
  const ::tt::target::LayoutDesc *layout = tensorRef->desc()->layout();
  const ::tt::target::TensorMemoryLayout targetMemoryLayout =
      layout->memory_desc()->memory_layout();
  assert(
      (targetMemoryLayout == ::tt::target::TensorMemoryLayout::Interleaved or
       targetMemoryLayout == ::tt::target::TensorMemoryLayout::BlockSharded) &&
      "Only interleaved and block sharded memory layouts are supported for L1 "
      "tensors");

  const ::flatbuffers::Vector<int32_t> *memoryDescShape =
      layout->memory_desc()->shape();
  assert(memoryDescShape->size() == 2 &&
         "Only 2D shard shape is supported in TTNN backend");

  CoreRangeSet coreRangeSet = toCoreRangeSet(layout->core_range_set());
  assert(coreRangeSet.size() == 1 &&
         "Currently only single core range/grid is supported");

  if (targetMemoryLayout == ::tt::target::TensorMemoryLayout::Interleaved) {
    return ::ttnn::L1_MEMORY_CONFIG;
  }

  std::array<uint32_t, 2> shardShape;
  std::copy(memoryDescShape->begin(), memoryDescShape->end(),
            shardShape.begin());
  assert((shardShape[0] % ::tt::constants::TILE_HEIGHT == 0 and
          shardShape[1] % ::tt::constants::TILE_WIDTH == 0) &&
         "Shard shape does not divide tile shape evenly");

  return createShardedMemoryConfig(targetMemoryLayout, coreRangeSet,
                                   shardShape);
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
          ::ttnn::typecast(::tilize(input), targetDataType);
      // untilize and return
      return ::untilize(converted);
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
    outputTensor = ::tilize(outputTensor);
  } else if (shouldUntilize) {
    outputTensor = ::untilize(outputTensor);
  }
  if (shouldConvertDataType) {
    outputTensor = convertDataType(outputTensor, targetDataType);
  }
  return outputTensor;
}

static void
handleToHostMemoryConfigOp(const ::ttnn::Tensor &inputTensor,
                           const ::ttnn::DataType &targetDataTypeTTNN,
                           uint32_t outputGlobalId,
                           ProgramTensorPool &tensorPool) {
  ::ttnn::Tensor result;
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
  if (tensorPool.contains(outputGlobalId)) {
    ::ttnn::Tensor &outputTensor = tensorPool.at(outputGlobalId);
    void *src = ::tt::tt_metal::get_raw_host_data_ptr(result);
    void *dst = ::tt::tt_metal::get_raw_host_data_ptr(outputTensor);
    std::uint32_t size = result.volume() * result.element_size();
    std::memcpy(dst, src, size);
  } else {
    tensorPool.insert_or_assign(outputGlobalId, std::move(result));
  }
}

static void handleToDramMemoryConfigOp(
    ::ttnn::Device &device, const ::ttnn::Tensor &inputTensor,
    const ::ttnn::DataType &targetDataTypeTTNN, uint32_t outputGlobalId,
    ProgramTensorPool &tensorPool) {
  ::tt::tt_metal::MemoryConfig memConfig = ::ttnn::DRAM_MEMORY_CONFIG;
  bool shouldTilize, shouldUntilize;
  if (isOnHost(inputTensor)) {
    ::ttnn::Tensor result = inputTensor;
    shouldTilize = true;
    shouldUntilize = false;
    // device tilize requires BFLOAT16, if not then tilize on host
    if (result.get_dtype() != ::ttnn::DataType::BFLOAT16) {
      result = ::tilize(result);
      shouldTilize = false;
    }
    result = ::ttnn::to_device(result, &device, memConfig);
    result = updateLayoutAndDataType(result, targetDataTypeTTNN, shouldTilize,
                                     shouldUntilize);
    tensorPool.insert_or_assign(outputGlobalId, std::move(result));
  } else if (isOnDevice(inputTensor)) {
    shouldTilize = false;
    shouldUntilize = false;
    ::ttnn::Tensor result = updateLayoutAndDataType(
        inputTensor, targetDataTypeTTNN, shouldTilize, shouldUntilize);
    result = ::ttnn::to_memory_config(result, memConfig, std::nullopt);
    tensorPool.insert_or_assign(outputGlobalId, std::move(result));
  }
}

static void handleToL1MemoryConfigOp(
    ::ttnn::Device &device, const ::ttnn::Tensor &inputTensor,
    const ::tt::target::TensorRef *outputTensorRef,
    const ::ttnn::DataType &targetDataTypeTTNN, ProgramTensorPool &tensorPool) {
  ::tt::tt_metal::MemoryConfig memConfig =
      createL1MemoryConfig(outputTensorRef);
  bool shouldTilize, shouldUntilize;
  if (isOnHost(inputTensor)) {
    ::ttnn::Tensor result = inputTensor;
    // device tilize requires BFLOAT16, if not then tilize on host
    if (result.get_dtype() != ::ttnn::DataType::BFLOAT16) {
      result = ::tilize(result);
      result = ::ttnn::to_device(result, &device, memConfig);
      shouldTilize = false;
      shouldUntilize = false;
      result = updateLayoutAndDataType(result, targetDataTypeTTNN, shouldTilize,
                                       shouldUntilize);
    } else {
      shouldTilize = true;
      shouldUntilize = false;
      // device tilize op requires height sharded or interleaved tensors
      result = ::ttnn::to_device(result, &device, std::nullopt);
      result = updateLayoutAndDataType(result, targetDataTypeTTNN, shouldTilize,
                                       shouldUntilize);
      result = ::ttnn::to_memory_config(result, memConfig, std::nullopt);
    }
    tensorPool.insert_or_assign(outputTensorRef->global_id(),
                                std::move(result));
  } else if (isOnDevice(inputTensor)) {
    shouldTilize = false;
    shouldUntilize = false;
    ::ttnn::Tensor result = updateLayoutAndDataType(
        inputTensor, targetDataTypeTTNN, shouldTilize, shouldUntilize);
    result = ::ttnn::to_memory_config(result, memConfig, std::nullopt);
    tensorPool.insert_or_assign(outputTensorRef->global_id(),
                                std::move(result));
  }
}

// TODO(bug #272): right now hardcoding tilize/untilize, should determine with
// tile shape blocked by issue #272
static void run(::tt::target::ttnn::ToMemoryConfigOp const *op,
                ::ttnn::Device &device, ProgramTensorPool &tensorPool) {

  const ::ttnn::Tensor &inputTensor = tensorPool.at(op->in0()->global_id());
  assert(isOnHost(inputTensor) or
         isOnDevice(inputTensor) && "Unsupported storage type");

  const ::tt::target::Dim2d *targetTileShape =
      op->out()->desc()->layout()->memory_desc()->tile_shape();
  assert(utils::isValidTileShape(targetTileShape) && "Invalid tile shape");

  ::tt::target::DataType targetDataType =
      op->out()->desc()->layout()->memory_desc()->data_type();
  ::ttnn::DataType targetDataTypeTTNN = utils::toTTNNDataType(targetDataType);

  const ::tt::target::MemorySpace targetMemorySpace =
      op->out()->desc()->layout()->memory_desc()->memory_space();

  switch (targetMemorySpace) {
  // This case should only be used when gathering outputs at the end of the
  // program
  case ::tt::target::MemorySpace::System:
  case ::tt::target::MemorySpace::SystemMMIO: {
    handleToHostMemoryConfigOp(inputTensor, targetDataTypeTTNN,
                               op->out()->global_id(), tensorPool);
    break;
  }
  case ::tt::target::MemorySpace::DeviceDRAM: {
    handleToDramMemoryConfigOp(device, inputTensor, targetDataTypeTTNN,
                               op->out()->global_id(), tensorPool);
    break;
  }
  case ::tt::target::MemorySpace::DeviceL1: {
    handleToL1MemoryConfigOp(device, inputTensor, op->out(), targetDataTypeTTNN,
                             tensorPool);
    break;
  }
  }
}

static void run(::tt::target::ttnn::EmptyOp const *op, ::ttnn::Device &device,
                ProgramTensorPool &tensorPool) {
  ::ttnn::DataType targetDataTypeTTNN = utils::toTTNNDataType(
      op->out()->desc()->layout()->memory_desc()->data_type());

  // TODO(bug #582): ttnn::empty doesn't work properly with tile layout,
  // using ROW_MAJOR until we fix it
  auto desiredLayout = ::ttnn::Layout::ROW_MAJOR;
  auto shape = ::ttnn::Shape(::tt::tt_metal::Shape(
      utils::toShapeFromFBShape(*op->out()->desc()->shape())));

  ::ttnn::Tensor out =
      ::ttnn::empty(shape, targetDataTypeTTNN, desiredLayout, device);
  // use try emplace here so the program output tensor doesn't get overwritten
  tensorPool.try_emplace(op->out()->global_id(), std::move(out));
}

static void run(::tt::target::ttnn::EltwiseOp const *op, ::ttnn::Device &device,
                ProgramTensorPool &tensorPool) {
  switch (op->type()) {
  /* Eltwise Binary */
  case ::tt::target::ttnn::EltwiseOpType::Add: {
    assert(op->ins()->size() == 2 && "Expected 2 inputs");
    const ::ttnn::Tensor &lhs = tensorPool.at(op->ins()->Get(0)->global_id());
    const ::ttnn::Tensor &rhs = tensorPool.at(op->ins()->Get(1)->global_id());
    ::ttnn::Tensor out = ::ttnn::add(lhs, rhs);
    tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Multiply: {
    assert(op->ins()->size() == 2 && "Expected 2 inputs");

    const ::ttnn::Tensor &lhs = tensorPool.at(op->ins()->Get(0)->global_id());
    const ::ttnn::Tensor &rhs = tensorPool.at(op->ins()->Get(1)->global_id());
    ::ttnn::Tensor out = ::ttnn::multiply(lhs, rhs);
    tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Subtract: {
    assert(op->ins()->size() == 2 && "Expected 2 inputs");

    const ::ttnn::Tensor &lhs = tensorPool.at(op->ins()->Get(0)->global_id());
    const ::ttnn::Tensor &rhs = tensorPool.at(op->ins()->Get(1)->global_id());
    ::ttnn::Tensor out = ::ttnn::subtract(lhs, rhs);
    tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::GreaterEqual: {
    assert(op->ins()->size() == 2 && "Expected 2 inputs");

    const ::ttnn::Tensor &lhs = tensorPool.at(op->ins()->Get(0)->global_id());
    const ::ttnn::Tensor &rhs = tensorPool.at(op->ins()->Get(1)->global_id());
    ::ttnn::Tensor out = ::ttnn::ge(lhs, rhs);
    tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Div: {
    assert(op->ins()->size() == 2 && "Expected 2 inputs");

    const ::ttnn::Tensor &lhs = tensorPool.at(op->ins()->Get(0)->global_id());
    const ::ttnn::Tensor &rhs = tensorPool.at(op->ins()->Get(1)->global_id());
    ::ttnn::Tensor out = ::ttnn::divide(lhs, rhs);
    tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
    break;
  }
  /* Eltwise Unary */
  case ::tt::target::ttnn::EltwiseOpType::Relu: {
    assert(op->ins()->size() == 1 && "Expected 1 input");
    const ::ttnn::Tensor &in = tensorPool.at(op->ins()->Get(0)->global_id());
    ::ttnn::Tensor out = ::ttnn::relu(in);
    tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Sqrt: {
    assert(op->ins()->size() == 1 && "Expected 1 input");
    const ::ttnn::Tensor &in = tensorPool.at(op->ins()->Get(0)->global_id());
    ::ttnn::Tensor out = ::ttnn::sqrt(in);
    tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Sigmoid: {
    assert(op->ins()->size() == 1 && "Expected 1 input");
    const ::ttnn::Tensor &in = tensorPool.at(op->ins()->Get(0)->global_id());
    ::ttnn::Tensor out = ::ttnn::sigmoid(in);
    tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Reciprocal: {
    assert(op->ins()->size() == 1 && "Expected 1 input");
    const ::ttnn::Tensor &in = tensorPool.at(op->ins()->Get(0)->global_id());
    ::ttnn::Tensor out = ::ttnn::reciprocal(in);
    tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Exp: {
    assert(op->ins()->size() == 1 && "Expected 1 input");
    const ::ttnn::Tensor &in = tensorPool.at(op->ins()->Get(0)->global_id());
    ::ttnn::Tensor out = ::ttnn::exp(in);
    tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
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
  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());

  const auto *dim_arg_fb_ptr = op->dim_arg();
  std::optional<vector<int>> dim_arg =
      dim_arg_fb_ptr ? std::make_optional(std::vector<int>(
                           dim_arg_fb_ptr->begin(), dim_arg_fb_ptr->end()))
                     : std::nullopt;

  ::ttnn::Tensor out =
      ttnnOp(in, dim_arg, op->keep_dim(), std::nullopt /* memory_config_arg */,
             std::nullopt /* compute_kernel_config */, 1.0f /* scalar */);

  tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
}

static void run(::tt::target::ttnn::ReductionOp const *op,
                ::ttnn::Device &device, ProgramTensorPool &tensorPool) {
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

static void run(::tt::target::ttnn::ReshapeOp const *op, ::ttnn::Device &device,
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
                ::ttnn::Device &device, ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &input = tensorPool.at(op->input()->global_id());
  const ::ttnn::Tensor &weight = tensorPool.at(op->weight()->global_id());

  ::ttnn::Tensor out = ::ttnn::embedding(input, weight);
  tensorPool.insert_or_assign(op->output()->global_id(), std::move(out));
}

static void run(::tt::target::ttnn::SoftmaxOp const *op, ::ttnn::Device &device,
                ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());
  int32_t dimension = op->dimension();

  ::ttnn::Tensor out = ::ttnn::softmax(in, dimension);
  tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
}

static void run(::tt::target::ttnn::TransposeOp const *op,
                ::ttnn::Device &device, ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());
  int32_t dim0 = op->dim0();
  int32_t dim1 = op->dim1();
  auto input_rank = in.get_shape().rank();
  // for the current version of permute, we need to work in 4D, so we add
  // leading dimensions of size 1
  std::vector<std::int64_t> dimensionOrder(4);
  std::iota(dimensionOrder.begin(), dimensionOrder.end(), 0);
  if (dim0 < 0) {
    dim0 += 4;
  } else {
    dim0 = dim0 + 4 - input_rank;
  }
  if (dim1 < 0) {
    dim1 += 4;
  } else {
    dim1 = dim1 + 4 - input_rank;
  }
  std::swap(dimensionOrder[dim0], dimensionOrder[dim1]);
  // Ideally this would use ttnn::transpose, but since ttnn::transpose doesn't
  // work at the moment, we use this temporary solution.
  auto unsqueezed_input = ::ttnn::unsqueeze_to_4D(in);
  ::ttnn::Tensor out = ::ttnn::permute(unsqueezed_input, dimensionOrder);
  tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
}

static void run(::tt::target::ttnn::ConcatOp const *op, ::ttnn::Device &device,
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
static void run(::tt::target::ttnn::MatmulOp const *op, ::ttnn::Device &device,
                ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &lhs = tensorPool.at(op->in0()->global_id());
  const ::ttnn::Tensor &rhs = tensorPool.at(op->in1()->global_id());
  ::ttnn::Tensor out = ::ttnn::operations::matmul::matmul(
      lhs, rhs, std::nullopt, ::ttnn::operations::matmul::Matmul{});
  tensorPool.insert_or_assign(op->out()->global_id(), std::move(out));
}
// ANCHOR_END: adding_an_op_matmul_runtime

static void run(::tt::target::ttnn::Conv2dOp const *op, ::ttnn::Device &device,
                ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &input = tensorPool.at(op->input()->global_id());
  const ::ttnn::Tensor &weight = tensorPool.at(op->weight()->global_id());
  std::optional<::ttnn::Tensor> bias =
      op->bias() ? std::make_optional(tensorPool.at(op->bias()->global_id()))
                 : std::nullopt;
  auto config = ::ttnn::operations::conv::conv2d::Conv2dConfig();
  config.dtype = input.dtype();
  config.weights_dtype = weight.dtype();

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

static void run(::tt::target::ttnn::DeallocOp const *op, ::ttnn::Device &device,
                ProgramTensorPool &tensorPool) {
  bool force = true;
  ::ttnn::Tensor &tensor = tensorPool.at(op->in()->global_id());
  tensor.deallocate(force);
  tensorPool.erase(op->in()->global_id());
}

static void run(::tt::target::ttnn::Operation const *op, ::ttnn::Device &device,
                ProgramTensorPool &tensorPool) {
  switch (op->type_type()) {
  case ::tt::target::ttnn::OpType::OpenDeviceOp: {
    // Skip for now, do we want device externally supplied?
    break;
  }
  case ::tt::target::ttnn::OpType::CloseDeviceOp: {
    // Skip for now, do we want device externally supplied?
    break;
  }
  case ::tt::target::ttnn::OpType::ToMemoryConfigOp: {
    return run(op->type_as_ToMemoryConfigOp(), device, tensorPool);
  }
  case ::tt::target::ttnn::OpType::EmptyOp: {
    return run(op->type_as_EmptyOp(), device, tensorPool);
  }
  case ::tt::target::ttnn::OpType::FullOp: {
    // Skip for now, we need an empty op
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseOp: {
    return run(op->type_as_EltwiseOp(), device, tensorPool);
  }
  case ::tt::target::ttnn::OpType::MatmulOp: {
    return run(op->type_as_MatmulOp(), device, tensorPool);
  }
  case ::tt::target::ttnn::OpType::ReductionOp: {
    return run(op->type_as_ReductionOp(), device, tensorPool);
  }
  case ::tt::target::ttnn::OpType::EmbeddingOp: {
    return run(op->type_as_EmbeddingOp(), device, tensorPool);
  }
  case ::tt::target::ttnn::OpType::SoftmaxOp: {
    return run(op->type_as_SoftmaxOp(), device, tensorPool);
  }
  case ::tt::target::ttnn::OpType::TransposeOp: {
    return run(op->type_as_TransposeOp(), device, tensorPool);
  }
  case ::tt::target::ttnn::OpType::Conv2dOp: {
    return run(op->type_as_Conv2dOp(), device, tensorPool);
  }
  case ::tt::target::ttnn::OpType::ConcatOp: {
    return run(op->type_as_ConcatOp(), device, tensorPool);
  case ::tt::target::ttnn::OpType::ReshapeOp: {
    return run(op->type_as_ReshapeOp(), device, tensorPool);
  }
  case ::tt::target::ttnn::OpType::DeallocOp: {
    return run(op->type_as_DeallocOp(), device, tensorPool);
  }
  default:
    throw std::runtime_error("Unsupported operation type");
  }
  }
}

// Nop is single input, output tensor where input is returned as output.
bool handleNopProgram(::tt::target::ttnn::Program const *program,
                      std::vector<::ttnn::Tensor *> const &inputs,
                      std::vector<::ttnn::Tensor *> const &outputs) {

  bool is_nop = program->inputs()->size() == 1 &&
                program->outputs()->size() == 1 &&
                program->inputs()->Get(0)->global_id() ==
                    program->outputs()->Get(0)->global_id();

  if (is_nop) {
    void *src = ::tt::tt_metal::get_raw_host_data_ptr(*inputs.at(0));
    void *dst = ::tt::tt_metal::get_raw_host_data_ptr(*outputs.at(0));
    std::uint32_t size = outputs[0]->volume() * outputs[0]->element_size();
    std::memcpy(dst, src, size);
  }
  return is_nop;
}

void runProgram(::ttnn::Device &device,
                ::tt::target::ttnn::Program const *program,
                std::vector<::ttnn::Tensor *> const &inputs,
                std::vector<::ttnn::Tensor *> const &outputs) {
  std::unordered_map<std::uint32_t, ::ttnn::Tensor *> liveTensors;

  int inputIndex = 0;
  assert(program->inputs()->size() == inputs.size());
  bool is_nop = handleNopProgram(program, inputs, outputs);
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
    assert((is_nop || inserted) && "Duplicate output tensor");
  }

  ProgramTensorPool tensorPool(std::move(liveTensors));
  for (::tt::target::ttnn::Operation const *op : *program->operations()) {
    run(op, device, tensorPool);
  }
}
} // namespace tt::runtime::ttnn
