// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "cpu.h"

#include "tt/runtime/detail/common/dylib.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttnn/tensor/tensor.hpp"

#include <type_traits>
#include <variant>
#include <vector>

namespace tt::runtime::ttnn::operations::cpu {

namespace {

// Create a borrowed TTNN tensor from raw data pointer with the given shape and
// dtype.
::ttnn::Tensor createBorrowedTensorFromPtr(std::shared_ptr<void> dataPtr,
                                           const ::ttnn::Shape &shape,
                                           ::ttnn::DataType dtype) {
  switch (dtype) {
  case ::ttnn::DataType::FLOAT32:
    return ::tt::runtime::ttnn::utils::createBorrowedTTNNTensor<float>(dataPtr,
                                                                       shape);
  case ::ttnn::DataType::INT32:
    return ::tt::runtime::ttnn::utils::createBorrowedTTNNTensor<int32_t>(
        dataPtr, shape);
  default:
    LOG_FATAL("Unsupported data type for CPU op output");
  }
}

// Executes the provided CPU-hoisted function for the inputs provided
// as raw data pointers, returning outputs of the function as TTNN tensors.
std::vector<::ttnn::Tensor> executeCPUFunction(
    common::WrappedFunc fn,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::ttnn::TensorRef>>
        *fbInputs,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::ttnn::TensorRef>>
        *fbOutputs,
    const std::vector<void *> &inputDataPtrs) {

  // Preparing CPU-hoisted function inputs.
  std::vector<std::vector<int64_t>> allSizesAndStrides;
  allSizesAndStrides.reserve(fbInputs->size());

  std::vector<common::WrappedTensor> packedInputs;
  packedInputs.reserve(fbInputs->size());

  for (size_t i = 0; i < fbInputs->size(); ++i) {
    const auto *tensorRef = fbInputs->Get(i);
    std::vector<int64_t> sizes = common::extractSizes(tensorRef);
    common::prepareSizesAndStrides(sizes, allSizesAndStrides);

    void *rawDataPtr = inputDataPtrs[i];
    packedInputs.emplace_back(rawDataPtr, rawDataPtr, 0,
                              allSizesAndStrides.back().data());
  }

  // Executing the CPU-hoisted function.
  common::WrappedTensor *outputArray = fn(packedInputs.data());

  // Callback for tensor creation from WrappedTensor.
  common::CreateTensorCallbackType<::ttnn::Tensor,
                                   ::tt::target::ttnn::TensorRef>
      createTensor = [](const tt::target::ttnn::TensorRef *ref,
                        std::shared_ptr<void> dataPtr) -> ::ttnn::Tensor {
    ::ttnn::Shape shape = utils::toTTNNShape(*ref->desc()->shape());
    ::ttnn::DataType dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(
        ref->desc()->layout()->memory_desc()->data_type());
    return createBorrowedTensorFromPtr(dataPtr, shape, dtype);
  };

  // Unpacking the function outputs into TTNN tensors.
  return common::unpackTensors<::ttnn::Tensor>(outputArray, fbOutputs->size(),
                                               fbOutputs, createTensor);
}

// Executes CPU-hoisted operation for single-chip workloads.
void runSingleChip(
    common::WrappedFunc fn,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::ttnn::TensorRef>>
        *fbInputs,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::ttnn::TensorRef>>
        *fbOutputs,
    ProgramContext &context) {
  LOG_DEBUG("CPU op: executing single-chip path...");

  std::vector<void *> inputDataPtrs;
  inputDataPtrs.reserve(fbInputs->size());

  for (size_t i = 0; i < fbInputs->size(); ++i) {
    const auto &tensor =
        context.getTensorPool().getTTNNTensorAndValidate(fbInputs->Get(i));
    inputDataPtrs.push_back(
        ::tt::runtime::ttnn::utils::getRawHostDataPtr(tensor));
  }

  std::vector<::ttnn::Tensor> outputs =
      executeCPUFunction(fn, fbInputs, fbOutputs, inputDataPtrs);

  for (size_t i = 0; i < outputs.size(); ++i) {
    context.getTensorPool().insertTTNNTensorAndValidate(fbOutputs->Get(i),
                                                        outputs[i]);
  }
}

// Executes CPU-hoisted operation for multi-chip workloads.
//
// This function handles sharded and replicated inputs across multiple devices.
// For each shard, it gathers input tensor data pointers, executes the
// CPU function, and finally, combines the outputs into a multi-device tensor.
//
// TODO(dmilinkovic):
// The current implementation assumes that input tensors are either:
// 1. Unsharded (single-device tensor with mesh_size == 1).
// 2. Fully sharded/replicated across all devices in the mesh.
//
// If there is a need to support partial sharding (e.g., sharded across
// subset of devices), the implementation will need to be updated to handle
// that case appropriately.
//
// Additionally, the implementation could be simplified if there was a way
// to detect whether a multi-device tensor is sharded or replicated; however, at
// the time of writing, all multi-device tensors are created through
// tt::runtime::createMultiDeviceHostTensor, and they all appear as sharded
// tensors from the TTNN perspective.
void runMultiChip(
    common::WrappedFunc fn,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::ttnn::TensorRef>>
        *fbInputs,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::ttnn::TensorRef>>
        *fbOutputs,
    ProgramContext &context) {
  LOG_DEBUG("CPU op: executing multi-chip path...");

  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();
  const ::ttnn::MeshShape &meshShape = meshDevice.shape();
  size_t numShards = meshDevice.num_devices();

  // Represents either individual shards of a multi-device tensor
  // (sharded/replicated across devices), or a single-device tensor.
  using InputTensorVariantType =
      std::variant<std::vector<::ttnn::Tensor>, ::ttnn::Tensor>;

  // Collect input tensors. For multi-device tensors, extract individual shards.
  // For single-device tensors, store the tensor directly.
  std::vector<InputTensorVariantType> inputTensors;
  inputTensors.reserve(fbInputs->size());

  for (size_t i = 0; i < fbInputs->size(); ++i) {
    const auto &tensor =
        context.getTensorPool().getTTNNTensorAndValidate(fbInputs->Get(i));
    size_t tensorMeshSize =
        tensor.tensor_topology().distribution_shape().mesh_size();

    LOG_ASSERT(tensorMeshSize == 1 || tensorMeshSize == numShards,
               "Tensor mesh size (", tensorMeshSize,
               ") must be either 1 (unsharded) or match mesh device size (",
               numShards, ")");

    if (tensorMeshSize > 1) {
      LOG_DEBUG("CPU op: input tensor ", i,
                " is multi-device tensor with mesh size ", tensorMeshSize);
      inputTensors.emplace_back(
          ::ttnn::distributed::get_device_tensors(tensor));
    } else {
      LOG_DEBUG("CPU op: input tensor ", i,
                " is single-device tensor (mesh size 1)");
      inputTensors.emplace_back(tensor);
    }
  }

  // Prepare output shards container.
  std::vector<std::vector<::ttnn::Tensor>> outputShards(fbOutputs->size());
  for (size_t i = 0; i < fbOutputs->size(); ++i) {
    outputShards[i].reserve(numShards);
  }

  // Execute CPU function for each shard.
  for (size_t shardIdx = 0; shardIdx < numShards; ++shardIdx) {
    std::vector<void *> inputDataPtrs;
    inputDataPtrs.reserve(fbInputs->size());

    for (size_t i = 0; i < fbInputs->size(); ++i) {
      void *dataPtr = std::visit(
          [shardIdx](const auto &input) -> void * {
            using T = std::decay_t<decltype(input)>;
            if constexpr (std::is_same_v<T, std::vector<::ttnn::Tensor>>) {
              // Multi-device tensor: get data from the specific shard.
              return ::tt::runtime::ttnn::utils::getRawHostDataPtr(
                  input[shardIdx]);
            } else {
              // Single-device tensor: use the same data for all shards.
              return ::tt::runtime::ttnn::utils::getRawHostDataPtr(input);
            }
          },
          inputTensors[i]);
      inputDataPtrs.push_back(dataPtr);
    }

    std::vector<::ttnn::Tensor> shardOutputs =
        executeCPUFunction(fn, fbInputs, fbOutputs, inputDataPtrs);

    for (size_t i = 0; i < shardOutputs.size(); ++i) {
      outputShards[i].push_back(std::move(shardOutputs[i]));
    }
  }

  // Combine output shards into multi-device tensors.
  for (size_t i = 0; i < fbOutputs->size(); ++i) {
    ::ttnn::Tensor multiDeviceTensor =
        ::ttnn::distributed::from_host_shards(outputShards[i], meshShape);
    context.getTensorPool().insertTTNNTensorAndValidate(fbOutputs->Get(i),
                                                        multiDeviceTensor);
  }
}

} // namespace

void run(const ::tt::target::ttnn::CpuOp *op, ProgramContext &context) {
  common::WrappedFunc fn = context.getDylibManager().getFunc(
      op->dylib_id(), op->func_name()->c_str());
  LOG_ASSERT(fn != nullptr);

  const auto *fbInputs = op->ins();
  const auto *fbOutputs = op->outs();

  bool isMultiChip = context.getMeshDevice().num_devices() > 1;

  if (isMultiChip) {
    runMultiChip(fn, fbInputs, fbOutputs, context);
  } else {
    runSingleChip(fn, fbInputs, fbOutputs, context);
  }
}

} // namespace tt::runtime::ttnn::operations::cpu
