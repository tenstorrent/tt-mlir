// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "cpu.h"

#include "tt/runtime/detail/common/dylib.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "tt-metalium/allocator.hpp"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/buffer_types.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "umd/device/tt_device/tt_device.hpp"
#include "x280_driver/driver.h"

#include <chrono>
#include <cstddef>
#include <thread>
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
  case ::ttnn::DataType::UINT32:
    return ::tt::runtime::ttnn::utils::createBorrowedTTNNTensor<uint32_t>(
        dataPtr, shape);
  case ::ttnn::DataType::INT32:
    return ::tt::runtime::ttnn::utils::createBorrowedTTNNTensor<int32_t>(
        dataPtr, shape);
  default:
    LOG_FATAL("Unsupported data type for CPU op output: ", dtype);
  }
}

// Callback for creating TTNN tensor from WrappedTensor output.
static const common::CreateTensorCallbackType<::ttnn::Tensor,
                                              ::tt::target::ttnn::TensorRef>
    createTensorFromWrapped = [](const tt::target::ttnn::TensorRef *ref,
                                 std::shared_ptr<void> dataPtr) {
      ::ttnn::Shape shape = utils::toTTNNShape(*ref->desc()->shape());
      ::ttnn::DataType dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(
          ref->desc()->layout()->memory_desc()->data_type());
      return createBorrowedTensorFromPtr(dataPtr, shape, dtype);
    };

// Executes the provided CPU-hoisted function for the inputs provided
// as raw data pointers, returning outputs of the function as TTNN tensors.
std::vector<::ttnn::Tensor> executeCPUHoistedFunction(
    common::WrappedFunc fn,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::ttnn::TensorRef>>
        *fbInputs,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::ttnn::TensorRef>>
        *fbOutputs,
    const std::vector<void *> &inputDataPtrs, bool onHost = true,
    ProgramContext *context = nullptr) {

  if (onHost) {
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

    common::WrappedTensor *outputArray = fn(packedInputs.data());

    return common::unpackTensors<::ttnn::Tensor>(
        outputArray, fbOutputs->size(), fbOutputs, createTensorFromWrapped);
  }

  // X280 path: dispatch the task to the L2CPU0 X280 core. The firmware was
  // already booted during submit() (see runtime.cpp); we just build a
  // Step5Task with bank tables for the input/output tensors and kick.
  LOG_ASSERT(context != nullptr);
  // Destination-passing style: fbInputs has the real input at [0] and the
  // pre-allocated output destination at [1]; fbOutputs has 1 entry that
  // references the same output tensor.
  LOG_ASSERT(fbInputs->size() == 2 && fbOutputs->size() == 1,
             "X280 DPS path expects 2 inputs (src + dst) and 1 output");

  ::ttnn::MeshDevice &meshDevice = context->getMeshDevice();

  const auto &inputTensor =
      context->getTensorPool().getTTNNTensorAndValidate(fbInputs->Get(0));
  const auto &outputTensor =
      context->getTensorPool().getTTNNTensorAndValidate(fbInputs->Get(1));

  const auto &inMeshBuf = inputTensor.mesh_buffer();
  const auto &outMeshBuf = outputTensor.mesh_buffer();
  tt::tt_metal::Buffer *inBuf = inMeshBuf.get_reference_buffer();
  tt::tt_metal::Buffer *outBuf = outMeshBuf.get_reference_buffer();
  LOG_ASSERT(inBuf && inBuf->buffer_type() == tt::tt_metal::BufferType::DRAM,
             "X280 path requires DRAM-resident input");
  LOG_ASSERT(outBuf && outBuf->buffer_type() == tt::tt_metal::BufferType::DRAM);

  const auto &allocator = meshDevice.allocator();
  uint32_t numBanks = allocator->get_num_banks(tt::tt_metal::BufferType::DRAM);
  LOG_ASSERT(numBanks > 0 && numBanks <= poc::kStep4MaxBanks);

  poc::Step5Task task{};
  task.kick = 0;
  task.num_banks = numBanks;
  task.aligned_page_size = inBuf->aligned_page_size();
  task.page_size = inBuf->page_size();
  task.num_pages = inMeshBuf.num_pages();

  for (uint32_t bankId = 0; bankId < numBanks; bankId++) {
    auto logical = meshDevice.logical_core_from_dram_channel(bankId);
    auto noc =
        meshDevice.virtual_core_from_logical_core(logical, tt::CoreType::DRAM);
    task.bank_x[bankId] = static_cast<uint32_t>(noc.x);
    task.bank_y[bankId] = static_cast<uint32_t>(noc.y);
    int32_t bankOffset =
        allocator->get_bank_offset(tt::tt_metal::BufferType::DRAM, bankId);
    task.input_bank_base[bankId] = static_cast<uint64_t>(
        static_cast<int64_t>(inMeshBuf.address()) + bankOffset);
    task.output_bank_base[bankId] = static_cast<uint64_t>(
        static_cast<int64_t>(outMeshBuf.address()) + bankOffset);
  }

  int deviceId = meshDevice.get_device_ids().at(0);
  auto dev = ::tt::umd::TTDevice::create(deviceId);

  // Write only up to the `done` field — the host must never touch `done`
  // because of a silicon quirk where host-written DRAM words shadow X280
  // writes on subsequent host reads (see driver.h).
  poc::NocWrite(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY, poc::kTaskAddr,
                &task, offsetof(poc::Step5Task, done));
  poc::NocWrite32(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY, poc::kTaskAddr,
                  poc::kKick);

  const uint64_t doneAddr = poc::kTaskAddr + offsetof(poc::Step5Task, done);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  uint32_t doneVal = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    doneVal =
        poc::NocRead32(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY, doneAddr);
    if (doneVal == poc::kDone) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  LOG_ASSERT(doneVal == poc::kDone, "X280 task timed out: done=0x", std::hex,
             doneVal);

  return {outputTensor};
}

// Executes CPU-hoisted function for single-chip workloads.
void runSingleChip(
    common::WrappedFunc fn,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::ttnn::TensorRef>>
        *fbInputs,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::ttnn::TensorRef>>
        *fbOutputs,
    ProgramContext &context) {

  std::vector<::ttnn::Tensor> outputs =
      executeCPUHoistedFunction(fn, fbInputs, fbOutputs, {},
                                /*onHost=*/false, &context);

  for (size_t i = 0; i < outputs.size(); ++i) {
    context.getTensorPool().insertTTNNTensorAndValidate(fbOutputs->Get(i),
                                                        outputs[i]);
  }
}

// Executes CPU-hoisted function for multi-chip workloads.
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

  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();
  const ::ttnn::MeshShape &meshShape = meshDevice.shape();
  size_t numShards = meshDevice.num_devices();

  // Collect input tensors.
  // For multi-device input tensors, extract individual shards.
  // For single-device input tensors, store as a size-1 vector.
  std::vector<std::vector<::ttnn::Tensor>> inputTensors;
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
      inputTensors.emplace_back(
          ::ttnn::distributed::get_device_tensors(tensor));
    } else {
      inputTensors.emplace_back(1, tensor);
    }
  }

  // Prepare output shards container.
  std::vector<std::vector<::ttnn::Tensor>> outputShards(fbOutputs->size());
  for (size_t i = 0; i < fbOutputs->size(); ++i) {
    outputShards[i].reserve(numShards);
  }

  // Execute CPU-hoisted function for each shard.
  std::vector<void *> inputDataPtrs;
  inputDataPtrs.reserve(fbInputs->size());

  for (size_t shardIdx = 0; shardIdx < numShards; ++shardIdx) {
    inputDataPtrs.clear();

    for (size_t i = 0; i < fbInputs->size(); ++i) {
      // For single device tensor inputs, always use index 0.
      // For sharded inputs, use the shard index.
      size_t tensorIdx = inputTensors[i].size() == 1 ? 0 : shardIdx;
      inputDataPtrs.push_back(::tt::runtime::ttnn::utils::getRawHostDataPtr(
          inputTensors[i][tensorIdx]));
    }

    std::vector<::ttnn::Tensor> shardOutputs =
        executeCPUHoistedFunction(fn, fbInputs, fbOutputs, inputDataPtrs);

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
  // common::WrappedFunc fn = context.getDylibManager().getFunc(
  //     op->dylib_id(), op->func_name()->c_str());
  // LOG_ASSERT(fn != nullptr);

  common::WrappedFunc fn = nullptr;

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
