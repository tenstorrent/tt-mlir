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
#include <cstring>
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
    ProgramContext *context = nullptr, uint32_t funcId = 0) {

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

  // X280 path: dispatch the task to the L2CPU0 X280 core via a Step6Task.
  // The firmware (persistent task loop) was booted during openMeshDevice();
  // the code blob was loaded during submit(). We build a Step6Task with
  // per-tensor DRAM bank metadata and use the sequence-number protocol.
  LOG_ASSERT(context != nullptr);

  // Sequence counter for the kick/done handshake. The firmware echoes the
  // kick value in done; incrementing ensures each task is distinguishable.
  static uint32_t seqCounter = 0;

  ::ttnn::MeshDevice &meshDevice = context->getMeshDevice();
  const auto &allocator = meshDevice.allocator();
  uint32_t numBanks = allocator->get_num_banks(tt::tt_metal::BufferType::DRAM);
  LOG_ASSERT(numBanks > 0 && numBanks <= poc::kStep4MaxBanks);

  // DPS convention: fbInputs contains all tensor arguments (real inputs
  // followed by pre-allocated output destinations). fbOutputs references
  // the output tensors (which are the last N entries of fbInputs).
  uint32_t numInputs = fbInputs->size() - fbOutputs->size();
  uint32_t numTensors = fbInputs->size();
  LOG_ASSERT(numTensors <= poc::kStep6MaxTensors);

  poc::Step6Task task{};
  task.num_tensors = numTensors;
  task.num_banks = numBanks;
  task.func_id = funcId;

  for (uint32_t bankId = 0; bankId < numBanks; bankId++) {
    auto logical = meshDevice.logical_core_from_dram_channel(bankId);
    auto noc =
        meshDevice.virtual_core_from_logical_core(logical, tt::CoreType::DRAM);
    task.bank_x[bankId] = static_cast<uint32_t>(noc.x);
    task.bank_y[bankId] = static_cast<uint32_t>(noc.y);
  }

  // Fill per-tensor metadata.
  std::vector<::ttnn::Tensor> outputTensors;
  for (uint32_t t = 0; t < numTensors; t++) {
    const auto *tensorRef = fbInputs->Get(t);
    const auto &tensor =
        context->getTensorPool().getTTNNTensorAndValidate(tensorRef);

    const auto &meshBuf = tensor.mesh_buffer();
    tt::tt_metal::Buffer *buf = meshBuf.get_reference_buffer();
    LOG_ASSERT(buf && buf->buffer_type() == tt::tt_metal::BufferType::DRAM,
               "X280 path requires DRAM-resident tensors");

    auto &tm = task.tensors[t];
    tm.aligned_page_size = buf->aligned_page_size();
    tm.page_size = buf->page_size();
    tm.num_pages = meshBuf.num_pages();
    tm.total_size_bytes = meshBuf.num_pages() * buf->page_size();
    tm.is_input = (t < numInputs) ? 1 : 0;
    tm.is_output = (t >= numInputs) ? 1 : 0;
    std::memset(tm.pad, 0, sizeof(tm.pad));

    // Extract memref sizes and strides from the tensor shape.
    std::vector<int64_t> sizes = common::extractSizes(tensorRef);
    std::vector<std::vector<int64_t>> allSizesAndStrides;
    common::prepareSizesAndStrides(sizes, allSizesAndStrides);
    tm.rank = static_cast<uint32_t>(sizes.size());
    LOG_ASSERT(tm.rank <= poc::kStep6MaxRank);
    std::memset(tm.sizes_and_strides, 0, sizeof(tm.sizes_and_strides));
    std::memcpy(tm.sizes_and_strides, allSizesAndStrides.back().data(),
                tm.rank * 2 * sizeof(int64_t));

    std::memset(tm.bank_base, 0, sizeof(tm.bank_base));
    for (uint32_t bankId = 0; bankId < numBanks; bankId++) {
      int32_t bankOffset =
          allocator->get_bank_offset(tt::tt_metal::BufferType::DRAM, bankId);
      tm.bank_base[bankId] = static_cast<uint64_t>(
          static_cast<int64_t>(meshBuf.address()) + bankOffset);
    }

    if (t >= numInputs) {
      outputTensors.push_back(tensor);
    }
  }

  int deviceId = meshDevice.get_device_ids().at(0);
  auto dev = ::tt::umd::TTDevice::create(deviceId);

  // Write the task body (after kick, up to done) then kick with the
  // sequence number. The host must never write to `done`.
  uint32_t seq = ++seqCounter;
  const size_t bodyOffset = sizeof(uint32_t); // skip kick
  const size_t bodySize = offsetof(poc::Step6Task, done) - bodyOffset;
  poc::NocWrite(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY,
                poc::kTaskAddr + bodyOffset,
                reinterpret_cast<const uint8_t *>(&task) + bodyOffset,
                bodySize);
  poc::NocWrite32(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY, poc::kTaskAddr,
                  seq);

  // Poll for done == seq.
  const uint64_t doneAddr = poc::kTaskAddr + offsetof(poc::Step6Task, done);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  uint32_t doneVal = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    doneVal =
        poc::NocRead32(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY, doneAddr);
    if (doneVal == seq) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  LOG_ASSERT(doneVal == seq, "X280 task timed out: done=", doneVal,
             " expected=", seq);

  return outputTensors;
}

// Executes CPU-hoisted function for single-chip workloads.
void runSingleChip(
    common::WrappedFunc fn,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::ttnn::TensorRef>>
        *fbInputs,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::ttnn::TensorRef>>
        *fbOutputs,
    ProgramContext &context, uint32_t funcId) {

  std::vector<::ttnn::Tensor> outputs =
      executeCPUHoistedFunction(fn, fbInputs, fbOutputs, {},
                                /*onHost=*/false, &context, funcId);

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
  uint32_t funcId = op->dylib_id();

  bool isMultiChip = context.getMeshDevice().num_devices() > 1;

  if (isMultiChip) {
    runMultiChip(fn, fbInputs, fbOutputs, context);
  } else {
    runSingleChip(fn, fbInputs, fbOutputs, context, funcId);
  }
}

} // namespace tt::runtime::ttnn::operations::cpu
