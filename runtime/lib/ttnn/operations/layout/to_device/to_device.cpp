// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "to_device.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::layout {
void run(const ::tt::target::ttnn::ToDeviceOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.tensorPool;
  DeviceMap &devicePool = context.devicePool;
  const ::ttnn::Tensor &inputTensor = tensorPool.at(op->in()->global_id());
  assert((utils::isOnHost(inputTensor) or utils::isOnDevice(inputTensor)) &&
         "Unsupported storage type");

  ::ttnn::TensorMemoryLayout tensorMemoryLayout =
      ::tt::runtime::ttnn::utils::toTTNNTensorMemoryLayout(
          op->memcfg()->tensor_memory_layout());

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
  CoreRangeSet ttnnCoreRangeSet = utils::toCoreRangeSet(targetCoreRangeSet);
  std::array<uint32_t, 2> ttnnShardShape;
  std::copy(targetShardShape->begin(), targetShardShape->end(),
            ttnnShardShape.begin());
  ::tt::tt_metal::ShardSpec shardSpec(
      ttnnCoreRangeSet, ttnnShardShape,
      ::tt::tt_metal::ShardOrientation::ROW_MAJOR, false);

  ::ttnn::MemoryConfig memoryConfig = {tensorMemoryLayout, bufferType,
                                       shardSpec};
  ::ttnn::Device &device = utils::getDevice(op->device(), devicePool);
  ::ttnn::Tensor out = ::ttnn::to_device(inputTensor, &device, memoryConfig);

  tensorPool.try_emplace(op->out()->global_id(), out);
}

} // namespace tt::runtime::ttnn::operations::layout
