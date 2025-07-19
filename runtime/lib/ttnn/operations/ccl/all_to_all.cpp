// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/all_to_all.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/experimental/ccl/all_to_all_async/all_to_all_async.hpp"

namespace tt::runtime::ttnn::operations::ccl {

void run(const ::tt::target::ttnn::AllToAllOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());
  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();
  int32_t in_dim = op->in_dim();
  int32_t out_dim = op->out_dim();

  // Creating intermediate and output buffers
  ::ttnn::Shape shape =
      operations::utils::toTTNNShape(*op->out()->desc()->shape());
  ::ttnn::DataType dataType = ::tt::runtime::ttnn::utils::toTTNNDataType(
      op->out()->desc()->layout()->memory_desc()->data_type());
  ::ttnn::Layout layout =
      ::tt::runtime::ttnn::utils::inferLayoutFromTileShape(op->out());
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(memoryConfig.has_value());
  ::ttnn::TensorSpec tensorSpec(
      shape, ::ttnn::TensorLayout(dataType, ::ttnn::PageConfig(layout),
                                  *memoryConfig));
  ::ttnn::Tensor intermediateBuffer =
      ::tt::tt_metal::create_device_tensor(tensorSpec, &meshDevice);
  ::ttnn::Tensor outputBuffer =
      ::tt::tt_metal::create_device_tensor(tensorSpec, &meshDevice);

  // creating multi device global semaphore
  auto semaphore = ::ttnn::global_semaphore::create_global_semaphore(
      &meshDevice,
      meshDevice.worker_cores(::tt::tt_metal::HalProgrammableCoreType::TENSIX,
                              tt::tt_metal::SubDeviceId{0}),
      0, tt::tt_metal::BufferType::L1);
  // running all_to_all
  auto out = ::ttnn::experimental::all_to_all_async(
      input, intermediateBuffer, outputBuffer, in_dim, out_dim, semaphore, 1,
      memoryConfig, ::ttnn::ccl::Topology::Ring, std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
