// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/point_to_point.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/distributed/types.hpp"
#include <optional>

/*
This is a temporary host fallback to ttnn::PointToPoint(..) API.
*/

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::PointToPointOp *op,
         ProgramContext &context) {
  DEBUG_ASSERT(!::tt::runtime::ttnn::utils::inSystemMemory(op->in()),
               "Input tensor of point to point must be on device.");

  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor =
      tensorPool.getTTNNTensorAndValidate(op->in());

  auto extractShardsToHost = [](const ::ttnn::Tensor &deviceTensor) {
    return ::ttnn::distributed::get_device_tensors(
        ::ttnn::from_device(deviceTensor));
  };

  std::vector<::ttnn::Tensor> inputTensorsHost =
      extractShardsToHost(inputTensor);

  std::vector<::ttnn::Tensor> outputTensorsHost;
  bool hasUserProvidedAccumTensor = op->accum_tensor();

  if (hasUserProvidedAccumTensor) {
    outputTensorsHost = extractShardsToHost(
        tensorPool.getTTNNTensorAndValidate(op->accum_tensor()));
  } else {
    outputTensorsHost = inputTensorsHost;
  }

  outputTensorsHost[op->receiver_id()] = inputTensorsHost[op->sender_id()];

  ::ttnn::Tensor outputTensor = ::ttnn::to_device(
      ::ttnn::distributed::from_host_shards(outputTensorsHost,
                                            inputTensor.mesh_device()->shape()),
      inputTensor.mesh_device(), inputTensor.memory_config());

  tensorPool.insertTTNNTensorAndValidate(op->out(), outputTensor);
}
} // namespace tt::runtime::ttnn::operations::ccl
