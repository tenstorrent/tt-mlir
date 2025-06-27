// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/point_to_point.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include <optional>
#include <ttnn/distributed/types.hpp>

/*
This is a temporary host fallback to ttnn::PointToPoint(..) API.
*/

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::PointToPointOp *op,
         ProgramContext &context) {
  DEBUG_ASSERT(!::tt::runtime::ttnn::utils::inSystemMemory(op->in()),
               "Calling ttnn::from_device on a host tensor");

  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor =
      tensorPool.getTTNNTensorAndValidate(op->in());

  auto extractShardsToHost = [](const ::ttnn::Tensor &deviceTensor) {
    return ::ttnn::distributed::get_device_tensors(
        ::ttnn::from_device(deviceTensor));
  };

  std::vector<::ttnn::Tensor> inputTensorsHost =
      extractShardsToHost(inputTensor);

  ::ttnn::Tensor outputTensor;
  bool hasUserProvidedOutputTensor = op->output_tensor();

  if (hasUserProvidedOutputTensor) {
    outputTensor = tensorPool.getTTNNTensorAndValidate(op->output_tensor());
  } else {
    outputTensor = ::tt::tt_metal::create_device_tensor(
        inputTensor.tensor_spec(), inputTensor.mesh_device());
  }

  std::vector<::ttnn::Tensor> outputTensorsHost =
      extractShardsToHost(outputTensor);

  outputTensorsHost[op->receiver_id()] = inputTensorsHost[op->sender_id()];

  ::ttnn::Tensor aggregatedTensor = ::ttnn::to_device(
      ::ttnn::distributed::aggregate_as_tensor(
          outputTensorsHost, inputTensor.distributed_tensor_config()),
      inputTensor.mesh_device(), inputTensor.memory_config());

  tensorPool.insertTTNNTensorAndValidate(op->out(), aggregatedTensor);
}
} // namespace tt::runtime::ttnn::operations::ccl
