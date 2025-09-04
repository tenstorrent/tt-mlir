// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/point_to_point.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/distributed/types.hpp"
#include <cstdint>
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

  ::ttnn::MeshShape meshShape = inputTensor.device()->shape();
  auto calcIdFromCoords =
      [&](const flatbuffers::Vector<uint32_t> *coords) -> size_t {
    DEBUG_ASSERT(coords->size() == meshShape.dims(),
                 "MeshShape and coords size mismatch");
    size_t id = 0;
    for (size_t i = 0; i < meshShape.dims(); i++) {
      id = id * meshShape[i] + (*coords)[i];
    }
    return id;
  };

  size_t sendId = calcIdFromCoords(op->send_coord());
  size_t recvId = calcIdFromCoords(op->receive_coord());

  outputTensorsHost[recvId] = inputTensorsHost[sendId];

  ::ttnn::Tensor outputTensor =
      ::ttnn::to_device(::ttnn::distributed::from_host_shards(
                            outputTensorsHost, inputTensor.device()->shape()),
                        inputTensor.device(), inputTensor.memory_config());

  tensorPool.insertTTNNTensorAndValidate(op->out(), outputTensor);
}
} // namespace tt::runtime::ttnn::operations::ccl
