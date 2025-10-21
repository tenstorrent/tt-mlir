// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/point_to_point.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/point_to_point/point_to_point.hpp"
#include <cstdint>
#include <optional>

namespace tt::runtime::ttnn::operations::ccl {

void run(const ::tt::target::ttnn::PointToPointOp *op,
         ProgramContext &context) {
  DEBUG_ASSERT(!::tt::runtime::ttnn::utils::inSystemMemory(op->in()),
               "Input tensor of point to point must be on device.");

  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor =
      tensorPool.getTTNNTensorAndValidate(op->in());

  const auto *sendCoordFb = op->sender_coord();
  const auto *receiveCoordFb = op->receiver_coord();

  std::vector<uint32_t> sendCoordVec(sendCoordFb->begin(), sendCoordFb->end());
  std::vector<uint32_t> receiveCoordVec(receiveCoordFb->begin(),
                                        receiveCoordFb->end());

  ::ttnn::MeshCoordinate sendCoord(ttsl::make_span(sendCoordVec));
  ::ttnn::MeshCoordinate receiveCoord(ttsl::make_span(receiveCoordVec));

  std::optional<::ttnn::Tensor> optionalOutputTensor = std::nullopt;
  if (op->optional_output_tensor()) {
    optionalOutputTensor =
        tensorPool.getTTNNTensorAndValidate(op->optional_output_tensor());
  }
  auto outputTensor = ::ttnn::point_to_point(
      inputTensor, receiveCoord, sendCoord, ::ttnn::ccl::Topology::Linear,
      optionalOutputTensor);

  tensorPool.insertTTNNTensorAndValidate(op->out(), outputTensor);
}
} // namespace tt::runtime::ttnn::operations::ccl
