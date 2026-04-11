// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/all_to_all_dispatch_metadata.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/experimental/ccl/all_to_all_dispatch_metadata/all_to_all_dispatch_metadata.hpp"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::AllToAllDispatchMetadataOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input_tensor());
  const ::ttnn::Tensor &expertIndices =
      tensorPool.getTTNNTensorAndValidate(op->expert_indices());
  const ::ttnn::Tensor &expertScores =
      tensorPool.getTTNNTensorAndValidate(op->expert_scores());
  const ::ttnn::Tensor &expertMapping =
      tensorPool.getTTNNTensorAndValidate(op->expert_mapping());

  std::optional<uint32_t> axis =
      std::make_optional<uint32_t>(op->cluster_axis());

  // The metal kernel requires drain_sync_tilizer_core to be provided explicitly
  // when persistent output tensors are not supplied. Read from flatbuffer.
  std::optional<tt::tt_metal::CoreCoord> drainCore;
  if (op->drain_core()) {
    drainCore = tt::tt_metal::CoreCoord(op->drain_core()->x(),
                                        op->drain_core()->y());
  }

  auto [dispatched, indices, scores] =
      ::ttnn::experimental::all_to_all_dispatch_metadata(
          input, expertIndices, expertScores, expertMapping,
          /*axis=*/axis,
          /*optional_output_tensors=*/std::nullopt,
          /*num_links=*/std::nullopt,
          /*drain_sync_tilizer_core=*/drainCore);

  tensorPool.insertTTNNTensorAndValidate(op->dispatched(), dispatched);
  tensorPool.insertTTNNTensorAndValidate(op->indices(), indices);
  tensorPool.insertTTNNTensorAndValidate(op->scores(), scores);
}
} // namespace tt::runtime::ttnn::operations::ccl
