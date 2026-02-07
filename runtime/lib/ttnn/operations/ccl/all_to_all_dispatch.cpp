// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/all_to_all_dispatch.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/ccl/all_to_all_dispatch/all_to_all_dispatch.hpp"

#include <limits>

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::AllToAllDispatchOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input_tensor());
  const ::ttnn::Tensor &expertIndices =
      tensorPool.getTTNNTensorAndValidate(op->expert_indices());
  const ::ttnn::Tensor &expertMapping =
      tensorPool.getTTNNTensorAndValidate(op->expert_mapping());

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  LOG_DEBUG("all_to_all_dispatch input shape: ", input.logical_shape(),
            " dtype: ", input.dtype(), " layout: ", input.layout());

  bool flattenMesh =
      (op->cluster_axis() == std::numeric_limits<uint32_t>::max());

  auto *meshDevice = input.device();

  if (flattenMesh) {
    g_originalMeshShape = meshDevice->shape();
    g_meshFlattened = true;
    uint32_t totalDevices = g_originalMeshShape[0] * g_originalMeshShape[1];
    meshDevice->reshape(
        tt::tt_metal::distributed::MeshShape({1, totalDevices}));
  }

  std::optional<uint32_t> axis =
      flattenMesh ? std::make_optional<uint32_t>(1)
                  : std::make_optional<uint32_t>(op->cluster_axis());

  auto [dispatched, metadata] =
      ::ttnn::all_to_all_dispatch(input, expertIndices, expertMapping,
                                  /*axis=*/axis,
                                  /*optional_output_tensors=*/std::nullopt,
                                  /*num_links=*/std::nullopt,
                                  /*topology=*/std::nullopt,
                                  /*memory_config=*/memoryConfig,
                                  /*subdevice_id=*/std::nullopt,
                                  /*output_concat_dim=*/std::nullopt);

  // Restore original mesh shape
  if (flattenMesh) {
    meshDevice->reshape(g_originalMeshShape);
  }

  tensorPool.insertTTNNTensorAndValidate(op->dispatched(), dispatched);
  tensorPool.insertTTNNTensorAndValidate(op->metadata(), metadata);
}
} // namespace tt::runtime::ttnn::operations::ccl
