// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/all_to_all_combine.h"
#include "operations/ccl/all_to_all_dispatch.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/ccl/all_to_all_combine/all_to_all_combine.hpp"

#include <limits>

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::AllToAllCombineOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input_tensor());
  const ::ttnn::Tensor &expertMapping =
      tensorPool.getTTNNTensorAndValidate(op->expert_mapping());
  const ::ttnn::Tensor &expertMetadata =
      tensorPool.getTTNNTensorAndValidate(op->expert_metadata());

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  bool flattenMesh =
      (op->cluster_axis() == std::numeric_limits<uint32_t>::max()) &&
      g_meshFlattened;

  auto *meshDevice = input.device();

  if (flattenMesh) {
    uint32_t totalDevices = g_originalMeshShape[0] * g_originalMeshShape[1];
    meshDevice->reshape(
        tt::tt_metal::distributed::MeshShape({1, totalDevices}));
  }

  std::optional<uint32_t> axis =
      flattenMesh ? std::make_optional<uint32_t>(1)
                  : std::make_optional<uint32_t>(op->cluster_axis());

  ::ttnn::Tensor output =
      ::ttnn::all_to_all_combine(input, expertMapping, expertMetadata,
                                 /*locally_reduced=*/false,
                                 /*num_links=*/std::nullopt,
                                 /*topology=*/std::nullopt,
                                 /*memory_config=*/memoryConfig,
                                 /*axis=*/axis,
                                 /*output_shard_dim=*/std::nullopt,
                                 /*subdevice_id=*/std::nullopt,
                                 /*optional_output_tensor=*/std::nullopt);

  // Restore original mesh shape
  if (flattenMesh) {
    meshDevice->reshape(g_originalMeshShape);
    g_meshFlattened = false;
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::ccl
