// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/distributed/distributed_tensor.hpp"

namespace tt::runtime::ttnn::operations::ccl {
using ::ttnn::distributed::MeshComposerConfig;
using ::ttnn::distributed::MeshToTensor;

void run(const ::tt::target::ttnn::AggregateTensorOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  LOG_ASSERT(
      ttnn::utils::isOnHost(input.storage_type()),
      "Input of aggregate_tensor must be HOST. id:", op->in()->global_id());
  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();

  MeshComposerConfig meshComposerConfig;
  meshComposerConfig.dims.assign(op->composer_config()->dims()->begin(),
                                 op->composer_config()->dims()->end());
  ::ttsl::SmallVector<uint32_t> meshShapeOverride(
      op->composer_config()->mesh_shape_override()->begin(),
      op->composer_config()->mesh_shape_override()->end());
  meshComposerConfig.mesh_shape_override = ::ttnn::MeshShape(meshShapeOverride);
  std::unique_ptr<MeshToTensor> meshComposer =
      ::ttnn::distributed::create_mesh_composer(meshDevice, meshComposerConfig);
  ::ttnn::Tensor out =
      ::ttnn::distributed::aggregate_tensor(input, *meshComposer);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
