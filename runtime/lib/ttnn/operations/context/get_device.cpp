// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/context/get_device.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::context {

using ::tt::tt_metal::distributed::MeshCoordinate;
using ::tt::tt_metal::distributed::MeshShape;

void run(const ::tt::target::ttnn::GetDeviceOp *op, ProgramContext &context) {
  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();
  const ::tt::target::Dim2d *subMeshShape = op->mesh();

  // Re-map mesh if subMeshShape cannot be a submesh of current shape
  MeshShape meshShape = meshDevice.shape();
  if (subMeshShape->y() > static_cast<int32_t>(meshShape[0]) ||
      subMeshShape->x() > static_cast<int32_t>(meshShape[1])) {
    meshDevice.reshape(MeshShape(subMeshShape->y(), subMeshShape->x()));
    LOG_INFO("remapped mesh device shape [", meshDevice.shape()[0], ", ",
             meshDevice.shape()[1], "]");
  }
}
} // namespace tt::runtime::ttnn::operations::context
