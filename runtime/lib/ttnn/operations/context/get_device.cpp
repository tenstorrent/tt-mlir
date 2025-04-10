// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/context/get_device.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::context {

using ::tt::tt_metal::distributed::MeshCoordinate;
using ::tt::tt_metal::distributed::MeshShape;

static std::shared_ptr<::ttnn::MeshDevice>
createSubMeshDevice(::ttnn::MeshDevice &parentMesh,
                    const ::tt::target::Dim2d *subMeshShape,
                    const ::tt::target::Dim2d *subMeshOffset) {
  // Carve out a submesh from the parentMesh
  MeshShape meshShape(subMeshShape->y(), subMeshShape->x());
  MeshCoordinate meshOffset(subMeshOffset->y(), subMeshOffset->x());
  return parentMesh.create_submesh(meshShape, meshOffset);
}

void run(const ::tt::target::ttnn::GetDeviceOp *op, ProgramContext &context) {
  ::ttnn::MeshDevice &meshDevice = context.getParentMesh();
  const ::tt::target::Dim2d *subMeshShape = op->mesh();
  const ::tt::target::Dim2d *subMeshOffset = op->offset();

  // Re-map mesh if subMeshShape cannot be a submesh of current shape
  MeshShape meshShape = meshDevice.shape();
  if (subMeshShape->y() > static_cast<int32_t>(meshShape[0]) ||
      subMeshShape->x() > static_cast<int32_t>(meshShape[1])) {
    meshDevice.reshape(MeshShape(subMeshShape->y(), subMeshShape->x()));
    LOG_INFO("remapped mesh device shape [", meshDevice.shape()[0], ", ",
             meshDevice.shape()[1], "]");
  }
  std::shared_ptr<::ttnn::MeshDevice> subMesh =
      createSubMeshDevice(meshDevice, subMeshShape, subMeshOffset);
  context.addSubMesh(op->out()->global_id(), subMesh);
}
} // namespace tt::runtime::ttnn::operations::context
