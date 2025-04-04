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

static MeshCoordinate
calculateMeshCoordinate(const ::ttnn::MeshDevice &parentMesh,
                        const std::unordered_set<uint32_t> &desiredDeviceIds,
                        const ::tt::target::Dim2d *subMeshShape) {
  for (uint32_t row = 0; row < parentMesh.shape()[0]; row++) {
    for (uint32_t col = 0; col < parentMesh.shape()[1]; col++) {
      const ::ttnn::IDevice *currDevice = parentMesh.get_device({row, col});
      if (desiredDeviceIds.contains(currDevice->id())) {
        return MeshCoordinate(row, col);
      }
    }
  }
  LOG_FATAL("Could not find any desired device in parent mesh");
}

static std::shared_ptr<::ttnn::MeshDevice>
createSubMesh(::ttnn::MeshDevice &parentMesh,
              const std::unordered_set<uint32_t> &desiredDeviceIds,
              const ::tt::target::Dim2d *subMeshShape) {
  // Carve out a submesh from the parentMesh
  MeshShape meshShape(subMeshShape->y(), subMeshShape->x());
  MeshCoordinate coordinate =
      calculateMeshCoordinate(parentMesh, desiredDeviceIds, subMeshShape);
  return parentMesh.create_submesh(meshShape, coordinate);
}

void run(const ::tt::target::ttnn::GetDeviceOp *op, ProgramContext &context) {
  ::ttnn::MeshDevice &meshDevice = context.getParentMesh();
  const ::tt::target::Dim2d *subMeshShape = op->mesh();
  const ::flatbuffers::Vector<uint32_t> *deviceIds = op->chip_ids();
  std::unordered_set<uint32_t> desiredDeviceIds(deviceIds->begin(),
                                                deviceIds->end());
  LOG_ASSERT(desiredDeviceIds.size() == deviceIds->size(),
             "Duplicate device ids in get device op");

  // Re-map mesh if subMeshShape cannot be a submesh of current shape
  MeshShape meshShape = meshDevice.shape();
  if (subMeshShape->y() > static_cast<int32_t>(meshShape[0]) ||
      subMeshShape->x() > static_cast<int32_t>(meshShape[1])) {
    meshDevice.reshape(MeshShape(subMeshShape->y(), subMeshShape->x()));
    LOG_INFO("remapped mesh device shape [", meshDevice.shape()[0], ", ",
             meshDevice.shape()[1], "]");
  }
  std::shared_ptr<::ttnn::MeshDevice> subMesh =
      createSubMesh(meshDevice, desiredDeviceIds, subMeshShape);
  context.addSubMesh(op->out()->global_id(), subMesh);
}
} // namespace tt::runtime::ttnn::operations::context
