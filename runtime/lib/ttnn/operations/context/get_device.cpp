// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "get_device.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::context {

using ::tt::tt_metal::distributed::MeshOffset;
using ::tt::tt_metal::distributed::MeshShape;
using ::tt::tt_metal::distributed::MeshType;

static MeshOffset
calculateMeshOffset(const ::ttnn::MeshDevice &parentMesh,
                    const std::unordered_set<uint32_t> &desiredDeviceIds,
                    const ::tt::target::Dim2d *subMeshShape) {
  for (size_t row = 0; row < parentMesh.num_rows(); row++) {
    for (size_t col = 0; col < parentMesh.num_cols(); col++) {
      const ::ttnn::Device *currDevice = parentMesh.get_device(row, col);
      if (desiredDeviceIds.contains(currDevice->id())) {
        return MeshOffset(row, col);
      }
    }
  }
  throw std::runtime_error("Could not find any desired device in parent mesh");
}

static std::shared_ptr<::ttnn::MeshDevice>
createSubMesh(::ttnn::MeshDevice &parentMesh,
              const std::unordered_set<uint32_t> &desiredDeviceIds,
              const ::tt::target::Dim2d *subMeshShape) {
  // Carve out a submesh from the parentMesh
  MeshShape meshShape(subMeshShape->y(), subMeshShape->x());
  MeshOffset offset =
      calculateMeshOffset(parentMesh, desiredDeviceIds, subMeshShape);
  return parentMesh.create_submesh(meshShape, offset, MeshType::RowMajor);
}

void run(const ::tt::target::ttnn::GetDeviceOp *op, ProgramContext &context) {
  ::ttnn::MeshDevice &meshDevice = context.getParentMesh();
  const ::tt::target::Dim2d *subMeshShape = op->mesh();
  LOG_ASSERT(
      subMeshShape->y() == 1,
      "Expected mesh row = 1 for get device op, got: ", subMeshShape->y());
  const ::flatbuffers::Vector<uint32_t> *deviceIds = op->chip_ids();
  std::unordered_set<uint32_t> desiredDeviceIds(deviceIds->begin(),
                                                deviceIds->end());
  LOG_ASSERT(desiredDeviceIds.size() == deviceIds->size(),
             "Duplicate device ids in get device op");
  std::shared_ptr<::ttnn::MeshDevice> subMesh =
      createSubMesh(meshDevice, desiredDeviceIds, subMeshShape);
  context.addSubMesh(op->out()->global_id(), subMesh);
}
} // namespace tt::runtime::ttnn::operations::context
