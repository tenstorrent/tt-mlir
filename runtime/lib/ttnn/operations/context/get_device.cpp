// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "get_device.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::context {

static std::pair<::tt::tt_metal::Coordinate, ::tt::tt_metal::Coordinate>
deriveMeshViewCoordinates(const ::ttnn::MeshDevice &meshDevice,
                          const std::unordered_set<uint32_t> &desiredDeviceIds,
                          const ::tt::target::Dim2d *meshViewShape) {
  ::tt::tt_metal::Coordinate topLeft, bottomRight;
  for (int row = 0; row < meshDevice.num_rows(); row++) {
    for (int col = 0; col < meshDevice.num_cols(); col++) {
      const ::ttnn::Device *currDevice = meshDevice.get_device(row, col);
      if (desiredDeviceIds.contains(currDevice->id())) {
        topLeft.row = row;
        topLeft.col = col;
        // coords are inclusive when constructing mesh view
        bottomRight.row = topLeft.row + meshViewShape->y() - 1;
        bottomRight.col = topLeft.col + meshViewShape->x() - 1;
        return std::make_pair(topLeft, bottomRight);
      }
    }
  }
  throw std::runtime_error("Device not found in mesh for get device op");
}

static std::unique_ptr<::ttnn::MeshDeviceView>
constructMeshView(const ::ttnn::MeshDevice &meshDevice,
                  const std::unordered_set<uint32_t> &desiredDeviceIds,
                  const ::tt::target::Dim2d *meshViewShape) {
  // Carve out a mesh view from MeshDevice
  auto [topLeft, bottomRight] =
      deriveMeshViewCoordinates(meshDevice, desiredDeviceIds, meshViewShape);

  return std::make_unique<::ttnn::MeshDeviceView>(meshDevice, topLeft,
                                                  bottomRight);
}

void run(const ::tt::target::ttnn::GetDeviceOp *op, ProgramContext &context) {
  const ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();
  const ::tt::target::Dim2d *meshViewShape = op->mesh();
  LOG_ASSERT(
      meshViewShape->y() == 1,
      "Expected mesh row = 1 for get device op, got: ", meshViewShape->y());
  const ::flatbuffers::Vector<uint32_t> *deviceIds = op->chip_ids();
  std::unordered_set<uint32_t> desiredDeviceIds(deviceIds->begin(),
                                                deviceIds->end());
  LOG_ASSERT(desiredDeviceIds.size() == deviceIds->size(),
             "Duplicate device ids in get device op");
  std::unique_ptr<::ttnn::MeshDeviceView> meshView =
      constructMeshView(meshDevice, desiredDeviceIds, meshViewShape);
  context.addMeshView(op->out()->global_id(), std::move(meshView));
}
} // namespace tt::runtime::ttnn::operations::context
