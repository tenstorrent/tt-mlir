// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/creation/empty.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/workarounds.h"

#include "ttnn/distributed/tensor_topology.hpp"
#include "ttnn/tensor/storage.hpp"
#include <tt-metalium/mesh_buffer.hpp>

namespace tt::runtime::ttnn::operations::creation {

// Allocates an empty device tensor using the given bottom_up direction.
// This replicates the ::ttnn::empty() path but exposes the bottom_up flag on
// DeviceLocalBufferConfig so that input-slot tensors can be placed at the
// opposite end of DRAM from trace-capture intermediates.
static ::ttnn::Tensor allocateEmptyTopDown(const ::ttnn::Shape &shape,
                                           ::ttnn::DataType dtype,
                                           ::ttnn::Layout layout,
                                           ::ttnn::MeshDevice &meshDevice,
                                           const ::ttnn::MemoryConfig &memCfg) {
  using namespace tt::tt_metal;
  using namespace tt::tt_metal::distributed;

  TensorSpec tensorSpec(shape, TensorLayout(dtype, PageConfig(layout), memCfg));

  DeviceLocalBufferConfig localConfig{
      .page_size = tensorSpec.compute_page_size_bytes(),
      .buffer_type = memCfg.buffer_type(),
      .sharding_args = tensorSpec.compute_buffer_sharding_args(),
      .bottom_up = false,
  };

  ReplicatedBufferConfig replicatedConfig{
      .size = tensorSpec.compute_packed_buffer_size_bytes(),
  };

  auto meshBuffer =
      MeshBuffer::create(replicatedConfig, localConfig, &meshDevice);

  std::vector<MeshCoordinate> coords;
  coords.reserve(meshDevice.shape().mesh_size());
  for (const auto &coord : MeshCoordinateRange(meshDevice.shape())) {
    coords.push_back(coord);
  }

  DeviceStorage deviceStorage(std::move(meshBuffer), coords);

  ttsl::SmallVector<MeshMapperConfig::Placement> placements(
      meshDevice.shape().dims());
  for (size_t i = 0; i < meshDevice.shape().dims(); i++) {
    placements[i] = MeshMapperConfig::Replicate{};
  }

  TensorTopology tensorTopology{meshDevice.shape(), placements, coords};

  return ::ttnn::Tensor(std::move(deviceStorage), tensorSpec, tensorTopology);
}

void run(const ::tt::target::ttnn::EmptyOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::ttnn::Shape shape = ::tt::runtime::ttnn::operations::utils::toTTNNShape(
      *op->out()->desc()->shape());
  ::ttnn::DataType dtype =
      ::tt::runtime::ttnn::utils::toTTNNDataType(op->dtype());
  ::ttnn::Layout layout =
      ::tt::runtime::ttnn::utils::toTTNNLayout(op->layout());

  ::ttnn::Tensor out;
  if (op->device()) {
    std::optional<::ttnn::MemoryConfig> memoryConfig =
        ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());
    LOG_ASSERT(memoryConfig.has_value(),
               "Memory config is required for device tensors");
    ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();
    if (context.getAllocateTopDown()) {
      out = allocateEmptyTopDown(shape, dtype, layout, meshDevice,
                                 memoryConfig.value());
    } else {
      out = ::ttnn::empty(shape, dtype, layout, &meshDevice,
                          memoryConfig.value());
    }
  } else {
    out = ::ttnn::zeros(shape, dtype, layout);
  }
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::creation
