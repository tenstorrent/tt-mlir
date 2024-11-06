// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "full.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/detail/workarounds.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::creation {
struct FullTensorConfig {
  ::ttnn::Shape shape;
  ::ttnn::DataType dtype;
  ::ttnn::Layout layout;
  float fillValue;
  uint32_t numShards;
  const ::tt::target::DistributionStrategy *strategy = nullptr;
  std::optional<::ttnn::MemoryConfig> memoryConfig = std::nullopt;

  FullTensorConfig(const ::tt::target::ttnn::FullOp *op)
      : shape(::tt::runtime::ttnn::utils::toShapeFromFBShape(
            *op->out()->desc()->shape())),
        dtype(::tt::runtime::ttnn::operations::utils::getDataType(op->out())),
        fillValue(op->fill_value()), numShards(op->num_shards()),
        strategy(op->strategy()) {

    layout = utils::inferLayoutFromTileShape(op->out());

    // TODO(bug #272), determine correct layout by tile shape in the future
    // currently tile shape is not set correctly, so as a workaround, hardcode
    // layout
    if (workaround::Env::get().ignoreTileShape) {
      layout = ::ttnn::Layout::TILE;
    }

    // TODO(bug #582): ttnn::empty doesn't work properly with tile layout,
    // using ROW_MAJOR until we fix it
    if (workaround::Env::get().fullOpForceRowMajor) {
      layout = ::ttnn::Layout::ROW_MAJOR;
    }

    if (!utils::inSystemMemory(op->out())) {
      memoryConfig =
          ::tt::runtime::ttnn::operations::utils::createMemoryConfig(op->out());
    }
    validate();
  }

  void validate() const {
    LOG_ASSERT(strategy, "Strategy must be provided");
    LOG_ASSERT(numShards > 0, "Number of shards must be greater than 0");
    LOG_ASSERT(numShards == 1 ||
                   strategy->strategy_type() !=
                       ::tt::target::DistributedTensorConfig::NONE,
               "Strategy must be provided when num shards is greater than 1");
    LOG_ASSERT(strategy->strategy_type() !=
                   ::tt::target::DistributedTensorConfig::AllGatherTensor,
               "AllGatherTensor is not supported");
  }

  ::tt::tt_metal::DistributedTensorConfig distributedTensorConfig() const {
    return ::tt::runtime::ttnn::operations::utils::
        distributedTensorConfigFromFlatbuffer(strategy);
  }
};

static ::ttnn::Tensor
createFullOnMultiDevice(FullTensorConfig &config,
                        ::ttnn::MeshDevice *meshDevice = nullptr) {
  ::tt::tt_metal::DistributedTensorConfig strategy =
      config.distributedTensorConfig();
  std::vector<::ttnn::Tensor> tensorShards;
  for (uint32_t i = 0; i < config.numShards; i++) {
    tensorShards.push_back(::ttnn::full(config.shape, config.fillValue,
                                        config.dtype, config.layout));
  }
  ::ttnn::Tensor out = ::ttnn::distributed::api::create_multi_device_tensor(
      tensorShards, ::tt::tt_metal::StorageType::MULTI_DEVICE_HOST, strategy);
  if (meshDevice) {
    out = ::ttnn::to_device(out, meshDevice, config.memoryConfig);
  }
  return out;
}

static ::ttnn::Tensor createFullOnSingleDevice(
    FullTensorConfig &config,
    std::optional<std::reference_wrapper<::ttnn::Device>> device =
        std::nullopt) {
  return ::ttnn::full(config.shape, config.fillValue, config.dtype,
                      config.layout, device, config.memoryConfig);
}

void run(const ::tt::target::ttnn::FullOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  FullTensorConfig config(op);
  std::optional<std::reference_wrapper<::ttnn::Device>> device = std::nullopt;
  ::ttnn::Tensor out;
  bool onDevice = !utils::inSystemMemory(op->out());
  if (config.numShards == 1 && !onDevice) {
    out = createFullOnSingleDevice(config);
  } else if (config.numShards == 1 && onDevice) {
    ::ttnn::MeshDevice &subMesh = context.getSubMesh(op->device()->global_id());
    LOG_ASSERT(subMesh.num_devices() == 1);
    device = std::make_optional(std::ref(*subMesh.get_device_index(0)));
    out = createFullOnSingleDevice(config, device);
  } else if (config.numShards > 1 && !onDevice) {
    out = createFullOnMultiDevice(config);
  } else if (config.numShards > 1 && onDevice) {
    ::ttnn::MeshDevice &subMesh = context.getSubMesh(op->device()->global_id());
    LOG_ASSERT(config.numShards == subMesh.num_devices(),
               "Num shards should equal the number of devices in mesh device");
    out = createFullOnMultiDevice(config, &subMesh);
  } else {
    throw std::invalid_argument(
        "Unsupported num shards and device combination");
  }
  utils::updateTensorPool(tensorPool, out, op->out()->global_id());
}
} // namespace tt::runtime::ttnn::operations::creation
