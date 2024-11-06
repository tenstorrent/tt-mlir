// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "empty.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/detail/workarounds.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::creation {
struct EmptyTensorConfig {
  ::ttnn::Shape shape;
  ::ttnn::DataType dtype;
  ::ttnn::Layout layout;
  uint32_t numShards;
  const ::tt::target::DistributionStrategy *strategy = nullptr;
  std::optional<::ttnn::MemoryConfig> memoryConfig = std::nullopt;

  EmptyTensorConfig(const ::tt::target::ttnn::EmptyOp *op)
      : shape(::tt::runtime::ttnn::utils::toShapeFromFBShape(
            *op->out()->desc()->shape())),
        dtype(::tt::runtime::ttnn::operations::utils::getDataType(op->out())),
        numShards(op->num_shards()), strategy(op->strategy()) {
    layout = ::tt::runtime::ttnn::utils::toTTNNLayout(op->layout());
    // TODO(bug #582): ttnn::empty doesn't work properly with tile layout,
    // using ROW_MAJOR until we fix it
    if (workaround::Env::get().emptyOpForceRowMajor) {
      layout = ::ttnn::Layout::ROW_MAJOR;
    }
    if (op->device()) {
      LOG_ASSERT(op->memcfg(),
                 "Memory config must be provided when device is provided");
      memoryConfig = ::tt::runtime::ttnn::operations::utils::createMemoryConfig(
          op->memcfg(), op->out());
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
createEmptyOnMultiDevice(EmptyTensorConfig &config,
                         ::ttnn::MeshDevice *meshDevice = nullptr) {
  ::tt::tt_metal::DistributedTensorConfig strategy =
      config.distributedTensorConfig();
  std::vector<::ttnn::Tensor> tensorShards;
  for (uint32_t i = 0; i < config.numShards; i++) {
    tensorShards.push_back(
        ::ttnn::zeros(config.shape, config.dtype, config.layout));
  }
  ::ttnn::Tensor out = ::ttnn::distributed::api::create_multi_device_tensor(
      tensorShards, ::tt::tt_metal::StorageType::MULTI_DEVICE_HOST, strategy);
  if (meshDevice) {
    out = ::ttnn::to_device(out, meshDevice, config.memoryConfig);
  }
  return out;
}

static ::ttnn::Tensor
createEmptyOnSingleDevice(EmptyTensorConfig &config,
                          ::ttnn::Device *device = nullptr) {
  if (device) {
    return ::ttnn::empty(config.shape, config.dtype, config.layout, device,
                         config.memoryConfig.value());
  }
  return ::ttnn::zeros(config.shape, config.dtype, config.layout);
}

void run(const ::tt::target::ttnn::EmptyOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  EmptyTensorConfig config(op);
  ::ttnn::Tensor out;
  if (config.numShards == 1 && !op->device()) {
    out = createEmptyOnSingleDevice(config);
  } else if (config.numShards == 1 && op->device()) {
    ::ttnn::MeshDevice &subMesh = context.getSubMesh(op->device()->global_id());
    LOG_ASSERT(subMesh.num_devices() == 1);
    ::ttnn::Device *device = subMesh.get_device_index(0);
    out = createEmptyOnSingleDevice(config, device);
  } else if (config.numShards > 1 && !op->device()) {
    out = createEmptyOnMultiDevice(config);
  } else if (config.numShards > 1 && op->device()) {
    ::ttnn::MeshDevice &subMesh = context.getSubMesh(op->device()->global_id());
    LOG_ASSERT(config.numShards == subMesh.num_devices(),
               "Num shards should equal the number of devices in mesh device");
    out = createEmptyOnMultiDevice(config, &subMesh);
  } else {
    throw std::invalid_argument(
        "Unsupported num shards and device combination");
  }
  utils::updateTensorPool(tensorPool, out, op->out()->global_id());
}
} // namespace tt::runtime::ttnn::operations::creation
