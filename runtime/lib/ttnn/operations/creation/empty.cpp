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
createEmptyOnMultiDevice(ProgramContext &context, EmptyTensorConfig &config,
                         const ::tt::target::DeviceRef *deviceRef) {
  ::tt::tt_metal::DistributedTensorConfig strategy =
      config.distributedTensorConfig();
  std::vector<::ttnn::Tensor> tensorShards;
  tensorShards.resize(config.numShards);
  std::generate_n(
      tensorShards.begin(), config.numShards, [&config]() -> ::ttnn::Tensor {
        return ::ttnn::zeros(config.shape, config.dtype, config.layout);
      });
  ::ttnn::Tensor out = ::ttnn::distributed::api::create_multi_device_tensor(
      tensorShards, ::tt::tt_metal::StorageType::MULTI_DEVICE_HOST, strategy);
  if (deviceRef) {
    ::ttnn::MeshDevice &meshDevice = context.getSubMesh(deviceRef->global_id());
    LOG_ASSERT(config.numShards == meshDevice.num_devices());
    out = ::ttnn::to_device(out, &meshDevice, config.memoryConfig);
  }
  return out;
}

static ::ttnn::Tensor
createEmptyOnSingleDevice(ProgramContext &context, EmptyTensorConfig &config,
                          const ::tt::target::DeviceRef *deviceRef) {
  if (deviceRef) {
    ::ttnn::MeshDevice &subMesh = context.getSubMesh(deviceRef->global_id());
    LOG_ASSERT(subMesh.num_devices() == 1);
    ::ttnn::Device *device = subMesh.get_device_index(0);
    return ::ttnn::empty(config.shape, config.dtype, config.layout, device,
                         config.memoryConfig.value());
  }
  return ::ttnn::zeros(config.shape, config.dtype, config.layout);
}

void run(const ::tt::target::ttnn::EmptyOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  EmptyTensorConfig config(op);
  ::ttnn::Tensor out;
  if (config.numShards == 1) {
    out = createEmptyOnSingleDevice(context, config, op->device());
  } else if (config.numShards > 1) {
    out = createEmptyOnMultiDevice(context, config, op->device());
  } else {
    LOG_FATAL("Unsupported num shards");
  }
  utils::updateTensorPool(tensorPool, out, op->out()->global_id());
}
} // namespace tt::runtime::ttnn::operations::creation
