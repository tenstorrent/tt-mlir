// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/cache/load_cached.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/program_executor.h"
#include "tt/runtime/detail/ttnn/types/global_tensor_cache.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/types.h"
#include "ttnn/distributed/tensor_topology.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/tensor/storage.hpp"
#include <string_view>
#include <tt-metalium/mesh_buffer.hpp>
#include <vector>

namespace tt::runtime::ttnn::operations::cache {

using LogType = ::tt::runtime::logger::LogType;

// Allocates a device tensor with bottom_up=false so it lives at high DRAM
// addresses, safely away from bottom-up trace intermediates.
static ::ttnn::Tensor allocateTopDown(const ::ttnn::Tensor &src,
                                      ::ttnn::MeshDevice &meshDevice) {
  using namespace tt::tt_metal;
  using namespace tt::tt_metal::distributed;

  const ::ttnn::MemoryConfig &memCfg = src.memory_config();
  TensorSpec tensorSpec(
      src.logical_shape(),
      TensorLayout(src.dtype(), PageConfig(src.layout()), memCfg));

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

void run(const ::tt::target::ttnn::LoadCachedOp *op, ProgramContext &context) {
  GlobalTensorCache &cache = GlobalTensorCache::getInstance();

  // Get the device ID from the parent mesh
  const int deviceId = context.getMeshDevice().id();
  const std::string &constEvalFuncname = op->callee_name()->str();

  std::vector<uint64_t> inputVersions;
  inputVersions.reserve(op->inputs()->size());
  // Extract versions for each input tensor.
  for (const auto *input : *op->inputs()) {
    const ::tt::runtime::ttnn::TTNNTensorWrapper &runtimeInput =
        context.getTensorPool().getTTNNTensorWrapperAndValidate(input);
    inputVersions.push_back(runtimeInput.getVersion());
  }

  const auto programHash = op->program_hash()->str();

  const auto cacheKey = CacheKey{deviceId, programHash, inputVersions};

  LOG_DEBUG("Running LoadCachedOp for function ", constEvalFuncname,
            " with hash: ", op->program_hash()->str());

  // Get the cached tensors, which will be empty if cache is invalid
  const std::vector<Tensor> *cachedOutputs = cache.getAll(cacheKey);

  if (cachedOutputs) {
    LOG_DEBUG("Cache hit for function: ", constEvalFuncname.c_str());

    LOG_ASSERT(cachedOutputs->size() == op->outputs()->size());
    for (size_t i = 0; i < cachedOutputs->size(); ++i) {
      context.getTensorPool().insertRuntimeTensorAndValidate(
          op->outputs()->Get(i), (*cachedOutputs)[i]);
    }

    return;
  }

  LOG_DEBUG("Cache miss or invalid cache for function: ", constEvalFuncname);

  // Collect the ::ttnn::Tensor objects for execution
  std::vector<::tt::runtime::Tensor> inputs;
  inputs.reserve(op->inputs()->size());
  for (const auto *input : *op->inputs()) {
    auto &tensor = context.getTensorPool().getRuntimeTensorAndValidate(input);
    inputs.emplace_back(tensor);
  }

  // Execute the function
  const size_t programIndex = op->program_idx();
  ProgramExecutor exec(context.getDeviceHandle(), context.getExecutableHandle(),
                       programIndex, inputs, /*constEvalProgram=*/true);
  exec.execute();
  LOG_DEBUG("executed sub-func: ", constEvalFuncname);
  std::vector<::tt::runtime::Tensor> outputs = exec.gatherOutputTensors();

  // Const-eval outputs (weights, constants) are retained permanently on device
  // and are inputs to traced graphs. Traces record writes to specific DRAM
  // addresses during capture; if a const-eval output lands at one of those
  // addresses, trace replay will corrupt it. To prevent this, we copy each
  // device output into a top-down-allocated tensor so that it lives at a high
  // DRAM address that traces will never write intermediates into.
  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();
  for (::tt::runtime::Tensor &output : outputs) {
    ::tt::runtime::ttnn::TTNNTensorWrapper &outputWrapper =
        output.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN);

    const ::ttnn::Tensor &original = outputWrapper.getTensor();
    if (original.storage_type() != ::ttnn::StorageType::DEVICE) {
      continue;
    }

    ::ttnn::Tensor topDown = allocateTopDown(original, meshDevice);
    ::ttnn::copy(original, topDown);
    output = utils::createRuntimeTensorFromTTNN(topDown);
  }

  // Const-eval outputs need to be retained
  for (::tt::runtime::Tensor &output : outputs) {
    ::tt::runtime::ttnn::TTNNTensorWrapper &outputWrapper =
        output.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN);
    outputWrapper.setRetain(true);
  }

  cache.store(cacheKey, inputs, outputs);

  for (size_t i = 0; i < outputs.size(); ++i) {
    context.getTensorPool().insertRuntimeTensorAndValidate(
        op->outputs()->Get(i), outputs[i]);
  }
}
} // namespace tt::runtime::ttnn::operations::cache
