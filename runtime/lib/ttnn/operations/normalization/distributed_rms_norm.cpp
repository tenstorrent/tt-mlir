// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/distributed_rms_norm.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/ccl/rms_allgather/rms_allgather.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace tt::runtime::ttnn::operations::distributed_rms_norm {
void run(const ::tt::target::ttnn::DistributedRMSNormOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());

  std::optional<::ttnn::Tensor> weight = std::nullopt;
  if (op->weight()) {
    weight = tensorPool.getTTNNTensorAndValidate(op->weight());
  }

  std::optional<::ttnn::Tensor> residual = std::nullopt;
  if (op->residual()) {
    residual = tensorPool.getTTNNTensorAndValidate(op->residual());
  }

  uint32_t clusterAxis = op->cluster_axis();
  float epsilon = op->epsilon();

  std::optional<::tt::tt_metal::SubDeviceId> subDeviceId =
      op->sub_device_id() ? std::make_optional<::tt::tt_metal::SubDeviceId>(
                                op->sub_device_id().value())
                          : std::nullopt;

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  std::optional<size_t> numLinks = std::nullopt;
  if (op->num_links()) {
    numLinks = static_cast<size_t>(op->num_links().value());
  }

  ::ttnn::ccl::Topology topology = ::ttnn::ccl::Topology::Linear;
  if (op->topology()) {
    topology = static_cast<::ttnn::ccl::Topology>(
        ::tt::runtime::common::toMetalTopology(op->topology().value()));
  }

  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig = std::nullopt;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();

  auto shardSpec = input.shard_spec();
  ::ttnn::prim::LayerNormProgramConfig programConfig =
      ::ttnn::prim::create_program_config(shardSpec);

  LOG_ASSERT(shardSpec.has_value(),
             "Input tensor must have shard spec for distributed_rms_norm");
  auto semaphore = ::ttnn::global_semaphore::create_global_semaphore(
      &meshDevice, shardSpec->grid, 0);

  // Stats scratch tensor: one tile (32x32) per device, width-sharded on core
  // (0,0). The fused kernel writes partial RMS statistics here and exchanges
  // them across devices via the allgather. The dtype must match the CB data
  // format: Float32 when fp32_dest_acc_en, BFloat16 otherwise.
  auto arch = meshDevice.arch();
  auto kernelConfigVal = ::ttnn::init_device_compute_kernel_config(
      arch, computeConfig, MathFidelity::HiFi4, true, false, false);
  auto [mathFidelity, mathApproxMode, fp32DestAccEn, packerL1Acc,
        dstFullSyncEn] =
      ::ttnn::get_compute_kernel_config_args(arch, kernelConfigVal);
  auto statsDataType = fp32DestAccEn ? ::ttnn::DataType::FLOAT32
                                     : ::ttnn::DataType::BFLOAT16;

  auto statsShardSpec = ::tt::tt_metal::ShardSpec(
      ::tt::tt_metal::CoreRangeSet(
          {::tt::tt_metal::CoreRange({0, 0}, {0, 0})}),
      {32, 32}, ::tt::tt_metal::ShardOrientation::ROW_MAJOR);
  auto statsMemConfig = ::ttnn::MemoryConfig(
      ::tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
      ::tt::tt_metal::BufferType::L1, statsShardSpec);
  ::ttnn::TensorSpec statsSpec(
      ::ttnn::Shape({1, 1, 32, 32}),
      ::ttnn::TensorLayout(statsDataType,
                           ::ttnn::PageConfig(::ttnn::Layout::TILE),
                           statsMemConfig));
  ::ttnn::Tensor statsTensor =
      ::tt::tt_metal::create_device_tensor(statsSpec, &meshDevice);

  ::ttnn::Tensor output = ::ttnn::fused_rms_minimal(
      input, programConfig, clusterAxis, meshDevice, semaphore,
      /*persistent_output_tensor=*/std::nullopt, numLinks, topology,
      subDeviceId,
      /*dtype=*/std::nullopt, computeConfig, memoryConfig, residual, epsilon,
      weight, statsTensor,
      /*use_noc1_only=*/false);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::distributed_rms_norm
