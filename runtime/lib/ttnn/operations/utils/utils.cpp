// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/workarounds.h"
#include "utils/utils.h"

namespace tt::runtime::ttnn::operations::utils {

void eventSync(::ttnn::MeshDevice *meshDevice, const ::ttnn::QueueId &recordCq,
               const ::ttnn::QueueId &waitCq) {
  ::ttnn::MeshEvent event =
      ::ttnn::events::record_mesh_event(meshDevice, recordCq);
  ::ttnn::events::wait_for_mesh_event(waitCq, event);
}

bool isTilized(const ::tt::target::ttnn::TensorRef *tensorRef) {
  const ::tt::target::Dim2d *tileShape =
      tensorRef->desc()->layout()->memory_desc()->tile_shape();
  return tileShape->x() == 32 && tileShape->y() == 32;
}

::ttnn::DataType getDataType(const ::tt::target::ttnn::TensorRef *tensorRef) {
  return unifiedOpLib::operations::utils::toTTNNDataType(
      tensorRef->desc()->layout()->memory_desc()->data_type());
}

::ttnn::operations::unary::UnaryWithParam
toTTNNUnaryWithParam(const ::tt::target::ttnn::UnaryWithParam &unaryWithParam) {
  ::tt::target::ttnn::UnaryWithParamT unaryWithParamT;
  unaryWithParam.UnPackTo(&unaryWithParamT);
  return unifiedOpLib::operations::utils::toTTNNUnaryWithParam(unaryWithParamT);
}

std::optional<::ttnn::operations::matmul::MatmulProgramConfig>
createMatmulProgramConfigIfNeeded(const ::tt::target::ttnn::MatmulOp *op) {
  if (!op->matmul_program_config()) {
    return std::nullopt;
  }

  ::ttnn::operations::matmul::MatmulProgramConfig matmulProgramConfig;
  switch (op->matmul_program_config_type()) {
  case ::tt::target::ttnn::MatmulProgramConfig::
      MatmulMultiCoreReuseProgramConfig: {
    auto *config =
        op->matmul_program_config_as_MatmulMultiCoreReuseProgramConfig();
    return ::ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig{
        .compute_with_storage_grid_size =
            unifiedOpLib::operations::utils::toTTNNCoreCoord(
                *config->compute_with_storage_grid_size()),
        .in0_block_w = config->in0_block_w(),
        .out_subblock_h = config->out_subblock_h(),
        .out_subblock_w = config->out_subblock_w(),
        .per_core_M = config->per_core_m(),
        .per_core_N = config->per_core_n()};
  }
  case ::tt::target::ttnn::MatmulProgramConfig::
      MatmulMultiCoreReuseMultiCastProgramConfig: {
    auto *config =
        op->matmul_program_config_as_MatmulMultiCoreReuseMultiCastProgramConfig();
    return ::ttnn::operations::matmul::
        MatmulMultiCoreReuseMultiCastProgramConfig{
            .compute_with_storage_grid_size =
                unifiedOpLib::operations::utils::toTTNNCoreCoord(
                    *config->compute_with_storage_grid_size()),
            .in0_block_w = config->in0_block_w(),
            .out_subblock_h = config->out_subblock_h(),
            .out_subblock_w = config->out_subblock_w(),
            .out_block_h = config->out_block_h(),
            .out_block_w = config->out_block_w(),
            .per_core_M = config->per_core_m(),
            .per_core_N = config->per_core_n(),
            .transpose_mcast = config->transpose_mcast(),
            .fused_activation =
                config->fused_activation()
                    ? std::optional<::ttnn::operations::unary::UnaryWithParam>(
                          toTTNNUnaryWithParam(*config->fused_activation()))
                    : std::nullopt,
            .fuse_batch = config->fuse_batch()};
  }
  case ::tt::target::ttnn::MatmulProgramConfig::
      MatmulMultiCoreReuseMultiCast1DProgramConfig: {
    auto *config =
        op->matmul_program_config_as_MatmulMultiCoreReuseMultiCast1DProgramConfig();
    target::ttnn::CoreRangeSetT hopCoresT;
    (*config->hop_cores()).UnPackTo(&hopCoresT);
    return ::ttnn::operations::matmul::
        MatmulMultiCoreReuseMultiCast1DProgramConfig{
            .compute_with_storage_grid_size =
                unifiedOpLib::operations::utils::toTTNNCoreCoord(
                    *config->compute_with_storage_grid_size()),
            .in0_block_w = config->in0_block_w(),
            .out_subblock_h = config->out_subblock_h(),
            .out_subblock_w = config->out_subblock_w(),
            .out_block_h = config->out_block_h(),
            .out_block_w = config->out_block_w(),
            .per_core_M = config->per_core_m(),
            .per_core_N = config->per_core_n(),
            .fuse_batch = config->fuse_batch(),
            .fused_activation =
                config->fused_activation()
                    ? std::optional<::ttnn::operations::unary::UnaryWithParam>(
                          toTTNNUnaryWithParam(*config->fused_activation()))
                    : std::nullopt,
            .mcast_in0 = config->mcast_in0(),
            .gather_in0 = config->gather_in0(),
            .hop_cores =
                unifiedOpLib::operations::utils::toTTNNCoreRangeSet(hopCoresT),
            .num_global_cb_receivers = config->num_global_cb_receivers(),
            .untilize_out = config->untilize_out()};
  }
  case ::tt::target::ttnn::MatmulProgramConfig::
      MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig: {
    auto *config =
        op->matmul_program_config_as_MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig();
    return ::ttnn::operations::matmul::
        MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig{
            .in0_block_w = config->in0_block_w(),
            .per_core_M = config->per_core_m(),
            .per_core_N = config->per_core_n(),
            .fused_activation =
                config->fused_activation()
                    ? std::optional<::ttnn::operations::unary::UnaryWithParam>(
                          toTTNNUnaryWithParam(*config->fused_activation()))
                    : std::nullopt,
        };
  }
  default:
    LOG_FATAL("Unsupported MatmulProgramConfig type");
  }
}

std::optional<::ttnn::operations::matmul::MatmulProgramConfig>
createMatmulProgramConfigIfNeeded(const ::tt::target::ttnn::LinearOp *op) {
  if (!op->matmul_program_config()) {
    return std::nullopt;
  }

  ::ttnn::operations::matmul::MatmulProgramConfig matmulProgramConfig;
  switch (op->matmul_program_config_type()) {
  case ::tt::target::ttnn::MatmulProgramConfig::
      MatmulMultiCoreReuseProgramConfig: {
    auto *config =
        op->matmul_program_config_as_MatmulMultiCoreReuseProgramConfig();
    return ::ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig{
        .compute_with_storage_grid_size =
            unifiedOpLib::operations::utils::toTTNNCoreCoord(
                *config->compute_with_storage_grid_size()),
        .in0_block_w = config->in0_block_w(),
        .out_subblock_h = config->out_subblock_h(),
        .out_subblock_w = config->out_subblock_w(),
        .per_core_M = config->per_core_m(),
        .per_core_N = config->per_core_n()};
  }
  case ::tt::target::ttnn::MatmulProgramConfig::
      MatmulMultiCoreReuseMultiCastProgramConfig: {
    auto *config =
        op->matmul_program_config_as_MatmulMultiCoreReuseMultiCastProgramConfig();
    return ::ttnn::operations::matmul::
        MatmulMultiCoreReuseMultiCastProgramConfig{
            .compute_with_storage_grid_size =
                unifiedOpLib::operations::utils::toTTNNCoreCoord(
                    *config->compute_with_storage_grid_size()),
            .in0_block_w = config->in0_block_w(),
            .out_subblock_h = config->out_subblock_h(),
            .out_subblock_w = config->out_subblock_w(),
            .out_block_h = config->out_block_h(),
            .out_block_w = config->out_block_w(),
            .per_core_M = config->per_core_m(),
            .per_core_N = config->per_core_n(),
            .transpose_mcast = config->transpose_mcast(),
            .fused_activation =
                config->fused_activation()
                    ? std::optional<::ttnn::operations::unary::UnaryWithParam>(
                          toTTNNUnaryWithParam(*config->fused_activation()))
                    : std::nullopt,
            .fuse_batch = config->fuse_batch()};
  }
  case ::tt::target::ttnn::MatmulProgramConfig::
      MatmulMultiCoreReuseMultiCast1DProgramConfig: {
    auto *config =
        op->matmul_program_config_as_MatmulMultiCoreReuseMultiCast1DProgramConfig();
    target::ttnn::CoreRangeSetT hopCoresT;
    (*config->hop_cores()).UnPackTo(&hopCoresT);
    return ::ttnn::operations::matmul::
        MatmulMultiCoreReuseMultiCast1DProgramConfig{
            .compute_with_storage_grid_size =
                unifiedOpLib::operations::utils::toTTNNCoreCoord(
                    *config->compute_with_storage_grid_size()),
            .in0_block_w = config->in0_block_w(),
            .out_subblock_h = config->out_subblock_h(),
            .out_subblock_w = config->out_subblock_w(),
            .out_block_h = config->out_block_h(),
            .out_block_w = config->out_block_w(),
            .per_core_M = config->per_core_m(),
            .per_core_N = config->per_core_n(),
            .fuse_batch = config->fuse_batch(),
            .fused_activation =
                config->fused_activation()
                    ? std::optional<::ttnn::operations::unary::UnaryWithParam>(
                          toTTNNUnaryWithParam(*config->fused_activation()))
                    : std::nullopt,
            .mcast_in0 = config->mcast_in0(),
            .gather_in0 = config->gather_in0(),
            .hop_cores =
                unifiedOpLib::operations::utils::toTTNNCoreRangeSet(hopCoresT),
            .num_global_cb_receivers = config->num_global_cb_receivers(),
            .untilize_out = config->untilize_out()};
  }
  case ::tt::target::ttnn::MatmulProgramConfig::
      MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig: {
    auto *config =
        op->matmul_program_config_as_MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig();
    return ::ttnn::operations::matmul::
        MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig{
            .in0_block_w = config->in0_block_w(),
            .per_core_M = config->per_core_m(),
            .per_core_N = config->per_core_n(),
            .fused_activation =
                config->fused_activation()
                    ? std::optional<::ttnn::operations::unary::UnaryWithParam>(
                          toTTNNUnaryWithParam(*config->fused_activation()))
                    : std::nullopt,
        };
  }
  default:
    LOG_FATAL("Unsupported MatmulProgramConfig type");
  }
}

::ttnn::Conv2dConfig
createConv2dConfig(const ::tt::target::ttnn::Conv2dConfig *config) {
  ::tt::target::ttnn::Conv2dConfigT conv2dConfigT;
  config->UnPackTo(&conv2dConfigT);
  return unifiedOpLib::operations::utils::createConv2dConfig(conv2dConfigT);
}

::ttnn::Conv2dSliceConfig
createConv2dSliceConfig(const ::tt::target::ttnn::Conv2dSliceConfig *config) {
  ::tt::target::ttnn::Conv2dSliceConfigT conv2dSliceConfigT;
  config->UnPackTo(&conv2dSliceConfigT);
  return unifiedOpLib::operations::utils::createConv2dSliceConfig(
      conv2dSliceConfigT);
}

::ttnn::DeviceComputeKernelConfig createDeviceComputeKernelConfig(
    const ::tt::target::ttnn::DeviceComputeKernelConfig *config) {
  ::tt::target::ttnn::DeviceComputeKernelConfigT deviceComputeKernelConfigT;
  config->UnPackTo(&deviceComputeKernelConfigT);
  return unifiedOpLib::operations::utils::createDeviceComputeKernelConfig(
      deviceComputeKernelConfigT);
}

::ttnn::operations::transformer::SDPAProgramConfig
createSDPAProgramConfig(const ::tt::target::ttnn::SDPAConfig *config) {
  ::ttnn::operations::transformer::SDPAProgramConfig sdpaConfig;

  sdpaConfig.compute_with_storage_grid_size =
      unifiedOpLib::operations::utils::toTTNNCoreCoord(
          *config->compute_with_storage_grid_size());

  if (config->sub_core_grids()) {
    sdpaConfig.sub_core_grids = ::tt::runtime::ttnn::utils::toTTNNCoreRangeSet(
        *config->sub_core_grids());
  }

  sdpaConfig.q_chunk_size = config->q_chunk_size();
  sdpaConfig.k_chunk_size = config->k_chunk_size();

  if (config->exp_approx_mode()) {
    sdpaConfig.exp_approx_mode = *config->exp_approx_mode();
  }

  if (config->max_cores_per_head_batch()) {
    sdpaConfig.max_cores_per_head_batch = *config->max_cores_per_head_batch();
  }

  return sdpaConfig;
}

::ttnn::prim::LayerNormProgramConfig
createLayerNormShardedMultiCoreProgramConfig(
    const ::tt::target::ttnn::LayerNormShardedMultiCoreProgramConfig *config) {
  const auto *gridSize = config->compute_with_storage_grid_size();
  return ::ttnn::prim::LayerNormShardedMultiCoreProgramConfig{
      .compute_with_storage_grid_size = {gridSize->x(), gridSize->y()},
      .subblock_w = config->subblock_w(),
      .block_h = config->block_h(),
      .block_w = config->block_w(),
      .inplace = config->inplace(),
  };
}

template <typename T>
static ::ttnn::Tensor toTTNNTensorImpl(
    const ::flatbuffers::Vector<uint8_t> *input, const ::ttnn::Shape &shape,
    const ::ttnn::DataType &outputDataType, ::ttnn::MeshDevice *device,
    const ::ttnn::Layout &layout, const ::ttnn::MemoryConfig &memoryConfig) {
  std::uint64_t numElements = shape.volume();
  size_t elementSize = sizeof(T);
  LOG_ASSERT(numElements * elementSize == input->size(), "Invalid data size");
  std::vector<T> dataVec(numElements);
  for (size_t i = 0; i < numElements; i++) {
    if constexpr (std::is_same_v<T, bfloat16>) {
      uint16_t raw =
          ::flatbuffers::IndirectHelper<uint16_t>::Read(input->data(), i);
      dataVec[i] = std::bit_cast<bfloat16>(raw);
    } else {
      dataVec[i] = ::flatbuffers::IndirectHelper<T>::Read(input->data(), i);
    }
  }
  return ::tt::runtime::ttnn::utils::createTTNNTensor<T>(
      dataVec.data(), shape, outputDataType, device, layout, memoryConfig);
}

::ttnn::Tensor toTTNNTensor(
    const ::flatbuffers::Vector<uint8_t> *input,
    const ::ttnn::DataType &inputDataType, const ::ttnn::Shape &shape,
    const ::ttnn::DataType &outputDataType,
    ::ttnn::MeshDevice *device = nullptr,
    const ::ttnn::Layout &layout = ::ttnn::Layout::ROW_MAJOR,
    const ::ttnn::MemoryConfig &memoryConfig = ::ttnn::DRAM_MEMORY_CONFIG) {
  switch (inputDataType) {
  case ::ttnn::DataType::FLOAT32: {
    return toTTNNTensorImpl<float>(input, shape, outputDataType, device, layout,
                                   memoryConfig);
  }
  case ::ttnn::DataType::BFLOAT16: {
    return toTTNNTensorImpl<bfloat16>(input, shape, outputDataType, device,
                                      layout, memoryConfig);
  }
  case ::ttnn::DataType::UINT32: {
    return toTTNNTensorImpl<uint32_t>(input, shape, outputDataType, device,
                                      layout, memoryConfig);
  }
  case ::ttnn::DataType::UINT16: {
    return toTTNNTensorImpl<uint16_t>(input, shape, outputDataType, device,
                                      layout, memoryConfig);
  }
  case ::ttnn::DataType::UINT8: {
    return toTTNNTensorImpl<uint8_t>(input, shape, outputDataType, device,
                                     layout, memoryConfig);
  }
  case ::ttnn::DataType::INT32: {
    return toTTNNTensorImpl<int32_t>(input, shape, outputDataType, device,
                                     layout, memoryConfig);
  }
  default:
    LOG_FATAL("Unsupported data type");
  }
}

::ttnn::Tensor
allocateTensorOnDevice(const ::tt::target::ttnn::TensorRef *tensorRef,
                       ::ttnn::MeshDevice &meshDevice) {
  ::ttnn::Shape ttnnShape = toTTNNShape(*tensorRef->desc()->shape());
  ::ttnn::DataType ttnnDataType =
      unifiedOpLib::operations::utils::toTTNNDataType(
          tensorRef->desc()->layout()->memory_desc()->data_type());
  ::ttnn::Layout ttnnLayout =
      ::tt::runtime::ttnn::utils::inferLayoutFromTileShape(tensorRef);
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(tensorRef));
  LOG_ASSERT(memoryConfig.has_value());
  ::ttnn::TensorSpec tensorSpec(
      ttnnShape,
      ::ttnn::TensorLayout(ttnnDataType, ::ttnn::PageConfig(ttnnLayout),
                           *memoryConfig));
  ::ttnn::Tensor deviceTensor =
      ::tt::tt_metal::create_device_tensor(tensorSpec, &meshDevice);
  return deviceTensor;
}

} // namespace tt::runtime::ttnn::operations::utils
