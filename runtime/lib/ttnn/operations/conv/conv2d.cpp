// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/conv/conv2d.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "ttnn/operations/experimental/quasar/conv2d/conv2d.hpp"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/types.hpp"

namespace tt::runtime::ttnn::operations::conv {
using ::ttnn::Conv2dResultWithOptions;
void run(const ::tt::target::ttnn::Conv2dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());

  std::optional<::ttnn::Tensor> bias =
      op->bias()
          ? std::make_optional(tensorPool.getTTNNTensorAndValidate(op->bias()))
          : std::nullopt;

  LOG_ASSERT(op->kernel_size()->size() == 2,
             "Kernel size expected to have 2 elements");
  LOG_ASSERT(op->stride()->size() == 2, "Stride expected to have 2 elements");
  LOG_ASSERT(op->padding()->size() == 2 || op->padding()->size() == 4,
             "Padding expected to have 2 or 4 elements");
  LOG_ASSERT(op->dilation()->size() == 2,
             "Dilation expected to have 2 elements");

  std::array<uint32_t, 2> kernelSize, stride, dilation;
  std::copy_n(op->kernel_size()->begin(), 2, kernelSize.begin());
  std::copy_n(op->stride()->begin(), 2, stride.begin());
  std::copy_n(op->dilation()->begin(), 2, dilation.begin());

  std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding;
  if (op->padding()->size() == 2) {
    std::array<uint32_t, 2> symPadding;
    std::copy_n(op->padding()->begin(), 2, symPadding.begin());
    padding = symPadding;
  } else {
    std::array<uint32_t, 4> asymPadding;
    std::copy_n(op->padding()->begin(), 4, asymPadding.begin());
    padding = asymPadding;
  }

  std::optional<::ttnn::DataType> outputDtype;
  if (op->output_dtype()) {
    outputDtype =
        ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->output_dtype()));
  }

  ::ttnn::Conv2dConfig conv2dConfig;
  if (op->conv2d_config()) {
    conv2dConfig = utils::createConv2dConfig(op->conv2d_config());
  }

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  std::optional<::ttnn::Conv2dSliceConfig> sliceConfig;
  if (op->conv2d_slice_config()) {
    sliceConfig = utils::createConv2dSliceConfig(op->conv2d_slice_config());
  }

  // Quasar's conv2d takes the identical argument list; the mainline conv program
  // factory builds a DataMovementKernel and TT_FATALs on Quasar.
  // This frontend emits no shard_layout at all (measured: 0 occurrences across all
  // 53 conv2d in ResNet-50), leaving Quasar's conv2d to choose for itself -- and it
  // fails there. tt-metal's own per-op Quasar conv2d test always passes one
  // explicitly. Same shape of bug as matmul: on Quasar, auto-selection is unsafe.
  //
  // Which one is not a free choice. Measured from the optimised pipeline's own IR,
  // where the compiler does assign it:
  //     224, 56 spatial -> height_sharded (14/14)
  //     28      spatial -> height_sharded (11), block_sharded (2)
  //     14, 7   spatial -> block_sharded  (26/26)
  // A narrow activation spread over many cores wants height sharding; once the
  // spatial extent collapses and the channel count dominates, block sharding is what
  // the reference uses. Splitting at 28 matches 50 of the 53 convolutions.
  //
  // BLOCK_SHARDED is what the tt-metal reference uses at 14/7, but it does not survive this
  // frontend's path: quasar::conv2d sets in0_block_w = full_K to contract K in one block
  // (dodging the matmul K-spill accumulate), and block sharding splits K across the grid
  // COLUMNS, so each core holds only full_K/ncols. quasar::matmul then rejects it with
  //   "shard_shape[1] / in0_tile.get_width() (8) must be divisible by in0_block_w (64)"
  // -- 64/8 being exactly the grid width. conv2d.cpp's own [#48552 Stage2 REVERTED-AGAIN]
  // note reaches the same conclusion: "HEIGHT_SHARDED is the only single-K-block shape".
  // TT_METAL_QSR_CONV_ALL_HS forces height sharding everywhere so that holds at 14/7 too.
  if (utils::isQuasar() && !conv2dConfig.shard_layout.has_value()) {
    const bool forceAllHeightSharded =
        std::getenv("TT_METAL_QSR_CONV_ALL_HS") != nullptr;
    conv2dConfig.shard_layout =
        (forceAllHeightSharded || op->input_height() >= 28)
            ? ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED
            : ::ttnn::TensorMemoryLayout::BLOCK_SHARDED;
  }

  // Gen2 (Quasar) cannot express the DRAM config-landing scratch buffer: halo's
  // reader binds `gather_scratch0` as both PRODUCER and CONSUMER, and
  // program_spec.cpp:1497 rejects self-looped DFBs for data-movement kernels on
  // Gen2. tt-metal defaults this to false; this frontend emits true on all 53
  // convolutions, which is what trips the assert. Not a policy choice -- the
  // DRAM-config path simply does not exist on Gen2.
  if (utils::isQuasar()) {
    conv2dConfig.config_tensors_in_dram = false;
  }

  // Note: forwarding the function *name* into a generic lambda would drop its
  // default arguments, so both calls are spelled out.
  Conv2dResultWithOptions result =
      utils::isQuasar()
          ? ::ttnn::operations::experimental::quasar::conv2d(
                input, weight, &targetDevice, op->in_channels(),
                op->out_channels(), op->batch_size(), op->input_height(),
                op->input_width(), kernelSize, stride, padding, dilation,
                op->groups(), outputDtype, bias, conv2dConfig, computeConfig,
                outputMemoryConfig, sliceConfig)
          : ::ttnn::conv2d(input, weight, &targetDevice, op->in_channels(),
                           op->out_channels(), op->batch_size(),
                           op->input_height(), op->input_width(), kernelSize,
                           stride, padding, dilation, op->groups(), outputDtype,
                           bias, conv2dConfig, computeConfig, outputMemoryConfig,
                           sliceConfig);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result));

  ::ttnn::Tensor out = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::conv
