// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/pool/upsample.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/operations/utils.h"

namespace tt::runtime::ttnn::operations::pool {
void run(const ::tt::target::ttnn::UpsampleOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(input.is_allocated());

  std::variant<int32_t, std::array<uint32_t, 2>> scaleFactor;
  if (op->scale_factor_type() == ::tt::target::ttnn::Scale2D::UniformScale2D) {
    scaleFactor = op->scale_factor_as_UniformScale2D()->scale();
  } else if (op->scale_factor_type() ==
             ::tt::target::ttnn::Scale2D::NonUniformScale2D) {
    std::array<uint32_t, 2> scaleHW;
    const ::flatbuffers::Vector<int32_t> *fbScaleFactor =
        op->scale_factor_as_NonUniformScale2D()->scale();
    std::copy(fbScaleFactor->begin(), fbScaleFactor->end(), scaleHW.begin());
    scaleFactor = scaleHW;
  } else {
    DEBUG_ASSERT(false);
  }

  std::string mode = op->mode()->str();
  std::optional<tt::tt_metal::MemoryConfig> memoryConfig =
      op->memory_config() ? std::make_optional(utils::createMemoryConfig(
                                op->memory_config(), op->out()))
                          : std::nullopt;

  ::ttnn::Tensor output = ::ttnn::upsample(input, scaleFactor, mode);

  tensorPool.insert_or_assign(op->out()->global_id(), std::move(output));
}
} // namespace tt::runtime::ttnn::operations::pool
