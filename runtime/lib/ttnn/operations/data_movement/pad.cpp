// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/pad.h"
#include "tt-metalium/span.hpp"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/workarounds.h"
#include "tt/runtime/ttnn/operations/utils.h"

#include <optional>

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::PadOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(in.is_allocated());

  float padValue = op->value();

  ::ttnn::Tensor out;

  std::vector<std::pair<uint32_t, uint32_t>> padding;
  for (uint32_t i = 0; i < op->padding()->size(); i += 2) {
    padding.push_back(
        std::make_pair(op->padding()->Get(i), op->padding()->Get(i + 1)));
  }

  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      op->memcfg() ? std::make_optional(
                         utils::createMemoryConfig(op->memcfg(), op->out()))
                   : std::nullopt;

  if (workaround::Env::get().usePaddingPairSignatureWithQueueId) {
    out = ::ttnn::pad(::ttnn::DefaultQueueId, in, padding, padValue,
                      op->use_multicore(), outputMemoryConfig);
  } else {
    LOG_FATAL("Currently, the only ::ttnn::pad signature which supports "
              "padding pairs as input has queue_id in its signature. The only "
              "way to execute the operation currently is to use the workaround "
              "enabled by the flag \"usePaddingPairSignatureWithQueueId\"");
  }

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
