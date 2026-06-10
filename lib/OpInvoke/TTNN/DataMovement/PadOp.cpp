// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/DataMovement/PadOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

PadResolvedParams resolvePadParams(const ::tt::target::ttnn::PadOpT &padOp) {
  PadResolvedParams params;

  for (uint32_t i = 0; i + 1 < padOp.padding.size(); i += 2) {
    params.padding.emplace_back(padOp.padding[i], padOp.padding[i + 1]);
  }

  if (padOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*padOp.out));
    LOG_ASSERT(operations::utils::inSystemMemory(*padOp.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createPadTuple(Tag tag, const ::tt::target::ttnn::PadOpT &padOp,
                    TensorArg input, const PadResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag), params.padding,
                         padOp.value, padOp.use_multicore,
                         params.outputMemoryConfig);
}

PadOpResult callPad(CallType callType, const ::tt::target::ttnn::PadOpT &padOp,
                    TensorArg input, ::ttnn::MeshDevice *device) {
  PadResolvedParams params = resolvePadParams(padOp);

  auto makeTuple = [&](auto tag) { return createPadTuple(tag, padOp, input, params); };

  return callOp<PadOpResult>(WRAP_OP(::ttnn::pad), callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
