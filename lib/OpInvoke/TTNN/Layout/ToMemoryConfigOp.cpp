// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Layout/ToMemoryConfigOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include <optional>

namespace ttnn_op_invoke {

ToMemoryConfigResolvedParams
resolveToMemoryConfigParams(const ::tt::target::ttnn::ToMemoryConfigOpT &op) {
  ToMemoryConfigResolvedParams params;

  TT_INVOKE_ASSERT(op.out, "ToMemoryConfigOp must have memory config");

  params.outputMemoryConfig = *operations::utils::createMemoryConfigIfNeeded(
      operations::utils::getTensorRefMemoryConfig(*op.out));

  return params;
}

template <typename Tag>
auto createToMemoryConfigTuple(Tag tag,
                               const ::tt::target::ttnn::ToMemoryConfigOpT &op,
                               TensorArg input,
                               const ToMemoryConfigResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag),
                         params.outputMemoryConfig, /*dtype=*/std::nullopt,
                         /*output_tensor=*/std::nullopt);
}

ToMemoryConfigOpResult
callToMemoryConfig(CallType callType,
                   const ::tt::target::ttnn::ToMemoryConfigOpT &op,
                   TensorArg input, ::ttnn::MeshDevice *device) {
  ToMemoryConfigResolvedParams params = resolveToMemoryConfigParams(op);

  auto makeTuple = [&](auto tag) {
    return createToMemoryConfigTuple(tag, op, input, params);
  };

  return callOp<ToMemoryConfigOpResult>(WRAP_OP(::ttnn::to_memory_config),
                                        callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
