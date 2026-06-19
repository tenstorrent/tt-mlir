// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Layout/ToLayoutOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

ToLayoutResolvedParams
resolveToLayoutParams(const ::tt::target::ttnn::ToLayoutOpT &op) {
  ToLayoutResolvedParams params;

  params.layout = operations::utils::toTTNNLayout(op.layout);

  if (op.out) {
    params.dtype = operations::utils::getDataType(*op.out);
  }

  if (op.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*op.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*op.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createToLayoutTuple(Tag tag, const ::tt::target::ttnn::ToLayoutOpT &op,
                         TensorArg input,
                         const ToLayoutResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag), params.layout,
                         params.dtype, params.outputMemoryConfig,
                         /*sub_core_grid=*/std::nullopt, /*pad_value=*/0.0f);
}

ToLayoutOpResult callToLayout(CallType callType,
                              const ::tt::target::ttnn::ToLayoutOpT &op,
                              TensorArg input, ::ttnn::MeshDevice *device) {
  ToLayoutResolvedParams params = resolveToLayoutParams(op);

  auto makeTuple = [&](auto tag) {
    return createToLayoutTuple(tag, op, input, params);
  };

  return callOp<ToLayoutOpResult>(WRAP_OP(::ttnn::to_layout), callType,
                                  makeTuple, device);
}

} // namespace ttnn_op_invoke
