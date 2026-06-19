// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Layout/TypecastOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

TypecastResolvedParams
resolveTypecastParams(const ::tt::target::ttnn::TypecastOpT &op) {
  TypecastResolvedParams params;

  params.dtype = operations::utils::toTTNNDataType(op.dtype);

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
auto createTypecastTuple(Tag tag, const ::tt::target::ttnn::TypecastOpT &op,
                         TensorArg input,
                         const TypecastResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag), params.dtype,
                         params.outputMemoryConfig,
                         /*optional_output_tensor=*/std::nullopt,
                         /*sub_core_grid=*/std::nullopt);
}

TypecastOpResult callTypecast(CallType callType,
                              const ::tt::target::ttnn::TypecastOpT &op,
                              TensorArg input, ::ttnn::MeshDevice *device) {
  TypecastResolvedParams params = resolveTypecastParams(op);

  auto makeTuple = [&](auto tag) {
    return createTypecastTuple(tag, op, input, params);
  };

  return callOp<TypecastOpResult>(WRAP_OP(::ttnn::typecast), callType,
                                  makeTuple, device);
}

} // namespace ttnn_op_invoke
