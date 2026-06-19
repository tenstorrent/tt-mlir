// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Layout/BitcastConvertOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

BitcastConvertResolvedParams
resolveBitcastConvertParams(const ::tt::target::ttnn::BitcastConvertOpT &op) {
  BitcastConvertResolvedParams params;

  params.dtype = ttnn::DataType::BFLOAT16;
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
auto createBitcastConvertTuple(Tag tag,
                               const ::tt::target::ttnn::BitcastConvertOpT &op,
                               TensorArg input,
                               const BitcastConvertResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag), params.dtype,
                         params.outputMemoryConfig);
}

BitcastConvertOpResult
callBitcastConvert(CallType callType,
                   const ::tt::target::ttnn::BitcastConvertOpT &op,
                   TensorArg input, ::ttnn::MeshDevice *device) {
  BitcastConvertResolvedParams params = resolveBitcastConvertParams(op);

  auto makeTuple = [&](auto tag) {
    return createBitcastConvertTuple(tag, op, input, params);
  };

  return callOp<BitcastConvertOpResult>(WRAP_OP(::ttnn::bitcast), callType,
                                        makeTuple, device);
}

} // namespace ttnn_op_invoke
