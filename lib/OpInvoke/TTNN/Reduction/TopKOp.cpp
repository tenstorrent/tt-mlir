// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Reduction/TopKOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

TopKResolvedParams resolveTopKParams(const ::tt::target::ttnn::TopKOpT &op) {
  TopKResolvedParams params;

  params.k = static_cast<uint32_t>(op.k);
  params.dim = static_cast<int8_t>(op.dim);

  if (!op.outputs.empty() && op.outputs[0]) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*op.outputs[0]));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*op.outputs[0]) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createTopKTuple(Tag tag, const ::tt::target::ttnn::TopKOpT &op,
                     TensorArg input, const TopKResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag), params.k, params.dim,
                         op.largest, op.sorted, params.outputMemoryConfig,
                         /*sub_core_grids=*/std::nullopt,
                         /*indices_tensor=*/std::nullopt,
                         /*preallocated_output_tensors=*/std::nullopt);
}

TopKOpResult callTopK(CallType callType, const ::tt::target::ttnn::TopKOpT &op,
                      TensorArg input, ::ttnn::MeshDevice *device) {
  TopKResolvedParams params = resolveTopKParams(op);

  auto makeTuple = [&](auto tag) {
    return createTopKTuple(tag, op, input, params);
  };

  return callOp<TopKOpResult>(WRAP_OP(::ttnn::topk), callType, makeTuple,
                              device);
}

} // namespace ttnn_op_invoke
