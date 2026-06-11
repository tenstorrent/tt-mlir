// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/DataMovement/SortOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

SortResolvedParams
resolveSortParams(const ::tt::target::ttnn::SortOpT &sortOp) {
  SortResolvedParams params;

  if (!sortOp.outputs.empty() && sortOp.outputs[0]) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*sortOp.outputs[0]));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*sortOp.outputs[0]) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createSortTuple(Tag tag, const ::tt::target::ttnn::SortOpT &sortOp,
                     TensorArg input, const SortResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag), sortOp.dim,
                         sortOp.descending, sortOp.stable,
                         params.outputMemoryConfig,
                         /*optional_output_tensors=*/std::nullopt);
}

SortOpResult callSort(CallType callType,
                      const ::tt::target::ttnn::SortOpT &sortOp,
                      TensorArg input, ::ttnn::MeshDevice *device) {
  SortResolvedParams params = resolveSortParams(sortOp);

  auto makeTuple = [&](auto tag) {
    return createSortTuple(tag, sortOp, input, params);
  };

  return callOp<SortOpResult>(WRAP_OP(::ttnn::sort), callType, makeTuple,
                              device);
}

} // namespace ttnn_op_invoke
