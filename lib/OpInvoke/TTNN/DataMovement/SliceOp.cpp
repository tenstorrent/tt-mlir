// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/DataMovement/SliceOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

SliceStaticResolvedParams
resolveSliceStaticParams(const ::tt::target::ttnn::SliceOpT &sliceOp) {
  SliceStaticResolvedParams params;

  const tt::target::ttnn::SliceStaticOpParamsT *staticParams =
      sliceOp.params.AsSliceStaticOpParams();
  TT_INVOKE_ASSERT(staticParams != nullptr,
                   "Expected SliceStaticOpParams for static slice");

  params.begins = {staticParams->begins.begin(), staticParams->begins.end()};
  params.ends = {staticParams->ends.begin(), staticParams->ends.end()};
  params.step = {sliceOp.step.begin(), sliceOp.step.end()};

  if (sliceOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*sliceOp.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*sliceOp.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createSliceStaticTuple(Tag tag,
                            const ::tt::target::ttnn::SliceOpT &sliceOp,
                            TensorArg input,
                            const SliceStaticResolvedParams &params) {
  auto beginsSpan = ::ttsl::make_const_span(params.begins);
  auto endsSpan = ::ttsl::make_const_span(params.ends);
  auto stepSpan = ::ttsl::make_const_span(params.step);
  return std::make_tuple(resolveTensorArg(input, tag), beginsSpan, endsSpan,
                         stepSpan, params.outputMemoryConfig);
}

SliceOpResult callSliceStatic(CallType callType,
                              const ::tt::target::ttnn::SliceOpT &sliceOp,
                              TensorArg input, ::ttnn::MeshDevice *device) {
  SliceStaticResolvedParams params = resolveSliceStaticParams(sliceOp);

  auto makeTuple = [&](auto tag) {
    return createSliceStaticTuple(tag, sliceOp, input, params);
  };

  return callOp<SliceOpResult>(WRAP_OP(::ttnn::slice), callType, makeTuple,
                               device);
}

} // namespace ttnn_op_invoke
