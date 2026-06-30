// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Conv/PrepareConv3dWeightsOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

PrepareConv3dWeightsResolvedParams resolvePrepareConv3dWeightsParams(
    const ::tt::target::ttnn::PrepareConv3dWeightsOpT &op) {
  PrepareConv3dWeightsResolvedParams params;

  return params;
}

template <typename Tag>
auto createPrepareConv3dWeightsTuple(
    Tag tag, const ::tt::target::ttnn::PrepareConv3dWeightsOpT &op,
    TensorArg weightTensor, ::ttnn::MeshDevice *device,
    const PrepareConv3dWeightsResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(weightTensor, tag), op.groups,
                         op.c_in_block, op.alignment, device);
}

PrepareConv3dWeightsOpResult
callPrepareConv3dWeights(CallType callType,
                         const ::tt::target::ttnn::PrepareConv3dWeightsOpT &op,
                         TensorArg weightTensor, ::ttnn::MeshDevice *device) {
  PrepareConv3dWeightsResolvedParams params =
      resolvePrepareConv3dWeightsParams(op);

  auto makeTuple = [&](auto tag) {
    return createPrepareConv3dWeightsTuple(tag, op, weightTensor, device,
                                           params);
  };

  return callOp<PrepareConv3dWeightsOpResult, false, false>(
      WRAP_OP(::ttnn::operations::experimental::conv3d::prepare_conv3d_weights),
      callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
