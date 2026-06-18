// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Experimental/ExperimentalEltwiseBinaryBackwardOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/experimental/unary_backward/gelu_backward/gelu_backward.hpp"

#include <optional>

namespace ttnn_op_invoke {

ExperimentalEltwiseBinaryBackwardResolvedParams
resolveExperimentalEltwiseBinaryBackwardParams(
    const ::tt::target::ttnn::ExperimentalEltwiseBinaryBackwardOpT &op) {
  ExperimentalEltwiseBinaryBackwardResolvedParams params;
  TT_INVOKE_ASSERT(
      op.type ==
          ::tt::target::ttnn::ExperimentalEltwiseBinaryBackwardOpType::GeluBW,
      "Expected GeluBW operation");

  params.approximate = op.approximate != "" ? op.approximate : "none";

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
auto createExperimentalEltwiseBinaryBackwardTuple(
    Tag tag, const ::tt::target::ttnn::ExperimentalEltwiseBinaryBackwardOpT &op,
    TensorArg grad, TensorArg input,
    const ExperimentalEltwiseBinaryBackwardResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(grad, tag),
                         resolveTensorArg(input, tag), params.approximate,
                         params.outputMemoryConfig,
                         /*input_grad_tensor=*/std::nullopt);
}

ExperimentalEltwiseBinaryBackwardOpResult callExperimentalEltwiseBinaryBackward(
    CallType callType,
    const ::tt::target::ttnn::ExperimentalEltwiseBinaryBackwardOpT &op,
    TensorArg grad, TensorArg input, ::ttnn::MeshDevice *device) {
  ExperimentalEltwiseBinaryBackwardResolvedParams params =
      resolveExperimentalEltwiseBinaryBackwardParams(op);

  auto makeTuple = [&](auto tag) {
    return createExperimentalEltwiseBinaryBackwardTuple(tag, op, grad, input,
                                                        params);
  };

  return callOp<ExperimentalEltwiseBinaryBackwardOpResult>(
      WRAP_OP(::ttnn::experimental::gelu_bw), callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
