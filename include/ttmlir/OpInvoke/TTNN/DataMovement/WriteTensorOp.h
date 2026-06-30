// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_DATA_MOVEMENT_WRITE_TENSOR_OP_H
#define TTNN_OP_INVOKE_DATA_MOVEMENT_WRITE_TENSOR_OP_H

#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/tensor/tensor.hpp"

#include <variant>

namespace ttnn_op_invoke {

using WriteTensorOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, std::monostate>;

struct WriteTensorResolvedParams {
  ttnn::QueueId cq_id;
};

WriteTensorResolvedParams resolveWriteTensorParams(
    const ::tt::target::ttnn::WriteTensorOpT &writeTensorOp);

WriteTensorOpResult
callWriteTensor(CallType callType,
                const ::tt::target::ttnn::WriteTensorOpT &writeTensorOp,
                TensorArg hostTensor, TensorArg deviceTensor,
                ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_DATA_MOVEMENT_WRITE_TENSOR_OP_H
