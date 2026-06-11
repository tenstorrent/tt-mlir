// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/DataMovement/WriteTensorOp.h"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <optional>

namespace ttnn_op_invoke {

WriteTensorResolvedParams resolveWriteTensorParams(
    const ::tt::target::ttnn::WriteTensorOpT &writeTensorOp) {
  WriteTensorResolvedParams params;
  params.cq_id = ::ttnn::QueueId(writeTensorOp.cq_id);
  return params;
}

template <typename Tag>
auto createWriteTensorTuple(
    Tag tag, const ::tt::target::ttnn::WriteTensorOpT &writeTensorOp,
    TensorArg hostTensor, TensorArg deviceTensor,
    const WriteTensorResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(hostTensor, tag),
                         resolveTensorArg(deviceTensor, tag),
                         std::make_optional(params.cq_id));
}

WriteTensorOpResult callWriteTensor(
    CallType callType, const ::tt::target::ttnn::WriteTensorOpT &writeTensorOp,
    TensorArg hostTensor, TensorArg deviceTensor, ::ttnn::MeshDevice *device) {
  WriteTensorResolvedParams params = resolveWriteTensorParams(writeTensorOp);

  auto makeTuple = [&](auto tag) {
    return createWriteTensorTuple(tag, writeTensorOp, hostTensor, deviceTensor,
                                  params);
  };

  // Note: copy_to_device replaced write_tensor and does not have a blocking
  // parameter. The operation is always blocking.
  return callOp<WriteTensorOpResult, false, false>(
      [&](auto &&...args) {
        ::tt::tt_metal::copy_to_device(std::forward<decltype(args)>(args)...);
        return std::monostate{};
      },
      callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
