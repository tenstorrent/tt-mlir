// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/DataMovement/AssignOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

AssignResolvedParams
resolveAssignParams(const ::tt::target::ttnn::AssignOpT &assignOp) {
  AssignResolvedParams params;

  if (assignOp.output_dtype) {
    params.outputDtype = operations::utils::getDataType(*assignOp.output);
  }

  if (assignOp.output) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*assignOp.output));
    LOG_ASSERT(operations::utils::inSystemMemory(*assignOp.output) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  if (!params.outputMemoryConfig) {
    params.outputMemoryConfig = ::ttnn::MemoryConfig{
        ::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM};
  }

  return params;
}

template <typename Tag>
auto createAssignTuple(Tag tag, const ::tt::target::ttnn::AssignOpT &assignOp,
                       TensorArg input, const AssignResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag),
                         params.outputMemoryConfig.value(), params.outputDtype,
                         /*optional_output_tensor=*/std::nullopt);
}

AssignOpResult callAssign(CallType callType,
                          const ::tt::target::ttnn::AssignOpT &assignOp,
                          TensorArg input, ::ttnn::MeshDevice *device) {
  AssignResolvedParams params = resolveAssignParams(assignOp);

  auto makeTuple = [&](auto tag) {
    return createAssignTuple(tag, assignOp, input, params);
  };

  return callOp<AssignOpResult>(WRAP_OP(::ttnn::assign), callType, makeTuple,
                                device);
}

} // namespace ttnn_op_invoke
