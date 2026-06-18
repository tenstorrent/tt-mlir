// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Experimental/DropoutOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/experimental/dropout/dropout.hpp"

#include <optional>

namespace ttnn_op_invoke {

DropoutResolvedParams
resolveDropoutParams(const ::tt::target::ttnn::DropoutOpT &op) {
  DropoutResolvedParams params;
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
auto createDropoutTuple(Tag tag, const ::tt::target::ttnn::DropoutOpT &op,
                        TensorArg input, const DropoutResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag), op.prob, op.scale,
                         op.seed, op.use_per_device_seed,
                         params.outputMemoryConfig,
                         /*optional_output_tensor=*/std::nullopt);
}

DropoutOpResult callDropout(CallType callType,
                            const ::tt::target::ttnn::DropoutOpT &op,
                            TensorArg input, ::ttnn::MeshDevice *device) {
  DropoutResolvedParams params = resolveDropoutParams(op);

  auto makeTuple = [&](auto tag) {
    return createDropoutTuple(tag, op, input, params);
  };

  return callOp<DropoutOpResult>(WRAP_OP(::ttnn::experimental::dropout),
                                 callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
