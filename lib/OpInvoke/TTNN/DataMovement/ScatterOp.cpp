// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/DataMovement/ScatterOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

ScatterResolvedParams
resolveScatterParams(const ::tt::target::ttnn::ScatterOpT &scatterOp) {
  ScatterResolvedParams params;

  switch (scatterOp.scatter_reduce_type) {
  case ::tt::target::ttnn::ScatterReduceType::Sum: // Sum -> add
    params.optReductionString = "add";
    break;
  case ::tt::target::ttnn::ScatterReduceType::Prod: // Prod -> multiply
    params.optReductionString = "multiply";
    break;
  case ::tt::target::ttnn::ScatterReduceType::Min: // Min -> amin
    params.optReductionString = "amin";
    break;
  case ::tt::target::ttnn::ScatterReduceType::Max: // Max -> amax
    params.optReductionString = "amax";
    break;
  case ::tt::target::ttnn::ScatterReduceType::Invalid: // Invalid -> no
                                                       // reduction
    params.optReductionString = std::nullopt;
    break;
  }

  if (scatterOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*scatterOp.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*scatterOp.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createScatterTuple(Tag tag,
                        const ::tt::target::ttnn::ScatterOpT &scatterOp,
                        TensorArg input, TensorArg index, TensorArg source,
                        const ScatterResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag), scatterOp.dim,
                         resolveTensorArg(index, tag),
                         resolveTensorArg(source, tag),
                         params.outputMemoryConfig, params.optReductionString,
                         /*sub_core_grid=*/std::nullopt);
}

ScatterOpResult callScatter(CallType callType,
                            const ::tt::target::ttnn::ScatterOpT &scatterOp,
                            TensorArg input, TensorArg index, TensorArg source,
                            ::ttnn::MeshDevice *device) {
  ScatterResolvedParams params = resolveScatterParams(scatterOp);

  auto makeTuple = [&](auto tag) {
    return createScatterTuple(tag, scatterOp, input, index, source, params);
  };

  return callOp<ScatterOpResult>(WRAP_OP(::ttnn::scatter), callType, makeTuple,
                                 device);
}

} // namespace ttnn_op_invoke
