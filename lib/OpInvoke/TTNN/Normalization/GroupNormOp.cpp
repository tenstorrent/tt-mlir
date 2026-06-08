// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Normalization/GroupNormOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/operations/normalization/groupnorm/groupnorm.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

GroupNormResolvedParams
resolveGroupNormParams(const ::tt::target::ttnn::GroupNormOpT &opT) {
  GroupNormResolvedParams params;
  if (opT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*opT.out));
    LOG_ASSERT(operations::utils::inSystemMemory(*opT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }
  return params;
}

template <typename Tag>
auto createGroupNormTuple(Tag tag, const ::tt::target::ttnn::GroupNormOpT &opT,
                          TensorArg input, std::optional<TensorArg> inputMask,
                          std::optional<TensorArg> weight,
                          std::optional<TensorArg> bias,
                          const GroupNormResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag), static_cast<int>(opT.num_groups),
      opT.epsilon,
      inputMask ? std::make_optional(resolveTensorArg(*inputMask, tag))
                : std::nullopt,
      weight ? std::make_optional(resolveTensorArg(*weight, tag))
             : std::nullopt,
      bias ? std::make_optional(resolveTensorArg(*bias, tag)) : std::nullopt,
      /*reciprocals=*/std::nullopt, params.outputMemoryConfig,
      /*dtype=*/std::nullopt, /*core_grid=*/std::nullopt,
      /*inplace=*/std::nullopt, /*output_layout=*/std::nullopt,
      /*num_out_blocks=*/std::nullopt, /*compute_kernel_config=*/std::nullopt,
      /*negative_mask=*/std::nullopt, /*use_welford=*/false);
}

GroupNormOpResult
callGroupNorm(CallType callType, const ::tt::target::ttnn::GroupNormOpT &opT,
              TensorArg input, std::optional<TensorArg> inputMask,
              std::optional<TensorArg> weight, std::optional<TensorArg> bias,
              ::ttnn::MeshDevice *device) {
  GroupNormResolvedParams params = resolveGroupNormParams(opT);

  auto makeTuple = [&](auto tag) {
    return createGroupNormTuple(tag, opT, input, inputMask, weight, bias,
                                params);
  };

  return callOp<GroupNormOpResult>(WRAP_OP(::ttnn::group_norm), callType,
                                   makeTuple, device);
}

} // namespace ttnn_op_invoke
