// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Pool/UpsampleOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/pool/upsample/upsample.hpp"

#include <optional>

namespace ttnn_op_invoke {

UpsampleResolvedParams
resolveUpsampleParams(const ::tt::target::ttnn::UpsampleOpT &op) {
  UpsampleResolvedParams params;

  if (op.scale_factor.type == ::tt::target::ttnn::Scale2D::UniformScale2D) {
    params.scaleFactor = op.scale_factor.AsUniformScale2D()->scale;
  } else if (op.scale_factor.type ==
             ::tt::target::ttnn::Scale2D::NonUniformScale2D) {
    std::array<int, 2> arr;
    const tt::target::ttnn::NonUniformScale2DT *nonUniform =
        op.scale_factor.AsNonUniformScale2D();
    std::copy(nonUniform->scale.begin(), nonUniform->scale.end(), arr.begin());
    params.scaleFactor = arr;
  } else {
    TT_INVOKE_ASSERT(false, "Invalid scale_factor type");
  }

  if (op.mode != "") {
    params.mode = op.mode;
  }

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
auto createUpsampleTuple(Tag tag, const ::tt::target::ttnn::UpsampleOpT &op,
                         TensorArg input,
                         const UpsampleResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(input, tag), params.scaleFactor,
                         params.mode, params.outputMemoryConfig,
                         /*compute_kernel_config=*/std::nullopt);
}

UpsampleOpResult callUpsample(CallType callType,
                              const ::tt::target::ttnn::UpsampleOpT &op,
                              TensorArg input, ::ttnn::MeshDevice *device) {
  UpsampleResolvedParams params = resolveUpsampleParams(op);

  auto makeTuple = [&](auto tag) {
    return createUpsampleTuple(tag, op, input, params);
  };

  return callOp<UpsampleOpResult>(WRAP_OP(::ttnn::upsample), callType,
                                  makeTuple, device);
}

} // namespace ttnn_op_invoke
