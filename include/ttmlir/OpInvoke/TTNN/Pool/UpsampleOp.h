// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_POOL_UPSAMPLEOP_H
#define TTMLIR_OPINVOKE_TTNN_POOL_UPSAMPLEOP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/pool_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/pool/upsample/upsample.hpp"

#include <optional>
#include <string>
#include <variant>

namespace ttnn_op_invoke {

using UpsampleOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct UpsampleResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::variant<int, std::array<int, 2>, float, std::array<float, 2>>
      scaleFactor;
  std::string mode = std::string("nearest");
};

UpsampleResolvedParams
resolveUpsampleParams(const ::tt::target::ttnn::UpsampleOpT &op);

UpsampleOpResult callUpsample(CallType callType,
                              const ::tt::target::ttnn::UpsampleOpT &op,
                              TensorArg input, ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_POOL_UPSAMPLEOP_H
