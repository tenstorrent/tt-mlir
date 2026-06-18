// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_EXPERIMENTAL_DROPOUTOP_H
#define TTMLIR_OPINVOKE_TTNN_EXPERIMENTAL_DROPOUTOP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/eltwise_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/experimental/dropout/dropout.hpp"

#include <optional>

namespace ttnn_op_invoke {

using DropoutOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct DropoutResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

DropoutResolvedParams
resolveDropoutParams(const ::tt::target::ttnn::DropoutOpT &op);

DropoutOpResult callDropout(CallType callType,
                            const ::tt::target::ttnn::DropoutOpT &op,
                            TensorArg input, ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_EXPERIMENTAL_DROPOUTOP_H
