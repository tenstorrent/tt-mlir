// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_RAND_RANDOP_H
#define TTMLIR_OPINVOKE_TTNN_RAND_RANDOP_H

#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/rand_generated.h"
#pragma clang diagnostic pop
#include "ttnn/operations/rand/rand.hpp"

#include <vector>

namespace ttnn_op_invoke {

using RandOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct RandResolvedParams {
  ::ttnn::Shape shape;
  ::ttnn::DataType dtype;
  ::ttnn::Layout layout;
  ::ttnn::MemoryConfig outputMemoryConfig;
};

RandResolvedParams resolveRandParams(const ::tt::target::ttnn::RandOpT &op);

RandOpResult callRand(CallType callType, const ::tt::target::ttnn::RandOpT &op,
                      ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_RAND_RANDOP_H
