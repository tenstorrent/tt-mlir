// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Rand/RandOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

RandResolvedParams resolveRandParams(const ::tt::target::ttnn::RandOpT &op) {
  RandResolvedParams params;

  params.shape = operations::utils::toTTNNShape(op.size);

  params.layout = operations::utils::toTTNNLayout(op.layout);

  params.dtype = ttnn::DataType::BFLOAT16;
  if (op.out) {
    params.dtype = operations::utils::getDataType(*op.out);
  }

  params.outputMemoryConfig = ::ttnn::types::DRAM_MEMORY_CONFIG;
  if (op.out) {
    auto memCfg = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*op.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*op.out) ||
                         memCfg.has_value(),
                     "Memory config must exist for device tensors");
    params.outputMemoryConfig = *memCfg;
  }

  return params;
}

template <typename Tag>
auto createRandTuple(Tag tag, const ::tt::target::ttnn::RandOpT &op,
                     const RandResolvedParams &params,
                     ::ttnn::MeshDevice *device) {
  return std::make_tuple(params.shape, std::ref(*device), params.dtype,
                         params.layout, params.outputMemoryConfig, op.low,
                         op.high, op.seed,
                         /*mesh_mapper=*/std::nullopt);
}

RandOpResult callRand(CallType callType, const ::tt::target::ttnn::RandOpT &op,
                      ::ttnn::MeshDevice *device) {
  RandResolvedParams params = resolveRandParams(op);

  auto makeTuple = [&](auto tag) {
    return createRandTuple(tag, op, params, device);
  };

  return callOp<RandOpResult, true, false>(WRAP_OP(::ttnn::rand), callType,
                                           makeTuple, device);
}

} // namespace ttnn_op_invoke
