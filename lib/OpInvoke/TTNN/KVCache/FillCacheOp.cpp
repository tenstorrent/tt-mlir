// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/KVCache/FillCacheOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

namespace ttnn_op_invoke {

FillCacheResolvedParams
resolveFillCacheParams(const ::tt::target::ttnn::FillCacheOpT &op) {
  FillCacheResolvedParams params;

  return params;
}

template <typename Tag>
auto createFillCacheTuple(Tag tag,
                          const ::tt::target::ttnn::FillCacheOpT &fillCacheOp,
                          TensorArg cache, TensorArg input,
                          const FillCacheResolvedParams &params) {
  return std::make_tuple(resolveTensorArg(cache, tag),
                         resolveTensorArg(input, tag),
                         fillCacheOp.batch_offset);
}

FillCacheOpResult callFillCache(CallType callType,
                                const ::tt::target::ttnn::FillCacheOpT &op,
                                TensorArg cache, TensorArg input,
                                ::ttnn::MeshDevice *device) {
  FillCacheResolvedParams params = resolveFillCacheParams(op);

  auto makeTuple = [&](auto tag) {
    return createFillCacheTuple(tag, op, cache, input, params);
  };

  return callOp<FillCacheOpResult>(WRAP_OP(::ttnn::fill_cache), callType,
                                   makeTuple, device);
}

} // namespace ttnn_op_invoke
