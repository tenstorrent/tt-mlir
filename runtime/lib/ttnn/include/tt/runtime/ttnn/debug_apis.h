// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TTNN_DEBUG_APIS_H
#define TT_RUNTIME_TTNN_DEBUG_APIS_H

#include "tt/runtime/detail/ttnn.h"
#include "ttmlir/Target/TTNN/Target.h"

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
#define RUNTIME_DEBUG_MAYBE_CONST_INLINE
#else
#define RUNTIME_DEBUG_MAYBE_CONST_INLINE                                       \
  inline __attribute__((always_inline, const))
#endif

namespace tt::runtime::ttnn::debug {

RUNTIME_DEBUG_MAYBE_CONST_INLINE void
checkTensorRefMatchesTTNNTensor(const ::tt::target::ttnn::TensorRef *tensorRef,
                                const ::ttnn::Tensor &ttnnTensor)
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    ;
#else
{
}
#endif

#undef RUNTIME_DEBUG_MAYBE_CONST_INLINE

} // namespace tt::runtime::ttnn::debug

#endif // TT_RUNTIME_TTNN_DEBUG_APIS_H
