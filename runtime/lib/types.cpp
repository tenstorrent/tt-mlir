// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/types.h"

#if defined(TT_RUNTIME_ENABLE_TTNN)
#include "tt/runtime/detail/ttnn.h"
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
#include "tt/runtime/detail/ttmetal.h"
#endif

namespace tt::runtime {

void Tensor::deallocate(bool force) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (this->matchesRuntime(DeviceRuntime::TTNN)) {
    ::tt::runtime::ttnn::deallocateTensor(*this, force);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (this->matchesRuntime(DeviceRuntime::TTMetal)) {
    throw std::runtime_error("Not implemented");
  }
#endif
  throw std::runtime_error("Runtime not enabled");
}

Tensor Tensor::cpu() const {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (this->matchesRuntime(DeviceRuntime::TTNN)) {
    return ::tt::runtime::ttnn::toCpu(*this);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (this->matchesRuntime(DeviceRuntime::TTMetal)) {
    throw std::runtime_error("Not implemented");
  }
#endif
  throw std::runtime_error("Runtime not enabled");
}
} // namespace tt::runtime
