// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/workarounds.h"

namespace tt::runtime::workaround {
const Env &Env::get(bool swapBinaryOperands,
                    bool readUpdateIndexFromDeviceForKVCache,
                    bool traceImplicitFromDevice, bool blackholeWorkarounds,
                    bool d2mReturnEvent) {
  static const Env config(
      swapBinaryOperands, readUpdateIndexFromDeviceForKVCache,
      traceImplicitFromDevice, blackholeWorkarounds, d2mReturnEvent);
  return config;
}
} // namespace tt::runtime::workaround
