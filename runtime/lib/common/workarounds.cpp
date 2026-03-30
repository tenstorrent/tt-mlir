// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/workarounds.h"

namespace tt::runtime::workaround {
const Env &Env::get(bool swapBinaryOperands,
                    bool readUpdateIndexFromDeviceForKVCache,
                    bool blackholeWorkarounds) {
  static const Env config(swapBinaryOperands,
                          readUpdateIndexFromDeviceForKVCache,
                          blackholeWorkarounds);
  return config;
}
} // namespace tt::runtime::workaround
