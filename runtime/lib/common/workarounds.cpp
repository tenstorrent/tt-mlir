// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/workarounds.h"

namespace tt::runtime::workaround {
#if defined(TT_RUNTIME_WORKAROUNDS) && TT_RUNTIME_WORKAROUNDS == 1
const Env &Env::get(bool maxpool2dPreshard, bool swapBinaryOperands,
                    bool readUpdateIndexFromDeviceForKVCache,
                    bool toDtypeOnHost, bool toLayoutAPIAssumeSingleChip) {
  static const Env config(maxpool2dPreshard, swapBinaryOperands,
                          readUpdateIndexFromDeviceForKVCache, toDtypeOnHost,
                          toLayoutAPIAssumeSingleChip);
  return config;
}
#endif
} // namespace tt::runtime::workaround
