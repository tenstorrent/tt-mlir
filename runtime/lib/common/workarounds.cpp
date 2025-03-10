// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/workarounds.h"

namespace tt::runtime::workaround {
#if defined(TT_RUNTIME_WORKAROUNDS) && TT_RUNTIME_WORKAROUNDS == 1
const Env &Env::get(bool swapBinaryOperands,
                    bool readUpdateIndexFromDeviceForKVCache,
                    bool toLayoutAPIAssumeSingleChip,
                    bool manualDeviceStorageFromBorrowedStorage) {
  static const Env config(
      swapBinaryOperands, readUpdateIndexFromDeviceForKVCache,
      toLayoutAPIAssumeSingleChip, bool manualDeviceStorageFromBorrowedStorage);
  return config;
}
#endif
} // namespace tt::runtime::workaround
