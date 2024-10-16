// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/workarounds.h"

namespace tt::runtime::workaround {
#if defined(TT_RUNTIME_WORKAROUNDS) && TT_RUNTIME_WORKAROUNDS == 1
const Env &Env::get(bool ignoreTileShape, bool emptyOpForceRowMajor,
                    bool fullOpForceRowMajor, bool maxpool2dPreshard,
                    bool setMatmul1DProgramConfig) {
  static const Env config(ignoreTileShape, emptyOpForceRowMajor,
                          fullOpForceRowMajor, maxpool2dPreshard,
                          setMatmul1DProgramConfig);
  return config;
}
#endif
} // namespace tt::runtime::workaround
