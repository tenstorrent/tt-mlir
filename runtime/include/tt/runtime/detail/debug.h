// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_DEBUG_H
#define TT_RUNTIME_DETAIL_DEBUG_H

#include <ostream>

namespace tt::runtime::debug {

struct Env {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  static Env const &
#else
  constexpr static Env
#endif
  get(bool loadKernelsFromDisk = false, bool disableAsyncTTNN = false)
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
      ;
#else
  {
    return Env(false, false);
  }
#endif

  bool loadKernelsFromDisk;
  bool disableAsyncTTNN;

private:
  constexpr Env(bool loadKernelsFromDisk, bool disableAsyncTTNN)
      : loadKernelsFromDisk(loadKernelsFromDisk),
        disableAsyncTTNN(disableAsyncTTNN) {}
};

inline std::ostream &operator<<(std::ostream &os, Env const &env) {
  os << "Env{\n"
     << "\t" << "loadKernelsFromDisk: " << env.loadKernelsFromDisk << ",\n"
     << "\t" << "disableAsyncTTNN: " << env.disableAsyncTTNN << "\n"
     << "}";
  return os;
}

} // namespace tt::runtime::debug

#endif // TT_RUNTIME_DETAIL_DEBUG_H
