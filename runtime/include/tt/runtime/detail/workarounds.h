// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_WORKAROUNDS_H
#define TT_RUNTIME_DETAIL_WORKAROUNDS_H

#include <ostream>

namespace tt::runtime::workaround {

struct Env {
#if defined(TT_RUNTIME_WORKAROUNDS) && TT_RUNTIME_WORKAROUNDS == 1
  static const Env &
#else
  constexpr static Env
#endif
  get(bool ignoreTileShape = true, bool emptyOpForceRowMajor = true,
      bool fullOpForceRowMajor = true, bool maxpool2dPreshard = true)
#if defined(TT_RUNTIME_WORKAROUNDS) && TT_RUNTIME_WORKAROUNDS == 1
      ;
#else
  {
    return Env(true, true, true, true);
  }
#endif
  // TODO(bug #272), determine correct layout by tile shape in the future
  // currently tile shape is not set correctly, so as a workaround, hardcode
  // layout
  bool ignoreTileShape;

  // TODO(bug #582): ttnn::empty doesn't work properly with tile layout,
  // using ROW_MAJOR until we fix it
  bool emptyOpForceRowMajor;

  // TODO(bug #582): ttnn::full doesn't work properly with tile layout,
  // using ROW_MAJOR until we fix it
  bool fullOpForceRowMajor;

  // TODO(bug #855): Ideally we should have an op that preshards for maxpool2d
  // instead of adding a method in runtime
  bool maxpool2dPreshard;

private:
  constexpr Env(bool ignoreTileShape, bool emptyOpForceRowMajor,
                bool fullOpForceRowMajor, bool maxpool2dPreshard)
      : ignoreTileShape(ignoreTileShape),
        emptyOpForceRowMajor(emptyOpForceRowMajor),
        fullOpForceRowMajor(fullOpForceRowMajor),
        maxpool2dPreshard(maxpool2dPreshard) {}
};

inline std::ostream &operator<<(std::ostream &os, const Env &env) {
  os << "workaround::Env{\n";
  os << "\t" << "ignoreTileShape: " << env.ignoreTileShape << ",\n";
  os << "\t" << "emptyOpForceRowMajor: " << env.emptyOpForceRowMajor << ",\n";
  os << "\t" << "fullOpForceRowMajor: " << env.fullOpForceRowMajor << ",\n";
  os << "\t" << "maxpool2dPreshard: " << env.maxpool2dPreshard << "\n";
  os << "}";
  return os;
}

} // namespace tt::runtime::workaround

#endif // TT_RUNTIME_DETAIL_WORKAROUNDS_H
