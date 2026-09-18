// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_WORKAROUNDS_H
#define TT_RUNTIME_WORKAROUNDS_H

#include <ostream>

namespace tt::runtime::workaround {

struct Env {
  static const Env &get(bool swapBinaryOperands = true,
                        bool blackholeWorkarounds = true);

  // TODO(bug #1124): We're currently swapping the operands for binary ops
  // in runtime if the lhs operand is smaller (and requires broadcast onto the
  // rhs operand). We should add this check in the compiler.
  bool swapBinaryOperands;

  // TODO(bug #3423): When link is down, get_connected_ethernet_core will throw
  // an exception.
  // TODO(bug #4023): untilize on device fails for blackhole. Falling back to
  // host for now.
  bool blackholeWorkarounds;

private:
  constexpr Env(bool swapBinaryOperands, bool blackholeWorkarounds)
      : swapBinaryOperands(swapBinaryOperands),
        blackholeWorkarounds(blackholeWorkarounds) {}
};

inline std::ostream &operator<<(std::ostream &os, const Env &env) {
  os << "workaround::Env{\n";
  os << "\t"
     << "swapBinaryOperands: " << env.swapBinaryOperands << ",\n";
  os << "\t"
     << "blackholeWorkarounds: " << env.blackholeWorkarounds << "\n";
  os << "}";
  return os;
}

} // namespace tt::runtime::workaround

#endif // TT_RUNTIME_WORKAROUNDS_H
