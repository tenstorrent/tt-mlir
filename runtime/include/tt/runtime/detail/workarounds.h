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
  get(bool maxpool2dPreshard = true, bool swapBinaryOperands = true,
      bool readUpdateIndexFromDeviceForKVCache = true,
      bool toDtypeOnHost = true)
#if defined(TT_RUNTIME_WORKAROUNDS) && TT_RUNTIME_WORKAROUNDS == 1
      ;
#else
  {
    return Env(true, true, true, true);
  }
#endif
  // TODO(bug #855): Ideally we should have an op that preshards for maxpool2d
  // instead of adding a method in runtime
  bool maxpool2dPreshard;

  // TODO(bug #1124): We're currently swapping the operands for binary ops
  // in runtime if the lhs operand is smaller (and requires broadcast onto the
  // rhs operand). We should add this check in the compiler.
  bool swapBinaryOperands;

  // TODO(bug #1510) ttnn::update_cache takes a single update index as a uint32
  // as a function argument. The tt-torch frontend and likely others model this
  // as a tensor with integer elements. For now, to get this op to work we need
  // to be able to pluck this update index from a runtime tensor.
  bool readUpdateIndexFromDeviceForKVCache;

  // TODO(bug #1658): We're currently use ttnn::to_dtype operation to cast the
  // data type of a tensor on host. Once we have improved the typecast operation
  // to handle this, we should remove this workaround.
  bool toDtypeOnHost;

private:
  constexpr Env(bool maxpool2dPreshard, bool swapBinaryOperands,
                bool readUpdateIndexFromDeviceForKVCache, bool toDtypeOnHost)
      : maxpool2dPreshard(maxpool2dPreshard),
        swapBinaryOperands(swapBinaryOperands),
        readUpdateIndexFromDeviceForKVCache(
            readUpdateIndexFromDeviceForKVCache),
        toDtypeOnHost(toDtypeOnHost) {}
};

inline std::ostream &operator<<(std::ostream &os, const Env &env) {
  os << "workaround::Env{\n";
  os << "\t"
     << "maxpool2dPreshard: " << env.maxpool2dPreshard << ",\n";
  os << "\t"
     << "swapBinaryOperands: " << env.swapBinaryOperands << ",\n";
  os << "\t"
     << "readUpdateIndexFromDeviceForKVCache: "
     << env.readUpdateIndexFromDeviceForKVCache << "\n";
  os << "\t"
     << "toDtypeOnHost: " << env.toDtypeOnHost << "\n";
  os << "}";
  return os;
}

} // namespace tt::runtime::workaround

#endif // TT_RUNTIME_DETAIL_WORKAROUNDS_H
