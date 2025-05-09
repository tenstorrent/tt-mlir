// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_WORKAROUNDS_H
#define TT_RUNTIME_WORKAROUNDS_H

#include <ostream>

namespace tt::runtime::workaround {

struct Env {
  static const Env &get(bool swapBinaryOperands = true,
                        bool readUpdateIndexFromDeviceForKVCache = true,
                        bool rawHostDataPointerWrapper = true) {
    static const Env config(swapBinaryOperands,
                            readUpdateIndexFromDeviceForKVCache,
                            rawHostDataPointerWrapper);
    return config;
  }

  // TODO(bug #1124): We're currently swapping the operands for binary ops
  // in runtime if the lhs operand is smaller (and requires broadcast onto the
  // rhs operand). We should add this check in the compiler.
  bool swapBinaryOperands;

  // TODO(bug #1510): ttnn::update_cache takes a single update index as a uint32
  // as a function argument. The tt-torch frontend and likely others model this
  // as a tensor with integer elements. For now, to get this op to work we need
  // to be able to pluck this update index from a runtime tensor.
  bool readUpdateIndexFromDeviceForKVCache;

  // TODO(bug #3176): With the new mesh device flow, all tensors read back to
  // host will be in MULTI_DEVICE_HOST storage. Currently metal does not have
  // support for getting the raw host data pointer for MULTI_DEVICE_HOST storage
  // nor do they support host to host memcpy directly. Working around this by
  // implementing a custom wrapper around get_raw_host_data_ptr until support is
  // added
  bool rawHostDataPointerWrapper;

private:
  constexpr Env(bool swapBinaryOperands,
                bool readUpdateIndexFromDeviceForKVCache,
                bool rawHostDataPointerWrapper)
      : swapBinaryOperands(swapBinaryOperands),
        readUpdateIndexFromDeviceForKVCache(
            readUpdateIndexFromDeviceForKVCache),
        rawHostDataPointerWrapper(rawHostDataPointerWrapper) {}
};

inline std::ostream &operator<<(std::ostream &os, const Env &env) {
  os << "workaround::Env{\n";
  os << "\t"
     << "swapBinaryOperands: " << env.swapBinaryOperands << ",\n";
  os << "\t"
     << "readUpdateIndexFromDeviceForKVCache: "
     << env.readUpdateIndexFromDeviceForKVCache << "\n";
  os << "\t"
     << "rawHostDataPointerWrapper: " << env.rawHostDataPointerWrapper << "\n";
  os << "}";
  return os;
}

} // namespace tt::runtime::workaround

#endif // TT_RUNTIME_WORKAROUNDS_H
