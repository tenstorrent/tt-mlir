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
                        bool traceImplicitFromDevice = true,
                        bool blackholeWorkarounds = true,
                        bool forceOutOfPlaceReshape = true);

  // TODO(bug #1124): We're currently swapping the operands for binary ops
  // in runtime if the lhs operand is smaller (and requires broadcast onto the
  // rhs operand). We should add this check in the compiler.
  bool swapBinaryOperands;

  // TODO(bug #1510): ttnn::update_cache takes a single update index as a uint32
  // as a function argument. The tt-torch frontend and likely others model this
  // as a tensor with integer elements. For now, to get this op to work we need
  // to be able to pluck this update index from a runtime tensor.
  bool readUpdateIndexFromDeviceForKVCache;

  // TODO(bug #3695): Currently ttnn only supports writing to a pre-allocated
  // device tensor from host. Therefore, the trace op implicitly reads any
  // device input back to host, then writes it to the designated buffer. This
  // should be updated in the future either when ttnn supports device to device
  // memcpy or when we model this behaviour in the compiler.
  bool traceImplicitFromDevice;

  // TODO(bug #3423): When link is down, get_connected_ethernet_core will throw
  // an exception.
  // TODO(bug #4023): untilize on device fails for blackhole. Falling back to
  // host for now.
  bool blackholeWorkarounds;

  // TODO(issue #5312): We currently model ttnn::reshape as out-of-place. For
  // some cases, ttnn::reshape performs a view instead, which is in-place. This
  // breaking of our assumption can causes issues with passing tensors between
  // programs and within programs themselves.
  //
  // This workaround will copy the input tensor to a reshape IF we know the
  // reshape is a view. That way the in-place operation is performed on a brand
  // new tensor, and we do not return the same tensor object as the input.
  bool forceOutOfPlaceReshape;

private:
  constexpr Env(bool swapBinaryOperands,
                bool readUpdateIndexFromDeviceForKVCache,
                bool traceImplicitFromDevice, bool blackholeWorkarounds,
                bool forceOutOfPlaceReshape)
      : swapBinaryOperands(swapBinaryOperands),
        readUpdateIndexFromDeviceForKVCache(
            readUpdateIndexFromDeviceForKVCache),
        traceImplicitFromDevice(traceImplicitFromDevice),
        blackholeWorkarounds(blackholeWorkarounds),
        forceOutOfPlaceReshape(forceOutOfPlaceReshape) {}
};

inline std::ostream &operator<<(std::ostream &os, const Env &env) {
  os << "workaround::Env{\n";
  os << "\t"
     << "swapBinaryOperands: " << env.swapBinaryOperands << ",\n";
  os << "\t"
     << "readUpdateIndexFromDeviceForKVCache: "
     << env.readUpdateIndexFromDeviceForKVCache << "\n";
  os << "\t"
     << "traceImplicitFromDevice: " << env.traceImplicitFromDevice << "\n";
  os << "\t"
     << "blackholeWorkarounds: " << env.blackholeWorkarounds << "\n";
  os << "\t"
     << "forceOutOfPlaceReshape: " << env.forceOutOfPlaceReshape << "\n";
  os << "}";
  return os;
}

} // namespace tt::runtime::workaround

#endif // TT_RUNTIME_WORKAROUNDS_H
