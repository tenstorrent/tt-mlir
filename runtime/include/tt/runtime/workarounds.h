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
                        bool toLayoutAPIAssumeSingleChip = true) {
    static const Env config(swapBinaryOperands,
                            readUpdateIndexFromDeviceForKVCache,
                            toLayoutAPIAssumeSingleChip);
    return config;
  }

  // TODO(bug #1124): We're currently swapping the operands for binary ops
  // in runtime if the lhs operand is smaller (and requires broadcast onto the
  // rhs operand). We should add this check in the compiler.
  bool swapBinaryOperands;

  // TODO(bug #1510) ttnn::update_cache takes a single update index as a uint32
  // as a function argument. The tt-torch frontend and likely others model this
  // as a tensor with integer elements. For now, to get this op to work we need
  // to be able to pluck this update index from a runtime tensor.
  bool readUpdateIndexFromDeviceForKVCache;

  // TODO(bug #1778): We currently don't have device grid information (mesh
  // shape, offset) in the flatbuffer TensorDesc nor in the mlir LayoutAttr. We
  // need to add this information to the tensorDesc so that the runtime toLayout
  // API can determine the correct devices. Enabling this workaround will assume
  // that a device tensor will reside in the L1/Dram of the first device (device
  // id 0) of the device grid. This should be removed once we add the device
  // grid information to the tensorDesc.
  bool toLayoutAPIAssumeSingleChip;

private:
  constexpr Env(bool swapBinaryOperands,
                bool readUpdateIndexFromDeviceForKVCache,
                bool toLayoutAPIAssumeSingleChip)
      : swapBinaryOperands(swapBinaryOperands),
        readUpdateIndexFromDeviceForKVCache(
            readUpdateIndexFromDeviceForKVCache),
        toLayoutAPIAssumeSingleChip(toLayoutAPIAssumeSingleChip) {}
};

inline std::ostream &operator<<(std::ostream &os, const Env &env) {
  os << "workaround::Env{\n";
  os << "\t"
     << "swapBinaryOperands: " << env.swapBinaryOperands << ",\n";
  os << "\t"
     << "readUpdateIndexFromDeviceForKVCache: "
     << env.readUpdateIndexFromDeviceForKVCache << "\n";
  os << "\t"
     << "toLayoutAPIAssumeSingleChip: " << env.toLayoutAPIAssumeSingleChip
     << "\n";
  os << "}";
  return os;
}

} // namespace tt::runtime::workaround

#endif // TT_RUNTIME_WORKAROUNDS_H
