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
      bool toDtypeOnHost = true, bool defaultStrideComputation = true,
      bool toLayoutAPIAssumeSingleChip = true)
#if defined(TT_RUNTIME_WORKAROUNDS) && TT_RUNTIME_WORKAROUNDS == 1
      ;
#else
  {
    return Env(true, true, true, true, true, true);
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

  // TODO(bug #2045): Our current stride calculation is incorrect for tilized
  // tensors. The current solution is to remove stride entirely from the
  // flatbuffer and calculate the stride in runtime assuming using the default
  // method ignoring details like grid, layout etc. Once we have a more
  // sophisticated way for handling this, we can remove this workaround.
  bool defaultStrideComputation;

  // TODO(bug #1778): We currently don't have device grid information (mesh
  // shape, offset) in the flatbuffer TensorDesc nor in the mlir LayoutAttr. We
  // need to add this information to the tensorDesc so that the runtime toLayout
  // API can determine the correct devices. Enabling this workaround will assume
  // that a device tensor will reside in the L1/Dram of the first device (device
  // id 0) of the device grid. This should be removed once we add the device
  // grid information to the tensorDesc.
  bool toLayoutAPIAssumeSingleChip;

private:
  constexpr Env(bool maxpool2dPreshard, bool swapBinaryOperands,
                bool readUpdateIndexFromDeviceForKVCache, bool toDtypeOnHost,
                bool defaultStrideComputation, bool toLayoutAPIAssumeSingleChip)
      : maxpool2dPreshard(maxpool2dPreshard),
        swapBinaryOperands(swapBinaryOperands),
        readUpdateIndexFromDeviceForKVCache(
            readUpdateIndexFromDeviceForKVCache),
        toDtypeOnHost(toDtypeOnHost),
        defaultStrideComputation(defaultStrideComputation),
        toLayoutAPIAssumeSingleChip(toLayoutAPIAssumeSingleChip) {}
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
  os << "\t"
     << "defaultStrideComputation: " << env.defaultStrideComputation << "\n";
  os << "\t"
     << "toLayoutAPIAssumeSingleChip: " << env.toLayoutAPIAssumeSingleChip
     << "\n";
  os << "}";
  return os;
}

} // namespace tt::runtime::workaround

#endif // TT_RUNTIME_DETAIL_WORKAROUNDS_H
