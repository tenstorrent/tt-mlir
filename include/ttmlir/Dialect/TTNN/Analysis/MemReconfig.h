// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_MEMRECONFIG_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_MEMRECONFIG_H

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>

namespace mlir::tt::ttnn {

class MemReconfigEntry {
public:
  // Map of consumer op config bit index to vector of valid reshard op output
  // configs. In the current implementation, vector will contain only one
  // element. In the future, we might want to keep all valid configs for optimal
  // resharding.
  llvm::DenseMap<std::size_t, llvm::SmallVector<OpConfig>>
      reshardOutputConfigMap;

  bool hasSelectedReshardOutputConfigBitIndex() const {
    return selectedReshardOutputConfigBitIndex >= 0;
  }

  void setSelectedReshardOutputConfigBitIndex(
      int32_t selectedReshardOutputConfigBitIndex) {
    assert(!hasSelectedReshardOutputConfigBitIndex() &&
           "Selected reshard output config bit index is already set!");
    this->selectedReshardOutputConfigBitIndex =
        selectedReshardOutputConfigBitIndex;
  }

  size_t getSelectedReshardOutputConfigBitIndex() const {
    assert(hasSelectedReshardOutputConfigBitIndex() &&
           "Selected reshard output config bit index is not set!");
    return static_cast<size_t>(selectedReshardOutputConfigBitIndex);
  }

  void setOverridenReconfig(bool overridenReconfig) {
    this->overridenReconfig = overridenReconfig;
  }

  bool hasOverridenReconfig() const { return overridenReconfig; }

private:
  // Index of the selected reshard output config bit in the map.
  int32_t selectedReshardOutputConfigBitIndex = -1;

  // Indicates if the reconfiguration is overriden by the user.
  bool overridenReconfig = false;

  friend llvm::raw_ostream &
  operator<<(llvm::raw_ostream &os, const MemReconfigEntry &memReconfigEntry);
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const MemReconfigEntry &memReconfigEntry) {
  os << "ReshardOutputConfigMap:\n";
  for (const auto &[idx, value] : memReconfigEntry.reshardOutputConfigMap) {
    os << "  Index " << idx << ":\n";
    for (const auto &config : value) {
      os << "    " << config.outputLayout << "\n";
    }
  }
  os << "SelectedReshardOutputConfigBitIndex: "
     << memReconfigEntry.selectedReshardOutputConfigBitIndex << "\n";
  os << "OverridenReconfig: " << memReconfigEntry.hasOverridenReconfig()
     << "\n";
  return os;
}

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_MEMRECONFIG_H
