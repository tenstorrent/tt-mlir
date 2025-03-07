// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_DYLIB_H
#define TT_RUNTIME_DETAIL_DYLIB_H

#include "tt/runtime/types.h"

#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
// Linux memfd_create syscall number, if not available in <sys/mman.h>
#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif
#ifndef SYS_memfd_create
#define SYS_memfd_create 319
#endif
#include <stdint.h>
namespace tt::runtime::common {
using DylibHandleMap = std::unordered_map<uint32_t, void *>;

class DylibManager {
public:
  // Constructor takes dylibs and loads them
  DylibManager(
      const ::flatbuffers::Vector<::flatbuffers::Offset<tt::target::DynamicLib>>
          *dylibs);

  // Destructor cleans up handles
  ~DylibManager();

  // Disable copy operations
  DylibManager(const DylibManager &) = delete;
  DylibManager &operator=(const DylibManager &) = delete;

  // Allow move operations
  DylibManager(DylibManager &&other) noexcept;
  DylibManager &operator=(DylibManager &&other) noexcept;

  // Access the handle map
  const DylibHandleMap &getHandles() const { return handles; }

  void *getHandle(const uint32_t key) {
    const auto it = handles.find(key);
    return (it == handles.end()) ? nullptr : it->second;
  }

private:
  DylibHandleMap handles;
};
} // namespace tt::runtime::common

#endif
