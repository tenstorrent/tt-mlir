// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/dylib.h"

namespace tt::runtime::common {
void *loadLibraryFromMemory(const uint8_t *data, size_t size) {
  // Create an in-memory file descriptor
  int memfd = memfd_create("dylib", MFD_CLOEXEC);
  if (memfd == -1) {
    perror("memfd_create");
    return nullptr;
  }
  if (write(memfd, data, size) != static_cast<ssize_t>(size)) {
    perror("write");
    close(memfd);
    return nullptr;
  }
  void *handle =
      dlopen(("/proc/self/fd/" + std::to_string(memfd)).c_str(), RTLD_LAZY);
  close(memfd); // Can close after dlopen
  if (!handle) {
    std::cerr << "dlopen failed: " << dlerror() << std::endl;
    return nullptr;
  }
  return handle;
}

DylibHandleMap openDylibHandles(
    const ::flatbuffers::Vector<::flatbuffers::Offset<tt::target::DynamicLib>>
        *dylibs) {
  DylibHandleMap dlHandleMap;

  if (dylibs) {
    for (const auto &dylib_offset : *dylibs) {
      const auto *dylib = dylib_offset; // Automatic dereferencing works here
      void *handle = loadLibraryFromMemory(dylib->raw_file()->data(),
                                           dylib->raw_file()->size());
      if (!handle) {
        throw std::runtime_error(
            "failed to open input dynamic library handle!");
      }
      dlHandleMap.emplace(dylib->dylib_id(), handle);
    }
  }

  return dlHandleMap;
}

void closeDylibHandles(DylibHandleMap handles) {
  for (const auto [_, handle] : handles) {
    dlclose(handle);
  }
}
} // namespace tt::runtime::common
