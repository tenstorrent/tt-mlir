// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/dylib.h"

#include "tt/runtime/detail/common/logger.h"

#include <dlfcn.h>

namespace tt::runtime::common {

void *loadLibraryFromMemory(const uint8_t *data, size_t size) {
  // Create an in-memory file descriptor
  int memfd = memfd_create("dylib", MFD_CLOEXEC);
  if (memfd == -1) {
    LOG_ERROR("memfd_create");
    return nullptr;
  }

  if (write(memfd, data, size) != static_cast<ssize_t>(size)) {
    LOG_ERROR("write");
    close(memfd);
    return nullptr;
  }

  void *handle =
      dlopen(("/proc/self/fd/" + std::to_string(memfd)).c_str(), RTLD_LAZY);
  close(memfd); // Can close after dlopen

  if (!handle) {
    LOG_ERROR("dlopen failed: ", dlerror());
    return nullptr;
  }

  return handle;
}

DylibManager::DylibManager(
    const ::flatbuffers::Vector<::flatbuffers::Offset<tt::target::DynamicLib>>
        *dylibs) {
  if (dylibs) {
    for (const auto &dylib_offset : *dylibs) {
      const auto *dylib = dylib_offset; // Automatic dereferencing works here
      void *handle = loadLibraryFromMemory(dylib->raw_file()->data(),
                                           dylib->raw_file()->size());
      if (!handle) {
        LOG_FATAL("failed to open input dynamic library handle!");
      }
      handles.emplace(dylib->dylib_id(), handle);
    }
  }
}

DylibManager::~DylibManager() {
  for (const auto &[_, handle] : handles) {
    dlclose(handle);
  }
}

DylibManager::DylibManager(DylibManager &&other) noexcept
    : handles(std::move(other.handles)) {
  // Clear the moved-from object's handles to prevent double-free
  other.handles.clear();
}

DylibManager &DylibManager::operator=(DylibManager &&other) noexcept {
  if (this != &other) {
    // First, close our current handles
    for (const auto &[_, handle] : handles) {
      dlclose(handle);
    }

    // Then take ownership of the other's handles
    handles = std::move(other.handles);

    // Clear the moved-from object's handles
    other.handles.clear();
  }
  return *this;
}

WrappedFunc DylibManager::getFunc(const uint32_t dylibId,
                                  const std::string &funcName) const {
  auto *dylibHandle = getHandle(dylibId);
  if (!dylibHandle) {
    LOG_FATAL("could not find dylib corresponding to id: " +
              std::to_string(dylibId));
  }

  WrappedFunc fn =
      reinterpret_cast<WrappedFunc>(dlsym(dylibHandle, funcName.c_str()));
  if (!fn) {
    LOG_FATAL("could not find requested op: \"" + funcName +
              "\" in dylib with id: " + std::to_string(dylibId));
  }
  return fn;
}

} // namespace tt::runtime::common
