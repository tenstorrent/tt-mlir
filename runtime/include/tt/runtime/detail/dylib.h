// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DYLIB_H
#define TT_RUNTIME_DYLIB_H

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
void *loadLibraryFromMemory(const uint8_t *data, size_t size);

DylibHandleMap openDylibHandles(
    const ::flatbuffers::Vector<::flatbuffers::Offset<tt::target::DynamicLib>>
        *dylibs);

void closeDylibHandles(DylibHandleMap handles);
} // namespace tt::runtime::common

#endif
