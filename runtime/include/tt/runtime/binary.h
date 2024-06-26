// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_BINARY_H
#define TT_RUNTIME_BINARY_H

#include "tt/runtime/types.h"

#include <string_view>

namespace tt::runtime::binary {

Flatbuffer loadFromData(void *data);
Flatbuffer loadFromPath(char const *path);
void store(Flatbuffer binary, char const *path);
std::string_view getFileIdentifier(Flatbuffer binary);
std::string getVersion(Flatbuffer binary);
std::string_view getTTMLIRGitHash(Flatbuffer binary);
std::string asJson(Flatbuffer binary);
std::vector<TensorDesc> getProgramInputs(Flatbuffer binary,
                                         std::uint32_t programIndex);
std::vector<TensorDesc> getProgramOutputs(Flatbuffer binary,
                                          std::uint32_t programIndex);

} // namespace tt::runtime::binary

#endif
