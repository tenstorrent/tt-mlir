// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_BINARY_H
#define TT_RUNTIME_BINARY_H

#include "tt/runtime/types.h"

namespace tt::runtime::binary {

Binary loadFromData(void *data);
Binary loadFromPath(char const *path);
std::string getVersion(Binary const &binary);
std::string getTTMLIRGitHash(Binary const &binary);
std::string asJson(Binary const &binary);

} // namespace tt::runtime::binary

#endif
