// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_COMMON_SYSTEM_DESC_H
#define TT_RUNTIME_DETAIL_COMMON_SYSTEM_DESC_H

#include "tt/runtime/types.h"

#define FMT_HEADER_ONLY
#include "tt-metalium/mesh_device.hpp"

namespace tt::runtime::system_desc {

::flatbuffers::Offset<tt::target::SystemDescRoot>
buildSystemDescRoot(::flatbuffers::FlatBufferBuilder &fbb,
                    const ::tt::tt_metal::distributed::MeshDevice &meshDevice);
} // namespace tt::runtime::system_desc

#endif // TT_RUNTIME_DETAIL_COMMON_SYSTEM_DESC_H
