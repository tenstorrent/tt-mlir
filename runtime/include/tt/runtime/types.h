// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TYPES_H
#define TT_RUNTIME_TYPES_H

#include <memory>
#include <vector>

namespace tt::runtime {

using Handle = std::shared_ptr<void>;
using DeviceIds = std::vector<int>;

struct Flatbuffer {
  Handle handle;
};

using SystemDesc = Flatbuffer;
using Binary = Flatbuffer;

struct Device {
  Handle handle;
};

struct Event {
  Handle handle;
};

struct Tensor {
  Handle handle;
};

} // namespace tt::runtime

#endif
