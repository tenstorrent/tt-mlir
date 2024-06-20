// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TYPES_H
#define TT_RUNTIME_TYPES_H

#include <memory>

namespace tt::runtime {

using Handle = std::shared_ptr<void>;

struct SystemDesc {
  Handle handle;
};

struct Binary {
  Handle handle;
};

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
