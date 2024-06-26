// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TYPES_H
#define TT_RUNTIME_TYPES_H

#include <memory>
#include <vector>

#include "tt/runtime/utils.h"
#include "ttmlir/Target/Common/types_generated.h"

namespace tt::runtime {

using DeviceIds = std::vector<int>;

struct ObjectImpl {
  std::shared_ptr<void> handle;

  ObjectImpl(std::shared_ptr<void> handle) : handle(handle) {}
  template <typename T>
  ObjectImpl(T &borrowed_reference)
      : handle(utils::unsafe_borrow_shared(&borrowed_reference)) {}

  template <typename T> T &as() { return *static_cast<T *>(handle.get()); }
  template <typename T> T const &as() const {
    return *static_cast<T const *>(handle.get());
  }
};

struct Flatbuffer : public ObjectImpl {
  using ObjectImpl::ObjectImpl;
};

using SystemDesc = Flatbuffer;
using Binary = Flatbuffer;

struct Device : public ObjectImpl {
  using ObjectImpl::ObjectImpl;
};

struct Event : public ObjectImpl {
  using ObjectImpl::ObjectImpl;
};

struct Tensor : public ObjectImpl {
  std::shared_ptr<void> data;
  Tensor(std::shared_ptr<void> handle, std::shared_ptr<void> data)
      : ObjectImpl(handle), data(data) {}
};

struct TensorDesc {
  std::vector<std::uint32_t> shape;
  std::vector<std::uint32_t> stride;
  std::uint32_t itemsize;
  ::tt::target::DataType dataType;
};

} // namespace tt::runtime

#endif
