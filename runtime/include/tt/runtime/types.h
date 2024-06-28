// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TYPES_H
#define TT_RUNTIME_TYPES_H

#include <memory>
#include <string_view>
#include <vector>

#include "tt/runtime/utils.h"
#include "ttmlir/Target/Common/system_desc_generated.h"
#include "ttmlir/Target/Common/types_generated.h"

namespace tt::runtime {

namespace detail {
struct ObjectImpl {
  std::shared_ptr<void> handle;

  ObjectImpl(std::shared_ptr<void> handle) : handle(handle) {}
  template <typename T> T &as() { return *static_cast<T *>(handle.get()); }
  template <typename T> T const &as() const {
    return *static_cast<T const *>(handle.get());
  }
};
} // namespace detail

struct TensorDesc {
  std::vector<std::uint32_t> shape;
  std::vector<std::uint32_t> stride;
  std::uint32_t itemsize;
  ::tt::target::DataType dataType;
};

using DeviceIds = std::vector<int>;

struct Flatbuffer : public detail::ObjectImpl {
  using detail::ObjectImpl::ObjectImpl;

  static Flatbuffer loadFromPath(char const *path);

  void store(char const *path) const;
  std::string_view getFileIdentifier() const;
  std::string getVersion() const;
  std::string_view getTTMLIRGitHash() const;
  std::string asJson() const;
};

struct SystemDesc : public Flatbuffer {
  using Flatbuffer::Flatbuffer;

  static SystemDesc loadFromPath(char const *path);

  ::tt::target::SystemDesc const *get() const {
    return ::tt::target::GetSizePrefixedSystemDescRoot(handle.get())
        ->system_desc();
  }
  ::tt::target::SystemDesc const *operator->() const { return get(); }
};

struct Binary : public Flatbuffer {
  using Flatbuffer::Flatbuffer;

  static Binary loadFromPath(char const *path);

  std::vector<TensorDesc> getProgramInputs(std::uint32_t programIndex) const;
  std::vector<TensorDesc> getProgramOutputs(std::uint32_t programIndex) const;
};

struct Device : public detail::ObjectImpl {
  using detail::ObjectImpl::ObjectImpl;

  template <typename T> static Device borrow(T &object) {
    return Device(utils::unsafe_borrow_shared(&object));
  }
};

struct Event : public detail::ObjectImpl {
  using detail::ObjectImpl::ObjectImpl;
};

struct Tensor : public detail::ObjectImpl {
  std::shared_ptr<void> data;
  Tensor(std::shared_ptr<void> handle, std::shared_ptr<void> data)
      : detail::ObjectImpl(handle), data(data) {}
};

} // namespace tt::runtime

#endif
