// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TYPES_H
#define TT_RUNTIME_TYPES_H

#include <cassert>
#include <memory>
#include <string_view>
#include <vector>

#include "tt/runtime/utils.h"
#include "ttmlir/Target/Common/system_desc_generated.h"
#include "ttmlir/Target/Common/types_generated.h"

namespace tt::runtime {

enum class DeviceRuntime {
  Disabled,
  TTNN,
  TTMetal,
};

namespace detail {
struct ObjectImpl {

  std::shared_ptr<void> handle;

  ObjectImpl(std::shared_ptr<void> handle) : handle(handle) {}
  template <typename T> T &as() { return *static_cast<T *>(handle.get()); }
  template <typename T> T const &as() const {
    return *static_cast<T const *>(handle.get());
  }
};

struct RuntimeCheckedObjectImpl {
  std::shared_ptr<void> handle;
  ::tt::runtime::DeviceRuntime associatedRuntime;

  RuntimeCheckedObjectImpl(std::shared_ptr<void> handle,
                           ::tt::runtime::DeviceRuntime runtime)
      : handle(handle), associatedRuntime(runtime) {}

  bool matchesRuntime(DeviceRuntime runtime) const {
    return associatedRuntime == runtime;
  }

  template <typename T> T &as(DeviceRuntime expectedRuntime) {
    assert(associatedRuntime == expectedRuntime &&
           "Associated runtime does not match expected runtime of cast");
    return *static_cast<T *>(handle.get());
  }
  template <typename T> T const &as(DeviceRuntime expectedRuntime) const {
    assert(associatedRuntime == expectedRuntime &&
           "Associated runtime does not match expected runtime of cast");
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

struct Device : public detail::RuntimeCheckedObjectImpl {
  using detail::RuntimeCheckedObjectImpl::RuntimeCheckedObjectImpl;

  template <typename T> static Device borrow(T &object, DeviceRuntime runtime) {
    return Device(utils::unsafe_borrow_shared(&object), runtime);
  }
};

struct Event : public detail::RuntimeCheckedObjectImpl {
  Event(std::shared_ptr<void> handle, DeviceRuntime runtime)
      : detail::RuntimeCheckedObjectImpl(handle, runtime) {}

  bool isTTNNEvent() const {
    return this->matchesRuntime(DeviceRuntime::TTNN) and this->handle.get();
  }

  bool isTTMetalEvent() const {
    return this->matchesRuntime(DeviceRuntime::TTMetal) and this->handle.get();
  }
};

struct Tensor : public detail::RuntimeCheckedObjectImpl {
  std::shared_ptr<void> data;
  Event event;

  Tensor(std::shared_ptr<void> handle, std::shared_ptr<void> data,
         DeviceRuntime runtime)
      : detail::RuntimeCheckedObjectImpl(handle, runtime), data(data),
        event(Event(nullptr, runtime)) {}

  Tensor(std::shared_ptr<void> handle, std::shared_ptr<void> data,
         DeviceRuntime runtime, Event event)
      : detail::RuntimeCheckedObjectImpl(handle, runtime), data(data),
        event(event) {}

  // Users need to manually deallocate tensors returned from submit
  // As the storage is now owned instead of borrowed
  void deallocate(bool force = false);

  Tensor cpu() const;
};

} // namespace tt::runtime

#endif
