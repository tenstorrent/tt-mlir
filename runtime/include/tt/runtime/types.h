// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TYPES_H
#define TT_RUNTIME_TYPES_H

#include <cassert>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/Common/debug_info_generated.h"
#include "ttmlir/Target/Common/system_desc_generated.h"
#include "ttmlir/Target/Common/types_generated.h"
#pragma clang diagnostic pop

namespace tt::runtime {
/*
MemoryBlockTable is a list of memory blocks in the following format:
[{"blockID": "0", "address": "0", "size": "0", "prevID": "0", "nextID": "0",
"allocated": true}] address: bytes size: bytes
*/
using MemoryBlockTable =
    std::vector<std::unordered_map<std::string, std::string>>;
using DeviceIds = std::vector<int>;

enum class MemoryBufferType {
  DRAM,
  L1,
  L1_SMALL,
  TRACE,
};

enum class DeviceRuntime {
  Disabled,
  TTNN,
  TTMetal,
};

enum class DispatchCoreType {
  WORKER,
  ETH,
};

namespace detail {
struct ObjectImpl {

  std::shared_ptr<void> handle;

  ObjectImpl(std::shared_ptr<void> handle) : handle(handle) {}
  template <typename T>
  T &as() {
    return *static_cast<T *>(handle.get());
  }
  template <typename T>
  T const &as() const {
    return *static_cast<T const *>(handle.get());
  }
  template <typename T>
  std::shared_ptr<T> handle_as() {
    return std::static_pointer_cast<T>(handle);
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

  template <typename T>
  T &as(DeviceRuntime expectedRuntime) {
    assert(associatedRuntime == expectedRuntime &&
           "Associated runtime does not match expected runtime of cast");
    return *static_cast<T *>(handle.get());
  }
  template <typename T>
  T const &as(DeviceRuntime expectedRuntime) const {
    assert(associatedRuntime == expectedRuntime &&
           "Associated runtime does not match expected runtime of cast");
    return *static_cast<T const *>(handle.get());
  }
  template <typename T>
  std::shared_ptr<T> handle_as(DeviceRuntime expectedRuntime) const {
    assert(associatedRuntime == expectedRuntime &&
           "Associated runtime does not match expected runtime of cast");
    return std::static_pointer_cast<T>(handle);
  }
};

} // namespace detail

struct TensorDesc {
  std::vector<std::uint32_t> shape;
  std::vector<std::uint32_t> stride;
  std::uint32_t itemsize;
  ::tt::target::DataType dataType;

  std::uint32_t size() const { return shape[0] * stride[0] * itemsize; }
};

struct MemoryView {
  std::uint64_t numBanks = 0;
  size_t totalBytesPerBank = 0;
  size_t totalBytesAllocatedPerBank = 0;
  size_t totalBytesFreePerBank = 0;
  size_t largestContiguousBytesFreePerBank = 0;
  MemoryBlockTable blockTable;
};

struct MeshDeviceOptions {
  std::vector<uint32_t> meshOffset{0, 0};
  std::vector<int> deviceIds{};
  size_t numHWCQs = 1;
  bool enableAsyncTTNN = false;
  bool enableProgramCache = false;
  std::optional<size_t> l1SmallSize = std::nullopt;
  std::optional<DispatchCoreType> dispatchCoreType = std::nullopt;
};

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
  const ::tt::target::GoldenTensor *getDebugInfoGolden(std::string &loc) const;
};

struct Device : public detail::RuntimeCheckedObjectImpl {
  using detail::RuntimeCheckedObjectImpl::RuntimeCheckedObjectImpl;
};

struct Event : public detail::RuntimeCheckedObjectImpl {
  using detail::RuntimeCheckedObjectImpl::RuntimeCheckedObjectImpl;
};

struct Tensor : public detail::RuntimeCheckedObjectImpl {
  std::shared_ptr<void> data;
  Event event;

  Tensor(std::shared_ptr<void> handle, std::shared_ptr<void> data,
         DeviceRuntime runtime)
      : detail::RuntimeCheckedObjectImpl(handle, runtime), data(data),
        event(nullptr, runtime) {}

  Tensor(std::shared_ptr<void> handle, std::shared_ptr<void> data,
         std::shared_ptr<void> eventHandle, DeviceRuntime runtime)
      : detail::RuntimeCheckedObjectImpl(handle, runtime), data(data),
        event(eventHandle, runtime) {}
};

struct Layout : public detail::RuntimeCheckedObjectImpl {
  using detail::RuntimeCheckedObjectImpl::RuntimeCheckedObjectImpl;
};

struct CallbackContext : public detail::RuntimeCheckedObjectImpl {
  using detail::RuntimeCheckedObjectImpl::RuntimeCheckedObjectImpl;
};

struct OpContext : public detail::RuntimeCheckedObjectImpl {
  using detail::RuntimeCheckedObjectImpl::RuntimeCheckedObjectImpl;
};

} // namespace tt::runtime

#endif
