// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TYPES_H
#define TT_RUNTIME_TYPES_H

#include <cassert>
#include <memory>
#include <numeric>
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

inline std::string toString(DeviceRuntime runtime) {
  switch (runtime) {
  case DeviceRuntime::TTNN:
    return "TTNN";
  case DeviceRuntime::TTMetal:
    return "TTMetal";
  case DeviceRuntime::Disabled:
    return "Disabled";
  }
}

enum class DispatchCoreType {
  WORKER,
  ETH,
};

enum class Arch { GRAYSKULL = 1, WORMHOLE_B0 = 2, BLACKHOLE = 3, QUASAR = 4 };

namespace detail {
struct ObjectImpl {

  std::shared_ptr<void> handle;

  ObjectImpl(std::shared_ptr<void> handle) : handle(handle) {}
  template <typename T>
  T &as() {
    assert(handle && "Handle should not be null");
    return *static_cast<T *>(handle.get());
  }
  template <typename T>
  const T &as() const {
    assert(handle && "Handle should not be null");
    return *static_cast<const T *>(handle.get());
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

  void assertMatchesRuntime(DeviceRuntime expectedRuntime) const {
    assert(matchesRuntime(expectedRuntime) &&
           "Associated runtime does not match expected runtime of cast");
  }

  template <typename T>
  T &as(DeviceRuntime expectedRuntime) {
    assert(handle && "Handle should not be null");
    assertMatchesRuntime(expectedRuntime);
    return *static_cast<T *>(handle.get());
  }

  template <typename T>
  const T &as(DeviceRuntime expectedRuntime) const {
    assert(handle && "Handle should not be null");
    assertMatchesRuntime(expectedRuntime);
    return *static_cast<const T *>(handle.get());
  }

  template <typename T>
  std::shared_ptr<T> asSharedPtr(DeviceRuntime expectedRuntime) const {
    assertMatchesRuntime(expectedRuntime);
    return std::static_pointer_cast<T>(handle);
  }
};

} // namespace detail

struct TensorDesc {
  std::vector<std::uint32_t> shape;
  std::vector<std::uint32_t> stride;
  std::uint32_t itemsize;
  ::tt::target::DataType dataType;

  TensorDesc() = default;
  TensorDesc(const std::vector<std::uint32_t> &shape,
             const std::vector<std::uint32_t> &stride, std::uint32_t itemsize,
             ::tt::target::DataType dataType)
      : shape(shape), stride(stride), itemsize(itemsize), dataType(dataType) {}

  std::int64_t volume() const {
    return std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1),
                           std::multiplies<int64_t>());
  }
  std::int64_t sizeBytes() const { return volume() * itemsize; }
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
  bool enableProgramCache = false;
  std::optional<size_t> l1SmallSize = std::nullopt;
  std::optional<size_t> traceRegionSize = std::nullopt;
  std::optional<DispatchCoreType> dispatchCoreType = std::nullopt;
};

struct TraceCache : public detail::RuntimeCheckedObjectImpl {
  using detail::RuntimeCheckedObjectImpl::RuntimeCheckedObjectImpl;
};

struct Flatbuffer : public detail::ObjectImpl {
  using detail::ObjectImpl::ObjectImpl;

  static Flatbuffer loadFromPath(const char *path);

  void store(const char *path) const;
  std::string_view getFileIdentifier() const;
  std::string getVersion() const;
  std::string_view getTTMLIRGitHash() const;
  std::string asJson() const;
};

struct SystemDesc : public Flatbuffer {
  using Flatbuffer::Flatbuffer;

  static SystemDesc loadFromPath(const char *path);

  const ::tt::target::SystemDesc *get() const {
    return ::tt::target::GetSizePrefixedSystemDescRoot(handle.get())
        ->system_desc();
  }
  const ::tt::target::SystemDesc *operator->() const { return get(); }
};

class TensorCache;
struct Binary : public Flatbuffer {
  Binary(Flatbuffer fb);
  Binary(std::shared_ptr<void> handle);
  Binary(const Binary &) = default;
  Binary(Binary &&) = default;

  Binary &operator=(const Binary &other) = default;
  Binary &operator=(Binary &&other) = default;
  Binary &operator=(Flatbuffer fb);
  Binary &operator=(std::shared_ptr<void> handle);

  using Flatbuffer::Flatbuffer;

  static Binary loadFromPath(const char *path);

  std::vector<TensorDesc> getProgramInputs(std::uint32_t programIndex) const;
  std::vector<TensorDesc> getProgramOutputs(std::uint32_t programIndex) const;
  const ::tt::target::GoldenTensor *getDebugInfoGolden(std::string &loc) const;

  std::uint64_t id() const;

  // Get the tensor cache associated with this binary
  std::shared_ptr<TensorCache> getConstEvalTensorCache() { return tensorCache; }

private:
  std::uint64_t nextBinaryId();

  std::uint64_t binaryId;

  // The tensor cache associated with this binary
  std::shared_ptr<TensorCache> tensorCache;
};

struct Device : public detail::RuntimeCheckedObjectImpl {

  Device(std::shared_ptr<void> handle, std::shared_ptr<TraceCache> traceCache,
         DeviceRuntime runtime)
      : detail::RuntimeCheckedObjectImpl(handle, runtime),
        traceCache(traceCache) {}

  std::shared_ptr<TraceCache> getTraceCache() { return traceCache; }

private:
  // The trace cache associated with this device.
  // This will get null initialized in the constructor
  // and properly initialized within the ttnn runtime
  // when it is needed to capture and replay a trace
  std::shared_ptr<TraceCache> traceCache;
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
