// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TYPES_H
#define TT_RUNTIME_TYPES_H

#include <cassert>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "tt/runtime/utils.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/Common/debug_info_generated.h"
#include "ttmlir/Target/Common/system_desc_generated.h"
#include "ttmlir/Target/Common/types_generated.h"
#pragma clang diagnostic pop

#include "tt/runtime/flatbuffer/flatbuffer.h"

namespace tt::runtime {

using ::tt::runtime::flatbuffer::DeviceRuntime;
using ::tt::runtime::flatbuffer::DispatchCoreType;
using ::tt::runtime::flatbuffer::FabricConfig;
using ::tt::runtime::flatbuffer::HostRuntime;

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

inline std::string toString(DeviceRuntime runtime) {
  return ::tt::runtime::flatbuffer::EnumNameDeviceRuntime(runtime);
}

inline std::string toString(HostRuntime runtime) {
  return ::tt::runtime::flatbuffer::EnumNameHostRuntime(runtime);
}

enum class DistributedMode {
  // Single process on the local host,
  // mainly for testing and debugging
  LocalSubprocess,

  // Multiple processes via MPI, may be same host or distributed
  // across multiple hosts
  MultiProcess,
};

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

  RuntimeCheckedObjectImpl()
      : handle(nullptr), associatedRuntime(DeviceRuntime::Disabled) {}
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

struct RuntimeCheckedConstObjectImpl {
  std::shared_ptr<const void> handle;
  ::tt::runtime::DeviceRuntime associatedRuntime;

  RuntimeCheckedConstObjectImpl(std::shared_ptr<const void> handle,
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
  const T &as(DeviceRuntime expectedRuntime) const {
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
  std::vector<uint32_t> shape = {}; // Logical.
  ::tt::target::DataType dataType = ::tt::target::DataType::MAX;
  uint32_t itemsize = 0;
  std::vector<uint32_t> stride = {}; // Potentially padded.
  uint64_t physicalVolume = 0;       // Potentially padded.

  TensorDesc() = default;

  TensorDesc(const std::vector<uint32_t> &shape,
             const ::tt::target::DataType dataType,
             const std::optional<uint32_t> itemsize = {},
             const std::optional<std::vector<uint32_t>> &stride = {},
             const std::optional<uint64_t> physicalVolume = {})
      : shape(shape), dataType(dataType) {
    this->itemsize = itemsize.value_or(utils::dataTypeElementSize(dataType));
    this->stride = stride.value_or(utils::calculateStride(shape));
    this->physicalVolume = physicalVolume.value_or(volume());
  }

  size_t volume() const { return utils::product(shape.cbegin(), shape.cend()); }

  size_t sizeBytes() const { return physicalVolume * itemsize; }

  bool isPadded() const { return physicalVolume > volume(); }
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
  std::optional<std::vector<uint32_t>> meshShape = std::nullopt;
  std::optional<size_t> l1SmallSize = std::nullopt;
  std::optional<size_t> traceRegionSize = std::nullopt;
  std::optional<::tt::runtime::DispatchCoreType> dispatchCoreType =
      std::nullopt;
};

class MultiProcessArgs {
public:
  static MultiProcessArgs create(std::string_view rankBindingPath) {
    return MultiProcessArgs(rankBindingPath);
  }

  MultiProcessArgs &withHosts(const std::vector<std::string> &hosts);
  MultiProcessArgs &withHostsFilePath(std::string_view path);

  std::string getRankBindingPath() const;
  MultiProcessArgs &withRankFilePath(std::string_view path);

  MultiProcessArgs &
  withMcaOptions(const std::map<std::string, std::string> &mcaOptions);

  MultiProcessArgs &withTagOutput(bool tagOutput);

  MultiProcessArgs &withAllowRunAsRoot(bool allowRunAsRoot);

  MultiProcessArgs &
  withExtraMpiArgs(const std::vector<std::string> &extraMpiArgs);

  std::string toArgString() const;

private:
  MultiProcessArgs(std::string_view rankBindingPath);

  std::filesystem::path rankBindingPath_;

  std::vector<std::string> hosts_;
  std::filesystem::path hostsFilePath_;

  std::filesystem::path rankFilePath_;

  std::map<std::string, std::string> mcaOptions_;

  bool tagOutput_ = true;

  bool allowRunAsRoot_ = false;

  std::vector<std::string> extraMpiArgs_;
};

struct DistributedOptions {
  uint16_t controllerPort = 0;
  DistributedMode mode = DistributedMode::LocalSubprocess;
  // Required for MultiProcess mode
  std::optional<MultiProcessArgs> multiProcessArgs = std::nullopt;
};

struct Flatbuffer : public detail::ObjectImpl {
  using detail::ObjectImpl::ObjectImpl;

  static Flatbuffer loadFromPath(const char *path);
  static Flatbuffer loadFromMemory(const void *memory, size_t size);

  void store(const char *path) const;

  template <typename T>
  void storeToMemory(std::vector<T> &serializedFlatbuffer) const;

  std::string getFileIdentifier() const;
  std::string getVersion() const;
  std::string getSchemaHash() const;
  bool checkSchemaHash() const;
  std::string getTTMLIRGitHash() const;
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
class ProgramDescCache;

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

  // Binary asJson functions are broken down to get individual flatbuffer
  // components, allowing for bypassing the golden_map in debug_info, the
  // loading and processing of which can use significant memory and time.
  std::uint32_t getNumPrograms() const;
  std::string getSystemDescAsJson() const;
  std::string getProgramName(std::uint32_t programIndex) const;
  bool isProgramPrivate(std::uint32_t programIndex) const;
  std::string getProgramOpsAsJson(std::uint32_t programIndex) const;
  std::string getProgramInputsAsJson(std::uint32_t programIndex) const;
  std::string getProgramOutputsAsJson(std::uint32_t programIndex) const;
  std::string getMlirAsJson() const;
  std::uint32_t getNumProgramInputs(std::uint32_t programIndex) const;
  std::uint32_t getNumProgramOutputs(std::uint32_t programIndex) const;
  std::vector<TensorDesc> getProgramInputs(std::uint32_t programIndex) const;
  std::vector<TensorDesc> getProgramOutputs(std::uint32_t programIndex) const;
  std::unordered_map<std::uint32_t, const ::tt::target::GoldenTensor *>
  getDebugInfoGolden(std::string &loc) const;
  const std::pair<std::uint32_t, std::uint32_t>
  getProgramMeshShape(std::uint32_t programIndex) const;

  void setId(std::uint64_t id) { binaryId = id; }
  std::uint64_t id() const;

  // Get the tensor cache associated with this binary
  std::shared_ptr<TensorCache> getConstEvalTensorCache() { return tensorCache; }

  // Get the program descriptor cache associated with this binary
  std::shared_ptr<tt::runtime::ProgramDescCache> getProgramDescCache() {
    return programDescCache;
  }

private:
  std::uint64_t nextBinaryId();

  std::uint64_t binaryId;

  // The tensor cache associated with this binary
  std::shared_ptr<TensorCache> tensorCache;

  // Program descriptor cache associated with this binary
  std::shared_ptr<tt::runtime::ProgramDescCache> programDescCache;
};

struct TraceCache : public detail::RuntimeCheckedObjectImpl {
  using detail::RuntimeCheckedObjectImpl::RuntimeCheckedObjectImpl;
};

struct Device : public detail::RuntimeCheckedObjectImpl {
  Device(DeviceRuntime runtime)
      : detail::RuntimeCheckedObjectImpl(nullptr, runtime),
        globalId(nextDeviceGlobalId()), traceCache(nullptr) {}

  Device(std::shared_ptr<void> handle, std::shared_ptr<TraceCache> traceCache,
         DeviceRuntime runtime)
      : detail::RuntimeCheckedObjectImpl(handle, runtime),
        globalId(nextDeviceGlobalId()), traceCache(traceCache) {}

  std::shared_ptr<TraceCache> getTraceCache() { return traceCache; }

  void setGlobalId(std::uint32_t id) { globalId = id; }
  std::uint32_t getGlobalId() const { return globalId; }

private:
  std::uint32_t nextDeviceGlobalId();

  std::uint32_t globalId;

  // The trace cache associated with this device.
  std::shared_ptr<TraceCache> traceCache;
};

struct Event : public detail::RuntimeCheckedObjectImpl {
  using detail::RuntimeCheckedObjectImpl::RuntimeCheckedObjectImpl;
};

struct Tensor : public detail::RuntimeCheckedObjectImpl {
  std::shared_ptr<void> data;
  Event event;
  Tensor() : globalId(nextTensorGlobalId()) {}
  Tensor(std::shared_ptr<void> handle, std::shared_ptr<void> data,
         DeviceRuntime runtime,
         std::optional<std::shared_ptr<void>> eventHandle = std::nullopt)
      : detail::RuntimeCheckedObjectImpl(handle, runtime), data(data),
        event(eventHandle.value_or(nullptr), runtime),
        globalId(nextTensorGlobalId()) {}

  void setGlobalId(std::uint64_t id) { globalId = id; }
  std::uint64_t getGlobalId() const { return globalId; }

private:
  std::uint64_t nextTensorGlobalId();

  std::uint64_t globalId;
};

struct TensorRef : public detail::RuntimeCheckedConstObjectImpl {
  using detail::RuntimeCheckedConstObjectImpl::RuntimeCheckedConstObjectImpl;
};

struct Layout : public detail::RuntimeCheckedObjectImpl {
  Layout(DeviceRuntime runtime)
      : detail::RuntimeCheckedObjectImpl(nullptr, runtime),
        globalId(nextLayoutGlobalId()) {}
  Layout(std::shared_ptr<void> handle, DeviceRuntime runtime)
      : detail::RuntimeCheckedObjectImpl(handle, runtime),
        globalId(nextLayoutGlobalId()) {}

  void setGlobalId(std::uint64_t id) { globalId = id; }
  std::uint64_t getGlobalId() const { return globalId; }

private:
  std::uint64_t nextLayoutGlobalId();

  std::uint64_t globalId;
};

struct CallbackContext : public detail::RuntimeCheckedObjectImpl {
  using detail::RuntimeCheckedObjectImpl::RuntimeCheckedObjectImpl;
};

struct OpContext : public detail::RuntimeCheckedObjectImpl {
  using detail::RuntimeCheckedObjectImpl::RuntimeCheckedObjectImpl;
};

} // namespace tt::runtime

#endif
