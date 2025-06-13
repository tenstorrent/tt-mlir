// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/runtime.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Version.h"
#include <atomic>

#if defined(TT_RUNTIME_ENABLE_TTNN) && (TT_RUNTIME_ENABLE_TTNN == 1)
#include "tt/runtime/detail/ttnn/ttnn.h"
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL) && (TT_RUNTIME_ENABLE_TTMETAL == 1)
#include "tt/runtime/detail/ttmetal/ttmetal.h"
#endif

#if defined(TT_RUNTIME_ENABLE_TTNN) && (TT_RUNTIME_ENABLE_TTNN == 1)
#define IF_TTNN_ENABLED(code) code
#else
#define IF_TTNN_ENABLED(code)
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL) && (TT_RUNTIME_ENABLE_TTMETAL == 1)
#define IF_TTMETAL_ENABLED(code) code
#else
#define IF_TTMETAL_ENABLED(code)
#endif

// This macro dispatches function calls to the current runtime implementation.
// An explicit retType is required because with auto deduction (-> auto),
// the compiler can't determine if the lambda should return the expected type
// or void in the default branch, causing compilation errors. Since ttnnImpl
// and/or ttmetalImpl may not exist, inferring return type from these functions
// is also not feasible, this is also why a macro is used over a templated
// approach.
#define DISPATCH_TO_CURRENT_RUNTIME(retType, ttnnImpl, ttmetalImpl)            \
  [&]() -> retType {                                                           \
    IF_TTNN_ENABLED(                                                           \
        static_assert(std::is_invocable_v<decltype((ttnnImpl))>,               \
                      "TTNN implementation must be invocable");                \
        static_assert(                                                         \
            std::is_same_v<retType,                                            \
                           std::invoke_result_t<decltype((ttnnImpl))>>,        \
            "Return type must match TTNN implementation return type");)        \
    IF_TTMETAL_ENABLED(                                                        \
        static_assert(std::is_invocable_v<decltype((ttmetalImpl))>,            \
                      "TTMetal implementation must be invocable");             \
        static_assert(                                                         \
            std::is_same_v<retType,                                            \
                           std::invoke_result_t<decltype((ttmetalImpl))>>,     \
            "Return type must match TTMetal implementation return type");)     \
    switch (::tt::runtime::getCurrentRuntime()) {                              \
      IF_TTNN_ENABLED(case ::tt::runtime::DeviceRuntime::TTNN                  \
                      : { return (ttnnImpl)(); })                              \
      IF_TTMETAL_ENABLED(case ::tt::runtime::DeviceRuntime::TTMetal            \
                         : { return (ttmetalImpl)(); })                        \
    default: {                                                                 \
      LOG_FATAL("Runtime is not enabled");                                     \
    }                                                                          \
    }                                                                          \
  }()

namespace tt::runtime {
namespace detail {
static std::atomic<DeviceRuntime> &currentRuntime() {
#if defined(TT_RUNTIME_ENABLE_TTNN) && (TT_RUNTIME_ENABLE_TTNN == 1)
  static std::atomic<DeviceRuntime> globalRuntime = DeviceRuntime::TTNN;
#elif defined(TT_RUNTIME_ENABLE_TTMETAL) && (TT_RUNTIME_ENABLE_TTMETAL == 1)
  static std::atomic<DeviceRuntime> globalRuntime = DeviceRuntime::TTMetal;
#else
  static std::atomic<DeviceRuntime> globalRuntime = DeviceRuntime::Disabled;
#endif
  return globalRuntime;
}

[[noreturn, maybe_unused]] static void
fatalNotImplemented(const std::string &funcName, DeviceRuntime runtime) {
  std::string message = "Function " + funcName + " is not implemented for " +
                        toString(runtime) + " runtime";
  LOG_FATAL(message);
}

void deallocateBuffers(Device device) {
  using RetType = void;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType, [&]() { ::tt::runtime::ttnn::deallocateBuffers(device); },
      [&]() { ::tt::runtime::ttmetal::deallocateBuffers(device); });
}

void dumpMemoryReport(Device device) {
  using RetType = void;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType, [&]() { ::tt::runtime::ttnn::dumpMemoryReport(device); },
      [&]() { ::tt::runtime::ttmetal::dumpMemoryReport(device); });
}

void dumpDeviceProfileResults(Device device) {
  using RetType = void;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType, [&]() { ::tt::runtime::ttnn::dumpDeviceProfileResults(device); },
      [&]() { ::tt::runtime::ttmetal::dumpDeviceProfileResults(device); });
}

using MemoryViewResult = std::unordered_map<::tt::runtime::MemoryBufferType,
                                            ::tt::runtime::MemoryView>;
MemoryViewResult getMemoryView(Device device) {
  using RetType = MemoryViewResult;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType { return ::tt::runtime::ttnn::getMemoryView(device); },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::getMemoryView(device);
      });
}
} // namespace detail

std::vector<DeviceRuntime> getAvailableRuntimes() {
  std::vector<DeviceRuntime> runtimes;
#if defined(TT_RUNTIME_ENABLE_TTNN) && (TT_RUNTIME_ENABLE_TTNN == 1)
  runtimes.push_back(DeviceRuntime::TTNN);
#endif
#if defined(TT_RUNTIME_ENABLE_TTMETAL) && (TT_RUNTIME_ENABLE_TTMETAL == 1)
  runtimes.push_back(DeviceRuntime::TTMetal);
#endif
  return runtimes;
}

DeviceRuntime getCurrentRuntime() {
  DeviceRuntime runtime =
      detail::currentRuntime().load(std::memory_order_relaxed);
#if !defined(TT_RUNTIME_ENABLE_TTNN) || (TT_RUNTIME_ENABLE_TTNN == 0)
  LOG_ASSERT(runtime != DeviceRuntime::TTNN);
#endif
#if !defined(TT_RUNTIME_ENABLE_TTMETAL) || (TT_RUNTIME_ENABLE_TTMETAL == 0)
  LOG_ASSERT(runtime != DeviceRuntime::TTMetal);
#endif
  return runtime;
}

void setCurrentRuntime(const DeviceRuntime &runtime) {
#if !defined(TT_RUNTIME_ENABLE_TTNN) || (TT_RUNTIME_ENABLE_TTNN == 0)
  LOG_ASSERT(runtime != DeviceRuntime::TTNN);
#endif
#if !defined(TT_RUNTIME_ENABLE_TTMETAL) || (TT_RUNTIME_ENABLE_TTMETAL == 0)
  LOG_ASSERT(runtime != DeviceRuntime::TTMetal);
#endif
  detail::currentRuntime().store(runtime, std::memory_order_relaxed);
}

void setCompatibleRuntime(const Binary &binary) {
#if defined(TT_RUNTIME_ENABLE_TTNN) && (TT_RUNTIME_ENABLE_TTNN == 1)
  if (binary.getFileIdentifier() ==
      ::tt::target::ttnn::TTNNBinaryIdentifier()) {
    return setCurrentRuntime(DeviceRuntime::TTNN);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL) && (TT_RUNTIME_ENABLE_TTMETAL == 1)
  if (binary.getFileIdentifier() ==
      ::tt::target::metal::TTMetalBinaryIdentifier()) {
    return setCurrentRuntime(DeviceRuntime::TTMetal);
  }
#endif
  LOG_FATAL("Unsupported binary file identifier or runtime not enabled");
}

SystemDesc
getCurrentSystemDesc(std::optional<DispatchCoreType> dispatchCoreType,
                     std::optional<Device> meshDevice) {
#if (defined(TT_RUNTIME_ENABLE_TTNN) && (TT_RUNTIME_ENABLE_TTNN == 1)) ||      \
    (defined(TT_RUNTIME_ENABLE_TTMETAL) && (TT_RUNTIME_ENABLE_TTMETAL == 1))
  return system_desc::getCurrentSystemDesc(dispatchCoreType, meshDevice);
#endif
  LOG_FATAL("runtime is not enabled");
}

Tensor createBorrowedHostTensor(void *data,
                                const std::vector<std::uint32_t> &shape,
                                const std::vector<std::uint32_t> &stride,
                                std::uint32_t itemsize,
                                ::tt::target::DataType dataType) {
  using RetType = Tensor;
  LOG_ASSERT(!shape.empty());
  LOG_ASSERT(!stride.empty());
  LOG_ASSERT(itemsize > 0);
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::createBorrowedHostTensor(
            data, shape, stride, itemsize, dataType);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::createBorrowedHostTensor(
            data, TensorDesc(shape, stride, itemsize, dataType));
      });
}

Tensor createOwnedHostTensor(const void *data,
                             const std::vector<std::uint32_t> &shape,
                             const std::vector<std::uint32_t> &stride,
                             std::uint32_t itemsize,
                             ::tt::target::DataType dataType) {
  using RetType = Tensor;
  LOG_ASSERT(!shape.empty());
  LOG_ASSERT(!stride.empty());
  LOG_ASSERT(itemsize > 0);
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::createOwnedHostTensor(data, shape, stride,
                                                          itemsize, dataType);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::createOwnedHostTensor(
            data, TensorDesc(shape, stride, itemsize, dataType));
      });
}

// TODO(mrakita): Should be deprecated but D2M path is using this, investigate
// if it can also use the new `createBorrowedHostTensor` function.
// https://github.com/tenstorrent/tt-mlir/issues/2757
Tensor createTensor(std::shared_ptr<void> data,
                    const std::vector<std::uint32_t> &shape,
                    const std::vector<std::uint32_t> &stride,
                    std::uint32_t itemsize, ::tt::target::DataType dataType) {
  using RetType = Tensor;
  LOG_ASSERT(!shape.empty());
  LOG_ASSERT(!stride.empty());
  LOG_ASSERT(itemsize > 0);
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::createBorrowedHostTensor(
            data.get(), shape, stride, itemsize, dataType);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::createBorrowedHostTensor(
            data, TensorDesc(shape, stride, itemsize, dataType));
      });
}

Tensor createMultiDeviceHostTensor(
    const std::vector<const void *> &data,
    const std::vector<std::uint32_t> &shape,
    const std::vector<std::uint32_t> &stride, std::uint32_t itemsize,
    ::tt::target::DataType dataType,
    const std::unordered_map<std::string, std::string> &strategy) {
  using RetType = Tensor;
  LOG_ASSERT(!shape.empty());
  LOG_ASSERT(!stride.empty());
  LOG_ASSERT(itemsize > 0);
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::createMultiDeviceHostTensor(
            data, shape, stride, itemsize, dataType, strategy);
      },
      [&]() -> RetType {
        detail::fatalNotImplemented(__FUNCTION__, DeviceRuntime::TTMetal);
      });
}

Tensor createMultiDeviceHostTensor(
    const std::vector<Tensor> &tensorShards,
    const std::unordered_map<std::string, std::string> &strategy) {
  using RetType = Tensor;
  LOG_ASSERT(!tensorShards.empty());
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::createMultiDeviceHostTensor(tensorShards,
                                                                strategy);
      },
      [&]() -> RetType {
        detail::fatalNotImplemented(__FUNCTION__, DeviceRuntime::TTMetal);
      });
}

Tensor createEmptyTensor(Device device, Layout layout,
                         const std::vector<std::uint32_t> &shape,
                         const std::vector<std::uint32_t> &stride,
                         std::uint32_t itemsize) {
  using RetType = Tensor;
  LOG_ASSERT(!shape.empty());
  LOG_ASSERT(!stride.empty());
  LOG_ASSERT(itemsize > 0);
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::createEmptyTensor(device, layout, shape,
                                                      stride, itemsize);
      },
      [&]() -> RetType {
        detail::fatalNotImplemented(__FUNCTION__, DeviceRuntime::TTMetal);
      });
}

bool isTensorAllocated(Tensor tensor) {
  using RetType = bool;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::isTensorAllocated(tensor);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::isTensorAllocated(tensor);
      });
}

::tt::target::DataType getTensorDataType(Tensor tensor) {
  using RetType = ::tt::target::DataType;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::getTensorDataType(tensor);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::getTensorDataType(tensor);
      });
}

std::vector<std::byte> getTensorDataBuffer(Tensor t) {
  using RetType = std::vector<std::byte>;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType { return ::tt::runtime::ttnn::getTensorDataBuffer(t); },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::getTensorDataBuffer(t);
      });
}

std::vector<std::uint32_t> getTensorShape(Tensor t) {
  using RetType = std::vector<std::uint32_t>;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType { return ::tt::runtime::ttnn::getTensorShape(t); },
      [&]() -> RetType { return ::tt::runtime::ttmetal::getTensorShape(t); });
}

std::vector<std::uint32_t> getTensorStride(Tensor t) {
  using RetType = std::vector<std::uint32_t>;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType { return ::tt::runtime::ttnn::getTensorStride(t); },
      [&]() -> RetType { return ::tt::runtime::ttmetal::getTensorStride(t); });
}

std::uint32_t getTensorElementSize(Tensor t) {
  using RetType = std::uint32_t;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType { return ::tt::runtime::ttnn::getTensorElementSize(t); },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::getTensorElementSize(t);
      });
}

std::uint32_t getTensorVolume(Tensor t) {
  using RetType = std::uint32_t;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType { return ::tt::runtime::ttnn::getTensorVolume(t); },
      [&]() -> RetType { return ::tt::runtime::ttmetal::getTensorVolume(t); });
}

TensorDesc getTensorDesc(Tensor t) {
  using RetType = TensorDesc;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType { return ::tt::runtime::ttnn::getTensorDesc(t); },
      [&]() -> RetType { return ::tt::runtime::ttmetal::getTensorDesc(t); });
}

bool getTensorRetain(Tensor tensor) {
  using RetType = bool;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType { return ::tt::runtime::ttnn::getTensorRetain(tensor); },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::getTensorRetain(tensor);
      });
}

void setTensorRetain(Tensor tensor, bool retain) {
  using RetType = void;
  DISPATCH_TO_CURRENT_RUNTIME(
      RetType, [&]() { ::tt::runtime::ttnn::setTensorRetain(tensor, retain); },
      [&]() { ::tt::runtime::ttmetal::setTensorRetain(tensor, retain); });
}

Arch getArch() {
  using RetType = Arch;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType, [&]() -> RetType { return ::tt::runtime::ttnn::getArch(); },
      [&]() -> RetType { return ::tt::runtime::ttmetal::getArch(); });
}

size_t getNumAvailableDevices() {
  using RetType = size_t;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::getNumAvailableDevices();
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::getNumAvailableDevices();
      });
}

Device openMeshDevice(const std::vector<uint32_t> &meshShape,
                      const MeshDeviceOptions &options) {
  using RetType = Device;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::openMeshDevice(meshShape, options);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::openMeshDevice(meshShape, options);
      });
}

void closeMeshDevice(Device parentMesh) {
  using RetType = void;
  DISPATCH_TO_CURRENT_RUNTIME(
      RetType, [&]() { ::tt::runtime::ttnn::closeMeshDevice(parentMesh); },
      [&]() { ::tt::runtime::ttmetal::closeMeshDevice(parentMesh); });
}

Device createSubMeshDevice(
    Device parentMesh, const std::vector<uint32_t> &meshShape,
    const std::optional<const std::vector<uint32_t>> &meshOffset) {
  using RetType = Device;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::createSubMeshDevice(parentMesh, meshShape,
                                                        meshOffset);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::createSubMeshDevice(
            parentMesh, meshShape, meshOffset);
      });
}

void releaseSubMeshDevice(Device subMesh) {
  using RetType = void;
  DISPATCH_TO_CURRENT_RUNTIME(
      RetType, [&]() { ::tt::runtime::ttnn::releaseSubMeshDevice(subMesh); },
      [&]() { ::tt::runtime::ttmetal::releaseSubMeshDevice(subMesh); });
}

void reshapeMeshDevice(Device meshDevice,
                       const std::vector<uint32_t> &meshShape) {
  using RetType = void;
  DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() { ::tt::runtime::ttnn::reshapeMeshDevice(meshDevice, meshShape); },
      [&]() {
        ::tt::runtime::ttmetal::reshapeMeshDevice(meshDevice, meshShape);
      });
}

std::vector<uint32_t> getMeshShape(Device meshDevice) {
  using RetType = std::vector<uint32_t>;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::getMeshShape(meshDevice);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::getMeshShape(meshDevice);
      });
}

std::vector<int> getDeviceIds(Device meshDevice) {
  using RetType = std::vector<int>;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::getDeviceIds(meshDevice);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::getDeviceIds(meshDevice);
      });
}

size_t getNumHwCqs(Device meshDevice) {
  using RetType = size_t;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType { return ::tt::runtime::ttnn::getNumHwCqs(meshDevice); },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::getNumHwCqs(meshDevice);
      });
}

bool isProgramCacheEnabled(Device meshDevice) {
  using RetType = bool;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::isProgramCacheEnabled(meshDevice);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::isProgramCacheEnabled(meshDevice);
      });
}

size_t getL1SmallSize(Device meshDevice) {
  using RetType = size_t;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::getL1SmallSize(meshDevice);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::getL1SmallSize(meshDevice);
      });
}

size_t getTraceRegionSize(Device meshDevice) {
  using RetType = size_t;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::getTraceRegionSize(meshDevice);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::getTraceRegionSize(meshDevice);
      });
}

size_t getNumDramChannels(Device meshDevice) {
  using RetType = size_t;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::getNumDramChannels(meshDevice);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::getNumDramChannels(meshDevice);
      });
}

size_t getDramSizePerChannel(Device meshDevice) {
  using RetType = size_t;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::getDramSizePerChannel(meshDevice);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::getDramSizePerChannel(meshDevice);
      });
}

size_t getL1SizePerCore(Device meshDevice) {
  using RetType = size_t;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::getL1SizePerCore(meshDevice);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::getL1SizePerCore(meshDevice);
      });
}

bool releaseTrace(Device meshDevice, std::uint64_t binaryId, size_t programId) {
  using RetType = bool;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::releaseTrace(meshDevice, binaryId,
                                                 programId);
      },
      [&]() -> RetType {
        detail::fatalNotImplemented(__FUNCTION__, DeviceRuntime::TTMetal);
      });
}

void wait(Event event) {
  using RetType = void;
  DISPATCH_TO_CURRENT_RUNTIME(
      RetType, [&]() { ::tt::runtime::ttnn::wait(event); },
      [&]() { ::tt::runtime::ttmetal::wait(event); });
}

void wait(Tensor tensor) {
  using RetType = void;
  DISPATCH_TO_CURRENT_RUNTIME(
      RetType, [&]() { ::tt::runtime::ttnn::wait(tensor); },
      [&]() { ::tt::runtime::ttmetal::wait(tensor); });
}

void wait(const std::vector<Tensor> &tensors) {
  using RetType = void;
  DISPATCH_TO_CURRENT_RUNTIME(
      RetType, [&]() { ::tt::runtime::ttnn::wait(tensors); },
      [&]() { ::tt::runtime::ttmetal::wait(tensors); });
}

std::vector<Tensor> toHost(Tensor tensor, bool untilize) {
  using RetType = std::vector<Tensor>;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::toHost(tensor, untilize);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::toHost(tensor, untilize);
      });
}

Tensor toLayout(Tensor tensor, Device device, Layout layout,
                std::optional<bool> retain) {
  using RetType = Tensor;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::toLayout(tensor, device, layout, retain);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::toLayout(tensor, device, layout, retain);
      });
}

Layout getLayout(Binary executableHandle, std::uint32_t programIndex,
                 std::uint32_t inputIndex) {
  using RetType = Layout;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::getLayout(executableHandle, programIndex,
                                              inputIndex);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::getLayout(executableHandle, programIndex,
                                                 inputIndex);
      });
}

void memcpy(void *dst, Tensor src) {
  using RetType = void;
  DISPATCH_TO_CURRENT_RUNTIME(
      RetType, [&]() { ::tt::runtime::ttnn::memcpy(dst, src); },
      [&]() { ::tt::runtime::ttmetal::memcpy(dst, src); });
}

void memcpy(Tensor dst, Tensor src) {
  using RetType = void;
  DISPATCH_TO_CURRENT_RUNTIME(
      RetType, [&]() { ::tt::runtime::ttnn::memcpy(dst, src); },
      [&]() { ::tt::runtime::ttmetal::memcpy(dst, src); });
}

void deallocateTensor(Tensor &tensor, bool force) {
  using RetType = void;
  DISPATCH_TO_CURRENT_RUNTIME(
      RetType, [&]() { ::tt::runtime::ttnn::deallocateTensor(tensor, force); },
      [&]() { ::tt::runtime::ttmetal::deallocateTensor(tensor, force); });
}

std::string getOpDebugString(OpContext opContextHandle) {
  using RetType = std::string;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::getOpDebugString(opContextHandle);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::getOpDebugString(opContextHandle);
      });
}

std::string getOpLocInfo(OpContext opContextHandle) {
  using RetType = std::string;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::getOpLocInfo(opContextHandle);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::getOpLocInfo(opContextHandle);
      });
}

Tensor getOpOutputTensor(OpContext opContextHandle,
                         CallbackContext programContextHandle) {
  using RetType = Tensor;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::getOpOutputTensor(opContextHandle,
                                                      programContextHandle);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::getOpOutputTensor(opContextHandle,
                                                         programContextHandle);
      });
}

std::vector<Tensor> submit(Device deviceHandle, Binary executableHandle,
                           std::uint32_t programIndex,
                           std::vector<Tensor> &inputs) {
  using RetType = std::vector<Tensor>;
  return DISPATCH_TO_CURRENT_RUNTIME(
      RetType,
      [&]() -> RetType {
        return ::tt::runtime::ttnn::submit(deviceHandle, executableHandle,
                                           programIndex, inputs);
      },
      [&]() -> RetType {
        return ::tt::runtime::ttmetal::submit(deviceHandle, executableHandle,
                                              programIndex, inputs);
      });
}

#undef IF_TTNN_ENABLED
#undef IF_TTMETAL_ENABLED
#undef DISPATCH_TO_CURRENT_RUNTIME

} // namespace tt::runtime
