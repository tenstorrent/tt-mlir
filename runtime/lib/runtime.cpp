// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/runtime.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Version.h"

#if defined(TT_RUNTIME_ENABLE_TTNN)
#include "tt/runtime/detail/ttnn/ttnn.h"
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
#include "tt/runtime/detail/ttmetal/ttmetal.h"
#endif

namespace tt::runtime {

namespace detail {
// NOLINTBEGIN
#if defined(TT_RUNTIME_ENABLE_TTNN)
DeviceRuntime globalCurrentRuntime = DeviceRuntime::TTNN;
#elif defined(TT_RUNTIME_ENABLE_TTMETAL)
DeviceRuntime globalCurrentRuntime = DeviceRuntime::TTMetal;
#else
DeviceRuntime globalCurrentRuntime = DeviceRuntime::Disabled;
#endif
// NOLINTEND

void deallocateBuffers(Device device) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::deallocateBuffers(device);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::deallocateBuffers(device);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

void dumpMemoryReport(Device device) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::dumpMemoryReport(device);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::dumpMemoryReport(device);
  }
#endif

  LOG_FATAL("runtime is not enabled");
}

std::unordered_map<tt::runtime::MemoryBufferType, tt::runtime::MemoryView>
getMemoryView(Device device, int deviceID) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getMemoryView(device, deviceID);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::getMemoryView(device, deviceID);
  }
#endif

  LOG_FATAL("runtime is not enabled");
}
} // namespace detail

DeviceRuntime getCurrentRuntime() {
#if !defined(TT_RUNTIME_ENABLE_TTNN)
  LOG_ASSERT(detail::globalCurrentRuntime != DeviceRuntime::TTNN);
#endif
#if !defined(TT_RUNTIME_ENABLE_TTMETAL)
  LOG_ASSERT(detail::globalCurrentRuntime != DeviceRuntime::TTMetal);
#endif
  return detail::globalCurrentRuntime;
}

std::vector<DeviceRuntime> getAvailableRuntimes() {
  std::vector<DeviceRuntime> runtimes;
#if defined(TT_RUNTIME_ENABLE_TTNN)
  runtimes.push_back(DeviceRuntime::TTNN);
#endif
#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  runtimes.push_back(DeviceRuntime::TTMetal);
#endif
  return runtimes;
}

void setCurrentRuntime(const DeviceRuntime &runtime) {
#if !defined(TT_RUNTIME_ENABLE_TTNN)
  LOG_ASSERT(runtime != DeviceRuntime::TTNN);
#endif
#if !defined(TT_RUNTIME_ENABLE_TTMETAL)
  LOG_ASSERT(runtime != DeviceRuntime::TTMetal);
#endif
  detail::globalCurrentRuntime = runtime;
}

void setCompatibleRuntime(const Binary &binary) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (binary.getFileIdentifier() ==
      ::tt::target::ttnn::TTNNBinaryIdentifier()) {
    return setCurrentRuntime(DeviceRuntime::TTNN);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (binary.getFileIdentifier() ==
      ::tt::target::metal::TTMetalBinaryIdentifier()) {
    return setCurrentRuntime(DeviceRuntime::TTMetal);
  }
#endif
  LOG_FATAL("Unsupported binary file identifier or runtime not enabled");
}

std::pair<SystemDesc, DeviceIds>
getCurrentSystemDesc(std::optional<DispatchCoreType> dispatchCoreType,
                     std::optional<Device> meshDevice) {
#if defined(TT_RUNTIME_ENABLE_TTNN) || defined(TT_RUNTIME_ENABLE_TTMETAL)
  return system_desc::getCurrentSystemDesc(dispatchCoreType, meshDevice);
#endif
  LOG_FATAL("runtime is not enabled");
}

Tensor createBorrowedHostTensor(void *data,
                                const std::vector<std::uint32_t> &shape,
                                const std::vector<std::uint32_t> &stride,
                                std::uint32_t itemsize,
                                ::tt::target::DataType dataType) {
  LOG_ASSERT(!shape.empty());
  LOG_ASSERT(!stride.empty());
  LOG_ASSERT(itemsize > 0);
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::createBorrowedHostTensor(data, shape, stride,
                                                         itemsize, dataType);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("TT Metal runtime does not support creating borrowed host tensor "
              "from direct pointer to data");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

Tensor createOwnedHostTensor(const void *data,
                             const std::vector<std::uint32_t> &shape,
                             const std::vector<std::uint32_t> &stride,
                             std::uint32_t itemsize,
                             ::tt::target::DataType dataType) {
  LOG_ASSERT(!shape.empty());
  LOG_ASSERT(!stride.empty());
  LOG_ASSERT(itemsize > 0);
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::createOwnedHostTensor(data, shape, stride,
                                                      itemsize, dataType);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("TT Metal runtime does not support creating owned tensors");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

// TODO(mrakita): Should be deprecated but D2M path is using this, investigate
// if it can also use the new `createBorrowedHostTensor` function.
// https://github.com/tenstorrent/tt-mlir/issues/2757
Tensor createTensor(std::shared_ptr<void> data,
                    const std::vector<std::uint32_t> &shape,
                    const std::vector<std::uint32_t> &stride,
                    std::uint32_t itemsize, ::tt::target::DataType dataType) {
  LOG_ASSERT(!shape.empty());
  LOG_ASSERT(!stride.empty());
  LOG_ASSERT(itemsize > 0);
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::createBorrowedHostTensor(
        data.get(), shape, stride, itemsize, dataType);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::createTensor(data, shape, stride, itemsize,
                                                dataType);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

Tensor createMultiDeviceHostTensor(
    const std::vector<const void *> &data,
    const std::vector<std::uint32_t> &shape,
    const std::vector<std::uint32_t> &stride, std::uint32_t itemsize,
    ::tt::target::DataType dataType,
    const std::unordered_map<std::string, std::string> &strategy) {
  LOG_ASSERT(!shape.empty());
  LOG_ASSERT(!stride.empty());
  LOG_ASSERT(itemsize > 0);
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::createMultiDeviceHostTensor(
        data, shape, stride, itemsize, dataType, strategy);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("Not implemented");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

Tensor createMultiDeviceHostTensor(
    const std::vector<Tensor> &tensorShards,
    const std::unordered_map<std::string, std::string> &strategy) {
  LOG_ASSERT(!tensorShards.empty());
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::createMultiDeviceHostTensor(tensorShards,
                                                            strategy);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("Not implemented");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

Tensor createEmptyTensor(Device device, Layout layout,
                         const std::vector<std::uint32_t> &shape,
                         const std::vector<std::uint32_t> &stride,
                         std::uint32_t itemsize) {
  LOG_ASSERT(!shape.empty());
  LOG_ASSERT(!stride.empty());
  LOG_ASSERT(itemsize > 0);
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::createEmptyTensor(device, layout, shape, stride,
                                                  itemsize);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("Not implemented");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

bool isTensorAllocated(Tensor tensor) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::isTensorAllocated(tensor);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::isTensorAllocated(tensor);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

tt::target::DataType getTensorDataType(Tensor tensor) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getTensorDataType(tensor);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::getTensorDataType(tensor);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

std::vector<std::byte> getTensorDataBuffer(Tensor t) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getTensorDataBuffer(t);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::getTensorDataBuffer(t);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

std::vector<std::uint32_t> getTensorShape(Tensor t) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getTensorShape(t);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::getTensorShape(t);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

std::vector<std::uint32_t> getTensorStride(Tensor t) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getTensorStride(t);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::getTensorStride(t);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

target::DataType getDtype(Tensor t) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getTensorDataType(t);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::getTensorDataType(t);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

std::uint32_t getTensorElementSize(Tensor t) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getTensorElementSize(t);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::getTensorElementSize(t);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

std::uint32_t getTensorVolume(Tensor t) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getTensorVolume(t);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::getTensorVolume(t);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

TensorDesc getTensorDesc(Tensor t) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getTensorDesc(t);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::getTensorDesc(t);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

bool getTensorRetain(Tensor tensor) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getTensorRetain(tensor);
  }
#endif
#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::getTensorRetain(tensor);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

void setTensorRetain(Tensor tensor, bool retain) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::setTensorRetain(tensor, retain);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::setTensorRetain(tensor, retain);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

size_t getNumAvailableDevices() {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getNumAvailableDevices();
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::getNumAvailableDevices();
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

Device openMeshDevice(const std::vector<uint32_t> &meshShape,
                      const MeshDeviceOptions &options) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::openMeshDevice(meshShape, options);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::openMeshDevice(meshShape, options);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

void closeMeshDevice(Device parentMesh) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::closeMeshDevice(parentMesh);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::closeMeshDevice(parentMesh);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

Device createSubMeshDevice(
    Device parentMesh, const std::vector<uint32_t> &meshShape,
    const std::optional<const std::vector<uint32_t>> &meshOffset) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::createSubMeshDevice(parentMesh, meshShape,
                                                    meshOffset);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::createSubMeshDevice(parentMesh, meshShape,
                                                       meshOffset);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

void releaseSubMeshDevice(Device subMesh) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::releaseSubMeshDevice(subMesh);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::releaseSubMeshDevice(subMesh);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

void reshapeMeshDevice(Device meshDevice,
                       const std::vector<uint32_t> &meshShape) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::reshapeMeshDevice(meshDevice, meshShape);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::reshapeMeshDevice(meshDevice, meshShape);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

void wait(Event event) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    LOG_WARNING("wait API will be deprecated for TTNN runtime.");
    return ::tt::runtime::ttnn::wait(event);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::wait(event);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

void wait(Tensor tensor) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::wait(tensor);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::wait(tensor);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

void wait(const std::vector<Tensor> &tensors) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::wait(tensors);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::wait(tensors);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

std::vector<Tensor> toHost(Tensor tensor, bool untilize) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::toHost(tensor, untilize);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("not implemented");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

Tensor toLayout(Tensor tensor, Device device, Layout layout,
                std::optional<bool> retain) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::toLayout(tensor, device, layout, retain);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("not implemented");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

Layout getLayout(Binary executableHandle, std::uint32_t programIndex,
                 std::uint32_t inputIndex) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getLayout(executableHandle, programIndex,
                                          inputIndex);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("not implemented");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

void memcpy(void *dst, Tensor src) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::memcpy(dst, src);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("not implemented");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

void memcpy(Tensor dst, Tensor src) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::memcpy(dst, src);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("not implemented");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

void deallocateTensor(Tensor &tensor, bool force) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::deallocateTensor(tensor, force);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("not implemented");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

std::string getOpDebugString(OpContext opContextHandle) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getOpDebugString(opContextHandle);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::getOpDebugString(opContextHandle);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

std::string getOpLocInfo(OpContext opContextHandle) {
#ifdef TT_RUNTIME_ENABLE_TTNN
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getOpLocInfo(opContextHandle);
  }
#endif

#ifdef TT_RUNTIME_ENABLE_TTMETAL
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::getOpLocInfo(opContextHandle);
  }
#endif
  throw std::runtime_error("runtime is not enabled");
}

Tensor getOpOutputTensor(OpContext opContextHandle,
                         CallbackContext programContextHandle) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getOpOutputTensor(opContextHandle,
                                                  programContextHandle);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::getOpOutputTensor(opContextHandle,
                                                     programContextHandle);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

std::vector<Tensor> submit(Device deviceHandle, Binary executableHandle,
                           std::uint32_t programIndex,
                           std::vector<Tensor> &inputs) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::submit(deviceHandle, executableHandle,
                                       programIndex, inputs);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("not implemented");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

Event submit(Device deviceHandle, Binary executableHandle,
             std::uint32_t programIndex,
             const std::vector<Tensor> &inputHandles,
             const std::vector<Tensor> &outputHandles) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    LOG_FATAL("This submit API is deprecated for TTNN. Please switch to the "
              "new API.");
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::submit(deviceHandle, executableHandle,
                                          programIndex, inputHandles,
                                          outputHandles);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

} // namespace tt::runtime
