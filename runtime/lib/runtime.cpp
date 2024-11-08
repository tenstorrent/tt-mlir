// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/runtime.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Version.h"

#if defined(TT_RUNTIME_ENABLE_TTNN)
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/types.h"
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
#include "tt/runtime/detail/ttmetal.h"
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
  throw std::runtime_error("runtime is not enabled");
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
  throw std::runtime_error(
      "Unsupported binary file identifier or runtime not enabled");
}

std::pair<SystemDesc, DeviceIds> getCurrentSystemDesc() {
#if defined(TT_RUNTIME_ENABLE_TTNN) || defined(TT_RUNTIME_ENABLE_TTMETAL)
  return system_desc::getCurrentSystemDesc();
#endif
  throw std::runtime_error("runtime is not enabled");
}

Tensor createTensor(std::shared_ptr<void> data,
                    std::vector<std::uint32_t> const &shape,
                    std::vector<std::uint32_t> const &stride,
                    std::uint32_t itemsize, ::tt::target::DataType dataType) {
  LOG_ASSERT(not shape.empty());
  LOG_ASSERT(not stride.empty());
  LOG_ASSERT(itemsize > 0);
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::createTensor(data, shape, stride, itemsize,
                                             dataType);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::createTensor(data, shape, stride, itemsize,
                                                dataType);
  }
#endif
  throw std::runtime_error("runtime is not enabled");
}

Tensor
createTensor(std::vector<std::shared_ptr<void>> &data,
             std::vector<std::uint32_t> const &shape,
             std::vector<std::uint32_t> const &stride, std::uint32_t itemsize,
             ::tt::target::DataType dataType,
             std::unordered_map<std::string, std::string> const &strategy) {
  LOG_ASSERT(not shape.empty());
  LOG_ASSERT(not stride.empty());
  LOG_ASSERT(itemsize > 0);
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::createTensor(data, shape, stride, itemsize,
                                             dataType, strategy);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    throw std::runtime_error("Not implemented");
  }
#endif
  throw std::runtime_error("runtime is not enabled");
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
  throw std::runtime_error("runtime is not enabled");
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
  throw std::runtime_error("runtime is not enabled");
}

Device openDevice(DeviceIds const &deviceIds, size_t numHWCQs) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::openDevice(deviceIds, numHWCQs);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::openDevice(deviceIds, numHWCQs);
  }
#endif
  throw std::runtime_error("runtime is not enabled");
}

void closeDevice(Device device) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::closeDevice(device);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::closeDevice(device);
  }
#endif
  throw std::runtime_error("runtime is not enabled");
}

Event submit(Device deviceHandle, Binary executableHandle,
             std::uint32_t programIndex,
             std::vector<Tensor> const &inputHandles,
             std::vector<Tensor> const &outputHandles) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::submit(deviceHandle, executableHandle,
                                       programIndex, inputHandles,
                                       outputHandles);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::submit(deviceHandle, executableHandle,
                                          programIndex, inputHandles,
                                          outputHandles);
  }
#endif
  throw std::runtime_error("runtime is not enabled");
}

void wait(Event event) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::wait(event);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::wait(event);
  }
#endif
  throw std::runtime_error("runtime is not enabled");
}

#if defined(TT_RUNTIME_ENABLE_TTNN)
std::vector<float> getOpOutputTensor(const void *context,
                                     const void *opContext) {
  auto *contextPtr = static_cast<const ttnn::ProgramContext *>(context);
  auto *opContextPtr =
      static_cast<const ::tt::target::ttnn::Operation *>(opContext);
  const ::ttnn::Tensor *outPtr = nullptr;
  const ttnn::ProgramTensorPool &tensorPool = contextPtr->getTensorPool();
  std::uint32_t globalId;

  switch (opContextPtr->type_type()) {
  case ::tt::target::ttnn::OpType::GetDeviceOp: {
    globalId = opContextPtr->type_as_GetDeviceOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::ToMemoryConfigOp: {
    globalId = opContextPtr->type_as_ToMemoryConfigOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::ToLayoutOp: {
    globalId = opContextPtr->type_as_ToLayoutOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::TypecastOp: {
    globalId = opContextPtr->type_as_TypecastOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::ToDeviceOp: {
    globalId = opContextPtr->type_as_ToDeviceOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::FromDeviceOp: {
    globalId = opContextPtr->type_as_FromDeviceOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::EmptyOp: {
    globalId = opContextPtr->type_as_EmptyOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::FullOp: {
    globalId = opContextPtr->type_as_FullOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseOp: {
    globalId = opContextPtr->type_as_EltwiseOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::MatmulOp: {
    globalId = opContextPtr->type_as_MatmulOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionOp: {
    globalId = opContextPtr->type_as_ReductionOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::EmbeddingOp: {
    globalId = opContextPtr->type_as_EmbeddingOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::SoftmaxOp: {
    globalId = opContextPtr->type_as_SoftmaxOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::TransposeOp: {
    globalId = opContextPtr->type_as_TransposeOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::ConcatOp: {
    globalId = opContextPtr->type_as_ConcatOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::ReshapeOp: {
    globalId = opContextPtr->type_as_ReshapeOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::SliceOp: {
    globalId = opContextPtr->type_as_SliceOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::Conv2dOp: {
    globalId = opContextPtr->type_as_Conv2dOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::DeallocOp: {
    LOG_WARNING("getting output tensor for DeallocOp is not supported");
    return {};
  }
  case ::tt::target::ttnn::OpType::MaxPool2dOp: {
    globalId = opContextPtr->type_as_MaxPool2dOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::AllGatherOp: {
    globalId = opContextPtr->type_as_AllGatherOp()->out()->global_id();
    break;
  }
  default: {
    throw std::runtime_error("Unsupported operation type");
  }
  }

  if (tensorPool.contains(globalId)) {
    outPtr = &tensorPool.at(globalId);
  } else {
    LOG_WARNING("Output tensor not found in tensor pool");
    return {};
  }
  ::ttnn::Tensor hostTensor = ::ttnn::from_device(*outPtr);
  ::ttnn::Tensor outCopy =
      ::ttnn::to_layout(hostTensor, ::ttnn::ROW_MAJOR_LAYOUT, std::nullopt,
                        std::nullopt, static_cast<::ttnn::Device *>(nullptr));
  std::uint32_t outCopySize = outCopy.volume() * outCopy.element_size();
  void *src = ::tt::tt_metal::get_raw_host_data_ptr(outCopy);
  void *dst = malloc(outCopySize);
  std::memcpy(dst, src, outCopySize);
  std::vector<float> outVec(static_cast<float *>(dst),
                            static_cast<float *>(dst) + outCopy.volume());

  return outVec;
}
#endif

#if defined(TT_RUNTIME_ENABLE_TTNN)
std::string getOpDebugString(const void *context, const void *opContext) {
  auto *opContextPtr =
      static_cast<const ::tt::target::ttnn::Operation *>(opContext);
  return std::string(opContextPtr->debug_info()->c_str());
}
#endif

} // namespace tt::runtime
