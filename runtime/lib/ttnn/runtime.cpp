// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt/runtime/runtime.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Version.h"
#include "utils.h"

namespace tt::runtime::ttnn {

using ::tt::runtime::DeviceRuntime;

template <typename T>
static BorrowedStorage createStorage(void *ptr, std::uint32_t numElements) {
  return BorrowedStorage(
      borrowed_buffer::Buffer<T>(static_cast<T *>(ptr), numElements), [] {},
      [] {});
}

static BorrowedStorage createStorage(void *ptr, std::uint32_t numElements,
                                     ::tt::target::DataType dataType) {
  switch (dataType) {
  case ::tt::target::DataType::Float32:
    return createStorage<float>(ptr, numElements);
  // case ::tt::target::DataType::Float16:
  //   return createStorage<float16>(ptr, numElements);
  case ::tt::target::DataType::BFloat16:
    return createStorage<bfloat16>(ptr, numElements);
  case ::tt::target::DataType::UInt32:
    return createStorage<std::uint32_t>(ptr, numElements);
  case ::tt::target::DataType::UInt16:
    return createStorage<std::uint16_t>(ptr, numElements);
  // case ::tt::target::DataType::UInt8:
  //   return createStorage<std::uint8_t>(ptr, numElements);
  default:
    throw std::runtime_error("Unsupported data type");
  }
}

Tensor createTensor(std::shared_ptr<void> data,
                    std::vector<std::uint32_t> const &shape,
                    std::vector<std::uint32_t> const &stride,
                    std::uint32_t itemsize, ::tt::target::DataType dataType) {
  std::uint32_t numElements = shape[0] * stride[0];
  auto tensor = std::make_shared<::ttnn::Tensor>(
      createStorage(data.get(), numElements, dataType), shape,
      utils::toTTNNDataType(dataType), ::ttnn::Layout::ROW_MAJOR);
  return Tensor(tensor, data, DeviceRuntime::TTNN);
}

tt::target::DataType getTensorDataType(Tensor tensor) {
  const ::ttnn::Tensor &nnTensor =
      tensor.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
  return utils::fromTTNNDataType(nnTensor.dtype());
}

void deallocateTensor(Tensor tensor, bool force) {
  ::ttnn::Tensor &ttnnTensor = tensor.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
  ttnnTensor.deallocate(force);
}

Tensor toCpu(Tensor tensor) {
  ::ttnn::Tensor &ttnnTensor = tensor.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
  std::shared_ptr<::ttnn::Tensor> cpuTensor =
      std::make_shared<::ttnn::Tensor>(ttnnTensor.cpu());
  void *dataPtr = ::tt::tt_metal::get_raw_host_data_ptr(*cpuTensor);
  return Tensor(cpuTensor, ::tt::runtime::utils::unsafe_borrow_shared(dataPtr),
                DeviceRuntime::TTNN);
}

Device openDevice(std::vector<int> const &deviceIds,
                  std::vector<std::uint8_t> const &numHWCQs) {
  assert(deviceIds.size() == 1 && "Only one device is supported for now");
  assert(numHWCQs.empty() && "HWCQs are not supported for now");
  auto &device = ::ttnn::open_device(deviceIds.front(), kL1SmallSize);
  return Device::borrow(device, DeviceRuntime::TTNN);
}

void closeDevice(Device device) {
  auto &ttnn_device = device.as<::ttnn::Device>(DeviceRuntime::TTNN);
  ::ttnn::close_device(ttnn_device);
}

void deallocateBuffers(Device deviceHandle) {
  deviceHandle.as<::ttnn::Device>(DeviceRuntime::TTNN).deallocate_buffers();
}

static ::tt::target::ttnn::TTNNBinary const *getBinary(Flatbuffer binary) {
  bool isTTNN = ::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
      binary.handle.get());
  if (not isTTNN) {
    throw std::runtime_error("Unsupported binary format");
  }
  return ::tt::target::ttnn::GetSizePrefixedTTNNBinary(binary.handle.get());
}

Event submit(Device deviceHandle, Binary executableHandle,
             std::uint32_t programIndex,
             std::vector<Tensor> const &inputHandles,
             std::vector<Tensor> const &outputHandles) {
  ::ttnn::Device &device = deviceHandle.as<::ttnn::Device>(DeviceRuntime::TTNN);
  ::tt::target::ttnn::TTNNBinary const &fbb = *getBinary(executableHandle);
  std::vector<::ttnn::Tensor *> inputs;
  inputs.reserve(inputHandles.size());
  for (auto &input : inputHandles) {
    assert(input.matchesRuntime(DeviceRuntime::TTNN));
    inputs.push_back(static_cast<::ttnn::Tensor *>(input.handle.get()));
  }
  std::vector<::ttnn::Tensor *> outputs;
  outputs.reserve(outputHandles.size());
  for (auto &output : outputHandles) {
    assert(output.matchesRuntime(DeviceRuntime::TTNN));
    outputs.push_back(static_cast<::ttnn::Tensor *>(output.handle.get()));
  }
  tt::runtime::ttnn::runProgram(device, fbb.programs()->Get(programIndex),
                                inputs, outputs);
  return Event(nullptr, DeviceRuntime::TTNN);
}

std::vector<Tensor> submit(Device deviceHandle, Binary executableHandle,
                           std::uint32_t programIndex,
                           std::vector<Tensor> const &inputHandles) {
  ::ttnn::Device &device = deviceHandle.as<::ttnn::Device>(DeviceRuntime::TTNN);
  ::tt::target::ttnn::TTNNBinary const &fbb = *getBinary(executableHandle);
  std::vector<::ttnn::Tensor *> inputs;
  inputs.reserve(inputHandles.size());
  for (auto &input : inputHandles) {
    assert(input.matchesRuntime(DeviceRuntime::TTNN));
    inputs.push_back(static_cast<::ttnn::Tensor *>(input.handle.get()));
  }
  std::vector<Tensor> outputs = ::tt::runtime::ttnn::runProgram(
      device, fbb.programs()->Get(programIndex), inputs);
  return outputs;
}

Tensor toLayout(Device device, Binary executable, std::uint32_t programIndex,
                std::uint32_t inputIndex, Tensor const &input) {

  const ::tt::target::ttnn::TTNNBinary *fbb = getBinary(executable);

  TT_FATAL(programIndex < fbb->programs()->size(),
           "Program index {} out of range {}", programIndex,
           fbb->programs()->size());
  const ::tt::target::ttnn::Program *program =
      fbb->programs()->Get(programIndex);

  return ::tt::runtime::ttnn::updateProgramTensorLayout(device, program,
                                                        inputIndex, input);
}

void wait(Event event) {
  // Not implemented
  assert(event.matchesRuntime(DeviceRuntime::TTNN));
}

} // namespace tt::runtime::ttnn
