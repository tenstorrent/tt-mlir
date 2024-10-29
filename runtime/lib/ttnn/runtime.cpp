// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt/runtime/runtime.h"
#include "tt/runtime/detail/debug.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/utils.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Version.h"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/types.hpp"
#include <dlfcn.h>
#include <memory>
#include <variant>
#include <vector>
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
  return utils::fromTTNNDataType(nnTensor.get_dtype());
}

size_t getNumAvailableDevices() {
  return ::tt::tt_metal::GetNumAvailableDevices();
}

Device openDevice(DeviceIds const &deviceIds, size_t numHWCQs) {
  LOG_ASSERT(deviceIds.size(), "No devices specified");
  ::tt::tt_metal::distributed::MeshShape grid =
      std::make_pair(1, deviceIds.size());
  std::shared_ptr<::ttnn::MeshDevice> meshDevice = ::ttnn::MeshDevice::create(
      grid, kL1SmallSize, DEFAULT_TRACE_REGION_SIZE, numHWCQs,
      ::tt::tt_metal::DispatchCoreType::WORKER);

  bool enableAsync = debug::Env::get().enableAsyncTTNN;
  for (::ttnn::Device *device : meshDevice->get_devices()) {
    device->enable_async(enableAsync);
  }

  return Device(std::static_pointer_cast<void>(meshDevice),
                DeviceRuntime::TTNN);
}

void closeDevice(Device device) {
  ::ttnn::MeshDevice &ttnnMeshDevice =
      device.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
#if defined(TT_RUNTIME_ENABLE_PERF_TRACE)
  for (::ttnn::Device *ttnnDevice : ttnnMeshDevice.get_devices()) {
    ::tt::tt_metal::detail::DumpDeviceProfileResults(ttnnDevice);
  }
#endif

  ttnnMeshDevice.close_devices();
}

void deallocateBuffers(Device deviceHandle) {
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  for (::ttnn::Device *device : meshDevice.get_devices()) {
    device->deallocate_buffers();
  }
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
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  ::tt::target::ttnn::TTNNBinary const &fbb = *getBinary(executableHandle);
  std::vector<::ttnn::Tensor *> inputs;
  inputs.reserve(inputHandles.size());
  for (auto &input : inputHandles) {
    LOG_ASSERT(input.matchesRuntime(DeviceRuntime::TTNN));
    inputs.push_back(static_cast<::ttnn::Tensor *>(input.handle.get()));
  }
  std::vector<::ttnn::Tensor *> outputs;
  outputs.reserve(outputHandles.size());
  for (auto &output : outputHandles) {
    LOG_ASSERT(output.matchesRuntime(DeviceRuntime::TTNN));
    outputs.push_back(static_cast<::ttnn::Tensor *>(output.handle.get()));
  }
  tt::runtime::ttnn::runProgram(meshDevice, fbb.programs()->Get(programIndex),
                                inputs, outputs);
  return Event(nullptr, DeviceRuntime::TTNN);
}

bool compareOuts(std::vector<Tensor> &lhs, std::vector<Tensor> &rhs) {
  std::vector<::ttnn::Tensor *> lhsTensors;
  std::vector<::ttnn::Tensor *> rhsTensors;

  for (auto &tensor : lhs) {
    lhsTensors.push_back(static_cast<::ttnn::Tensor *>(tensor.handle.get()));
  }
  for (auto &tensor : rhs) {
    rhsTensors.push_back(static_cast<::ttnn::Tensor *>(tensor.handle.get()));
  }

  LOG_ASSERT(lhsTensors.size() == rhsTensors.size());
  for (size_t i = 0; i < lhsTensors.size(); i++) {
    auto lhsTensor = lhsTensors[i];
    auto rhsTensor = rhsTensors[i];
    std::cout << "Dtype: " << (int)lhsTensor->get_dtype() << ", "
              << (int)rhsTensor->get_dtype() << std::endl;
    LOG_ASSERT(lhsTensor->get_dtype() == rhsTensor->get_dtype());
    std::cout << "Shape: " << lhsTensor->get_shape() << ", "
              << rhsTensor->get_shape() << std::endl;
    LOG_ASSERT(lhsTensor->get_shape() == rhsTensor->get_shape());
    std::cout << "Layout: " << (int)lhsTensor->get_layout() << ", "
              << (int)rhsTensor->get_layout() << std::endl;
    LOG_ASSERT(lhsTensor->get_layout() == rhsTensor->get_layout());
    std::cout << "Logical shape: " << lhsTensor->get_logical_shape() << ", "
              << rhsTensor->get_logical_shape() << std::endl;
    LOG_ASSERT(lhsTensor->get_logical_shape() ==
               rhsTensor->get_logical_shape());

    // LOG_ASSERT(lhsTensor == rhsTensor);

    std::cout << "Printing LHS:" << std::endl;
    lhsTensor->print();
    std::cout << std::endl << std::endl;
    std::cout << "Printing RHS:" << std::endl;
    rhsTensor->print();
    std::cout << "Done" << std::endl << std::endl;
  }

  return true;
}

void wait(Event event) {
  // Not implemented
  LOG_ASSERT(event.matchesRuntime(DeviceRuntime::TTNN));
}

std::vector<Tensor> do_stuff(void *so, std::string func_name,
                             std::vector<Tensor> inputs) {
  // Convert inputs to TTNN tensors using .as method
  //
  std::vector<::ttnn::Tensor> ttnnInputs;
  for (auto &input : inputs) {
    LOG_ASSERT(input.matchesRuntime(DeviceRuntime::TTNN));
    ttnnInputs.push_back(input.as<::ttnn::Tensor>(DeviceRuntime::TTNN));
  }

  // Clear previous error
  //
  dlerror();

  // Get function from shared object
  //
  using ForwardFunction =
      std::vector<::ttnn::Tensor> (*)(std::vector<::ttnn::Tensor>);
  std::cout << "before" << std::endl;
  ForwardFunction forwardFunc = (ForwardFunction)dlsym(so, func_name.c_str());
  std::cout << "after" << std::endl;

  const char *dlsym_error = dlerror();
  if (dlsym_error) {
    std::cerr << "Failed to load symbol: " << dlsym_error << std::endl;
    dlclose(so);
    throw std::runtime_error("Failed to load symbol");
  }

  // Call function
  //
  std::vector<::ttnn::Tensor> ttnnOutputs = forwardFunc(ttnnInputs);

  // Convert outputs to Tensor using Tensor constructor
  //
  std::vector<Tensor> outputs;
  for (::ttnn::Tensor &output : ttnnOutputs) {
    // using Storage = std::variant<OwnedStorage, DeviceStorage,
    // BorrowedStorage, MultiDeviceHostStorage, MultiDeviceStorage>;
    if (std::holds_alternative<OwnedStorage>(
            output.tensor_attributes->storage)) {
      std::cout << "OwnedStorage" << std::endl;
    } else if (std::holds_alternative<DeviceStorage>(
                   output.tensor_attributes->storage)) {
      std::cout << "DeviceStorage" << std::endl;
    } else if (std::holds_alternative<BorrowedStorage>(
                   output.tensor_attributes->storage)) {
      std::cout << "BorrowedStorage" << std::endl;
    } else if (std::holds_alternative<MultiDeviceHostStorage>(
                   output.tensor_attributes->storage)) {
      std::cout << "MultiDeviceHostStorage" << std::endl;
    } else if (std::holds_alternative<MultiDeviceStorage>(
                   output.tensor_attributes->storage)) {
      std::cout << "MultiDeviceStorage" << std::endl;
    } else {
      std::cout << "Unknown" << std::endl;
    }

    // BorrowedBuffer borrowedBuffer =
    //     std::get<BorrowedStorage>(output.tensor_attributes->storage).buffer;
    // std::visit(
    //     [&outputs, &output](auto &&buffer) {
    //       outputs.push_back(
    //           Tensor(std::make_shared<::ttnn::Tensor>(std::move(output)),
    //                  std::shared_ptr<void>(static_cast<void
    //                  *>(buffer.data()),
    //                                        [](void *) {}),
    //                  DeviceRuntime::TTNN));
    //     },
    //     borrowedBuffer);

    OwnedStorage ownedStorage =
        std::get<OwnedStorage>(output.tensor_attributes->storage).buffer;

    std::visit(
        [&outputs, &output](auto &&buffer) {
          outputs.push_back(
              Tensor(std::make_shared<::ttnn::Tensor>(std::move(output)),
                     std::shared_ptr<void>(static_cast<void *>(buffer.data()),
                                           [](void *) {}),
                     DeviceRuntime::TTNN));
        },
        ownedStorage.get_buffer());
  }

  return outputs;
}

} // namespace tt::runtime::ttnn
