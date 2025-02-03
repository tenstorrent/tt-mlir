// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/test/dylib.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/test/utils.h"
#include "tt/runtime/ttnn/types.h"
#include "tt/runtime/ttnn/utils.h"
#include "tt/runtime/types.h"

#include <dlfcn.h>

namespace tt::runtime::ttnn::test {

void *openSo(std::string path) {
  LOG_ASSERT(getCurrentRuntime() == DeviceRuntime::TTNN);

  void *handle = dlopen(path.c_str(), RTLD_LAZY);
  if (!handle) {
    std::cerr << "Failed to load shared object: " << dlerror() << std::endl;
    throw std::runtime_error("Failed to load shared object");
  }

  dlerror();
  return handle;
}

std::vector<Tensor> runSoProgram(void *so, std::string func_name,
                                 std::vector<Tensor> inputs, Device device) {
  LOG_ASSERT(getCurrentRuntime() == DeviceRuntime::TTNN);

  ::ttnn::MeshDevice &ttnnMeshDevice =
      device.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);

  // In this path, we only ever test with a single device (for now), hence
  // MeshDevice is unpacked to get the Device
  //
  assert(ttnnMeshDevice.get_devices().size() == 1);
  ::ttnn::IDevice *ttnnDevice = ttnnMeshDevice.get_devices()[0];

  // Convert inputs to TTNN tensors using .as method
  //
  std::vector<::ttnn::Tensor> ttnnInputs;
  for (auto &input : inputs) {
    LOG_ASSERT(input.matchesRuntime(DeviceRuntime::TTNN));
    ttnnInputs.push_back(input.as<::ttnn::Tensor>(DeviceRuntime::TTNN));
  }

  // Clear previous errors
  //
  dlerror();

  // Get function from the shared object
  //
  using ForwardFunction = std::vector<::ttnn::Tensor> (*)(
      std::vector<::ttnn::Tensor>, ::ttnn::IDevice *);
  ForwardFunction forwardFunc =
      reinterpret_cast<ForwardFunction>(dlsym(so, func_name.c_str()));

  // Check for errors
  //
  const char *dlsym_error = dlerror();
  if (dlsym_error) {
    dlclose(so);
    LOG_FATAL("Failed to load symbol: ", dlsym_error);
  }

  // Call program/function
  //
  std::vector<::ttnn::Tensor> ttnnOutputs = forwardFunc(ttnnInputs, ttnnDevice);

  // Convert TTNN Tensors to Runtime Tensors
  //
  std::vector<Tensor> outputs;
  for (::ttnn::Tensor &output : ttnnOutputs) {
    outputs.push_back(utils::createRuntimeTensorFromTTNN(output));
  }

  return outputs;
}

bool compareOuts(std::vector<Tensor> &lhs, std::vector<Tensor> &rhs) {
  LOG_ASSERT(getCurrentRuntime() == DeviceRuntime::TTNN);

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
    auto *lhsTensor = lhsTensors[i];
    auto *rhsTensor = rhsTensors[i];

    // Compare various tensor properties
    //
    LOG_ASSERT(lhsTensor->get_dtype() == rhsTensor->get_dtype(),
               "DType: ", static_cast<int>(lhsTensor->get_dtype()), ", ",
               static_cast<int>(rhsTensor->get_dtype()));
    LOG_ASSERT(lhsTensor->get_logical_shape() == rhsTensor->get_logical_shape(),
               "Shape: ", lhsTensor->get_logical_shape(), ", ",
               rhsTensor->get_logical_shape());
    LOG_ASSERT(lhsTensor->get_layout() == rhsTensor->get_layout(),
               "Layout: ", static_cast<int>(lhsTensor->get_layout()), ", ",
               static_cast<int>(rhsTensor->get_layout()));
    LOG_ASSERT(lhsTensor->get_logical_shape() == rhsTensor->get_logical_shape(),
               "Logical shape: ", lhsTensor->get_logical_shape(), ", ",
               rhsTensor->get_logical_shape());
    LOG_ASSERT(lhsTensor->volume() == rhsTensor->volume(),
               "Volume: ", lhsTensor->volume(), ", ", rhsTensor->volume());
    LOG_ASSERT(lhsTensor->element_size() == rhsTensor->element_size(),
               "Element size in bytes: ", lhsTensor->element_size(), ", ",
               rhsTensor->element_size());

    // Compare tensor data
    //
    uint8_t *lhsData = static_cast<uint8_t *>(
        ::tt::tt_metal::get_raw_host_data_ptr(*lhsTensor));
    uint8_t *rhsData = static_cast<uint8_t *>(
        ::tt::tt_metal::get_raw_host_data_ptr(*rhsTensor));
    for (size_t i = 0; i < lhsTensor->volume() * lhsTensor->element_size();
         i++) {
      if (lhsData[i] != rhsData[i]) {
        LOG_FATAL("Mismatch at byte number: ", i, ": ",
                  static_cast<int>(lhsData[i]),
                  " != ", static_cast<int>(rhsData[i]));
        return false;
      }
    }
  }

  return true;
}
} // namespace tt::runtime::ttnn::test
