// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/test/ttnn/dylib.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"

#include <dlfcn.h>

namespace tt::runtime::test::ttnn {

static constexpr const char *POTENTIAL_MANGLING_ADDITIONS[] = {
    "",
    "PNS1_11distributed10MeshDeviceE",
};

// If the model function name is `main`, it can't be emitted as such, because
// compiler will complain about the parameter and return type, as `main` has a
// special semantics in C/C++. Hence, in `TTNNModifySignaturesForDylib` pass,
// the `main` is prepended with a `_`.
static std::string getEmittedFuncName(std::string_view funcName) {
  if (funcName == "main") {
    return "_main";
  }
  return std::string(funcName);
}

static std::string getMangledName(std::string_view funcName,
                                  std::string_view suffix) {
  std::string mangledName = "_Z";
  // The function name and the name with which it is emitted can be different
  // (see the explanation in the `getEmittedFuncName`).
  std::string emittedFuncName = getEmittedFuncName(funcName);
  mangledName += std::to_string(emittedFuncName.size());
  mangledName += emittedFuncName;
  mangledName += "St6vectorIN2tt8tt_metal6TensorESaIS2_EE";
  mangledName += suffix;
  return mangledName;
}

void *openSo(const std::string &path) {
  LOG_ASSERT(getCurrentRuntime() == DeviceRuntime::TTNN);

  void *handle = dlopen(path.data(), RTLD_LAZY);
  if (!handle) {
    std::cerr << "Failed to load shared object: " << dlerror() << std::endl;
    throw std::runtime_error("Failed to load shared object");
  }

  dlerror();
  return handle;
}

void closeSo(void *handle) {
  int ret = dlclose(handle);

  if (ret != 0) {
    std::cerr << "Failed to close shared object: " << dlerror() << std::endl;
    exit(ret);
  }
}

std::vector<::tt::runtime::Tensor>
runSoProgram(void *so, const std::string &funcName,
             std::vector<::tt::runtime::Tensor> inputs, Device device) {
  LOG_ASSERT(getCurrentRuntime() == DeviceRuntime::TTNN);

  ::ttnn::MeshDevice &ttnnMeshDevice =
      device.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);

  // In this path, we only ever test with a single device (for now) in CI, but
  // locally we may have 2 devices.
  assert(ttnnMeshDevice.get_devices().size() > 0);

  // Clear any previous errors.
  //
  dlerror();

  // Call setDevice function from dylib.
  //
  void *setDeviceSymbol = dlsym(so, "setDevice");
  const char *setDeviceError = dlerror();
  if (setDeviceError) {
    dlclose(so);
    LOG_FATAL("Failed to find setDevice function in dylib.");
  }
  using SetDeviceFunction = void (*)(::ttnn::MeshDevice *);
  auto setDeviceFunc = reinterpret_cast<SetDeviceFunction>(setDeviceSymbol);
  setDeviceFunc(&ttnnMeshDevice);

  // Convert inputs to TTNN tensors using .as method
  //
  std::vector<::ttnn::Tensor> ttnnInputs;
  for (auto &input : inputs) {
    LOG_ASSERT(input.matchesRuntime(DeviceRuntime::TTNN));
    ttnnInputs.push_back(
        input.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
            .getTensor());
  }

  // Get function from the shared object.
  //
  using ForwardFunction =
      std::vector<::ttnn::Tensor> (*)(std::vector<::ttnn::Tensor>);

  const char *dlsymError;
  void *symbol;
  std::string mangledName;
  for (const char *addition : POTENTIAL_MANGLING_ADDITIONS) {
    mangledName = getMangledName(funcName, addition);
    symbol = dlsym(so, mangledName.c_str());
    dlsymError = dlerror();
    if (!dlsymError) {
      break;
    }
  }
  if (dlsymError) {
    dlclose(so);
    LOG_FATAL("Failed to load symbol: ", dlsymError);
  }

  // Call program/function.
  //
  std::vector<::ttnn::Tensor> ttnnOutputs;
  auto forwardFunc = reinterpret_cast<ForwardFunction>(symbol);
  ttnnOutputs = forwardFunc(ttnnInputs);

  // Convert TTNN Tensors to Runtime Tensors.
  //
  std::vector<::tt::runtime::Tensor> outputs;
  for (::ttnn::Tensor &output : ttnnOutputs) {
    outputs.push_back(
        ::tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(output));
  }

  return outputs;
}

using SupportedTypes =
    std::variant<uint8_t, uint16_t, int32_t, uint32_t, float, bfloat16>;

static SupportedTypes getValueForDType(::ttnn::DataType dtype, void *data) {
  switch (dtype) {
  case ::ttnn::DataType::UINT8:
    return *static_cast<uint8_t *>(data);
  case ::ttnn::DataType::UINT16:
    return *static_cast<uint16_t *>(data);
  case ::ttnn::DataType::INT32:
    return *static_cast<int32_t *>(data);
  case ::ttnn::DataType::UINT32:
    return *static_cast<uint32_t *>(data);
  case ::ttnn::DataType::FLOAT32:
    return *static_cast<float *>(data);
  case ::ttnn::DataType::BFLOAT16:
    return *static_cast<bfloat16 *>(data);
  // Defaults to uint8_t, i.e. raw data.
  default:
    return *static_cast<uint8_t *>(data);
  }
}

static std::string toString(SupportedTypes &v) {
  return std::visit(
      ::tt::runtime::utils::overloaded{
          [](uint8_t v) { return std::to_string(static_cast<int>(v)); },
          [](uint16_t v) { return std::to_string(static_cast<int>(v)); },
          [](bfloat16 v) { return std::to_string(v.to_float()); },
          [](auto &&v) { return std::to_string(v); },
      },
      v);
}

// Compare two values of the same type for equality. We compare byte by byte, so
// that even special values are compared correctly.
//
static bool areEqual(const SupportedTypes &lhs, const SupportedTypes &rhs) {
  size_t size = std::visit(
      ::tt::runtime::utils::overloaded{
          [](auto &&val) { return sizeof(val); },
      },
      lhs);
  for (size_t i = 0; i < size; i++) {
    if (reinterpret_cast<const uint8_t *>(&lhs)[i] !=
        reinterpret_cast<const uint8_t *>(&rhs)[i]) {
      return false;
    }
  }
  return true;
}

static bool operator==(const SupportedTypes &lhs, const SupportedTypes &rhs) {
  return areEqual(lhs, rhs);
}

static bool operator!=(const SupportedTypes &lhs, const SupportedTypes &rhs) {
  return !(lhs == rhs);
}

using IndexTy = std::vector<size_t>;

IndexTy getIndex(const ::ttnn::Shape &shape, size_t idx) {
  IndexTy result(shape.size());

  size_t remaining = idx;
  size_t stride = 1;

  assert(shape.size() > 0 && "Shape must have at least one dimension");
  for (int32_t i = shape.size() - 1; i >= 0; i--) {
    result[i] = (remaining / stride) % shape[i];
    stride *= shape[i];
  }

  return result;
}

static std::string toString(const IndexTy &v) {
  std::string result = "[";
  for (size_t i = 0; i < v.size(); i++) {
    result += std::to_string(v[i]);
    if (i != v.size() - 1) {
      result += ", ";
    }
  }
  result += "]";
  return result;
}

bool compareOuts(std::vector<::tt::runtime::Tensor> &lhs,
                 std::vector<::tt::runtime::Tensor> &rhs) {
  LOG_ASSERT(getCurrentRuntime() == DeviceRuntime::TTNN);

  std::vector<::ttnn::Tensor *> lhsTensors;
  std::vector<::ttnn::Tensor *> rhsTensors;

  for (auto &tensor : lhs) {
    lhsTensors.push_back(
        &(tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(getCurrentRuntime())
              .getTensor()));
  }
  for (auto &tensor : rhs) {
    rhsTensors.push_back(
        &(tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(getCurrentRuntime())
              .getTensor()));
  }
  LOG_ASSERT(lhsTensors.size() == rhsTensors.size());

  for (size_t i = 0; i < lhsTensors.size(); i++) {
    auto *lhsTensor = lhsTensors[i];
    auto *rhsTensor = rhsTensors[i];

    // Compare various tensor properties
    //
    LOG_ASSERT(lhsTensor->dtype() == rhsTensor->dtype(),
               "DType: ", lhsTensor->dtype(), ", ", rhsTensor->dtype());
    LOG_ASSERT(lhsTensor->layout() == rhsTensor->layout(),
               "Layout: ", static_cast<int>(lhsTensor->layout()), ", ",
               static_cast<int>(rhsTensor->layout()));
    LOG_ASSERT(lhsTensor->logical_shape() == rhsTensor->logical_shape(),
               "Logical shape: ", lhsTensor->logical_shape(), ", ",
               rhsTensor->logical_shape());

    // Compare tensor data
    //
    size_t elementSize = lhsTensor->element_size();
    uint8_t *lhsData = static_cast<uint8_t *>(
        ::tt::runtime::ttnn::utils::getRawHostDataPtr(*lhsTensor));
    uint8_t *rhsData = static_cast<uint8_t *>(
        ::tt::runtime::ttnn::utils::getRawHostDataPtr(*rhsTensor));
    for (size_t i = 0; i < lhsTensor->padded_volume(); ++i) {
      SupportedTypes lhsVal =
          getValueForDType(lhsTensor->dtype(), lhsData + i * elementSize);
      SupportedTypes rhsVal =
          getValueForDType(rhsTensor->dtype(), rhsData + i * elementSize);
      if (lhsVal != rhsVal) {
        LOG_FATAL("Mismatch at index ",
                  toString(getIndex(lhsTensor->logical_shape(), i)), ": ",
                  toString(lhsVal), " != ", toString(rhsVal));
        return false;
      }
    }
  }

  return true;
}
} // namespace tt::runtime::test::ttnn
