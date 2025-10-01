// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/common/socket.h"
#include "tt/runtime/detail/distributed/controller/command_factory.h"
#include "tt/runtime/detail/distributed/flatbuffer/flatbuffer.h"
#include "tt/runtime/detail/distributed/utils/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"

#include <gtest/gtest.h>

#include <chrono>
#include <future>

using tt::runtime::distributed::controller::CommandFactory;

static auto runWorkerSubprocess(uint16_t port) {
  std::string command =
      tt::runtime::distributed::utils::getWorkerExecutableCommand(port);
  return std::async(std::launch::async,
                    [command]() { return std::system(command.c_str()); });
}

static ::tt::runtime::SizedBuffer readResponseAndValidate(
    std::unique_ptr<tt::runtime::Socket> &workerSocket,
    tt::runtime::distributed::flatbuffer::ResponseType responseType) {

  tt::runtime::SizedBuffer responseBuffer = workerSocket->sizePrefixedRead();

  const tt::runtime::distributed::flatbuffer::Response *response =
      tt::runtime::distributed::flatbuffer::GetResponse(responseBuffer.data());

  EXPECT_EQ(response->type_type(), responseType);

  return responseBuffer;
}

static void sendShutdownCommandAndValidate(
    std::unique_ptr<tt::runtime::Socket> &workerSocket) {
  flatbuffers::FlatBufferBuilder fbb;

  CommandFactory::buildShutdownCommand(fbb);
  workerSocket->sizePrefixedWrite(fbb.GetBufferPointer(), fbb.GetSize());

  tt::runtime::SizedBuffer responseBuffer = workerSocket->sizePrefixedRead();
  const tt::runtime::distributed::flatbuffer::Response *response =
      tt::runtime::distributed::flatbuffer::GetResponse(responseBuffer.data());

  EXPECT_EQ(
      response->type_type(),
      tt::runtime::distributed::flatbuffer::ResponseType::ShutdownResponse);
}

TEST(RuntimeWorkerTest, TestSystemDesc) {

  tt::runtime::ControllerSocket controllerSocket;

  uint16_t port = controllerSocket.port();

  auto futureResult = runWorkerSubprocess(port);

  std::unique_ptr<tt::runtime::Socket> workerSocket =
      std::move(controllerSocket.connectToWorkers(1)[0]);

  flatbuffers::FlatBufferBuilder fbb;

  CommandFactory::buildGetSystemDescCommand(
      fbb, tt::runtime::DispatchCoreType::WORKER);
  workerSocket->sizePrefixedWrite(fbb.GetBufferPointer(), fbb.GetSize());

  tt::runtime::SizedBuffer responseBuffer = readResponseAndValidate(
      workerSocket, tt::runtime::distributed::flatbuffer::ResponseType::
                        GetSystemDescResponse);

  const tt::runtime::distributed::flatbuffer::Response *response =
      tt::runtime::distributed::flatbuffer::GetResponse(responseBuffer.data());

  const tt::runtime::distributed::flatbuffer::GetSystemDescResponse
      *getSystemDescResponse = response->type_as_GetSystemDescResponse();

  const auto *systemDescVec = getSystemDescResponse->system_desc();

  sendShutdownCommandAndValidate(workerSocket);

  futureResult.wait_for(std::chrono::seconds(5));

  auto localSystemDesc = tt::runtime::system_desc::getCurrentSystemDesc(
      tt::runtime::DispatchCoreType::WORKER, std::nullopt);

  std::vector<uint8_t> localSystemDescBuffer;
  localSystemDesc.storeToMemory(localSystemDescBuffer);

  std::vector<uint8_t> remoteSystemDescBuffer(systemDescVec->begin(),
                                              systemDescVec->end());

  EXPECT_EQ(localSystemDescBuffer, remoteSystemDescBuffer);
}

TEST(RuntimeWorkerTest, TestSubmit) {
  tt::runtime::ControllerSocket controllerSocket;

  uint16_t port = controllerSocket.port();

  auto futureResult = runWorkerSubprocess(port);

  std::unique_ptr<tt::runtime::Socket> workerSocket =
      std::move(controllerSocket.connectToWorkers(1)[0]);

  flatbuffers::FlatBufferBuilder fbb;

  std::string binaryPath = ::tt::runtime::utils::getMlirHome() +
                           "/build/test/ttmlir/Runtime/TTNN/n150/consteval/"
                           "Output/binary_ops.mlir.tmp.ttnn";
  ::tt::runtime::Binary binary =
      ::tt::runtime::Binary::loadFromPath(binaryPath.c_str());

  // Create input tensors
  std::vector<uint32_t> inputShape = {32, 32};
  std::vector<uint32_t> inputStride =
      tt::runtime::utils::calculateStride(inputShape);

  constexpr size_t numInputs = 4;
  std::vector<std::vector<uint16_t>> inputs(numInputs);

  for (size_t i = 0; i < numInputs; i++) {
    for (size_t j = 0; j < inputShape[0] * inputShape[1]; j++) {
      inputs[i].push_back(i + j);
    }
  }

  std::vector<::tt::runtime::Tensor> inputTensorWrappers;
  for (size_t i = 0; i < numInputs; i++) {
    inputTensorWrappers.push_back(::tt::runtime::Tensor(
        nullptr, nullptr, tt::runtime::DeviceRuntime::TTNN));
  }

  for (size_t i = 0; i < numInputs; i++) {
    CommandFactory::buildCreateHostTensorCommand(
        fbb, inputTensorWrappers[i], inputs[i].data(), inputShape, inputStride,
        sizeof(uint16_t), tt::target::DataType::BFloat16);
    workerSocket->sizePrefixedWrite(fbb.GetBufferPointer(), fbb.GetSize());
    readResponseAndValidate(workerSocket,
                            tt::runtime::distributed::flatbuffer::ResponseType::
                                CreateHostTensorResponse);
    fbb.Clear();
  }

  // Open mesh device
  ::tt::runtime::Device deviceWrapper(nullptr, nullptr,
                                      tt::runtime::DeviceRuntime::TTNN);
  ::tt::runtime::MeshDeviceOptions meshDeviceOptions;
  meshDeviceOptions.meshShape = {1, 1};
  CommandFactory::buildOpenMeshDeviceCommand(fbb, deviceWrapper,
                                             meshDeviceOptions);
  workerSocket->sizePrefixedWrite(fbb.GetBufferPointer(), fbb.GetSize());
  readResponseAndValidate(workerSocket,
                          tt::runtime::distributed::flatbuffer::ResponseType::
                              OpenMeshDeviceResponse);
  fbb.Clear();

  // To layout
  std::vector<::tt::runtime::Tensor> inputWithLayoutTensorWrappers;
  for (size_t i = 0; i < numInputs; i++) {
    inputWithLayoutTensorWrappers.push_back(::tt::runtime::Tensor(
        nullptr, nullptr, tt::runtime::DeviceRuntime::TTNN));
  }

  std::vector<::tt::runtime::Layout> layoutWrappers;
  for (size_t i = 0; i < numInputs; i++) {
    ::tt::runtime::Layout layoutWrapper =
        ::tt::runtime::Layout(nullptr, tt::runtime::DeviceRuntime::TTNN);
    layoutWrappers.push_back(layoutWrapper);
    CommandFactory::buildGetLayoutCommand(fbb, binary, 0, i, layoutWrapper);
    workerSocket->sizePrefixedWrite(fbb.GetBufferPointer(), fbb.GetSize());
    readResponseAndValidate(
        workerSocket,
        tt::runtime::distributed::flatbuffer::ResponseType::GetLayoutResponse);
    fbb.Clear();
  }

  for (size_t i = 0; i < numInputs; i++) {
    CommandFactory::buildToLayoutCommand(
        fbb, inputTensorWrappers[i], deviceWrapper, layoutWrappers[i],
        inputWithLayoutTensorWrappers[i], true);
    workerSocket->sizePrefixedWrite(fbb.GetBufferPointer(), fbb.GetSize());
    readResponseAndValidate(
        workerSocket,
        tt::runtime::distributed::flatbuffer::ResponseType::ToLayoutResponse);
    fbb.Clear();
  }

  // Submit
  std::vector<::tt::runtime::Tensor> submitOutputTensors;
  const auto *program = tt::runtime::ttnn::utils::getProgram(binary, 0);
  for (size_t i = 0; i < program->outputs()->size(); i++) {
    ::tt::runtime::Tensor outputTensorWrapper(nullptr, nullptr,
                                              tt::runtime::DeviceRuntime::TTNN);
    submitOutputTensors.push_back(outputTensorWrapper);
  }

  CommandFactory::buildSubmitCommand(fbb, deviceWrapper, binary, 0,
                                     inputWithLayoutTensorWrappers,
                                     submitOutputTensors);
  workerSocket->sizePrefixedWrite(fbb.GetBufferPointer(), fbb.GetSize());
  readResponseAndValidate(
      workerSocket,
      tt::runtime::distributed::flatbuffer::ResponseType::SubmitResponse);
  fbb.Clear();

  // To host
  std::vector<::tt::runtime::Tensor> outputTensorsHost;
  for (size_t i = 0; i < submitOutputTensors.size(); i++) {
    outputTensorsHost.push_back(::tt::runtime::Tensor(
        nullptr, nullptr, tt::runtime::DeviceRuntime::TTNN));
    CommandFactory::buildToHostCommand(fbb, submitOutputTensors[i], true, true,
                                       {outputTensorsHost[i]});
    workerSocket->sizePrefixedWrite(fbb.GetBufferPointer(), fbb.GetSize());
    readResponseAndValidate(
        workerSocket,
        tt::runtime::distributed::flatbuffer::ResponseType::ToHostResponse);
    fbb.Clear();
  }

  // Memcpy
  std::vector<std::vector<uint8_t>> remoteOutputTensors;

  for (size_t i = 0; i < outputTensorsHost.size(); i++) {
    CommandFactory::buildMemcpyCommand(fbb, outputTensorsHost[i], std::nullopt,
                                       std::nullopt);
    workerSocket->sizePrefixedWrite(fbb.GetBufferPointer(), fbb.GetSize());
    tt::runtime::SizedBuffer responseBuffer = readResponseAndValidate(
        workerSocket,
        tt::runtime::distributed::flatbuffer::ResponseType::MemcpyResponse);
    const tt::runtime::distributed::flatbuffer::Response *response =
        tt::runtime::distributed::flatbuffer::GetResponse(
            responseBuffer.data());

    const tt::runtime::distributed::flatbuffer::MemcpyResponse *memcpyResponse =
        response->type_as_MemcpyResponse();
    remoteOutputTensors.push_back(std::vector<uint8_t>(
        memcpyResponse->data()->begin(), memcpyResponse->data()->end()));
    fbb.Clear();
  }

  // Shutdown
  sendShutdownCommandAndValidate(workerSocket);
  futureResult.wait_for(std::chrono::seconds(5));

  // Submit locally and compare
  ::tt::runtime::Device localDevice =
      ::tt::runtime::openMeshDevice(meshDeviceOptions);

  std::vector<::tt::runtime::Tensor> localInputs;
  for (size_t i = 0; i < numInputs; i++) {
    localInputs.push_back(::tt::runtime::createOwnedHostTensor(
        inputs[i].data(), inputShape, inputStride, sizeof(uint16_t),
        tt::target::DataType::BFloat16));
  }

  std::vector<::tt::runtime::Layout> localLayouts;
  for (size_t i = 0; i < numInputs; i++) {
    localLayouts.push_back(::tt::runtime::getLayout(binary, 0, i));
  }

  std::vector<::tt::runtime::Tensor> localInputsWithLayout;
  for (size_t i = 0; i < numInputs; i++) {
    localInputsWithLayout.push_back(::tt::runtime::toLayout(
        localInputs[i], localDevice, localLayouts[i], true));
  }

  std::vector<::tt::runtime::Tensor> localOutputTensors =
      ::tt::runtime::submit(localDevice, binary, 0, localInputsWithLayout);

  EXPECT_EQ(remoteOutputTensors.size(), localOutputTensors.size());

  for (size_t i = 0; i < remoteOutputTensors.size(); i++) {
    std::vector<std::byte> localTensor =
        ::tt::runtime::getTensorDataBuffer(localOutputTensors[i]);
    EXPECT_EQ(remoteOutputTensors[i].size(), localTensor.size());
    EXPECT_EQ(std::memcmp(remoteOutputTensors[i].data(), localTensor.data(),
                          remoteOutputTensors[i].size()),
              0);
  }
}
