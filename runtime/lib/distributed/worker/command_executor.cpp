// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/distributed/worker/command_executor.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/common/runtime_context.h"
#include "tt/runtime/detail/common/socket.h"
#include "tt/runtime/detail/distributed/worker/response_factory.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unistd.h>
#include <unordered_map>

namespace tt::runtime::distributed::worker {

namespace fb = ::tt::runtime::distributed::flatbuffer;

// Memory-tracing instrumentation for the distributed worker process.
//
// Enabled by setting TT_RUNTIME_DISTRIBUTED_MEM_TRACE to a non-empty,
// non-"0" value. When enabled, the worker logs (at INFO level) the size and
// logical byte footprint of its tensor pool together with the worker process
// resident set size (RSS) after tensor-mutating commands. This makes it
// possible to observe that host input tensors created via
// CreateHostTensor / CreateMultiDeviceHostTensorFromShards remain resident in
// the pool after ToLayout migrates them to device (i.e. they are not freed
// until an explicit DeallocateTensor command, which tt-xla never sends).
static bool memTraceEnabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("TT_RUNTIME_DISTRIBUTED_MEM_TRACE");
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
  }();
  return enabled;
}

// Reads the worker process resident set size in bytes from /proc/self/statm.
// Returns 0 if the value cannot be read. The second field of statm is the
// resident set size expressed in pages.
static size_t readProcSelfRssBytes() {
  std::FILE *f = std::fopen("/proc/self/statm", "r");
  if (f == nullptr) {
    return 0;
  }
  long totalPages = 0;
  long residentPages = 0;
  const int matched = std::fscanf(f, "%ld %ld", &totalPages, &residentPages);
  std::fclose(f);
  if (matched != 2) {
    return 0;
  }
  static const long pageSize = ::sysconf(_SC_PAGESIZE);
  return static_cast<size_t>(residentPages) * static_cast<size_t>(pageSize);
}

// Logs the current tensor-pool footprint and worker RSS. No-op unless
// TT_RUNTIME_DISTRIBUTED_MEM_TRACE is enabled.
static void
logPoolFootprint(const char *tag, uint64_t commandId,
                 const std::unordered_map<uint64_t, ::tt::runtime::Tensor>
                     &tensorPool) {
  if (!memTraceEnabled()) {
    return;
  }

  size_t totalLogicalBytes = 0;
  size_t allocatedCount = 0;
  for (const auto &entry : tensorPool) {
    const ::tt::runtime::Tensor &tensor = entry.second;
    try {
      if (::tt::runtime::isTensorAllocated(tensor)) {
        ++allocatedCount;
      }
    } catch (...) {
      // Best-effort: ignore tensors that cannot report allocation state.
    }
    try {
      totalLogicalBytes += ::tt::runtime::getTensorDesc(tensor).sizeBytes();
    } catch (...) {
      // Best-effort: some tensors may not support a desc query.
    }
  }

  const size_t rssBytes = readProcSelfRssBytes();
  constexpr double kMiB = 1024.0 * 1024.0;
  LOG_INFO("[MEM_TRACE] after ", tag, " cmd=", commandId,
           " pool_size=", tensorPool.size(), " allocated=", allocatedCount,
           " pool_logical_MB=", static_cast<double>(totalLogicalBytes) / kMiB,
           " worker_rss_MB=", static_cast<double>(rssBytes) / kMiB);
}

template <typename Builder, typename... Args>
static std::unique_ptr<::flatbuffers::FlatBufferBuilder>
buildResponse(const Builder &builderFunc, uint64_t commandId, Args &&...args) {

  auto responseBuilder = std::make_unique<::flatbuffers::FlatBufferBuilder>();

  builderFunc(*responseBuilder, commandId, std::forward<Args>(args)...);

  return responseBuilder;
}

static const fb::Command *getCommand(const SizedBuffer &command) {
  bool isDistributedCommand = fb::CommandBufferHasIdentifier(command.data());
  LOG_ASSERT(isDistributedCommand, "Command is not a distributed command");
  return fb::GetCommand(command.data());
}

void CommandExecutor::connect(const std::string &host, uint16_t port) {
  LOG_ASSERT(!workerSocket_, "WorkerSocket already connected");
  workerSocket_ = std::make_unique<WorkerSocket>(host, port);
}

::tt::runtime::Binary
CommandExecutor::getOrCreateBinary(const flatbuffers::Vector<uint8_t> *binary,
                                   uint64_t binaryId) {
  ::tt::runtime::Binary executable(nullptr);
  if (binaryPool_.contains(binaryId)) {
    executable = binaryPool_.at(binaryId);
  } else {
    executable =
        ::tt::runtime::Binary::loadFromMemory(binary->data(), binary->size());
    executable.setId(binaryId);
    binaryPool_.insert_or_assign(binaryId, executable);
  }
  return executable;
}

void CommandExecutor::run() {
  launchCommandReceiver();
  launchResponseSender();
  while (!shutdownRequested_.load(std::memory_order_relaxed)) {
    SizedBuffer commandData = commandQueue_.popBlocking();
    const fb::Command *command = getCommand(commandData);

    LOG_DEBUG("Executing command: ",
              fb::EnumNameCommandType(command->type_type()));

    executeCommand(command);

    LOG_DEBUG("Finished executing command: ",
              fb::EnumNameCommandType(command->type_type()));
  }
}

void CommandExecutor::launchCommandReceiver() {
  LOG_ASSERT(!commandReceiverThread_.joinable(),
             "Command receiver thread already running");
  commandReceiverThread_ = std::thread([this]() { receiveCommands(); });
}

// This will get run on the command receiver thread
void CommandExecutor::receiveCommands() {
  while (!shutdownRequested_.load(std::memory_order_relaxed)) {
    if (!workerSocket_->hasDataToRead()) {
      continue;
    }
    SizedBuffer commandData = workerSocket_->sizePrefixedRead();
    LOG_ASSERT(commandData.size(), "Read null command from worker socket");
    commandQueue_.push(commandData);
  }
}

void CommandExecutor::launchResponseSender() {
  LOG_ASSERT(!responseSenderThread_.joinable(),
             "Response sender thread already running");
  responseSenderThread_ = std::thread([this]() { sendResponses(); });
}

// This will get run on the response sender thread
void CommandExecutor::sendResponses() {
  while (!(shutdownRequested_.load(std::memory_order_acquire) &&
           responseQueue_.empty())) {
    std::optional<std::unique_ptr<::flatbuffers::FlatBufferBuilder>>
        responseBuilderOpt = responseQueue_.popWithTimeout();
    if (!responseBuilderOpt.has_value()) {
      continue;
    }
    std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
        std::move(responseBuilderOpt.value());
    size_t responseSize = responseBuilder->GetSize();
    LOG_ASSERT(responseSize > 0, "Unexpected empty response");
    workerSocket_->sizePrefixedWrite(responseBuilder->GetBufferPointer(),
                                     responseSize);
  }
}

void CommandExecutor::execute(
    uint64_t commandId, const fb::ConfigureRuntimeContextCommand *command) {

  ::tt::runtime::RuntimeContext::instance().setMlirHome(
      command->mlir_home()->c_str());
  ::tt::runtime::RuntimeContext::instance().setMetalHome(
      command->metal_home()->c_str());
  ::tt::runtime::RuntimeContext::instance().setCurrentDeviceRuntime(
      command->current_device_runtime());
  ::tt::runtime::RuntimeContext::instance().setMemoryLogLevel(
      command->memory_log_level());

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildConfigureRuntimeContextResponse,
                    commandId);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::SetMemoryLogLevelCommand *command) {
  ::tt::runtime::RuntimeContext::instance().setMemoryLogLevel(
      command->memory_log_level());

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildSetMemoryLogLevelResponse, commandId);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::GetSystemDescCommand *command) {

  std::optional<::tt::runtime::DispatchCoreType> dispatchCoreType =
      std::nullopt;
  if (command->dispatch_core_type()) {
    dispatchCoreType = command->dispatch_core_type().value();
  }

  std::optional<Device> device = std::nullopt;
  if (command->device()) {
    device = devicePool_.at(command->device()->global_id());
  }

  ::tt::runtime::SystemDesc systemDesc =
      ::tt::runtime::system_desc::getCurrentSystemDesc(dispatchCoreType,
                                                       device);

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildGetSystemDescResponse, commandId,
                    systemDesc);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::SetFabricConfigCommand *command) {

  ::tt::runtime::setFabricConfig(command->config());

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildSetFabricConfigResponse, commandId);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(
    uint64_t commandId, const fb::GetNumAvailableDevicesCommand *command) {

  size_t numDevices = ::tt::runtime::getNumAvailableDevices();

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildGetNumAvailableDevicesResponse,
                    commandId, numDevices);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::OpenMeshDeviceCommand *command) {

  uint32_t deviceGlobalId = command->device_global_id();
  const ::tt::runtime::flatbuffer::MeshDeviceOptions *options =
      command->options();

  ::tt::runtime::MeshDeviceOptions meshDeviceOptions;
  if (options->mesh_offset()) {
    LOG_ASSERT(options->mesh_offset()->size() == 2,
               "Currently only 2D mesh offsets are supported");
    meshDeviceOptions.meshOffset = {options->mesh_offset()->Get(0),
                                    options->mesh_offset()->Get(1)};
  }

  if (options->device_ids()) {
    meshDeviceOptions.deviceIds.resize(options->device_ids()->size());
    std::copy_n(options->device_ids()->begin(), options->device_ids()->size(),
                meshDeviceOptions.deviceIds.begin());
  }

  meshDeviceOptions.numHWCQs = options->num_hw_cqs();
  meshDeviceOptions.enableProgramCache = options->enable_program_cache();

  if (options->mesh_shape()) {
    LOG_ASSERT(options->mesh_shape()->size() == 2,
               "Currently only 2D mesh shapes are supported");
    meshDeviceOptions.meshShape = {options->mesh_shape()->Get(0),
                                   options->mesh_shape()->Get(1)};
  }

  if (options->l1_small_size().has_value()) {
    meshDeviceOptions.l1SmallSize = options->l1_small_size().value();
  }

  if (options->trace_region_size().has_value()) {
    meshDeviceOptions.traceRegionSize = options->trace_region_size().value();
  }

  if (options->dispatch_core_type().has_value()) {
    meshDeviceOptions.dispatchCoreType = options->dispatch_core_type().value();
  }

  ::tt::runtime::Device device =
      ::tt::runtime::openMeshDevice(meshDeviceOptions);

  device.setGlobalId(deviceGlobalId);

  devicePool_.insert_or_assign(deviceGlobalId, device);

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildOpenMeshDeviceResponse, commandId,
                    device);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::CloseMeshDeviceCommand *command) {

  uint32_t deviceGlobalId = command->device()->global_id();

  ::tt::runtime::Device device = devicePool_.at(deviceGlobalId);
  LOG_ASSERT(device.getGlobalId() == deviceGlobalId,
             "Parent mesh global id mismatch");

  ::tt::runtime::closeMeshDevice(device);

  devicePool_.erase(deviceGlobalId);

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildCloseMeshDeviceResponse, commandId);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::CreateSubMeshDeviceCommand *command) {

  ::tt::runtime::Device parentMesh =
      devicePool_.at(command->parent_mesh()->global_id());
  LOG_ASSERT(parentMesh.getGlobalId() == command->parent_mesh()->global_id(),
             "Parent mesh global id mismatch");

  std::vector<uint32_t> meshShape(command->mesh_shape()->begin(),
                                  command->mesh_shape()->end());

  std::optional<std::vector<uint32_t>> meshOffset = std::nullopt;
  if (command->mesh_offset()) {
    meshOffset = std::vector<uint32_t>(command->mesh_offset()->begin(),
                                       command->mesh_offset()->end());
  }

  uint32_t subMeshGlobalId = command->submesh_global_id();
  ::tt::runtime::Device subMesh =
      ::tt::runtime::createSubMeshDevice(parentMesh, meshShape, meshOffset);
  subMesh.setGlobalId(subMeshGlobalId);
  devicePool_.insert_or_assign(subMeshGlobalId, subMesh);

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildCreateSubMeshDeviceResponse,
                    commandId, subMesh);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::ReleaseSubMeshDeviceCommand *command) {

  uint32_t subMeshGlobalId = command->sub_mesh()->global_id();

  ::tt::runtime::Device subMesh = devicePool_.at(subMeshGlobalId);
  LOG_ASSERT(subMesh.getGlobalId() == subMeshGlobalId,
             "Submesh global id mismatch");

  ::tt::runtime::releaseSubMeshDevice(subMesh);
  devicePool_.erase(subMeshGlobalId);

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildReleaseSubMeshDeviceResponse,
                    commandId);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::GetMeshShapeCommand *command) {

  ::tt::runtime::Device device = devicePool_.at(command->device()->global_id());

  std::vector<uint32_t> shape = ::tt::runtime::getMeshShape(device);

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildGetMeshShapeResponse, commandId,
                    shape);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::CreateHostTensorCommand *command) {
  uint64_t tensorGlobalId = command->output_global_id();
  const uint8_t *tensorData = command->data()->data();
  std::vector<uint32_t> shape(command->shape()->begin(),
                              command->shape()->end());
  std::vector<int64_t> stride(command->stride()->begin(),
                              command->stride()->end());
  uint32_t itemSize = command->item_size();
  ::tt::target::DataType dataType = command->data_type();

  ::tt::runtime::Tensor tensor = ::tt::runtime::createOwnedHostTensor(
      tensorData, shape, stride, itemSize, dataType);

  tensor.setGlobalId(tensorGlobalId);

  tensorPool_.insert_or_assign(tensorGlobalId, tensor);

  logPoolFootprint("CreateHostTensor", commandId, tensorPool_);

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildCreateHostTensorResponse, commandId);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(
    uint64_t commandId,
    const fb::CreateMultiDeviceHostTensorFromShardsCommand *command) {
  std::vector<::tt::runtime::Tensor> tensorShards;
  for (const auto &inputGlobalId : *command->input_global_ids()) {
    tensorShards.push_back(tensorPool_.at(inputGlobalId));
  }

  std::unordered_map<std::string, std::string> strategy;
  for (const auto &strategyEntry : *command->strategy_map()) {
    strategy[strategyEntry->key()->c_str()] = strategyEntry->value()->c_str();
  }

  std::vector<uint32_t> meshShape(command->mesh_shape()->begin(),
                                  command->mesh_shape()->end());

  ::tt::runtime::Tensor multiDeviceHostTensor =
      ::tt::runtime::createMultiDeviceHostTensor(tensorShards, strategy,
                                                 meshShape);

  multiDeviceHostTensor.setGlobalId(command->output_global_id());
  tensorPool_.insert_or_assign(command->output_global_id(),
                               multiDeviceHostTensor);

  logPoolFootprint("CreateMultiDeviceHostTensorFromShards", commandId,
                   tensorPool_);

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(
          ResponseFactory::buildCreateMultiDeviceHostTensorFromShardsResponse,
          commandId);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(
    uint64_t commandId,
    const fb::CreateUnsafeBorrowedHostTensorCommand *command) {
  uint64_t sourceGlobalId = command->source_tensor_global_id();
  uint64_t outputGlobalId = command->output_global_id();

  LOG_ASSERT(tensorPool_.contains(sourceGlobalId),
             "Source tenso with global id ", sourceGlobalId,
             " not found in tensor pool");

  ::tt::runtime::Tensor sourceTensor = tensorPool_.at(sourceGlobalId);
  ::tt::runtime::Tensor borrowedTensor =
      ::tt::runtime::createUnsafeBorrowedHostTensor(sourceTensor);
  borrowedTensor.setGlobalId(outputGlobalId);
  tensorPool_.insert_or_assign(outputGlobalId, borrowedTensor);

  logPoolFootprint("CreateUnsafeBorrowedHostTensor", commandId, tensorPool_);

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(
          ResponseFactory::buildCreateUnsafeBorrowedHostTensorResponse,
          commandId);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::IsTensorAllocatedCommand *command) {
  uint64_t tensorGlobalId = command->tensor_global_id();

  bool allocated =
      tensorPool_.contains(tensorGlobalId)
          ? ::tt::runtime::isTensorAllocated(tensorPool_.at(tensorGlobalId))
          : false;

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildIsTensorAllocatedResponse, commandId,
                    allocated);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::GetTensorVolumeCommand *command) {
  uint64_t tensorGlobalId = command->tensor_global_id();
  ::tt::runtime::Tensor tensor = tensorPool_.at(tensorGlobalId);

  uint32_t volume = ::tt::runtime::getTensorVolume(tensor);

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildGetTensorVolumeResponse, commandId,
                    volume);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::GetTensorRetainCommand *command) {

  uint64_t tensorGlobalId = command->tensor_global_id();
  ::tt::runtime::Tensor tensor = tensorPool_.at(tensorGlobalId);

  bool retain = ::tt::runtime::getTensorRetain(tensor);

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildGetTensorRetainResponse, commandId,
                    retain);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::SetTensorRetainCommand *command) {
  uint64_t tensorGlobalId = command->tensor_global_id();
  ::tt::runtime::Tensor tensor = tensorPool_.at(tensorGlobalId);

  ::tt::runtime::setTensorRetain(tensor, command->retain());

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildSetTensorRetainResponse, commandId);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::SeedProgramBinaryCommand *command) {

  getOrCreateBinary(command->binary(), command->binary_id());

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildSeedProgramBinaryResponse, commandId);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::GetLayoutCommand *command) {

  uint64_t binaryId = command->binary_id();
  LOG_ASSERT(binaryPool_.contains(binaryId),
             "Binary with id ", binaryId,
             " not found in binary pool; it must be seeded via "
             "SeedProgramBinaryCommand before issuing a GetLayoutCommand");
  ::tt::runtime::Binary binary = binaryPool_.at(binaryId);

  ::tt::runtime::Layout layout = ::tt::runtime::getLayout(
      binary, command->program_id(), command->input_id());

  layout.setGlobalId(command->output_layout_id());
  layoutPool_.insert_or_assign(command->output_layout_id(), layout);

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildGetLayoutResponse, commandId);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::ToLayoutCommand *command) {

  uint64_t inputGlobalId = command->input_global_id();
  uint64_t outputGlobalId = command->output_global_id();
  ::tt::runtime::Device device = devicePool_.at(command->device()->global_id());

  ::tt::runtime::Tensor inputTensor = tensorPool_.at(inputGlobalId);

  std::optional<bool> retain = std::nullopt;
  if (command->retain().has_value()) {
    retain = command->retain().value();
  }

  ::tt::runtime::Layout layout = layoutPool_.at(command->layout_global_id());

  ::tt::runtime::Tensor resultTensor =
      ::tt::runtime::toLayout(inputTensor, device, layout, retain);

  resultTensor.setGlobalId(outputGlobalId);

  tensorPool_.insert_or_assign(outputGlobalId, resultTensor);

  if (memTraceEnabled()) {
    const bool inputStillInPool = tensorPool_.contains(inputGlobalId);
    bool inputAllocated = false;
    bool inputRetain = false;
    if (inputStillInPool) {
      const ::tt::runtime::Tensor &pooledInput = tensorPool_.at(inputGlobalId);
      try {
        inputAllocated = ::tt::runtime::isTensorAllocated(pooledInput);
      } catch (...) {
      }
      try {
        inputRetain = ::tt::runtime::getTensorRetain(pooledInput);
      } catch (...) {
      }
    }
    LOG_INFO("[MEM_TRACE] ToLayout input_global_id=", inputGlobalId,
             " still_in_pool=", inputStillInPool,
             " input_allocated=", inputAllocated,
             " input_retain=", inputRetain,
             " output_global_id=", outputGlobalId);
  }

  logPoolFootprint("ToLayout", commandId, tensorPool_);

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildToLayoutResponse, commandId);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::SubmitCommand *command) {
  LOG_INFO("Worker executing SubmitCommand id=", commandId,
           " program=", command->program_id(),
           " inputs=", command->input_global_ids()->size(),
           " outputs=", command->output_global_ids()->size());

  ::tt::runtime::Device device = devicePool_.at(command->device()->global_id());

  std::vector<::tt::runtime::Tensor> inputTensors;
  for (const auto &inputGlobalId : *command->input_global_ids()) {
    inputTensors.push_back(tensorPool_.at(inputGlobalId));
  }

  ::tt::runtime::Binary executable =
      getOrCreateBinary(command->binary(), command->binary_id());

  std::vector<::tt::runtime::Tensor> outputTensors = ::tt::runtime::submit(
      device, executable, command->program_id(), inputTensors);

  LOG_ASSERT(outputTensors.size() == command->output_global_ids()->size(),
             "Output tensors from submit does not match the number of output "
             "global ids: ",
             outputTensors.size(),
             " != ", command->output_global_ids()->size());

  for (size_t i = 0; i < outputTensors.size(); i++) {
    outputTensors[i].setGlobalId(command->output_global_ids()->Get(i));
    tensorPool_.insert_or_assign(command->output_global_ids()->Get(i),
                                 outputTensors[i]);
  }

  logPoolFootprint("Submit", commandId, tensorPool_);

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildSubmitResponse, commandId);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::GetNumShardsCommand *command) {
  uint64_t tensorGlobalId = command->input_global_id();
  ::tt::runtime::Tensor tensor = tensorPool_.at(tensorGlobalId);

  uint32_t numBuffers = ::tt::runtime::detail::getNumShards(tensor);

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildGetNumShardsResponse, commandId,
                    numBuffers);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::ToHostCommand *command) {

  uint64_t inputGlobalId = command->input_global_id();

  ::tt::runtime::Tensor inputTensor = tensorPool_.at(inputGlobalId);

  bool untilize = command->untilize();
  bool blocking = command->blocking();

  std::vector<::tt::runtime::Tensor> outputTensors =
      ::tt::runtime::toHost(inputTensor, untilize, blocking);

  LOG_ASSERT(outputTensors.size() == command->output_global_ids()->size(),
             "Output tensors from toHost does not match the number of output "
             "global ids: ",
             outputTensors.size(),
             " != ", command->output_global_ids()->size());

  for (size_t i = 0; i < outputTensors.size(); i++) {
    outputTensors[i].setGlobalId(command->output_global_ids()->Get(i));
    tensorPool_.insert_or_assign(command->output_global_ids()->Get(i),
                                 outputTensors[i]);
  }

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildToHostResponse, commandId);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::MemcpyCommand *command) {
  uint64_t srcGlobalId = command->src_global_id();
  ::tt::runtime::Tensor srcTensor = tensorPool_.at(srcGlobalId);

  std::optional<std::vector<std::uint8_t>> dstDataBuffer = std::nullopt;

  if (command->dst_global_id().has_value()) {
    ::tt::runtime::Tensor dstTensor =
        tensorPool_.at(command->dst_global_id().value());
    ::tt::runtime::memcpy(dstTensor, srcTensor);
  } else {
    size_t volume = ::tt::runtime::getTensorVolume(srcTensor);
    size_t dstElementSize = ::tt::runtime::getTensorElementSize(srcTensor);

    std::optional<::tt::target::DataType> dstDataType = std::nullopt;
    if (command->dst_data_type().has_value()) {
      dstDataType = command->dst_data_type().value();
      dstElementSize =
          ::tt::runtime::utils::dataTypeElementSize(dstDataType.value());
    }

    size_t dstSizeBytes = volume * dstElementSize;

    dstDataBuffer = std::vector<std::uint8_t>(dstSizeBytes);

    ::tt::runtime::memcpy(dstDataBuffer.value().data(), srcTensor, dstDataType);
  }

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildMemcpyResponse, commandId,
                    dstDataBuffer);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::DeallocateTensorCommand *command) {
  uint64_t tensorGlobalId = command->tensor_global_id();
  ::tt::runtime::Tensor tensor = tensorPool_.at(tensorGlobalId);

  if (::tt::runtime::getTensorRetain(tensor)) {
    LOG_WARNING("Tensor is retained, will not be deallocated");
  } else {
    ::tt::runtime::deallocateTensor(tensor, command->force());
    tensorPool_.erase(tensorGlobalId);
  }

  logPoolFootprint("DeallocateTensor", commandId, tensorPool_);

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildDeallocateTensorResponse, commandId);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::ShutdownCommand *command) {

  LOG_INFO("Shutdown command received, shutting down command executor");

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildShutdownResponse, commandId);

  responseQueue_.push(std::move(responseBuilder));

  shutdownRequested_.store(true, std::memory_order_release);
  commandReceiverThread_.join();
  responseSenderThread_.join();

  LOG_INFO("Command executor shutdown complete");
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::GetTensorDescCommand *command) {
  uint64_t tensorGlobalId = command->tensor_global_id();
  ::tt::runtime::Tensor tensor = tensorPool_.at(tensorGlobalId);
  TensorDesc tensorDesc = ::tt::runtime::getTensorDesc(tensor);

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildGetTensorDescResponse, commandId,
                    tensorDesc);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::HasLayoutCommand *command) {
  ::tt::runtime::Tensor tensor = tensorPool_.at(command->tensor_global_id());
  ::tt::runtime::Layout layout = layoutPool_.at(command->layout_global_id());

  const std::shared_ptr<::tt::runtime::ttnn::LayoutDesc> tensorLayoutDesc =
      ::tt::runtime::ttnn::LayoutDesc::fromTensor(tensor);
  const ::tt::runtime::ttnn::LayoutDesc &desiredLayoutDesc =
      layout.as<::tt::runtime::ttnn::LayoutDesc>(DeviceRuntime::TTNN);
  bool hasLayout = *tensorLayoutDesc == desiredLayoutDesc;

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildHasLayoutResponse, commandId,
                    hasLayout);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::IsProgramCacheEnabledCommand *command) {
  uint32_t deviceGlobalId = command->device()->global_id();
  ::tt::runtime::Device device = devicePool_.at(deviceGlobalId);

  bool isEnabled = ::tt::runtime::isProgramCacheEnabled(device);

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildIsProgramCacheEnabledResponse,
                    commandId, isEnabled);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::execute(uint64_t commandId,
                              const fb::ClearProgramCacheCommand *command) {
  uint32_t deviceGlobalId = command->device()->global_id();
  ::tt::runtime::Device device = devicePool_.at(deviceGlobalId);

  ::tt::runtime::clearProgramCache(device);

  std::unique_ptr<::flatbuffers::FlatBufferBuilder> responseBuilder =
      buildResponse(ResponseFactory::buildClearProgramCacheResponse, commandId);

  responseQueue_.push(std::move(responseBuilder));
}

void CommandExecutor::executeCommand(const fb::Command *command) {
  switch (command->type_type()) {
  case fb::CommandType::ConfigureRuntimeContextCommand: {
    return execute(command->command_id(),
                   command->type_as_ConfigureRuntimeContextCommand());
  }
  case fb::CommandType::SetMemoryLogLevelCommand: {
    return execute(command->command_id(),
                   command->type_as_SetMemoryLogLevelCommand());
  }
  case fb::CommandType::GetSystemDescCommand: {
    return execute(command->command_id(),
                   command->type_as_GetSystemDescCommand());
  }
  case fb::CommandType::SetFabricConfigCommand: {
    return execute(command->command_id(),
                   command->type_as_SetFabricConfigCommand());
  }
  case fb::CommandType::GetNumAvailableDevicesCommand: {
    return execute(command->command_id(),
                   command->type_as_GetNumAvailableDevicesCommand());
  }
  case fb::CommandType::OpenMeshDeviceCommand: {
    return execute(command->command_id(),
                   command->type_as_OpenMeshDeviceCommand());
  }
  case fb::CommandType::CloseMeshDeviceCommand: {
    return execute(command->command_id(),
                   command->type_as_CloseMeshDeviceCommand());
  }
  case fb::CommandType::CreateSubMeshDeviceCommand: {
    return execute(command->command_id(),
                   command->type_as_CreateSubMeshDeviceCommand());
  }
  case fb::CommandType::ReleaseSubMeshDeviceCommand: {
    return execute(command->command_id(),
                   command->type_as_ReleaseSubMeshDeviceCommand());
  }
  case fb::CommandType::GetMeshShapeCommand: {
    return execute(command->command_id(),
                   command->type_as_GetMeshShapeCommand());
  }
  case fb::CommandType::CreateHostTensorCommand: {
    return execute(command->command_id(),
                   command->type_as_CreateHostTensorCommand());
  }
  case fb::CommandType::CreateMultiDeviceHostTensorFromShardsCommand: {
    return execute(
        command->command_id(),
        command->type_as_CreateMultiDeviceHostTensorFromShardsCommand());
  }
  case fb::CommandType::CreateUnsafeBorrowedHostTensorCommand: {
    return execute(command->command_id(),
                   command->type_as_CreateUnsafeBorrowedHostTensorCommand());
  }
  case fb::CommandType::IsTensorAllocatedCommand: {
    return execute(command->command_id(),
                   command->type_as_IsTensorAllocatedCommand());
  }
  case fb::CommandType::GetTensorVolumeCommand: {
    return execute(command->command_id(),
                   command->type_as_GetTensorVolumeCommand());
  }
  case fb::CommandType::GetTensorRetainCommand: {
    return execute(command->command_id(),
                   command->type_as_GetTensorRetainCommand());
  }
  case fb::CommandType::SetTensorRetainCommand: {
    return execute(command->command_id(),
                   command->type_as_SetTensorRetainCommand());
  }
  case fb::CommandType::SeedProgramBinaryCommand: {
    return execute(command->command_id(),
                   command->type_as_SeedProgramBinaryCommand());
  }
  case fb::CommandType::GetLayoutCommand: {
    return execute(command->command_id(), command->type_as_GetLayoutCommand());
  }
  case fb::CommandType::ToLayoutCommand: {
    return execute(command->command_id(), command->type_as_ToLayoutCommand());
  }
  case fb::CommandType::SubmitCommand: {
    return execute(command->command_id(), command->type_as_SubmitCommand());
  }
  case fb::CommandType::GetNumShardsCommand: {
    return execute(command->command_id(),
                   command->type_as_GetNumShardsCommand());
  }
  case fb::CommandType::ToHostCommand: {
    return execute(command->command_id(), command->type_as_ToHostCommand());
  }
  case fb::CommandType::MemcpyCommand: {
    return execute(command->command_id(), command->type_as_MemcpyCommand());
  }
  case fb::CommandType::DeallocateTensorCommand: {
    return execute(command->command_id(),
                   command->type_as_DeallocateTensorCommand());
  }
  case fb::CommandType::ShutdownCommand: {
    return execute(command->command_id(), command->type_as_ShutdownCommand());
  }
  case fb::CommandType::GetTensorDescCommand: {
    return execute(command->command_id(),
                   command->type_as_GetTensorDescCommand());
  }
  case fb::CommandType::HasLayoutCommand: {
    return execute(command->command_id(), command->type_as_HasLayoutCommand());
  }
  case fb::CommandType::IsProgramCacheEnabledCommand: {
    return execute(command->command_id(),
                   command->type_as_IsProgramCacheEnabledCommand());
  }
  case fb::CommandType::ClearProgramCacheCommand: {
    return execute(command->command_id(),
                   command->type_as_ClearProgramCacheCommand());
  }
  case fb::CommandType::NONE: {
    LOG_FATAL("Unhandled command type: ",
              fb::EnumNameCommandType(command->type_type()));
  }
  } // end switch

  LOG_FATAL("Unreachable code path, all commands should be handled in switch "
            "statement");
}
} // namespace tt::runtime::distributed::worker
