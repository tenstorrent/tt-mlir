// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "executor.h"
#include "arguments.h"
#include "executor_utils.h"
#include "kernels.h"
#include "meshshard_utils.h"

#include "tools/profiler/op_profiler.hpp"
#include "tracy/Tracy.hpp"
#include "tt/runtime/debug.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/dylib.h"
#include "tt/runtime/detail/common/fabric_config.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttmetal/profiler.h"
#include "tt/runtime/detail/ttmetal/ttmetal.h"
#include "tt/runtime/perf.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/utils.h"
#include "tt/runtime/workarounds.h"

#include "ttmlir/Target/TTMetal/Target.h"
#include "ttmlir/Target/TTMetal/types_generated.h"
#include "ttmlir/Version.h"
#include "types_generated.h"

#include <cstdint>
#include <string>
#include <unordered_map>

#include <iostream>

namespace tt::runtime::ttmetal {

namespace target = ::tt::target;
namespace tt_metal = ::tt::tt_metal;
namespace distributed = ::tt::tt_metal::distributed;

namespace {
class MCQExecutor {
public:
  MCQExecutor(
      distributed::MeshDevice *meshDevice,
      const flatbuffers::Vector<
          flatbuffers::Offset<tt::target::metal::BufferRef>> *programInputs,
      const std::vector<Tensor> &inputs, common::DylibManager &&dylibManager,
      bool blockingCQ);

  const std::vector<Tensor> &getOutputs() const { return outputs; }

  void execute(const target::metal::CommandQueue *commandQueue);

private:
  void execute(const target::metal::Command *command);
  void execute(const target::metal::HostAllocCommand *command);
  void execute(const target::metal::ReturnCommand *command);
  void execute(const target::metal::EnqueueProgramCommand *command,
               const char *loc, const char *debugInfo);
  void execute(const target::metal::EnqueueWriteBufferCommand *command);
  void execute(const target::metal::EnqueueReadBufferCommand *command);
  void execute(const target::metal::CreateBufferCommand *command);
  void execute(const target::metal::DeallocateBufferCommand *command);
  void execute(const target::metal::EnqueueRecordEventCommand *command);
  void execute(const target::metal::EnqueueWaitForEventCommand *command);
  void execute(const target::metal::EventSynchronizeCommand *command);
  void execute(const target::metal::MemrefCopyCommand *command);
  void execute(const target::metal::CpuCommand *command);
  void execute(const target::metal::FinishCommand *command);
  void execute(const target::metal::MeshShardCommand *command);
  void execute(const target::metal::CreateGlobalSemaphoreCommand *command);
  void execute(const target::metal::ResetGlobalSemaphoreCommand *command);
  void execute(const target::metal::CreateLocalSemaphoreCommand *command);

  std::uint64_t getUniqueProgramRuntimeId() { return nextProgramRuntimeId++; }

private:
  // Reference to the device.
  distributed::MeshDevice *meshDevice;

  // Events for synchronizing before executing the first command.
  std::vector<std::shared_ptr<distributed::MeshEvent>> initMeshEvents;

  // Buffers that live on the host. Indexed by global_id.
  std::unordered_map<std::uint32_t, Tensor> hostBuffers;

  // Buffers that live on the mesh. Indexed by global_id. They are created by
  // ttmetal.create_buffer().
  std::unordered_map<std::uint32_t, std::shared_ptr<distributed::MeshBuffer>>
      meshBuffers;

  // Global semaphores. Indexed by global_id. They are created by
  // ttmetal.create_global_semaphore().
  std::unordered_map<std::uint32_t, tt_metal::GlobalSemaphore>
      global_semaphores;

  // Local semaphores. Indexed by global_id. We only store their initial value
  // here and lookup during their creation.
  std::unordered_map<std::uint32_t, std::uint32_t> local_semaphores;

  // Mesh events created by EnqueueRecordEventCommand. Used by
  // EnqueueWaitForEventCommand and EventSynchronizeCommand. Indexed by
  // global_id.
  std::unordered_map<std::uint32_t, std::shared_ptr<distributed::MeshEvent>>
      meshEvents;

  // Output tensors to be returned.
  std::vector<Tensor> outputs;

  // Other state data structures for executing commands.
  distributed::MeshCommandQueue *mcq;
  bool blockingCQ;
  const char *currentProgramName;
  DeviceAddressValidator deviceAddressValidator;
  common::DylibManager dylibManager;
  std::uint64_t nextProgramRuntimeId = 10000; // Start at a greppable number.
};
} // namespace

MCQExecutor::MCQExecutor(
    distributed::MeshDevice *meshDevice,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::metal::BufferRef>>
        *programInputs,
    const std::vector<Tensor> &inputs, common::DylibManager &&dylibManager,
    bool blockingCQ)
    : meshDevice(meshDevice), blockingCQ(blockingCQ),
      deviceAddressValidator(meshDevice->get_devices().at(0)),
      dylibManager(std::move(dylibManager)) {

  // Walk through the program inputs and populate hostBuffers and meshBuffers.
  // We need to keep track of them because later commands may reference them by
  // their global_id.
  std::uint32_t inputIndex = 0;
  for (const Tensor &input : inputs) {
    const target::metal::BufferRef *ref = programInputs->Get(inputIndex++);
    std::visit(utils::overloaded{
                   [&](const TensorDesc &) {
                     auto [_, inserted] =
                         this->hostBuffers.try_emplace(ref->global_id(), input);
                     LOG_ASSERT(inserted);
                     std::cout << "tensordesc" << std::endl;
                   },
                   [&](const HostBuffer &hostBuffer) {
                     auto [_, inserted] =
                         this->hostBuffers.try_emplace(ref->global_id(), input);
                     LOG_ASSERT(inserted);
                     std::cout << "host buffer" << std::endl;
                   },
                   [&](const DistributedHostBuffer &distributedHostBuffer) {
                     auto [_, inserted] =
                         this->hostBuffers.try_emplace(ref->global_id(), input);
                     LOG_ASSERT(inserted);
                     std::cout << "distributed host buffer" << std::endl;
                   },
                   [&](const MeshBuffer &meshBuffer) {
                     auto [_, inserted] = this->meshBuffers.try_emplace(
                         ref->global_id(), meshBuffer);
                     LOG_ASSERT(inserted);
                     std::cout << "mesh buffer" << std::endl;
                   },
               },
               input.as<MetalTensor>(DeviceRuntime::TTMetal));

    // If the input has an associated mesh event, we need to synchronize on it
    // before executing any command.
    this->initMeshEvents.reserve(inputs.size());
    auto meshEvent =
        input.event.asSharedPtr<distributed::MeshEvent>(DeviceRuntime::TTMetal);
    if (meshEvent) {
      this->initMeshEvents.push_back(meshEvent);
    }
  }
}

void MCQExecutor::execute(const target::metal::CommandQueue *commandQueue) {
  this->currentProgramName = commandQueue->name()->c_str();
  this->mcq = &meshDevice->mesh_command_queue(commandQueue->queue_id());

  for (const auto &mesh_event : initMeshEvents) {
    distributed::EventSynchronize(*mesh_event);
  }
  this->initMeshEvents.clear();

  for (const target::metal::Command *command : *commandQueue->commands()) {
    LOG_TRACE(logger::LogRuntimeTTMetalCommand,
              "Executing command: ", EnumNameCommandType(command->type_type()),
              "\n\t", command->debug_info()->c_str());
    execute(command);
  }
}

void MCQExecutor::execute(const target::metal::Command *command) {
  switch (command->type_type()) {
  case target::metal::CommandType::HostAllocCommand: {
    execute(command->type_as_HostAllocCommand());
    break;
  }
  case target::metal::CommandType::ReturnCommand: {
    execute(command->type_as_ReturnCommand());
    break;
  }
  case target::metal::CommandType::EnqueueProgramCommand: {
    execute(command->type_as_EnqueueProgramCommand(), command->loc()->c_str(),
            command->debug_info()->c_str());
    break;
  }
  case target::metal::CommandType::EnqueueWriteBufferCommand: {
    execute(command->type_as_EnqueueWriteBufferCommand());
    break;
  }
  case target::metal::CommandType::EnqueueReadBufferCommand: {
    execute(command->type_as_EnqueueReadBufferCommand());
    break;
  }
  case target::metal::CommandType::CreateBufferCommand: {
    execute(command->type_as_CreateBufferCommand());
    break;
  }
  case target::metal::CommandType::DeallocateBufferCommand: {
    execute(command->type_as_DeallocateBufferCommand());
    break;
  }
  case target::metal::CommandType::EnqueueRecordEventCommand: {
    execute(command->type_as_EnqueueRecordEventCommand());
    break;
  }
  case target::metal::CommandType::EnqueueWaitForEventCommand: {
    execute(command->type_as_EnqueueWaitForEventCommand());
    break;
  }
  case target::metal::CommandType::EventSynchronizeCommand: {
    execute(command->type_as_EventSynchronizeCommand());
    break;
  }
  case target::metal::CommandType::MemrefCopyCommand: {
    execute(command->type_as_MemrefCopyCommand());
    break;
  }
  case target::metal::CommandType::CpuCommand: {
    execute(command->type_as_CpuCommand());
    break;
  }
  case target::metal::CommandType::FinishCommand: {
    execute(command->type_as_FinishCommand());
    break;
  }
  case target::metal::CommandType::MeshShardCommand: {
    execute(command->type_as_MeshShardCommand());
    break;
  }
  case target::metal::CommandType::CreateGlobalSemaphoreCommand: {
    execute(command->type_as_CreateGlobalSemaphoreCommand());
    break;
  }
  case target::metal::CommandType::ResetGlobalSemaphoreCommand: {
    execute(command->type_as_ResetGlobalSemaphoreCommand());
    break;
  }
  case target::metal::CommandType::CreateLocalSemaphoreCommand: {
    execute(command->type_as_CreateLocalSemaphoreCommand());
    break;
  }
  case target::metal::CommandType::NONE: {
    LOG_FATAL("Unsupported CommandType::NONE");
    break;
  }
  }
}

void MCQExecutor::execute(const target::metal::HostAllocCommand *command) {
  // Get buffer description from the command.
  LOG_ASSERT(command->dst()->address() == 0);
  const auto *bufferDesc = command->dst()->desc();
  LOG_ASSERT(bufferDesc->shape()->size() > 0);

  // Create a TensorDesc from the buffer description.
  TensorDesc desc = createTensorDescFromBufferDesc(bufferDesc);
  const size_t size = desc.sizeBytes();

  // Default to zero-fill.
  auto data = utils::callocShared(size);
  if (!data) {
    LOG_FATAL("HostAllocCommand: Failed to allocate host memory.");
  }

  // If the command has initial data, copy it to the allocated memory. We assume
  // the size of the initial data matches the buffer size.
  if (command->data() != nullptr) {
    assert(command->data()->size() == size);
    std::memcpy(data.get(), command->data()->data(), size);
  }

  // Add the allocated buffer to hostBuffers. Use global id as the key so that
  // later commands can reference it.
  auto meshShape = meshDevice->shape();
  if (meshShape.mesh_size() == 1 || !bufferDesc->mesh()) {
    auto [_, inserted] = this->hostBuffers.try_emplace(
        command->dst()->global_id(),
        std::static_pointer_cast<void>(std::make_shared<MetalTensor>(desc)),
        data, DeviceRuntime::TTMetal);
    LOG_ASSERT(inserted);
  } else {
    auto distributedHostBufferPtr =
        std::make_shared<tt_metal::DistributedHostBuffer>(
            tt_metal::DistributedHostBuffer::create(meshDevice->shape()));
    for (const auto &coord :
         tt_metal::distributed::MeshCoordinateRange(meshShape)) {
      const auto hostBuffer = createMetalHostBuffer(
          data.get(), desc.shape, desc.sizeBytes(), desc.dataType);
      distributedHostBufferPtr->emplace_shard(
          coord, [&buffer = *hostBuffer]() { return buffer; });
    }
    auto [_, inserted] = this->hostBuffers.try_emplace(
        command->dst()->global_id(),
        std::static_pointer_cast<void>(
            std::make_shared<MetalTensor>(distributedHostBufferPtr)),
        nullptr, DeviceRuntime::TTMetal);
    LOG_ASSERT(inserted);
  }
}

void MCQExecutor::execute(const target::metal::ReturnCommand *command) {
  auto meshEvent = std::make_shared<distributed::MeshEvent>(
      this->mcq->enqueue_record_event_to_host());

  LOG_ASSERT(this->outputs.empty(),
             "Unexpected outputs, multiple returns not supported");
  this->outputs.reserve(command->results()->size());

  // For each result, check if it's a mesh buffer or a host buffer, and
  // construct the corresponding output Tensor. We identify them by looking up
  // their global_id in meshBuffers and hostBuffers.
  for (const auto *result : *command->results()) {
    auto meshBufferIter = this->meshBuffers.find(result->global_id());
    bool meshBufferFound = meshBufferIter != meshBuffers.end();
    auto hostBufferIter = this->hostBuffers.find(result->global_id());
    bool hostBufferFound = hostBufferIter != hostBuffers.end();
    LOG_ASSERT(meshBufferFound != hostBufferFound);
    if (meshBufferFound) {
      this->outputs.emplace_back(
          std::static_pointer_cast<void>(meshBufferIter->second), nullptr,
          DeviceRuntime::TTMetal, std::static_pointer_cast<void>(meshEvent));
    } else {
      this->outputs.emplace_back(hostBufferIter->second);
      this->outputs.back().event = Event(
          std::static_pointer_cast<void>(meshEvent), DeviceRuntime::TTMetal);
    }
  }
}

void MCQExecutor::execute(
    const target::metal::CreateGlobalSemaphoreCommand *command) {
  ZoneScopedN("CreateGlobalSemaphoreCommand");
  LOG_ASSERT(this->global_semaphores.find(command->ref()->global_id()) ==
                 this->global_semaphores.end(),
             "Global semaphore with id ", command->ref()->global_id(),
             " already exists.");
  auto global_semaphore = tt::tt_metal::experimental::CreateGlobalSemaphore(
      this->meshDevice, common::toCoreRangeSet(command->core_range_set()),
      command->initial_value(), tt_metal::BufferType::L1,
      deviceAddressValidator(command->ref()->address(),
                             target::BufferType::L1));
  LOG_ASSERT(global_semaphore.address() == command->ref()->address());
  this->global_semaphores.emplace(command->ref()->global_id(),
                                  std::move(global_semaphore));
}

void MCQExecutor::execute(
    const target::metal::ResetGlobalSemaphoreCommand *command) {
  ZoneScopedN("ResetGlobalSemaphoreCommand");
  LOG_ASSERT(this->global_semaphores.find(command->ref()->global_id()) !=
                 this->global_semaphores.end(),
             "Global semaphore with id ", command->ref()->global_id(),
             " does not exist.");
  this->global_semaphores.at(command->ref()->global_id())
      .reset_semaphore_value(command->value());
}

void MCQExecutor::execute(
    const target::metal::CreateLocalSemaphoreCommand *command) {
  LOG_ASSERT(this->local_semaphores.find(command->ref()->global_id()) ==
                 this->local_semaphores.end(),
             "Local semaphore with id ", command->ref()->global_id(),
             " already exists.");
  this->local_semaphores.emplace(command->ref()->global_id(),
                                 command->ref()->initial_value());
}

void MCQExecutor::execute(const target::metal::EnqueueProgramCommand *command,
                          const char *loc, const char *debugInfo) {
  ZoneScopedN("EnqueueProgramCommand");
  auto meshWorkload = distributed::MeshWorkload();
  auto deviceRange = distributed::MeshCoordinateRange(meshDevice->shape());

  // Iterate through all devices. For each device:
  // 1. Create a tt_metal::Program.
  // 2. Create kernels for each program.
  // 3. Create circular buffers for each CBRef in the program.
  // 4. Add fabric config args to kernels that need them.
  for (auto deviceCoord : deviceRange) {
    tt_metal::Program program = tt_metal::CreateProgram();

    // Iterate through all kernels in program.
    for (const target::metal::KernelConfig *kernelConfig :
         *command->program()->kernels()) {
      const target::metal::KernelSource *kernelSource =
          kernelConfig->kernel_as_KernelSource();
      LOG_ASSERT(kernelSource, "Only source kernels supported for now");
      std::string kernelSourceString(kernelSource->source()->c_str(),
                                     kernelSource->source()->size());

      tt::tt_metal::CoreRangeSet coreRangeSet =
          common::toCoreRangeSet(kernelConfig->core_range_set());
      auto createSemaphore = [&](std::uint32_t initialValue) -> std::uint32_t {
        return tt_metal::CreateSemaphore(program, coreRangeSet, initialValue);
      };

      // Generate compile time args.
      std::vector<uint32_t> compileArgs = processCompileArgs(
          kernelConfig->args()->ct_args(), command->arg_refs(), command->cbs(),
          this->hostBuffers, this->meshBuffers, this->global_semaphores,
          this->local_semaphores, this->deviceAddressValidator,
          createSemaphore);

      // Generate kernel config.
      auto generatedKernelConfig =
          createKernelConfig(kernelConfig, compileArgs);

      // Generate kernel handle.
      tt_metal::KernelHandle handle = createKernel(
          program, kernelSourceString, coreRangeSet, generatedKernelConfig,
          this->currentProgramName, debugInfo,
          kernelConfig->debug_info()->c_str(),
          kernelConfig->loc() ? kernelConfig->loc()->c_str() : nullptr);

      // Generate runtime args.
      std::vector<uint32_t> rtArgsVec = processRuntimeArgs(
          kernelConfig->args()->rt_args(), command->arg_refs(), command->cbs(),
          this->hostBuffers, this->meshBuffers, this->global_semaphores,
          this->local_semaphores, this->deviceAddressValidator,
          createSemaphore);

      // If the kernel has fabric connection config and it's NocConfig, we need
      // to append fabric config args and set them for each core.
      if (command->fabric_connection_config() &&
          kernelConfig->type_type() ==
              target::metal::KernelConfigType::NocConfig &&
          command->fabric_connection_config()->noc_index() ==
              kernelConfig->type_as_NocConfig()->noc_index()) {
        auto fabricConfigArgs = common::appendFabricConfigArgs(
            command->fabric_connection_config(), kernelConfig, program, handle,
            deviceCoord, this->meshDevice, rtArgsVec, coreRangeSet);

        for (auto core : tt::tt_metal::corerange_to_cores(coreRangeSet)) {
          tt_metal::SetRuntimeArgs(program, handle, core,
                                   fabricConfigArgs[core]);
        }
      } else {
        tt_metal::SetRuntimeArgs(program, handle, coreRangeSet, rtArgsVec);
      }
    }

    // Iterate through all CBRefs in program and create circular buffers for
    // them.
    for (const target::metal::CBRef *cbRef : *command->cbs()) {
      const target::metal::BufferDesc *bufferDesc = cbRef->buffer_ref()->desc();
      LOG_ASSERT(bufferDesc->buffer_detail_type() ==
                 target::metal::BufferDetail::MetalBuffer);
      const target::metal::MetalBuffer *metalBuffer =
          bufferDesc->buffer_detail_as_MetalBuffer();

      assert((metalBuffer->buffer_config_type() !=
                  target::metal::BufferConfig::InterleavedBufferConfig ||
              !metalBuffer->circular_buffer_config()) &&
             "Interleaved buffer configs should not have a CB config");

      // Skip init if CircularBufferConfig is not present.
      if (!metalBuffer->circular_buffer_config()) {
        continue;
      }

      tt::tt_metal::CoreRangeSet coreRangeSet = common::toCoreRangeSet(
          metalBuffer->circular_buffer_config()->core_range_set());
      tt_metal::CircularBufferConfig config =
          createCircularBufferConfig(cbRef, this->meshBuffers);
      tt_metal::CreateCircularBuffer(program, coreRangeSet, config);
    }

    // Fabric connected cores all have separate runtime args so we add a
    // separate program for each device.
    if (command->fabric_connection_config()) {
      meshWorkload.add_program(distributed::MeshCoordinateRange(deviceCoord),
                               std::move(program));
    } else {
      meshWorkload.add_program(deviceRange, std::move(program));
      break;
    }
  }

  if (perf::Env::get().enablePerfTrace) {
    auto devices = meshDevice->get_devices();
    auto meshShape = meshDevice->shape();

    for (auto &[range, program] : meshWorkload.get_programs()) {
      for (auto coord : range) {
        size_t linearIdx = coord.to_linear_index(meshShape);
        auto deviceId = devices[linearIdx]->id();
        program.set_runtime_id(getUniqueProgramRuntimeId());
        profiler::addProgramProfileHostMetadata(deviceId, program, loc);
      }
    }
  }

  distributed::EnqueueMeshWorkload(*this->mcq, meshWorkload, this->blockingCQ);

  if (perf::Env::get().enablePerfTrace) {
    ::tt::tt_metal::ReadMeshDeviceProfilerResults(*meshDevice);
  }
}

void MCQExecutor::execute(
    const target::metal::EnqueueWriteBufferCommand *command) {
  ZoneScopedN("EnqueueWriteBufferCommand");

  auto input = this->hostBuffers.at(command->src()->global_id());
  auto meshBuffer = this->meshBuffers.at(command->dst()->global_id());
  checkHostTensorSizeMatchWithMeshBufferSize(input, meshBuffer);
  writeHostTensorToMeshBuffer(this->mcq, input, meshBuffer, this->blockingCQ);
}

void MCQExecutor::execute(
    const target::metal::EnqueueReadBufferCommand *command) {
  ZoneScopedN("EnqueueReadBufferCommand");

  auto meshBuffer = this->meshBuffers.at(command->src()->global_id());
  auto output = this->hostBuffers.at(command->dst()->global_id());
  checkHostTensorSizeMatchWithMeshBufferSize(output, meshBuffer);
  readHostTensorFromMeshBuffer(this->mcq, meshBuffer, output, this->blockingCQ);
}

void MCQExecutor::execute(const target::metal::CreateBufferCommand *command) {
  ZoneScopedN("CreateBufferCommand");
  if (this->meshBuffers.find(command->ref()->global_id()) ==
      this->meshBuffers.end()) {
    this->meshBuffers[command->ref()->global_id()] =
        createMeshBufferFromBufferRef(this->meshDevice, command->ref(),
                                      this->deviceAddressValidator);
  }
}

void MCQExecutor::execute(
    const target::metal::DeallocateBufferCommand *command) {
  ZoneScopedN("DeallocateBufferCommand");
  auto meshBufferIter = this->meshBuffers.find(command->ref()->global_id());
  LOG_ASSERT(meshBufferIter != this->meshBuffers.end(), "Buffer not allocated");
  LOG_ASSERT(meshBufferIter->second != nullptr, "Buffer already deallocated");
  auto meshBuffer = meshBufferIter->second;
  meshBuffer->deallocate();
  this->meshBuffers.erase(meshBufferIter);
}

void MCQExecutor::execute(
    const target::metal::EnqueueRecordEventCommand *command) {
  ZoneScopedN("EnqueueRecordEventCommand");
  this->meshEvents[command->ref()->global_id()] =
      std::make_shared<distributed::MeshEvent>(mcq->enqueue_record_event());
}

void MCQExecutor::execute(
    const target::metal::EnqueueWaitForEventCommand *command) {
  ZoneScopedN("EnqueueWaitForEventCommand");
  auto mesh_event = this->meshEvents.at(command->ref()->global_id());
  this->mcq->enqueue_wait_for_event(*mesh_event);
}

void MCQExecutor::execute(
    const target::metal::EventSynchronizeCommand *command) {
  ZoneScopedN("EventSynchronizeCommand");
  auto mesh_event = this->meshEvents.at(command->ref()->global_id());
  distributed::EventSynchronize(*mesh_event);
}

void MCQExecutor::execute(const target::metal::MemrefCopyCommand *command) {
  auto srcIt = this->hostBuffers.find(command->src()->global_id());
  LOG_ASSERT(srcIt != this->hostBuffers.end());
  auto dstIt = this->hostBuffers.find(command->dst()->global_id());
  LOG_ASSERT(dstIt != this->hostBuffers.end());
  ttmetal::memcpy(
      dstIt->second, createTensorDescFromBufferDesc(command->dst()->desc()),
      srcIt->second, createTensorDescFromBufferDesc(command->src()->desc()));
}

void MCQExecutor::execute(const target::metal::CpuCommand *command) {
  std::vector<std::vector<int64_t>> allSizesAndStrides;
  auto dataFuncPtr =
      std::function<void *(const tt::target::metal::BufferRef *)>(
          [this](const tt::target::metal::BufferRef *ref) -> void * {
            auto it = hostBuffers.find(ref->global_id());
            LOG_ASSERT(
                it != hostBuffers.end(),
                "Cannot invoke cpu op on tensor which is not in cpu tensors.");
            const Tensor &tens = it->second;
            return tens.data.get();
          });

  auto packedInputs = tt::runtime::common::packTensors(
      command->ins(), dataFuncPtr, allSizesAndStrides);

  common::WrappedFunc func =
      dylibManager.getFunc(command->dylib_id(), command->func_name()->c_str());

  // Call the CPU function and get returned outputs.
  common::WrappedTensor *outputArray = func(packedInputs.data());

  common::CreateTensorCallbackType<Tensor, tt::target::metal::BufferRef>
      createTensor = [](const tt::target::metal::BufferRef *ref,
                        std::shared_ptr<void> dataPtr) -> Tensor {
    TensorDesc desc = createTensorDescFromBufferDesc(ref->desc());
    return ttmetal::createBorrowedHostTensor(dataPtr, desc);
  };

  // Unpack outputs and insert into hostBuffers.
  auto outputs = common::unpackTensors<Tensor>(
      outputArray, command->outs()->size(), command->outs(), createTensor);

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto [_, inserted] = hostBuffers.try_emplace(
        command->outs()->Get(i)->global_id(), std::move(outputs[i]));
    LOG_ASSERT(inserted);
  }
}

void MCQExecutor::execute(const target::metal::FinishCommand *) {
  ZoneScopedN("FinishCommand");
  distributed::Finish(*this->mcq);
}

void MCQExecutor::execute(const target::metal::MeshShardCommand *command) {
  LOG_ASSERT(command->src()->desc()->buffer_detail_type() ==
                 tt::target::metal::BufferDetail::SystemBuffer,
             "MeshShardCommand requires system memory as input");
  LOG_ASSERT(command->dst()->desc()->buffer_detail_type() ==
                 tt::target::metal::BufferDetail::SystemBuffer,
             "MeshShardCommand requires system memory as output");
  const auto dstDataType = command->dst()->desc()->data_type();
  const auto *fbTensorShape = command->src()->desc()->shape();
  const std::vector<size_t> tensorShape(fbTensorShape->begin(),
                                        fbTensorShape->end());
  const auto *fbShardDims = command->shard_dims();
  const std::vector<int64_t> meshShardDims(fbShardDims->begin(),
                                           fbShardDims->end());
  const auto meshShardType = command->shard_type();

  auto srcBufferIter = hostBuffers.find(command->src()->global_id());
  LOG_ASSERT(srcBufferIter != hostBuffers.end(),
             "Input host buffer not found.");
  const Tensor input = srcBufferIter->second;

  auto putHostTensor = [&](const Tensor &output) -> void {
    LOG_ASSERT(hostBuffers.find(command->dst()->global_id()) ==
                   hostBuffers.end(),
               "Output host buffer already exists.");
    auto [_, inserted] =
        hostBuffers.try_emplace(command->dst()->global_id(), output);
    LOG_ASSERT(inserted);
  };

  if (meshShardType == target::MeshShardType::Identity) {
    // Identity: copy from src tensor to dst tensor
    putHostTensor(input);
    return;
  }

  if (command->shard_direction() ==
      target::MeshShardDirection::FullToShardShape) {
    auto distributedHostBufferPtr = meshshard_utils::tensorFullToShard(
        input, meshDevice->shape(), dstDataType, tensorShape, meshShardType,
        meshShardDims);
    putHostTensor(
        Tensor(std::static_pointer_cast<void>(
                   std::make_shared<MetalTensor>(distributedHostBufferPtr)),
               nullptr, DeviceRuntime::TTMetal));
  } else {
    auto hostBufferPtr = meshshard_utils::tensorShardToFull(
        input, meshDevice->shape(), dstDataType, tensorShape, meshShardType,
        meshShardDims);
    putHostTensor(Tensor(std::static_pointer_cast<void>(
                             std::make_shared<MetalTensor>(hostBufferPtr)),
                         nullptr, DeviceRuntime::TTMetal));
  }
}

std::vector<Tensor>
executeMeshDeviceProgram(distributed::MeshDevice *meshDevice,
                         const target::metal::DeviceProgram *program,
                         const std::vector<Tensor> &inputs,
                         common::DylibManager &&dylibs) {
  LOG_ASSERT(program->command_queues()->size() == 1, "Only one MCQ supported");

  MCQExecutor executor(meshDevice, program->inputs(), inputs, std::move(dylibs),
                       debug::Env::get().blockingCQ);
  for (const target::metal::CommandQueue *cq : *program->command_queues()) {
    FrameMark;
    ZoneScoped;
    std::string zoneName =
        "executeCommandQueue_mcq_" + std::to_string(cq->queue_id());
    ZoneName(zoneName.c_str(), zoneName.size());
    perf::Env::get().tracyLogProgramMetadata(
        perf::Env::get().tracyProgramMetadata);

    executor.execute(cq);

    FrameMark;
  }

  return executor.getOutputs();
}
} // namespace tt::runtime::ttmetal
