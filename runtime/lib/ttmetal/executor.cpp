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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tt::runtime::ttmetal {

namespace target = ::tt::target;
namespace tt_metal = ::tt::tt_metal;
namespace distributed = ::tt::tt_metal::distributed;

namespace {
struct CachedKernelBinding {
  tt_metal::KernelHandle handle;
  tt_metal::CoreRangeSet coreRangeSet;
  std::vector<std::uint32_t> compileArgs;
  std::vector<std::uint32_t> commonRuntimeArgs;
  std::vector<std::uint32_t> runtimeArgs;
};

struct CachedCircularBufferBinding {
  tt_metal::CBHandle handle;
  std::uint32_t bufferGlobalId;
};

struct CachedProgramBinding {
  std::vector<CachedKernelBinding> kernels;
  std::vector<CachedCircularBufferBinding> circularBuffers;
};

struct CachedMeshWorkloadState {
  std::unordered_map<distributed::MeshCoordinateRange, CachedProgramBinding>
      programs;
};

using CachedMeshWorkload = tt_metal::program_cache::detail::CachedMeshWorkload<
    CachedMeshWorkloadState>;

struct ReusableInputKey {
  std::uint32_t deviceId;
  std::uint64_t binaryId;
  std::uint32_t programIndex;
  std::uint32_t deviceProgramIndex;
  std::uint32_t inputIndex;
  std::uint32_t destinationGlobalId;

  bool operator==(const ReusableInputKey &other) const = default;
};

struct ReusableInputKeyHash {
  std::size_t operator()(const ReusableInputKey &key) const {
    std::size_t hash = std::hash<std::uint32_t>{}(key.deviceId);
    auto combine = [&hash](auto value) {
      hash ^= std::hash<decltype(value)>{}(value) + 0x9e3779b9 + (hash << 6) +
              (hash >> 2);
    };
    combine(key.binaryId);
    combine(key.programIndex);
    combine(key.deviceProgramIndex);
    combine(key.inputIndex);
    combine(key.destinationGlobalId);
    return hash;
  }
};

struct ReusableInputCache {
  std::mutex mutex;
  std::unordered_map<ReusableInputKey, MeshBuffer, ReusableInputKeyHash>
      buffers;
  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
  std::uint64_t uploadedBytes = 0;
};

std::uint64_t bufferSpanPerBank(const tt_metal::Buffer *buffer) {
  const auto &distribution = buffer->buffer_distribution_spec();
  if (distribution) {
    return distribution->max_num_dev_pages_per_core() *
           buffer->aligned_page_size();
  }
  return buffer->aligned_size_per_bank();
}

std::optional<tt_metal::CoreRangeSet>
getBufferCoreRangeSet(const tt_metal::Buffer *buffer) {
  if (buffer->buffer_distribution_spec()) {
    return tt_metal::CoreRangeSet(
        buffer->buffer_distribution_spec()->cores_with_data());
  }
  if (buffer->has_shard_spec()) {
    return buffer->shard_spec().grid();
  }
  return std::nullopt;
}

bool buffersOverlap(const MeshBuffer &lhs, const MeshBuffer &rhs) {
  const tt_metal::Buffer *lhsBuffer = lhs->get_reference_buffer();
  const tt_metal::Buffer *rhsBuffer = rhs->get_reference_buffer();
  if (lhsBuffer->buffer_type() != rhsBuffer->buffer_type()) {
    return false;
  }

  const std::uint64_t lhsBegin = lhsBuffer->address();
  const std::uint64_t lhsEnd = lhsBegin + bufferSpanPerBank(lhsBuffer);
  const std::uint64_t rhsBegin = rhsBuffer->address();
  const std::uint64_t rhsEnd = rhsBegin + bufferSpanPerBank(rhsBuffer);
  if (lhsBegin >= rhsEnd || rhsBegin >= lhsEnd) {
    return false;
  }

  // L1 addresses are core-local. Equal ranges on disjoint shard grids do not
  // overlap, while an unsharded L1 buffer is conservatively treated as using
  // every participating core.
  if (lhsBuffer->is_l1()) {
    const auto lhsCores = getBufferCoreRangeSet(lhsBuffer);
    const auto rhsCores = getBufferCoreRangeSet(rhsBuffer);
    if (lhsCores && rhsCores) {
      return lhsCores->intersects(*rhsCores);
    }
  }
  return true;
}

class ReusableInputRegistry {
public:
  static ReusableInputRegistry &instance() {
    static ReusableInputRegistry registry;
    return registry;
  }

  void insert(std::uint32_t deviceId, const MeshBuffer &buffer,
              const std::shared_ptr<ReusableInputCache> &cache) {
    std::lock_guard<std::mutex> lock(mutex);
    auto &deviceBuffers = buffers[deviceId];
    for (auto it = deviceBuffers.begin(); it != deviceBuffers.end();) {
      auto existing = it->lock();
      if (!existing) {
        it = deviceBuffers.erase(it);
        continue;
      }
      if (existing.get() == buffer.get()) {
        return;
      }
      ++it;
    }
    deviceBuffers.push_back(buffer);

    auto &deviceCaches = caches[deviceId];
    for (auto it = deviceCaches.begin(); it != deviceCaches.end();) {
      auto existing = it->lock();
      if (!existing) {
        it = deviceCaches.erase(it);
        continue;
      }
      if (existing.get() == cache.get()) {
        return;
      }
      ++it;
    }
    deviceCaches.push_back(cache);
  }

  void assertNoOverlap(std::uint32_t deviceId, const MeshBuffer &buffer) {
    std::lock_guard<std::mutex> lock(mutex);
    auto deviceIt = buffers.find(deviceId);
    if (deviceIt == buffers.end()) {
      return;
    }

    for (auto it = deviceIt->second.begin(); it != deviceIt->second.end();) {
      auto reusable = it->lock();
      if (!reusable) {
        it = deviceIt->second.erase(it);
        continue;
      }
      LOG_ASSERT(reusable.get() == buffer.get() ||
                     !buffersOverlap(reusable, buffer),
                 "D2M transient allocation overlaps a reusable input buffer; "
                 "the compiled memory plan does not leave enough device "
                 "memory for input reuse");
      ++it;
    }
  }

  void clearDevice(std::uint32_t deviceId) {
    std::vector<std::shared_ptr<ReusableInputCache>> liveCaches;
    {
      std::lock_guard<std::mutex> lock(mutex);
      auto cacheIt = caches.find(deviceId);
      if (cacheIt != caches.end()) {
        liveCaches.reserve(cacheIt->second.size());
        for (const auto &cache : cacheIt->second) {
          if (auto liveCache = cache.lock()) {
            liveCaches.push_back(std::move(liveCache));
          }
        }
        caches.erase(cacheIt);
      }
      buffers.erase(deviceId);
    }

    for (const auto &cache : liveCaches) {
      std::lock_guard<std::mutex> lock(cache->mutex);
      for (auto it = cache->buffers.begin(); it != cache->buffers.end();) {
        if (it->first.deviceId == deviceId) {
          it = cache->buffers.erase(it);
        } else {
          ++it;
        }
      }
    }
  }

private:
  std::mutex mutex;
  std::unordered_map<std::uint32_t,
                     std::vector<std::weak_ptr<distributed::MeshBuffer>>>
      buffers;
  std::unordered_map<std::uint32_t,
                     std::vector<std::weak_ptr<ReusableInputCache>>>
      caches;
};

struct ProgramInputProvenance {
  std::uint32_t inputIndex;
  std::shared_ptr<ReusableInputCache> cache;
};

class MCQExecutor {
public:
  MCQExecutor(
      distributed::MeshDevice *meshDevice,
      const flatbuffers::Vector<
          flatbuffers::Offset<tt::target::metal::BufferRef>> *programInputs,
      const std::vector<Tensor> &inputs, common::DylibManager &&dylibManager,
      bool blockingCQ, std::uint32_t deviceId, std::uint64_t binaryId,
      std::uint32_t programIndex, std::uint32_t deviceProgramIndex);

  const std::vector<Tensor> &getOutputs() const { return outputs; }

  void execute(const target::metal::CommandQueue *commandQueue);

private:
  void analyzeReusableInputs(const target::metal::CommandQueue *commandQueue);
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

  tt_metal::program_cache::detail::ProgramCacheKey
  createProgramCacheKey(const target::metal::EnqueueProgramCommand *command,
                        std::uint32_t commandIndex) const;
  void refreshCachedMeshWorkload(
      CachedMeshWorkload &cached,
      const target::metal::EnqueueProgramCommand *command);
  void enqueueMeshWorkload(distributed::MeshWorkload &workload,
                           const char *loc);

  ReusableInputKey
  createReusableInputKey(const ProgramInputProvenance &input,
                         std::uint32_t destinationGlobalId) const;
  void
  bindReusableInput(const target::metal::EnqueueWriteBufferCommand *command,
                    const ProgramInputProvenance &input);

  std::uint64_t generateUniqueProgramRuntimeId() {
    return nextProgramRuntimeId++;
  }

private:
  distributed::MeshDevice *meshDevice;
  std::vector<std::shared_ptr<distributed::MeshEvent>> initMeshEvents;
  std::unordered_map<std::uint32_t, std::shared_ptr<distributed::MeshBuffer>>
      meshBuffers;
  std::unordered_map<std::uint32_t, tt_metal::GlobalSemaphore>
      global_semaphores;

  // Local semaphores. Indexed by global_id. We only store their initial value
  // here and lookup during their creation.
  std::unordered_map<std::uint32_t, std::uint32_t> local_semaphore_initializer;

  // Buffers that live on the host. Indexed by global_id.
  std::unordered_map<std::uint32_t, Tensor> hostBuffers;
  // Maps host buffers and their derived memref views back to program inputs.
  std::unordered_map<std::uint32_t, ProgramInputProvenance>
      hostBufferInputProvenance;
  // Host buffers and copies that are unnecessary once every device write for
  // their immutable source input has a reusable cache entry.
  std::unordered_set<std::uint32_t> skippedHostBufferIds;
  std::unordered_set<std::uint32_t> skippedHostCopyInputIndices;
  // Destination aliases that refer to tensor-owned reusable buffers. Their
  // scheduled deallocation removes the alias without freeing the cache entry.
  std::unordered_set<std::uint32_t> reusableBufferAliases;
  std::unordered_map<std::uint32_t, std::shared_ptr<distributed::MeshEvent>>
      meshEvents;
  std::vector<Tensor> outputs;
  distributed::MeshCommandQueue *mcq;
  bool blockingCQ;
  const char *currentProgramName;
  DeviceAddressValidator deviceAddressValidator;
  common::DylibManager dylibManager;
  std::uint32_t deviceId;
  std::uint64_t binaryId;
  std::uint32_t programIndex;
  std::uint32_t deviceProgramIndex;
  std::uint64_t nextProgramRuntimeId = 10000; // Start at a greppable number.
  std::uint32_t nextEnqueueProgramIndex = 0;
};
} // namespace

std::shared_ptr<void> createReusableInputCache() {
  return std::make_shared<ReusableInputCache>();
}

TensorReuseStats
getReusableInputCacheStats(const std::shared_ptr<void> &reusableInputCache) {
  if (!reusableInputCache) {
    return {};
  }

  auto cache = std::static_pointer_cast<ReusableInputCache>(reusableInputCache);
  std::lock_guard<std::mutex> lock(cache->mutex);
  return TensorReuseStats{cache->hits, cache->misses, cache->uploadedBytes,
                          static_cast<std::uint64_t>(cache->buffers.size())};
}

void clearReusableInputCachesForDevice(std::uint32_t deviceId) {
  ReusableInputRegistry::instance().clearDevice(deviceId);
}

MCQExecutor::MCQExecutor(
    distributed::MeshDevice *meshDevice,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::metal::BufferRef>>
        *programInputs,
    const std::vector<Tensor> &inputs, common::DylibManager &&dylibManager,
    bool blockingCQ, std::uint32_t deviceId, std::uint64_t binaryId,
    std::uint32_t programIndex, std::uint32_t deviceProgramIndex)
    : meshDevice(meshDevice), blockingCQ(blockingCQ),
      deviceAddressValidator(meshDevice->get_devices().at(0)),
      dylibManager(std::move(dylibManager)), deviceId(deviceId),
      binaryId(binaryId), programIndex(programIndex),
      deviceProgramIndex(deviceProgramIndex) {
  initMeshEvents.reserve(inputs.size());

  std::uint32_t inputIndex = 0;
  for (const Tensor &input : inputs) {
    const std::uint32_t currentInputIndex = inputIndex++;
    const target::metal::BufferRef *ref = programInputs->Get(currentInputIndex);
    std::visit(utils::overloaded{
                   [&](const std::uint32_t &) {
                     auto [_, inserted] =
                         this->hostBuffers.try_emplace(ref->global_id(), input);
                     LOG_ASSERT(inserted);
                   },
                   [&](const TensorDesc &) {
                     auto [_, inserted] =
                         hostBuffers.try_emplace(ref->global_id(), input);
                     LOG_ASSERT(inserted);
                   },
                   [&](const HostBuffer &hostBuffer) {
                     auto [_, inserted] =
                         hostBuffers.try_emplace(ref->global_id(), input);
                     LOG_ASSERT(inserted);
                   },
                   [&](const DistributedHostBuffer &distributedHostBuffer) {
                     auto [_, inserted] =
                         hostBuffers.try_emplace(ref->global_id(), input);
                     LOG_ASSERT(inserted);
                   },
                   [&](const MeshBuffer &meshBuffer) {
                     auto [_, inserted] =
                         meshBuffers.try_emplace(ref->global_id(), meshBuffer);
                     LOG_ASSERT(inserted);
                   },
               },
               input.as<MetalTensor>(DeviceRuntime::TTMetal));

    std::shared_ptr<ReusableInputCache> reusableInputCache;
    if (input.metadata) {
      std::lock_guard<std::mutex> lock(input.metadata->mutex);
      if (input.metadata->reusable) {
        LOG_ASSERT(input.metadata->reusableInputCache,
                   "Reusable tensor cache is not initialized");
        reusableInputCache = std::static_pointer_cast<ReusableInputCache>(
            input.metadata->reusableInputCache);
      }
    }
    if (reusableInputCache &&
        hostBuffers.find(ref->global_id()) != hostBuffers.end()) {
      hostBufferInputProvenance.try_emplace(
          ref->global_id(),
          ProgramInputProvenance{currentInputIndex,
                                 std::move(reusableInputCache)});
    }

    auto meshEvent =
        input.event.asSharedPtr<distributed::MeshEvent>(DeviceRuntime::TTMetal);
    if (meshEvent) {
      initMeshEvents.push_back(meshEvent);
    }
  }
}

void MCQExecutor::analyzeReusableInputs(
    const target::metal::CommandQueue *commandQueue) {
  struct ReusePlan {
    ProgramInputProvenance input;
    std::unordered_set<std::uint32_t> destinationGlobalIds;
    std::unordered_set<std::uint32_t> derivedHostBufferIds;
    bool requiresHostMaterialization = false;
  };

  auto provenance = hostBufferInputProvenance;
  std::unordered_map<std::uint32_t, ReusePlan> plans;
  for (const auto &[globalId, input] : provenance) {
    plans.try_emplace(input.inputIndex, ReusePlan{input, {}, {}, false});
  }

  auto markHostUse = [&](const target::metal::BufferRef *ref) {
    auto source = provenance.find(ref->global_id());
    if (source != provenance.end()) {
      plans.at(source->second.inputIndex).requiresHostMaterialization = true;
    }
  };

  for (const target::metal::Command *command : *commandQueue->commands()) {
    switch (command->type_type()) {
    case target::metal::CommandType::HostAllocCommand: {
      provenance.erase(command->type_as_HostAllocCommand()->dst()->global_id());
      break;
    }
    case target::metal::CommandType::MemrefCopyCommand: {
      const auto *copy = command->type_as_MemrefCopyCommand();
      auto source = provenance.find(copy->src()->global_id());
      if (source == provenance.end()) {
        provenance.erase(copy->dst()->global_id());
        break;
      }
      provenance.insert_or_assign(copy->dst()->global_id(), source->second);
      plans.at(source->second.inputIndex)
          .derivedHostBufferIds.insert(copy->dst()->global_id());
      break;
    }
    case target::metal::CommandType::EnqueueWriteBufferCommand: {
      const auto *write = command->type_as_EnqueueWriteBufferCommand();
      auto source = provenance.find(write->src()->global_id());
      if (source != provenance.end()) {
        plans.at(source->second.inputIndex)
            .destinationGlobalIds.insert(write->dst()->global_id());
      }
      break;
    }
    case target::metal::CommandType::EnqueueReadBufferCommand: {
      const auto *read = command->type_as_EnqueueReadBufferCommand();
      markHostUse(read->dst());
      provenance.erase(read->dst()->global_id());
      break;
    }
    case target::metal::CommandType::CpuCommand: {
      const auto *cpu = command->type_as_CpuCommand();
      for (const target::metal::BufferRef *input : *cpu->ins()) {
        markHostUse(input);
      }
      for (const target::metal::BufferRef *output : *cpu->outs()) {
        provenance.erase(output->global_id());
      }
      break;
    }
    case target::metal::CommandType::MeshShardCommand: {
      const auto *meshShard = command->type_as_MeshShardCommand();
      auto source = provenance.find(meshShard->src()->global_id());
      if (source == provenance.end()) {
        provenance.erase(meshShard->dst()->global_id());
        break;
      }
      provenance.insert_or_assign(meshShard->dst()->global_id(),
                                  source->second);
      plans.at(source->second.inputIndex)
          .derivedHostBufferIds.insert(meshShard->dst()->global_id());
      break;
    }
    case target::metal::CommandType::ReturnCommand: {
      for (const target::metal::BufferRef *result :
           *command->type_as_ReturnCommand()->results()) {
        markHostUse(result);
      }
      break;
    }
    default:
      break;
    }
  }

  hostBufferInputProvenance = provenance;

  for (const auto &[inputIndex, plan] : plans) {
    if (!plan.input.cache || plan.requiresHostMaterialization ||
        plan.destinationGlobalIds.empty()) {
      continue;
    }

    const auto &cache = plan.input.cache;
    std::lock_guard<std::mutex> lock(cache->mutex);
    const bool fullyCached = std::all_of(
        plan.destinationGlobalIds.begin(), plan.destinationGlobalIds.end(),
        [&](std::uint32_t destination) {
          return cache->buffers.contains(
              createReusableInputKey(plan.input, destination));
        });
    if (!fullyCached) {
      continue;
    }

    // Bind cache hits before command execution so the corresponding
    // CreateBufferCommand becomes a no-op. Otherwise every reuse would still
    // allocate and immediately release the compiled transient destination
    // before EnqueueWriteBufferCommand replaced it with the cached buffer.
    for (std::uint32_t destination : plan.destinationGlobalIds) {
      const MeshBuffer &buffer =
          cache->buffers.at(createReusableInputKey(plan.input, destination));
      auto [existing, inserted] = meshBuffers.try_emplace(destination, buffer);
      LOG_ASSERT(inserted || existing->second.get() == buffer.get(),
                 "Reusable input destination is already bound to a different "
                 "device buffer");
      reusableBufferAliases.insert(destination);
    }

    skippedHostCopyInputIndices.insert(inputIndex);
    skippedHostBufferIds.insert(plan.derivedHostBufferIds.begin(),
                                plan.derivedHostBufferIds.end());
  }
}

void MCQExecutor::execute(const target::metal::CommandQueue *commandQueue) {
  currentProgramName = commandQueue->name()->c_str();
  mcq = &meshDevice->mesh_command_queue(commandQueue->queue_id());
  analyzeReusableInputs(commandQueue);

  for (const auto &mesh_event : initMeshEvents) {
    distributed::EventSynchronize(*mesh_event);
  }
  initMeshEvents.clear();

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
  if (skippedHostBufferIds.contains(command->dst()->global_id())) {
    return;
  }

  LOG_ASSERT(command->dst()->address() == 0);
  const auto *bufferDesc = command->dst()->desc();

  TensorDesc desc = createTensorDescFromBufferDesc(bufferDesc);
  const size_t size = desc.sizeBytes();

  // Default to zero-fill.
  auto data = utils::callocShared(size);
  if (!data) {
    LOG_FATAL("HostAllocCommand: Failed to allocate host memory.");
  }
  if (command->data() != nullptr) {
    assert(command->data()->size() == size);
    std::memcpy(data.get(), command->data()->data(), size);
  }

  auto meshShape = meshDevice->shape();
  if (meshShape.mesh_size() == 1 || !bufferDesc->mesh()) {
    auto [_, inserted] = hostBuffers.try_emplace(
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
    auto [_, inserted] = hostBuffers.try_emplace(
        command->dst()->global_id(),
        std::static_pointer_cast<void>(
            std::make_shared<MetalTensor>(distributedHostBufferPtr)),
        nullptr, DeviceRuntime::TTMetal);
    LOG_ASSERT(inserted);
  }
}

void MCQExecutor::execute(const target::metal::ReturnCommand *command) {
  auto meshEvent = std::make_shared<distributed::MeshEvent>(
      mcq->enqueue_record_event_to_host());

  LOG_ASSERT(outputs.empty(),
             "Unexpected outputs, multiple returns not supported");
  outputs.reserve(command->results()->size());
  for (const auto *result : *command->results()) {
    auto meshBufferIter = meshBuffers.find(result->global_id());
    bool meshBufferFound = meshBufferIter != meshBuffers.end();
    auto hostBufferIter = hostBuffers.find(result->global_id());
    bool hostBufferFound = hostBufferIter != hostBuffers.end();
    LOG_ASSERT(meshBufferFound != hostBufferFound);
    if (meshBufferFound) {
      outputs.emplace_back(
          std::static_pointer_cast<void>(meshBufferIter->second), nullptr,
          DeviceRuntime::TTMetal, std::static_pointer_cast<void>(meshEvent));
    } else {
      outputs.emplace_back(hostBufferIter->second);
      outputs.back().event = Event(std::static_pointer_cast<void>(meshEvent),
                                   DeviceRuntime::TTMetal);
    }
  }
}

void MCQExecutor::execute(
    const target::metal::CreateGlobalSemaphoreCommand *command) {
  ZoneScopedN("CreateGlobalSemaphoreCommand");
  LOG_ASSERT(global_semaphores.find(command->ref()->global_id()) ==
                 global_semaphores.end(),
             "Global semaphore with id ", command->ref()->global_id(),
             " already exists.");
  auto global_semaphore = tt::tt_metal::experimental::CreateGlobalSemaphore(
      meshDevice, common::toCoreRangeSet(command->core_range_set()),
      command->initial_value(), tt_metal::BufferType::L1,
      deviceAddressValidator(command->ref()->address(),
                             target::BufferType::L1));
  LOG_ASSERT(global_semaphore.address() == command->ref()->address());
  global_semaphores.emplace(command->ref()->global_id(),
                            std::move(global_semaphore));
}

void MCQExecutor::execute(
    const target::metal::ResetGlobalSemaphoreCommand *command) {
  ZoneScopedN("ResetGlobalSemaphoreCommand");
  LOG_ASSERT(global_semaphores.find(command->ref()->global_id()) !=
                 global_semaphores.end(),
             "Global semaphore with id ", command->ref()->global_id(),
             " does not exist.");
  global_semaphores.at(command->ref()->global_id())
      .reset_semaphore_value(command->value());
}

void MCQExecutor::execute(
    const target::metal::CreateLocalSemaphoreCommand *command) {
  LOG_ASSERT(
      this->local_semaphore_initializer.find(command->ref()->global_id()) ==
          this->local_semaphore_initializer.end(),
      "Local semaphore with id ", command->ref()->global_id(),
      " already exists.");
  this->local_semaphore_initializer.emplace(command->ref()->global_id(),
                                            command->ref()->initial_value());
}

tt_metal::program_cache::detail::ProgramCacheKey
MCQExecutor::createProgramCacheKey(
    const target::metal::EnqueueProgramCommand *command,
    std::uint32_t commandIndex) const {
  std::string canonical = "ttmlir.ttmetal:";
  canonical += std::to_string(binaryId) + ":" + std::to_string(programIndex) +
               ":" + std::to_string(deviceProgramIndex) + ":" +
               std::to_string(commandIndex);

  // A scalar compile argument changes the materialized kernel, while runtime
  // scalars are refreshed in place on cache hits.
  for (const target::metal::KernelConfig *kernelConfig :
       *command->program()->kernels()) {
    const auto *compileArgs = kernelConfig->args()->ct_args();
    if (!compileArgs) {
      continue;
    }
    for (const target::metal::KernelArg *kernelArg : *compileArgs) {
      if (kernelArg->arg_type() !=
          target::metal::KernelArgType::KernelArgScalar) {
        continue;
      }

      const auto *arg = kernelArg->arg_as_KernelArgScalar();
      LOG_ASSERT(command->arg_refs_type()->Get(arg->operand_idx()) ==
                 target::metal::ArgRef::BufferRef);
      const auto *buffer = reinterpret_cast<const target::metal::BufferRef *>(
          command->arg_refs()->Get(arg->operand_idx()));
      const MetalTensor &metalTensor =
          hostBuffers.at(buffer->global_id())
              .as<MetalTensor>(DeviceRuntime::TTMetal);
      canonical += ":" + std::to_string(std::get<std::uint32_t>(metalTensor));
    }
  }

  return {std::hash<std::string>{}(canonical), std::move(canonical)};
}

void MCQExecutor::refreshCachedMeshWorkload(
    CachedMeshWorkload &cached,
    const target::metal::EnqueueProgramCommand *command) {
  auto &programs = cached.workload.get_programs();
  LOG_ASSERT(programs.size() == cached.shared_variables.programs.size(),
             "Cached program binding count mismatch");

  for (auto &[range, program] : programs) {
    const CachedProgramBinding &programBinding =
        cached.shared_variables.programs.at(range);
    LOG_ASSERT(programBinding.kernels.size() ==
                   command->program()->kernels()->size(),
               "Cached kernel binding count mismatch");

    std::function<std::uint32_t(std::uint32_t)> preserveLocalSemaphores;
    for (std::size_t i = 0; i < programBinding.kernels.size(); ++i) {
      const target::metal::KernelConfig *kernelConfig =
          command->program()->kernels()->Get(i);
      const CachedKernelBinding &binding = programBinding.kernels[i];

      std::vector<std::uint32_t> compileArgs = processCompileArgs(
          kernelConfig->args()->ct_args(), command->arg_refs_type(),
          command->arg_refs(), meshBuffers, global_semaphores,
          local_semaphore_initializer, command->cbs(), deviceAddressValidator,
          std::function<std::uint32_t(std::uint32_t)>(), hostBuffers,
          &binding.compileArgs);
      LOG_ASSERT(compileArgs == binding.compileArgs,
                 "Cached compile-time kernel arguments changed between "
                 "submissions");

      std::vector<std::uint32_t> commonRuntimeArgs = refreshRuntimeArgs(
          binding.commonRuntimeArgs, kernelConfig->args()->crt_args(),
          command->arg_refs_type(), command->arg_refs(), meshBuffers,
          global_semaphores, local_semaphore_initializer, command->cbs(),
          deviceAddressValidator, preserveLocalSemaphores, hostBuffers);
      tt_metal::RuntimeArgsData &cachedCommonRuntimeArgs =
          tt_metal::GetCommonRuntimeArgs(program, binding.handle);
      LOG_ASSERT(cachedCommonRuntimeArgs.size() == commonRuntimeArgs.size(),
                 "Cached common runtime argument count mismatch");
      std::copy(commonRuntimeArgs.begin(), commonRuntimeArgs.end(),
                cachedCommonRuntimeArgs.data());

      std::vector<std::uint32_t> runtimeArgs = refreshRuntimeArgs(
          binding.runtimeArgs, kernelConfig->args()->rt_args(),
          command->arg_refs_type(), command->arg_refs(), meshBuffers,
          global_semaphores, local_semaphore_initializer, command->cbs(),
          deviceAddressValidator, preserveLocalSemaphores, hostBuffers);
      for (const tt_metal::CoreCoord core :
           tt_metal::corerange_to_cores(binding.coreRangeSet)) {
        tt_metal::RuntimeArgsData &cachedRuntimeArgs =
            tt_metal::GetRuntimeArgs(program, binding.handle, core);
        LOG_ASSERT(cachedRuntimeArgs.size() >= runtimeArgs.size(),
                   "Cached runtime argument count mismatch");
        std::copy(runtimeArgs.begin(), runtimeArgs.end(),
                  cachedRuntimeArgs.data());
      }
    }

    for (const CachedCircularBufferBinding &binding :
         programBinding.circularBuffers) {
      const auto &meshBuffer = meshBuffers.at(binding.bufferGlobalId);
      tt_metal::UpdateDynamicCircularBufferAddress(
          program, binding.handle, *meshBuffer->get_reference_buffer());
    }
  }
}

void MCQExecutor::enqueueMeshWorkload(distributed::MeshWorkload &workload,
                                      const char *loc) {
  if (perf::Env::get().enablePerfTrace) {
    if (loc) {
      perf::Env::get().tracyLogOpLocation(std::string(loc));
    }

    auto devices = meshDevice->get_devices();
    auto meshShape = meshDevice->shape();

    // All programs in a workload represent one serialized enqueue operation.
    auto opId = generateUniqueProgramRuntimeId();
    for (auto &[range, program] : workload.get_programs()) {
      program.set_runtime_id(opId);
      for (auto coord : range) {
        size_t linearIdx = coord.to_linear_index(meshShape);
        auto deviceId = devices[linearIdx]->id();
        profiler::addProgramProfileHostMetadata(deviceId, program, loc);
      }
    }
  }

  distributed::EnqueueMeshWorkload(*mcq, workload, blockingCQ);
}

void MCQExecutor::execute(const target::metal::EnqueueProgramCommand *command,
                          const char *loc, const char *debugInfo) {
  ZoneScopedN("EnqueueProgramCommand");
  if (loc) {
    std::string_view zoneText(loc);
    constexpr std::string_view namePrefix = "loc(\"";
    constexpr std::string_view nameSuffix = "\")";
    if (zoneText.size() >= namePrefix.size() + nameSuffix.size() &&
        zoneText.substr(0, namePrefix.size()) == namePrefix &&
        zoneText.substr(zoneText.size() - nameSuffix.size()) == nameSuffix) {
      zoneText = zoneText.substr(namePrefix.size(), zoneText.size() -
                                                        namePrefix.size() -
                                                        nameSuffix.size());
    }
    ZoneText(zoneText.data(), zoneText.size());
  }
  LOG_TRACE(logger::LogRuntimeTTMetalCommand, "Executing program: ", loc, "\n",
            debugInfo);

  const std::uint32_t commandIndex = nextEnqueueProgramIndex++;
  auto &programCache = meshDevice->get_program_cache();

  const bool cacheEligible = programCache.is_enabled();
  tt_metal::program_cache::detail::ProgramCacheKey programKey;
  if (cacheEligible) {
    programKey = createProgramCacheKey(command, commandIndex);
    if (programCache.contains(programKey)) {
      auto &cachedProgramFactory = programCache.get(programKey);
      auto &cached =
          cachedProgramFactory.cached_program.get<CachedMeshWorkload>();
      refreshCachedMeshWorkload(cached, command);
      enqueueMeshWorkload(cached.workload, loc);
      return;
    }

    LOG_ASSERT(programCache.cache_misses_allowed(),
               "TTMetal program cache miss occurred while misses are disabled");
  }

  CachedMeshWorkloadState cacheState;

  auto meshWorkload = distributed::MeshWorkload();
  auto deviceRange = distributed::MeshCoordinateRange(meshDevice->shape());
  for (auto deviceCoord : deviceRange) {
    CachedProgramBinding cacheProgramBinding;
    tt_metal::Program program = tt_metal::CreateProgram();
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

      auto materializedKernelConfig = createKernelConfig(
          kernelConfig, command->arg_refs_type(), command->arg_refs(),
          meshBuffers, global_semaphores, local_semaphore_initializer,
          command->cbs(), deviceAddressValidator, createSemaphore, hostBuffers);
      const std::vector<std::uint32_t> &compileArgs = std::visit(
          [](const auto &config) -> const std::vector<std::uint32_t> & {
            return config.compile_args;
          },
          materializedKernelConfig);
      tt_metal::KernelHandle handle = createKernel(
          program, kernelSourceString, coreRangeSet, materializedKernelConfig,
          currentProgramName, debugInfo, kernelConfig->debug_info()->c_str(),
          kernelConfig->loc() ? kernelConfig->loc()->c_str() : nullptr);

      std::vector<uint32_t> commonRtArgsVec = processRuntimeArgs(
          kernelConfig->args()->crt_args(), command->arg_refs_type(),
          command->arg_refs(), meshBuffers, global_semaphores,
          local_semaphore_initializer, command->cbs(), deviceAddressValidator,
          createSemaphore, hostBuffers);
      tt_metal::SetCommonRuntimeArgs(program, handle, commonRtArgsVec);

      std::vector<uint32_t> rtArgsVec = processRuntimeArgs(
          kernelConfig->args()->rt_args(), command->arg_refs_type(),
          command->arg_refs(), meshBuffers, global_semaphores,
          local_semaphore_initializer, command->cbs(), deviceAddressValidator,
          createSemaphore, hostBuffers);

      if (command->fabric_connection_config() &&
          kernelConfig->type_type() ==
              target::metal::KernelConfigType::NocConfig &&
          command->fabric_connection_config()->noc_index() ==
              kernelConfig->type_as_NocConfig()->noc_index()) {
        auto fabricConfigArgs = common::appendFabricConfigArgs(
            command->fabric_connection_config(), kernelConfig, program, handle,
            deviceCoord, meshDevice, rtArgsVec, coreRangeSet);

        for (auto core : tt::tt_metal::corerange_to_cores(coreRangeSet)) {
          auto it = fabricConfigArgs.find(core);
          tt_metal::SetRuntimeArgs(program, handle, core,
                                   it != fabricConfigArgs.end() ? it->second
                                                                : rtArgsVec);
        }
      } else {
        tt_metal::SetRuntimeArgs(program, handle, coreRangeSet, rtArgsVec);
      }

      if (cacheEligible) {
        cacheProgramBinding.kernels.push_back(
            {handle, coreRangeSet, compileArgs, std::move(commonRtArgsVec),
             std::move(rtArgsVec)});
      }
    }

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

      // skip init if CircularBufferConfig is not present
      if (!metalBuffer->circular_buffer_config()) {
        continue;
      }

      tt::tt_metal::CoreRangeSet coreRangeSet = common::toCoreRangeSet(
          metalBuffer->circular_buffer_config()->core_range_set());
      tt_metal::CircularBufferConfig config =
          createCircularBufferConfig(cbRef, meshBuffers);
      tt_metal::CBHandle handle =
          tt_metal::CreateCircularBuffer(program, coreRangeSet, config);
      if (cacheEligible) {
        cacheProgramBinding.circularBuffers.push_back(
            {handle, cbRef->buffer_ref()->global_id()});
      }
    }

    // Fabric connection arguments are device-specific, so fabric workloads
    // need one Program per device.
    distributed::MeshCoordinateRange programRange =
        command->fabric_connection_config()
            ? distributed::MeshCoordinateRange(deviceCoord)
            : deviceRange;
    if (cacheEligible) {
      cacheState.programs.emplace(programRange, std::move(cacheProgramBinding));
    }
    meshWorkload.add_program(programRange, std::move(program));
    if (!command->fabric_connection_config()) {
      break;
    }
  }

  if (cacheEligible) {
    CachedMeshWorkload cached(std::move(meshWorkload), std::move(cacheState));
    programCache.insert(programKey,
                        tt_metal::program_cache::detail::CachedProgramFactory{
                            std::move(cached), 0});
    auto &cachedProgramFactory = programCache.get(programKey);
    auto &cachedWorkload =
        cachedProgramFactory.cached_program.get<CachedMeshWorkload>();
    enqueueMeshWorkload(cachedWorkload.workload, loc);
  } else {
    enqueueMeshWorkload(meshWorkload, loc);
  }
}

ReusableInputKey
MCQExecutor::createReusableInputKey(const ProgramInputProvenance &input,
                                    std::uint32_t destinationGlobalId) const {
  return ReusableInputKey{deviceId,         binaryId,
                          programIndex,     deviceProgramIndex,
                          input.inputIndex, destinationGlobalId};
}

void MCQExecutor::bindReusableInput(
    const target::metal::EnqueueWriteBufferCommand *command,
    const ProgramInputProvenance &input) {
  LOG_ASSERT(input.cache, "Expected reusable input cache");
  const auto &cache = input.cache;

  const ReusableInputKey key =
      createReusableInputKey(input, command->dst()->global_id());
  MeshBuffer reusableBuffer;
  bool cacheHit = false;
  {
    std::lock_guard<std::mutex> lock(cache->mutex);
    auto cached = cache->buffers.find(key);
    if (cached != cache->buffers.end()) {
      reusableBuffer = cached->second;
      ++cache->hits;
      cacheHit = true;
    } else {
      reusableBuffer = createOwnedMeshBufferFromBufferRef(
          meshDevice, command->dst(), deviceAddressValidator);

      // Owned reusable allocations and D2M's explicit-address plan are managed
      // by different allocators. Check all currently-live transient buffers
      // before publishing the cache entry, then check every future transient
      // allocation through ReusableInputRegistry.
      for (const auto &[globalId, liveBuffer] : meshBuffers) {
        LOG_ASSERT(!buffersOverlap(reusableBuffer, liveBuffer),
                   "Reusable input allocation overlaps live D2M buffer ",
                   globalId,
                   "; the compiled memory plan does not leave "
                   "enough device memory for input reuse");
      }

      auto &hostInput = hostBuffers.at(command->src()->global_id());
      checkHostTensorSizeMatchWithMeshBufferSize(hostInput, reusableBuffer);
      writeHostTensorToMeshBuffer(mcq, hostInput, reusableBuffer, blockingCQ);

      cache->uploadedBytes += reusableBuffer->size();
      ++cache->misses;
      cache->buffers.emplace(key, reusableBuffer);
      ReusableInputRegistry::instance().insert(deviceId, reusableBuffer, cache);
    }
  }

  auto destination = meshBuffers.find(command->dst()->global_id());
  LOG_ASSERT(destination != meshBuffers.end(),
             "Reusable input destination was not allocated");
  destination->second = reusableBuffer;
  reusableBufferAliases.insert(command->dst()->global_id());

  LOG_DEBUG(logger::LogRuntimeTTMetalBufferCreation, "D2M reusable input ",
            input.inputIndex, cacheHit ? " cache hit" : " cache miss",
            ", destination buffer ", command->dst()->global_id(), ", ",
            reusableBuffer->size(), " bytes at ",
            logger::Address(reusableBuffer->address()));
}

void MCQExecutor::execute(
    const target::metal::EnqueueWriteBufferCommand *command) {
  ZoneScopedN("EnqueueWriteBufferCommand");

  const auto provenance =
      hostBufferInputProvenance.find(command->src()->global_id());
  if (provenance != hostBufferInputProvenance.end()) {
    bindReusableInput(command, provenance->second);
    return;
  }

  auto &input = hostBuffers.at(command->src()->global_id());
  auto meshBuffer = meshBuffers.at(command->dst()->global_id());
  checkHostTensorSizeMatchWithMeshBufferSize(input, meshBuffer);
  writeHostTensorToMeshBuffer(mcq, input, meshBuffer, blockingCQ);
}

void MCQExecutor::execute(
    const target::metal::EnqueueReadBufferCommand *command) {
  ZoneScopedN("EnqueueReadBufferCommand");

  auto &output = hostBuffers.at(command->dst()->global_id());
  auto meshBuffer = meshBuffers.at(command->src()->global_id());
  checkHostTensorSizeMatchWithMeshBufferSize(output, meshBuffer);
  readHostTensorFromMeshBuffer(mcq, meshBuffer, output, blockingCQ);
}

void MCQExecutor::execute(const target::metal::CreateBufferCommand *command) {
  ZoneScopedN("CreateBufferCommand");
  if (meshBuffers.find(command->ref()->global_id()) == meshBuffers.end()) {
    auto meshBuffer = createMeshBufferFromBufferRef(meshDevice, command->ref(),
                                                    deviceAddressValidator);
    ReusableInputRegistry::instance().assertNoOverlap(deviceId, meshBuffer);
    meshBuffers[command->ref()->global_id()] = std::move(meshBuffer);
  }
}

void MCQExecutor::execute(
    const target::metal::DeallocateBufferCommand *command) {
  ZoneScopedN("DeallocateBufferCommand");
  auto meshBufferIter = meshBuffers.find(command->ref()->global_id());
  LOG_ASSERT(meshBufferIter != meshBuffers.end(), "Buffer not allocated");
  LOG_ASSERT(meshBufferIter->second != nullptr, "Buffer already deallocated");
  if (reusableBufferAliases.erase(command->ref()->global_id()) != 0) {
    meshBuffers.erase(meshBufferIter);
    return;
  }
  auto meshBuffer = meshBufferIter->second;
  meshBuffer->deallocate();
  meshBuffers.erase(meshBufferIter);
}

void MCQExecutor::execute(
    const target::metal::EnqueueRecordEventCommand *command) {
  ZoneScopedN("EnqueueRecordEventCommand");
  meshEvents[command->ref()->global_id()] =
      std::make_shared<distributed::MeshEvent>(mcq->enqueue_record_event());
}

void MCQExecutor::execute(
    const target::metal::EnqueueWaitForEventCommand *command) {
  ZoneScopedN("EnqueueWaitForEventCommand");
  auto mesh_event = meshEvents.at(command->ref()->global_id());
  mcq->enqueue_wait_for_event(*mesh_event);
}

void MCQExecutor::execute(
    const target::metal::EventSynchronizeCommand *command) {
  ZoneScopedN("EventSynchronizeCommand");
  auto mesh_event = meshEvents.at(command->ref()->global_id());
  distributed::EventSynchronize(*mesh_event);
}

void MCQExecutor::execute(const target::metal::MemrefCopyCommand *command) {
  const auto input =
      hostBufferInputProvenance.find(command->src()->global_id());
  if (input != hostBufferInputProvenance.end() &&
      skippedHostCopyInputIndices.contains(input->second.inputIndex)) {
    hostBufferInputProvenance.insert_or_assign(command->dst()->global_id(),
                                               input->second);
    return;
  }

  auto srcIt = hostBuffers.find(command->src()->global_id());
  LOG_ASSERT(srcIt != hostBuffers.end());
  auto dstIt = hostBuffers.find(command->dst()->global_id());
  LOG_ASSERT(dstIt != hostBuffers.end());
  ttmetal::memcpy(
      dstIt->second, createTensorDescFromBufferDesc(command->dst()->desc()),
      srcIt->second, createTensorDescFromBufferDesc(command->src()->desc()));

  if (input == hostBufferInputProvenance.end()) {
    hostBufferInputProvenance.erase(command->dst()->global_id());
  } else {
    hostBufferInputProvenance.insert_or_assign(command->dst()->global_id(),
                                               input->second);
  }
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
  distributed::Finish(*mcq);
}

void MCQExecutor::execute(const target::metal::MeshShardCommand *command) {
  ZoneScopedN("MeshShardCommand");
  if (skippedHostBufferIds.contains(command->dst()->global_id())) {
    return;
  }
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
                         common::DylibManager &&dylibs, std::uint32_t deviceId,
                         std::uint64_t binaryId, std::uint32_t programIndex,
                         std::uint32_t deviceProgramIndex) {
  LOG_ASSERT(program->command_queues()->size() == 1, "Only one MCQ supported");

  MCQExecutor executor(meshDevice, program->inputs(), inputs, std::move(dylibs),
                       debug::Env::get().blockingCQ, deviceId, binaryId,
                       programIndex, deviceProgramIndex);
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
