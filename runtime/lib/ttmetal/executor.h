// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTMETAL_EXECUTOR_H
#define RUNTIME_LIB_TTMETAL_EXECUTOR_H

#include "executor_utils.h"

#define FMT_HEADER_ONLY
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include "tt/runtime/detail/common/dylib.h"
#include "tt/runtime/types.h"
#include "ttmlir/Target/TTMetal/Target.h"

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

  std::uint64_t getUniqueProgramRuntimeId() { return nextProgramRuntimeId++; }

private:
  distributed::MeshDevice *meshDevice;
  std::vector<std::shared_ptr<distributed::MeshEvent>> initMeshEvents;
  std::unordered_map<std::uint32_t, std::shared_ptr<distributed::MeshBuffer>>
      meshBuffers;
  std::unordered_map<std::uint32_t, tt_metal::GlobalSemaphore>
      global_semaphores;
  std::unordered_map<std::uint32_t, Tensor> hostBuffers;
  std::unordered_map<std::uint32_t, std::shared_ptr<distributed::MeshEvent>>
      meshEvents;
  std::vector<Tensor> outputs;
  distributed::MeshCommandQueue *mcq;
  bool blockingCQ;
  const char *currentProgramName;
  DeviceAddressValidator deviceAddressValidator;
  common::DylibManager dylibManager;
  std::uint64_t nextProgramRuntimeId = 10000; // Start at a greppable number.
};
} // namespace

std::vector<Tensor>
executeMeshDeviceProgram(::tt::tt_metal::distributed::MeshDevice *meshDevice,
                         const ::tt::target::metal::DeviceProgram *program,
                         const std::vector<Tensor> &inputs,
                         common::DylibManager &&dylibs);

} // namespace tt::runtime::ttmetal

#endif // RUNTIME_LIB_TTMETAL_EXECUTOR_H
