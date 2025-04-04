// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/debug.h"
#include "tt/runtime/detail/dylib.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/types.h"
#include "tt/runtime/ttnn/utils.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::executor {
using LogType = ::tt::runtime::logger::LogType;

static void tracyLogOpLocation(const ::tt::target::ttnn::Operation *op) {
#ifdef TT_RUNTIME_ENABLE_PERF_TRACE
  TracyMessage(op->loc_info()->c_str(), op->loc_info()->size());
#endif
}

class ProgramExecutor {
public:
  ProgramExecutor(const ::tt::target::ttnn::Program *program,
                  const Binary &executableHandle,
                  const std::vector<::ttnn::Tensor *> &programInputs,
                  ::ttnn::MeshDevice *meshDevice)
      : program(program) {
    LOG_ASSERT(program, "Program must be provided for execution");

    std::vector<uint32_t> programInputIds;
    int inputIndex = 0;
    std::unordered_map<uint32_t, ::ttnn::Tensor *> liveTensors;
    LOG_ASSERT(program->inputs()->size() == programInputs.size(),
               "Program input size mismatch: ", program->inputs()->size(),
               " != ", programInputs.size());
    for (const ::tt::target::ttnn::TensorRef *input : *program->inputs()) {
      auto [iter, inserted] = liveTensors.try_emplace(
          input->global_id(), programInputs[inputIndex++]);
      LOG_ASSERT(inserted, "Duplicate input tensor");
      programInputIds.push_back(input->global_id());
    }

    std::vector<uint32_t> programOutputIds;
    for (const ::tt::target::ttnn::TensorRef *output : *program->outputs()) {
      programOutputIds.push_back(output->global_id());
    }

    context = std::make_unique<ProgramContext>(
        programInputIds, programOutputIds, std::move(liveTensors),
        common::DylibManager(program->dylibs()), meshDevice, executableHandle,
        program->name()->str());
  }

  // Constructor that accepts an external cache and input versions
  ProgramExecutor(const ::tt::target::ttnn::Program *program,
                  const Binary &executableHandle,
                  const std::vector<::ttnn::Tensor *> &programInputs,
                  ::ttnn::MeshDevice *meshDevice,
                  std::shared_ptr<TensorCache> externalCache,
                  std::vector<uint64_t> &&inputVersions)
      : program(program) {
    LOG_ASSERT(program, "Program must be provided for execution");

    std::vector<uint32_t> programInputIds;
    int inputIndex = 0;
    std::unordered_map<uint32_t, ::ttnn::Tensor *> liveTensors;
    LOG_ASSERT(program->inputs()->size() == programInputs.size(),
               "Program input size mismatch: ", program->inputs()->size(),
               " != ", programInputs.size());
    for (const ::tt::target::ttnn::TensorRef *input : *program->inputs()) {
      auto [iter, inserted] = liveTensors.try_emplace(
          input->global_id(), programInputs[inputIndex++]);
      LOG_ASSERT(inserted, "Duplicate input tensor");
      programInputIds.push_back(input->global_id());
    }

    std::vector<uint32_t> programOutputIds;
    for (const ::tt::target::ttnn::TensorRef *output : *program->outputs()) {
      programOutputIds.push_back(output->global_id());
    }

    context = std::make_unique<ProgramContext>(
        programInputIds, programOutputIds, std::move(liveTensors),
        common::DylibManager(program->dylibs()), meshDevice, executableHandle,
        externalCache, std::move(inputVersions), program->name()->str());
  }

  void runCallback(const ::tt::target::ttnn::Operation *opContext,
                   ProgramContext *programContext);

  void execute() {
    for (const ::tt::target::ttnn::Operation *op : *program->operations()) {
      LOG_DEBUG(LogType::LogRuntimeTTNN,
                "Executing operation: ", op->debug_info()->c_str());
      tracyLogOpLocation(op);
      runOperation(op);
      runCallback(op, context.get());
    }
  }

  ProgramContext &getContext() { return *context; }

  std::vector<Tensor> gatherOutputTensors() {
    return context->getTensorPool().gatherOutputTensors();
  }

  std::vector<::ttnn::Tensor *> gatherTTNNOutputTensors() {
    const std::vector<uint32_t> &outputIdxs =
        context->getTensorPool().getProgramOutputIds();
    std::vector<::ttnn::Tensor *> outputs(outputIdxs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      // LOG_INFO("")
      outputs[i] = &context->getTensorPool().getAndValidate(outputIdxs[i]);
    }
    return outputs;
  }

private:
  const ::tt::target::ttnn::Program *program;
  std::unique_ptr<ProgramContext> context;
  void runOperation(const ::tt::target::ttnn::Operation *op);
  void runEltwiseOperation(const ::tt::target::ttnn::EltwiseOp *op);
};
} // namespace tt::runtime::ttnn::executor
