// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/debug/debug.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include <Python.h>

namespace tt::runtime::ttnn::operations::debug {

void invoke_pdb() { PyRun_SimpleString("import pdb; pdb.set_trace()"); }

void run(const ::tt::target::ttnn::AnnotateOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &operand =
      tensorPool.getTTNNTensorAndValidate(op->operand());
  tensorPool.insertTTNNTensorAndValidate(op->result(), operand);
}

void run(const ::tt::target::ttnn::BreakpointOp *op, ProgramContext &context) {
  invoke_pdb();
}

void run(const ::tt::target::ttnn::PrintOp *op, ProgramContext &context) {
  LOG_INFO(op->message()->str());
}

void run(const ::tt::target::ttnn::MemorySnapshotOp *op,
         ProgramContext &context) {
  constexpr std::array<tt::runtime::MemoryBufferType, 4> MEMORY_TYPES = {
      tt::runtime::MemoryBufferType::DRAM, tt::runtime::MemoryBufferType::L1,
      tt::runtime::MemoryBufferType::L1_SMALL,
      tt::runtime::MemoryBufferType::TRACE};

  std::string filePath = op->file_path()->str();
  std::ofstream outFile(filePath, std::ios::out | std::ios::trunc);
  if (!outFile.is_open()) {
    throw std::runtime_error("Failed to open file: " + filePath);
  }

  auto memoryState =
      ::tt::runtime::ttnn::utils::getMemoryView(context.getDeviceHandle());
  for (const auto &memoryType : MEMORY_TYPES) {
    outFile << "Device " << toString(memoryType)
            << " memory state: " << memoryState.at(memoryType).toString()
            << "\n";
  }
  outFile.close();
}

void run(const ::tt::target::ttnn::RegionStartOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &operand =
      tensorPool.getTTNNTensorAndValidate(op->operand());
  tensorPool.insertTTNNTensorAndValidate(op->result(), operand);
}

void run(const ::tt::target::ttnn::RegionEndOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &operand =
      tensorPool.getTTNNTensorAndValidate(op->operand());
  tensorPool.insertTTNNTensorAndValidate(op->result(), operand);
}

} // namespace tt::runtime::ttnn::operations::debug
