// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/perf.h"

#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1
#include "tracy/Tracy.hpp"
#endif

namespace tt::runtime::perf {

#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1
Env &Env::get(std::uint32_t dumpDeviceRate, bool enablePerfTrace,
              const std::string &tracyProgramMetadata) {
  static Env config(dumpDeviceRate, enablePerfTrace, tracyProgramMetadata);
  return config;
}
#endif

void Env::tracyLogOpLocation(std::string locInfo) {
#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1
  std::string message =
      perf::toString(perf::TracyLogTag::MLIR_OP_LOCATION) + ";" + locInfo;
  TracyMessage(message.c_str(), message.size());
#endif
}

void Env::tracyLogConstEvalProgram(bool constEvalOp) {
#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1
  std::string message = perf::toString(perf::TracyLogTag::MLIR_CONST_EVAL_OP) +
                        ";" + std::string(constEvalOp ? "true" : "false");
  TracyMessage(message.c_str(), message.size());
#endif
}

void Env::tracyLogProgramMetadata(std::string metaData) {
#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1
  std::string message =
      perf::toString(perf::TracyLogTag::MLIR_PROGRAM_METADATA) + ";" + metaData;
  TracyMessage(message.c_str(), message.size());
#endif
}

void Env::setProgramMetadata(const std::string &programMetadata) {
  tracyProgramMetadata = programMetadata;
}

} // namespace tt::runtime::perf
