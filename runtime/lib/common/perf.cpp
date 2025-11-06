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

void Env::tracyLogOpLocation(const std::string &locInfo) const {
#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1
  // Log to stderr for CI debugging (will appear in test output)
  static bool first_call = true;
  if (first_call) {
    fprintf(stderr, "DEBUG: tracyLogOpLocation called (first time)\n");
    first_call = false;
  }
  if (locInfo.empty()) {
    fprintf(stderr, "DEBUG: tracyLogOpLocation called with EMPTY locInfo!\n");
  }
  std::string message =
      perf::toString(perf::TracyLogTag::MLIR_OP_LOCATION) + ";" + locInfo;
  TracyMessage(message.c_str(), message.size());
#endif
}

void Env::tracyLogConstEvalProgram(bool constEvalOp) const {
#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1
  std::string message = perf::toString(perf::TracyLogTag::MLIR_CONST_EVAL_OP) +
                        ";" + std::string(constEvalOp ? "true" : "false");
  TracyMessage(message.c_str(), message.size());
#endif
}

void Env::tracyLogProgramMetadata(const std::string &metaData) const {
#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1
  std::string message =
      perf::toString(perf::TracyLogTag::MLIR_PROGRAM_METADATA) + ";" + metaData;
  TracyMessage(message.c_str(), message.size());
#endif
}

void Env::tracyLogInputLayoutConversion(bool inputLayoutConversionOp) const {
#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1
  std::string message =
      perf::toString(perf::TracyLogTag::MLIR_INPUT_LAYOUT_CONVERSION_OP) + ";" +
      std::string(inputLayoutConversionOp ? "true" : "false");
  TracyMessage(message.c_str(), message.size());
#endif
}

void Env::setProgramMetadata(const std::string &programMetadata) {
  tracyProgramMetadata = programMetadata;
}

} // namespace tt::runtime::perf
