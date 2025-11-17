// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_PERF_H
#define TT_RUNTIME_PERF_H

#include <cstdint>
#include <ostream>

namespace tt::runtime::perf {

enum class TracyLogTag {
  MLIR_OP_LOCATION,
  MLIR_CONST_EVAL_OP,
  MLIR_PROGRAM_METADATA,
  MLIR_INPUT_LAYOUT_CONVERSION_OP
};

inline std::string toString(TracyLogTag tracyLogTag) {
  switch (tracyLogTag) {
  case TracyLogTag::MLIR_OP_LOCATION:
    return "MLIR_OP_LOCATION";
  case TracyLogTag::MLIR_CONST_EVAL_OP:
    return "MLIR_CONST_EVAL_OP";
  case TracyLogTag::MLIR_PROGRAM_METADATA:
    return "MLIR_PROGRAM_METADATA";
  case TracyLogTag::MLIR_INPUT_LAYOUT_CONVERSION_OP:
    return "MLIR_INPUT_LAYOUT_CONVERSION_OP";
  }
}

struct Env {
#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1
  static Env &
#else
  static Env
#endif
  get(std::uint32_t dumpDeviceRate = 1000, bool enablePerfTrace = false,
      const std::string &tracyProgramMetadata = "")
#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1
      ;
#else
  {
    return Env(1000, false, "");
  }
#endif

  std::uint32_t dumpDeviceRate;
  bool enablePerfTrace;
  std::string tracyProgramMetadata;

#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1
  Env(const Env &) = delete;
  Env &operator=(const Env &) = delete;
  Env(Env &&) = delete;
  Env &operator=(Env &&) = delete;
#endif

  void tracyLogOpLocation(const std::string &locInfo) const;
  void tracyLogConstEvalProgram(bool constEvalOp) const;
  void tracyLogInputLayoutConversion(bool inputLayoutConversionOp) const;
  void tracyLogProgramMetadata(const std::string &metaData) const;
  void setProgramMetadata(const std::string &programMetadata);

private:
  Env(std::uint32_t dumpDeviceRate, bool enablePerfTrace,
      const std::string &tracyProgramMetadata)
      : dumpDeviceRate(dumpDeviceRate), enablePerfTrace(enablePerfTrace),
        tracyProgramMetadata(tracyProgramMetadata) {}
};

inline std::ostream &operator<<(std::ostream &os, const Env &env) {
  os << "perf::Env{\n"
     << "\t"
     << "dumpDeviceRate: " << env.dumpDeviceRate << "\n"
     << "\t"
     << "enablePerfTrace: " << env.enablePerfTrace << "\n"
     << "\t"
     << "tracyProgramMetadata: " << env.tracyProgramMetadata << "\n"
     << "}";
  return os;
}
} // namespace tt::runtime::perf

#endif
