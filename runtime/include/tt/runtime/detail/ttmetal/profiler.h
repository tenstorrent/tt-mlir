// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTMETAL_PROFILER_H
#define TT_RUNTIME_DETAIL_TTMETAL_PROFILER_H

#define FMT_HEADER_ONLY
#include "tools/profiler/op_profiler_serialize.hpp"
#include "tt/runtime/detail/common/logger.h"

// [todo] move into op_profiler_serialize.hpp includes after build break from
// tt-metal b13938c See https://github.com/tenstorrent/tt-mlir/issues/4004
#include "tt/runtime/detail/ttmetal/ttmetal.h"

#include <cstdint>

namespace tt::runtime::ttmetal::profiler {

#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1
#if !defined(TRACY_ENABLE)
#error "TRACY_ENABLE is not defined"
#endif

inline std::string op_meta_data_serialized_json(
    ChipId deviceId, const tt::tt_metal::Program &program, const char *loc) {
  auto runtime_id = tt_metal::detail::EncodePerDeviceProgramID(
      program.get_runtime_id(), deviceId, /*is_host_fallback_op=*/false);

  tt::tt_metal::op_profiler::OpProfileData data;
  data.operation_id = runtime_id;
  data.op_name = loc;

  return tt::tt_metal::op_profiler::assemble_device_op_json(
      data, /*program_hash=*/0, deviceId, /*program_cache_hit=*/false, program);
}

inline void addProgramProfileHostMetadata(int deviceId,
                                          const tt::tt_metal::Program &program,
                                          const char *loc) {
  ZoneScopedN("TT_DNN_DEVICE_OP");
  std::string op_message =
      tt::runtime::ttmetal::profiler::op_meta_data_serialized_json(
          deviceId, program, loc);
  auto runtime_id = tt_metal::detail::EncodePerDeviceProgramID(
      program.get_runtime_id(), deviceId, /*is_host_fallback_op=*/false);
  std::string op_text = fmt::format("id:{}", runtime_id);
  ZoneText(op_text.c_str(), op_text.size());
  TracyMessage(op_message.c_str(), op_message.size());
}

#else

inline void addProgramProfileHostMetadata(int deviceId,
                                          const tt::tt_metal::Program &program,
                                          const char *loc) {
  LOG_WARNING_ONCE("TT_RUNTIME_ENABLE_PERF_TRACE is not enabled in build, no "
                   "perf trace will be generated!");
}

#endif // TT_RUNTIME_ENABLE_PERF_TRACE

} // namespace tt::runtime::ttmetal::profiler

#endif // TT_RUNTIME_DETAIL_TTMETAL_PROFILER_H
