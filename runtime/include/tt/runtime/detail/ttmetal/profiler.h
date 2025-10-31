// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTMETAL_PROFILER_H
#define TT_RUNTIME_DETAIL_TTMETAL_PROFILER_H

#define FMT_HEADER_ONLY
#include "tools/profiler/op_profiler.hpp"
#include "tt/runtime/detail/common/logger.h"

// [todo] move into op_profiler.hpp includes after build break from tt-metal
// b13938c See https://github.com/tenstorrent/tt-mlir/issues/4004
#include "tt/runtime/detail/ttmetal/ttmetal.h"

#include <cstdint>

namespace tt::runtime::ttmetal::profiler {

#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1
#if !defined(TRACY_ENABLE)
#error "TRACY_ENABLE is not defined"
#endif

//
// The following code is based on the code from tt_metal's
// ttnn/tools/profiler/op_profiler.hpp
//
// tt_metal's code is overfit to ttnn ops so much of what is done below is just
// to stub it out and make it as compatible as possible.
//

struct device_operation_t {
  struct operation_attributes_t {
    const char *loc;

    operation_attributes_t(const char *loc) : loc(loc) {}
    ttsl::reflection::Attributes attributes() const { return {}; }
  };
  using tensor_args_t = std::vector<tt::tt_metal::Tensor>;
  using tensor_return_value_t = std::vector<tt::tt_metal::Tensor>;

  static std::string get_type_name(const operation_attributes_t &attributes) {
    return attributes.loc;
  }
};

inline std::string op_meta_data_serialized_json(
    ChipId deviceId, const tt::tt_metal::Program &program, const char *loc) {
  std::uint32_t programHash = 0;

  auto runtime_id = tt_metal::detail::EncodePerDeviceProgramID(
      program.get_runtime_id(), deviceId);
  device_operation_t::operation_attributes_t attributes(loc);
  device_operation_t::tensor_return_value_t tmpLValue{};
  auto j = tt::tt_metal::op_profiler::get_base_json<device_operation_t>(
      runtime_id, attributes, device_operation_t::tensor_args_t{}, tmpLValue);
  j["op_type"] = "ttmlir_program";
  j["device_id"] = deviceId;
  j["op_hash"] = programHash;
  j["kernel_info"] =
      tt::tt_metal::op_profiler::get_kernels_json(deviceId, program);
  j["optional_input_tensors"] = std::vector<json>{};

  std::string short_str =
      fmt::format("`TT_DNN_DEVICE_OP: {}, {}, {}, ", j["op_code"].dump(),
                  programHash, deviceId);
  std::string ser = j.dump(4);
  return fmt::format("{}{} ->\n{}`", short_str, runtime_id, ser);
}

inline void addProgramProfileHostMetadata(int deviceId,
                                          const tt::tt_metal::Program &program,
                                          const char *loc) {
  ZoneScopedN("TT_DNN_DEVICE_OP");
  std::string op_message =
      tt::runtime::ttmetal::profiler::op_meta_data_serialized_json(
          deviceId, program, loc);
  auto runtime_id = tt_metal::detail::EncodePerDeviceProgramID(
      program.get_runtime_id(), deviceId);
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
