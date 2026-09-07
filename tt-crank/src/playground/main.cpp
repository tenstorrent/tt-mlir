// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "assert.hpp"
#include "cast.hpp"
#include "engine/compile.hpp"

#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include <format>
#include <limits>
#include <tt-logger/tt-logger.hpp>

#include <tt-logger/tt-logger-initializer.hpp>

constexpr std::string_view k_trivial_add_ttir = R"mlir(
func.func @add(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}
)mlir";

// NOLINTBEGIN

// Playground.
// Enable compilation in cmake with TT_KURBLA_BUILD_MAIN
// Feel free to test new code here.
int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv) {
    tt::kurbla::CompileOptions opts;
    tt::kurbla::CompiledProgram &program =
        *tt::kurbla::compile_ttir_to_ttnn_flatbuffer(k_trivial_add_ttir, opts).program;

    log_debug(tt::LogAlways, "log_debug");
    log_info(tt::LogAlways, "log_info");
    log_warning(tt::LogAlways, "log_warning");
    log_error(tt::LogAlways, "log_error");
    log_critical(tt::LogAlways, "log_critical");
    log_fatal(tt::LogAlways, "log_fatal");

    return as<int>(std::numeric_limits<uint32_t>::max());
}

// NOLINTEND
