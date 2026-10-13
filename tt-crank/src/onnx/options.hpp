// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "engine/compile_options.hpp"
#include "onnxruntime_c_api.h"

namespace tt::crank::onnx {

CompileOptions parse_compile_options(const OrtSessionOptions *session_options);

// Whether the session requested a precompiled (EPContext) model to be written
// out — ORT's "ep.context_enable" session option.
bool parse_ep_ctx_enabled(const OrtSessionOptions *session_options);

} // namespace tt::crank::onnx
