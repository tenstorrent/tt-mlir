// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "assert.hpp"
#include "onnxruntime_c_api.h"

#include <exception>
#include <string>

namespace tt::crank::onnx {

const OrtApi &ort_api();
const OrtEpApi &ep_api();

OrtMemoryInfo *meminfo();

// Throws if an ORT call failed.
inline void check_call(OrtStatus *status) {
    if (status == nullptr) {
        return;
    }
    std::string message = ort_api().GetErrorMessage(status);
    OrtErrorCode code = ort_api().GetErrorCode(status);
    ort_api().ReleaseStatus(status);
    TT_THROW("ORT call failed: {} ({})", message, code);
}

// Converts exception into ORT failure status.
inline OrtStatus *fail_status(const std::exception &e) {
    return ort_api().CreateStatus(ORT_EP_FAIL, e.what());
}

} // namespace tt::crank::onnx
