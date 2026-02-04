// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_PYTHON_PROFILER_PROFILER_H
#define TT_RUNTIME_PYTHON_PROFILER_PROFILER_H

#include "tt/runtime/detail/python/nanobind_headers.h"

namespace tt::runtime::python {
namespace nb = nanobind;
void registerProfilerBindings(nb::module_ &m);
} // namespace tt::runtime::python

#endif // TT_RUNTIME_PYTHON_PROFILER_PROFILER_H