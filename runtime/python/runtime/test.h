// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_PYTHON_RUNTIME_TEST_H
#define TT_RUNTIME_PYTHON_RUNTIME_TEST_H

#if (!defined(TTMLIR_ENABLE_RUNTIME_TESTS) || TTMLIR_ENABLE_RUNTIME_TESTS == 0)
#error(TTMLIR_ENABLE_RUNTIME_TESTS must be set to 1 to enable runtime tests)
#endif

#include "tt/runtime/detail/python/nanobind_headers.h"

namespace tt::runtime::python {
namespace nb = nanobind;
void registerRuntimeTestBindings(nb::module_ &m);
} // namespace tt::runtime::python

#endif // TT_RUNTIME_PYTHON_RUNTIME_TEST_H
