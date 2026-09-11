// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#define FS_CONFIG_IMPL
#include "config.hpp" // NOLINT

#include <cstdlib>

// Global logger types env setter.
// Sets TT_LOGGER_TYPES env var if it is not already set.
const int logger_init = [] { return ::setenv("TT_LOGGER_TYPES", logger_types_config(), 0); }();
