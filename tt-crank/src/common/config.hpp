// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "preproc.hpp"
#include <cstdlib>
#include <string>

// Feature switches section.

// Enables backtrace for exceptions.
FS_WITH_DISABLER(backtrace, true, "TT_KURBLA_BACKTRACE_DISABLED");

// Aborts on assert.
FS_WITH_ENABLER(assert_abort, false, "TT_KURBLA_ASSERT_ABORT_ENABLED");

// Prints monitor info whenever any monitored condition is accessed.
FS_WITH_ENABLER(monitor_verbose_print, false, "TT_KURBLA_MONITOR_VERBOSE_PRINT_ENABLED");

// Enables compiler cache.
FS_WITH_DISABLER(comp_cache, true, "TT_KURBLA_COMP_CACHE_DISABLED");

// Enables compiler cache on disk.
FS_WITH_DISABLER(comp_cache_on_disk, true, "TT_KURBLA_COMP_CACHE_ON_DISK_DISABLED");

// Enables tt ir printing.
FS_WITH_ENABLER(print_tt_ir, false, "TT_KURBLA_PRINT_TT_IR_ENABLED");

// Enables ttnn ir printing.
FS_WITH_ENABLER(print_ttnn_ir, false, "TT_KURBLA_PRINT_TTNN_IR_ENABLED");

// Enables printing of compile options.
FS_WITH_ENABLER(print_compile_options, false, "TT_KURBLA_PRINT_COMPILE_OPTIONS_ENABLED");

// Enables tt-sim simulator.
FS_WITH_ENABLER(use_sim, false, "TT_KURBLA_USE_SIMULATOR");

// Enables logging of a CPU fallbacked ops.
FS_WITH_ENABLER(log_fallback, false, "TT_KURBLA_LOG_FALLBACK_ENABLED");

// Enables compiler consteval.
FS_WITH_DISABLER(comp_consteval, true, "TT_KURBLA_COMP_CONSTEVAL_DISABLED");

// Enables tensor borrowing.
FS_WITH_DISABLER(tensor_borrowing, true, "TT_KURBLA_TENSOR_BORROWING_DISABLED");

// Warns user that a borrowed tensor's in-place writes cannot be detected.
FS_WITH_DISABLER(warn_on_unsafe_borrow, true, "TT_KURBLA_WARN_ON_UNSAFE_BORROW_DISABLED");

// Config section.

constexpr size_t cfg_monitor_max_slots = 64;

// Logger types for program.
// Kurbla uses Always type logs in codebase and it is default logger setting.
// Also, Always silences all logs in third party code, except an important ones.
// To see all logs, set TT_LOGGER_TYPES=All
// Available log filters: All, Always, UMD, Metal, BuildKernels, Fabric, Distributed, Device
CONFIG_STR(logger_types, "Always", "TT_LOGGER_TYPES")

// Compile cache directory where all compiled binaries are loaded/stored.
CONFIG_STR(compile_cache_dir, ".data/bin_cache/", "TT_KURBLA_COMPILE_CACHE_DIR")

// Directory where the artifacts should be stored.
CONFIG_STR(artifacts_dir, ".data/artifacts/", "TT_KURBLA_ARTIFACTS_DIR")
