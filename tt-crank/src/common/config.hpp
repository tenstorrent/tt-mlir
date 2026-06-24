#pragma once

#include "preproc.hpp"
#include <cstdlib>

// Feature switches section.

// Enables backtrace for exceptions.
FS_WITH_DISABLER(backtrace, true, "TT_KURBLA_BACKTRACE_DISABLED");

// Aborts on assert.
FS_WITH_ENABLER(assert_abort, false, "TT_KURBLA_ASSERT_ABORT_ENABLED");

// Prints monitor info whenever any monitored condition is accessed.
FS_WITH_ENABLER(monitor_verbose_print, false, "TT_KURBLA_MONITOR_VERBOSE_PRINT_ENABLED");

// Enables compiler cache.
FS_WITH_DISABLER(comp_cache, true, "TT_KURBLA_COMP_CACHE_DISABLED");

// Enables tt ir printing.
FS_WITH_ENABLER(print_tt_ir, false, "TT_KURBLA_PRINT_TT_IR_ENABLED");

// Enables ttnn ir printing.
FS_WITH_ENABLER(print_ttnn_ir, false, "TT_KURBLA_PRINT_TTNN_IR_ENABLED");

// Enables tt-sim simulator.
FS_WITH_ENABLER(use_sim, false, "TT_KURBLA_USE_SIMULATOR");

// Enables logging of a CPU fallbacked ops.
FS_WITH_ENABLER(log_fallback, false, "TT_KURBLA_LOG_FALLBACK_ENABLED");

// Enables compiler consteval.
FS_WITH_DISABLER(comp_consteval, true, "TT_KURBLA_COMP_CONSTEVAL_DISABLED");

// Config section.

constexpr size_t cfg_monitor_max_slots = 64;
