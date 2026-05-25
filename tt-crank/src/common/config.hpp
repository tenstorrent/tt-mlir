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

// Config section.

constexpr size_t cfg_monitor_max_slots = 64;
