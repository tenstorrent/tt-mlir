#pragma once

#include "preproc.hpp"
#include <cstdlib>

// Feature switches section.

FS_WITH_DISABLER(backtrace, true, "TT_KURBLA_BACKTRACE_DISABLED");
FS_WITH_ENABLER(assert_abort, false, "TT_KURBLA_ASSERT_ABORT_ENABLED");

// Config section.

constexpr size_t cfg_placeholder = 1;
