// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_COMMON_GRAPH_TRACKER_LOCK_H
#define TTMLIR_COMMON_GRAPH_TRACKER_LOCK_H

#include <mutex>

namespace tt::ttmlir::common {

// Process-wide mutex used to serialize all interactions with tt-metal's
// `tt::tt_metal::GraphTracker` singleton.  The singleton's `processors`
// vector is not internally thread-safe, and tt-mlir uses graph capture
// (push_processor / track_function_* / pop_processor) from two distinct
// places that may run concurrently on different XLA worker threads:
//
//   - The optimizer's compile-time constraint queries
//     (lib/OpModel/TTNN/TTNNOpModel.cpp::executeConstraintQuery), which
//     wrap each ttnn op invocation in a ScopedGraphCapture.
//
//   - The runtime's program execution (runtime/lib/ttnn/runtime.cpp::submit),
//     which dispatches ttnn ops that fire `track_*` callbacks on whichever
//     processor is currently on the singleton's stack.
//
// Without serialization, the runtime thread can iterate `processors` while
// the compile thread is mid-mutation - segfaulting on stale pointers.
//
// Both call sites acquire this mutex for the entire push -> callable -> pop
// (compile side) or for the entire `submit` body (runtime side), so there
// is at most one thread interacting with GraphTracker at any given time.
//
// The lock lives in an inline function so the static-local has vague linkage
// (COMDAT-deduplicated across translation units within the same process).
inline std::mutex &graphTrackerLock() {
  static std::mutex m;
  return m;
}

} // namespace tt::ttmlir::common

#endif // TTMLIR_COMMON_GRAPH_TRACKER_LOCK_H
