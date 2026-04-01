# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression test for GIL-safety of DebugHooks callback copies.

Problem: getPreOperatorCallback() / getPostOperatorCallback() return
std::optional<CallbackFn> by value, which copies the std::function wrapping a
captured nb::callable. Copying nb::callable manipulates Python refcounts. When
the calling thread doesn't hold the GIL (e.g. tt-xla's C++ execution thread),
this causes a segfault.

This test exercises the callback registration and copy paths from multiple
Python threads. While the GIL serializes Python-level execution, the rapid
concurrent registration stresses the copy/destruction cycle of the
std::function<nb::callable> objects and can surface refcount corruption.

The definitive reproduction requires calling getPreOperatorCallback() from a
C++ thread without the GIL (the tt-xla scenario). This pure-Python test serves
as a smoke test for the callback copy path.
"""

import threading
import pytest
import ttrt
import ttrt.runtime


def _is_debug_enabled():
    return str(ttrt.runtime.DebugStats.get()) != "DebugStats Disabled"


@pytest.mark.skipif(not _is_debug_enabled(), reason="Requires TT_RUNTIME_DEBUG=1")
def test_hooks_concurrent_registration():
    """Stress-test concurrent callback registration from multiple threads.

    Each call to DebugHooks.get(pre, post) internally copies the previous
    callbacks (return-by-value in the current implementation). Concurrent
    threads doing this exercises the copy/destruction path that is unsafe
    without the GIL.
    """
    call_count = 0
    lock = threading.Lock()

    def pre_op(binary, program_context, op_context):
        nonlocal call_count
        with lock:
            call_count += 1

    def post_op(binary, program_context, op_context):
        nonlocal call_count
        with lock:
            call_count += 1

    # Initial registration
    hooks = ttrt.runtime.DebugHooks.get(pre_op, post_op)
    assert hooks is not None, "DebugHooks.get() returned None despite debug being enabled"

    errors = []
    num_threads = 8
    iterations_per_thread = 500

    def thread_func(thread_id):
        """Re-register callbacks repeatedly, triggering internal copies."""
        try:
            for i in range(iterations_per_thread):
                # Each call copies the previous callback std::function internally
                # before replacing it. This is the copy that segfaults without GIL.
                ttrt.runtime.DebugHooks.get(pre_op, post_op)
        except Exception as e:
            errors.append((thread_id, e))

    threads = [
        threading.Thread(target=thread_func, args=(i,)) for i in range(num_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    # Check all threads completed without error
    for t in threads:
        assert not t.is_alive(), f"Thread {t.name} did not complete within timeout"
    assert not errors, f"Threads raised exceptions: {errors}"

    ttrt.runtime.unregister_hooks()


@pytest.mark.skipif(not _is_debug_enabled(), reason="Requires TT_RUNTIME_DEBUG=1")
def test_hooks_register_unregister_cycle():
    """Stress-test register/unregister cycles from multiple threads.

    Interleaving registration and unregistration exercises both the copy
    (registration) and destruction (unregistration) of the nb::callable
    wrappers.
    """

    def noop(binary, program_context, op_context):
        pass

    errors = []
    num_threads = 4
    iterations = 200

    def thread_func(thread_id):
        try:
            for _ in range(iterations):
                ttrt.runtime.DebugHooks.get(noop, noop)
                ttrt.runtime.unregister_hooks()
        except Exception as e:
            errors.append((thread_id, e))

    threads = [
        threading.Thread(target=thread_func, args=(i,)) for i in range(num_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    for t in threads:
        assert not t.is_alive(), f"Thread {t.name} did not complete within timeout"
    assert not errors, f"Threads raised exceptions: {errors}"

    # Clean up
    ttrt.runtime.unregister_hooks()
