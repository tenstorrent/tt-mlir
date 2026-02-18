# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import atexit
import json
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from multiprocessing import queues
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from ttmlir.dialects import stablehlo
from ttmlir.ir import Context, Module, OpView


class ModuleDialect(Enum):
    """
    Enum for available dialects used in modules.

    Named like this to avoid collision with builtin `Dialect`.
    """

    STABLE_HLO = "stablehlo"
    TTIR = "ttir"
    TTNN = "ttnn"
    TT = "tt"

    @staticmethod
    def detect(module_or_op: str | OpView | Module) -> ModuleDialect:
        """
        Factory method. Detects dialect used in the mlir module or op string
        representation.
        """
        str_repr = str(module_or_op)

        if "= stablehlo." in str_repr:
            return ModuleDialect.STABLE_HLO
        elif '= "stablehlo.' in str_repr:
            return ModuleDialect.STABLE_HLO
        elif '= "ttir.' in str_repr:
            return ModuleDialect.TTIR
        elif '= "ttnn.' in str_repr:
            return ModuleDialect.TTNN
        else:
            # Fallback to returning `tt` dialect if nothing else succeeds. It bundles
            # together all builtin dialects.
            return ModuleDialect.TT


def create_mlir_module_from_string(module_str: str) -> Module:
    """
    Within a temporary context registers necessary dialects and parses `module_str`
    returning Module instance.
    """

    def register_dialect(dialect: ModuleDialect, ctx: Context) -> None:
        """
        Detects dialect used in `module_str` and registers it with context `ctx`.

        Note that only `stablehlo` needs to be registered this way. All custom TT
        dialects are registered automatically.
        """
        if dialect == ModuleDialect.STABLE_HLO:
            stablehlo.register_dialect(ctx)
        elif dialect not in [ModuleDialect.TTIR, ModuleDialect.TTNN, ModuleDialect.TT]:
            raise ValueError(f"Unknown dialect: {dialect.name}")

    with Context() as ctx:
        dialect = ModuleDialect.detect(module_str)
        # Must register dialect in order for parsing to work.
        register_dialect(dialect, ctx)
        return Module.parse(module_str)


class Status(Enum):
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class CompilationProcessResult:
    """Result of a compilation process."""

    status: Status
    module_str: str = None
    err: str = None

    @staticmethod
    def success(module_str: str) -> CompilationProcessResult:
        return CompilationProcessResult(Status.SUCCESS, module_str=module_str)

    @staticmethod
    def error(error: str) -> CompilationProcessResult:
        return CompilationProcessResult(Status.ERROR, err=error)


@dataclass
class TranslationProcessResult:
    """Result of a translation process."""

    status: Status
    fb_file_path: str = None
    err: str = None

    @staticmethod
    def success(fb_file_path: str) -> TranslationProcessResult:
        return TranslationProcessResult(Status.SUCCESS, fb_file_path=fb_file_path)

    @staticmethod
    def error(error: str) -> TranslationProcessResult:
        return TranslationProcessResult(Status.ERROR, err=error)


# Convenience alias.
Result = CompilationProcessResult | TranslationProcessResult


@dataclass
class Task:
    """Utility dataclass storing worker and its arguments."""

    worker_fn: Callable
    worker_args_without_queue: Tuple

    @property
    def name(self) -> str:
        return self.worker_fn.__name__ if not self.is_exit() else "EXIT"

    def __call__(self, result_queue: queues.Queue) -> Result:
        """Executes worker function and stores results in `result_queue`."""
        return self.worker_fn(*self.worker_args_without_queue, result_queue)

    @staticmethod
    def exit() -> Task:
        """
        Factory method for special no-op task ("exit task") indicating worker should
        stop looping.
        """
        return Task(None, None)

    def is_exit(self) -> bool:
        """Returns True if self is an "exit task"."""
        return self.worker_fn is None and self.worker_args_without_queue is None


def _persistent_worker(task_queue: queues.Queue, result_queue: queues.Queue):
    """
    Worker function looping indefinitely waiting for next task to execute.

    It blocks waiting for a task to arrive and then executes it. It can be shut down
    by sending an "exit task" to it.
    """
    while True:
        # Wait indefinitely for the next task.
        task: Task = task_queue.get()
        if task.is_exit():
            break

        # Execute task.
        task(result_queue)


class ProcessManager:
    """
    Manages compilation workers using multiprocessing for performance.

    Uses persistent worker processes with task queues. Processes exit via
    os._exit() (intentionally bypasses destructors) to avoid issues with
    inherited state from parent.

    For hardware workers requiring clean shutdown, use run_subprocess_worker()
    instead (defined at end of this module).
    """

    # ----- Public methods -----

    def __init__(self, mp_start_method: str = "forkserver") -> None:
        self.ctx = mp.get_context(mp_start_method)
        self.task_queue: queues.Queue = self.ctx.Queue()
        self.result_queue: queues.Queue = self.ctx.Queue()
        self.process: mp.Process = None

    def run(
        self, worker_fn: Callable, worker_args_without_queue: Tuple, timeout: float = 60
    ) -> Result:
        """
        Runs `worker_fn` in a separate process, returns whatever worker returned through
        queue if no errors happened, otherwise raises RuntimeError.
        """
        self._ensure_process_is_alive()

        # Pass the task to the worker.
        task = Task(worker_fn, worker_args_without_queue)
        self.task_queue.put(task)

        try:
            # Block waiting for result.
            result: Result = self.result_queue.get(timeout=timeout)
        except queue.Empty:
            # Worker failed to fill result queue before timeout.
            if self._is_process_running():
                self.stop()
                raise RuntimeError(f"Worker `{task.name}` timed out")
            else:
                # Something that wasn't caught by try-except occured, like a segfault,
                # that killed the process. Raise proper python error that can be handled
                # in try-except somewhere above in call stack.
                raise RuntimeError(f"Worker `{task.name}` crashed unexpectedly.")

        # Process must still be running if it managed to return a result.
        if not self._is_process_running():
            raise RuntimeError(f"Worker `{task.name}` crashed unexpectedly.")

        if result.status == Status.ERROR:
            # Errors caught by try-except in worker. Re-raise them as proper python
            # errors that can be handled in try-except somewhere above in call stack.
            raise RuntimeError(f"Worker `{task.name}` failed: {result.err}")

        return result

    def stop(self) -> None:
        """Gracefully stops the process by sending it an "exit task"."""
        if not self._is_process_running():
            return

        self.task_queue.put(Task.exit())
        self.process.join()

    # ----- Private methods -----

    def _is_process_running(self) -> bool:
        """Returns True if process is alive."""
        return self.process is not None and self.process.is_alive()

    def _ensure_process_is_alive(self) -> None:
        """
        Ensures process is alive.

        It instantiates a new Process if `self.process` is not running.

        Process might have been killed unexpectedly by previous task that it was
        executing.
        """
        if self._is_process_running():
            return

        self.process = self.ctx.Process(
            target=_persistent_worker,
            args=(self.task_queue, self.result_queue),
        )
        self.process.start()


# Private singleton instance of process manager. Do not use directly. Use provided
# factory function below instead.
_process_manager: ProcessManager = None


def get_process_manager() -> ProcessManager:
    """Returns singleton instance of process manager."""
    global _process_manager

    if _process_manager is None:
        _process_manager = ProcessManager()
        # Register an exit function to be executed upon normal program termination.
        # This is needed in order to avoid the need to manually stop process manager
        # from caller, because singleton instance is module-level and will continue
        # to live until program terminates.
        atexit.register(_process_manager.stop)

    return _process_manager


# ---------- Subprocess worker execution ----------


def run_subprocess_worker(
    runner_script: str,
    args: tuple,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """
    Execute worker script in subprocess with clean shutdown for hardware cleanup.

    Subprocess exits normally (unlike multiprocessing which calls os._exit),
    allowing C++ destructors to run.

    Parameters
    ----------
    runner_script : str
        Worker script name in common/ directory.
    args : tuple
        Arguments passed to the worker.
    timeout : float
        Timeout in seconds.

    Returns
    -------
    dict
        Result with "status" field ("success" or "error").
    """
    result_fd, result_path = tempfile.mkstemp(suffix=".json")
    os.close(result_fd)

    runner_path = Path(__file__).parent / runner_script
    if not runner_path.exists():
        raise FileNotFoundError(f"Worker script not found: {runner_path}")

    try:
        cmd = [sys.executable, str(runner_path)] + list(args) + [result_path]
        subprocess.run(cmd, timeout=timeout, check=False)

        with open(result_path) as f:
            result = json.load(f)

    except Exception as e:
        raise RuntimeError(f"Worker failed: {runner_script}") from e
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass

    return result
