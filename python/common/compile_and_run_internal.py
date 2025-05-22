# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum
from multiprocessing import Process, Queue
from typing import Any, Callable

from ttmlir.compile_and_run_internal import *
from ttmlir.ir import Module
from ttmlir.passes import (
    stablehlo_to_ttir_pipeline,
    ttir_to_ttmetal_backend_pipeline,
    ttir_to_ttnn_backend_pipeline,
    ttmetal_to_flatbuffer_file,
    ttnn_to_flatbuffer_file,
)
from ttmlir.utils import create_mlir_module_from_string
from ttrt.common.api import API
from ttrt.common.util import Binary, FileManager, Logger


class Status(Enum):
    SUCCESS = "success"
    ERROR = "error"


def _run_worker_in_separate_process(
    worker_fn: Callable,
    worker_args_without_queue: tuple = (),
) -> Any:
    """
    Runs `worker_fn` in a separate process, returns whatever worker returned through
    queue if no errors happend, otherwise raises RuntimeError.
    """
    q = Queue()
    p = Process(target=worker_fn, args=(*worker_args_without_queue, q))
    p.start()
    p.join()

    if p.exitcode != 0:
        # Something that wasn't caught by try-except occured, like a segfault. Raise
        # proper python error that can be handled in try-except somewhere above in call
        # stack.
        raise RuntimeError(f"Worker `{worker_fn.__name__}` crashed unexpectedly.")

    assert not q.empty(), f"Process `{worker_fn.__name__}` expected to return a result"

    result = q.get()

    if result.status == Status.ERROR:
        # Errors caught by try-except in worker. Re-raise them as proper python errors
        # that can be handled in try-except somewhere above in call stack.
        raise RuntimeError(f"Worker `{worker_fn.__name__}` failed: {result.error}")

    return result


# ---------- Utility wrappers around compiler passes ----------


@dataclass
class CompilationProcessResult:
    status: Status
    module_str: str = None
    error: str = None


def stablehlo_to_ttir_pipeline_worker(module_str: str, result_queue: Queue) -> None:
    """
    Wrapper around `stablehlo_to_ttir_pipeline` pybound pass.

    It is not resistant to segfaults, i.e. some unpredictable errors that can happen
    inside the pybound call. Thus it is meant to be used as a worker for a Process
    which will guard the caller from such errors.
    """
    try:
        module = create_mlir_module_from_string(module_str)

        stablehlo_to_ttir_pipeline(module)

        result_queue.put(CompilationProcessResult(Status.SUCCESS, str(module)))
    except Exception as e:
        result_queue.put(CompilationProcessResult(Status.ERROR, error=str(e)))


def ttir_to_ttnn_backend_pipeline_worker(
    module_str: str, system_desc: str, result_queue: Queue
) -> None:
    """
    Wrapper around `ttir_to_ttnn_backend_pipeline` pybound pass.

    It is not resistant to segfaults, i.e. some unpredictable errors that can happen
    inside the pybound call. Thus it is meant to be used as a worker for a Process
    which will guard the caller from such errors.
    """
    try:
        module = create_mlir_module_from_string(module_str)

        ttir_to_ttnn_backend_pipeline(module, f"system-desc-path={system_desc}")

        result_queue.put(CompilationProcessResult(Status.SUCCESS, str(module)))
    except Exception as e:
        result_queue.put(CompilationProcessResult(Status.ERROR, error=str(e)))


def ttir_to_ttmetal_backend_pipeline_worker(
    module_str: str, system_desc: str, result_queue: Queue
) -> None:
    """
    Wrapper around `ttir_to_ttmetal_backend_pipeline` pybound pass.

    It is not resistant to segfaults, i.e. some unpredictable errors that can happen
    inside the pybound call. Thus it is meant to be used as a worker for a Process
    which will guard the caller from such errors.
    """
    try:
        module = create_mlir_module_from_string(module_str)

        ttir_to_ttmetal_backend_pipeline(module, f"system-desc-path={system_desc}")

        result_queue.put(CompilationProcessResult(Status.SUCCESS, str(module)))
    except Exception as e:
        result_queue.put(CompilationProcessResult(Status.ERROR, error=str(e)))


def run_compilation_process(
    worker_fn: Callable,
    worker_args_without_queue: tuple = (),
) -> Module:
    """
    Runs `worker_fn` (function doing some compilation pass from above) in a separate
    process, returns produced Module if no errors happend, otherwise raises
    RuntimeError.
    """
    result: CompilationProcessResult = _run_worker_in_separate_process(
        worker_fn, worker_args_without_queue
    )
    return create_mlir_module_from_string(result.module_str)


# ---------- Utility wrappers around translation passes ----------


@dataclass
class TranslationProcessResult:
    status: Status
    fb_file_path: str = None
    error: str = None


def ttnn_to_flatbuffer_file_worker(
    module_str: str, output_file_name: str, result_queue: Queue
) -> None:
    """
    Wrapper around `ttnn_to_flatbuffer_file` pybound pass.

    It is not resistant to segfaults, i.e. some unpredictable errors that can happen
    inside the pybound call. Thus it is meant to be used as a worker for a Process
    which will guard the caller from such errors.
    """
    try:
        module = create_mlir_module_from_string(module_str)

        ttnn_to_flatbuffer_file(module, output_file_name)

        result_queue.put(TranslationProcessResult(Status.SUCCESS, output_file_name))
    except Exception as e:
        result_queue.put(TranslationProcessResult(Status.ERROR, error=str(e)))


def ttmetal_to_flatbuffer_file_worker(
    module_str: str, output_file_name: str, result_queue: Queue
) -> None:
    """
    Wrapper around `ttmetal_to_flatbuffer_file` pybound pass.

    It is not resistant to segfaults, i.e. some unpredictable errors that can happen
    inside the pybound call. Thus it is meant to be used as a worker for a Process
    which will guard the caller from such errors.
    """
    try:
        module = create_mlir_module_from_string(module_str)

        ttmetal_to_flatbuffer_file(module, output_file_name)

        result_queue.put(TranslationProcessResult(Status.SUCCESS, output_file_name))
    except Exception as e:
        result_queue.put(TranslationProcessResult(Status.ERROR, error=str(e)))


def run_translation_process(
    worker_fn: Callable,
    worker_args_without_queue: tuple = (),
) -> Binary:
    """
    Runs `worker_fn` (function doing some translation pass from above) in a separate
    process, returns produced flatbuffer `Binary` instance if no errors happend,
    otherwise raises RuntimeError.
    """
    result: TranslationProcessResult = _run_worker_in_separate_process(
        worker_fn, worker_args_without_queue
    )

    logger = Logger()
    file_manager = FileManager(logger)
    return Binary(logger, file_manager, result.fb_file_path)


# ---------- Utility wrappers around ttrt ----------


@dataclass
class RunProcessResult:
    status: Status
    return_code: int = None
    error: str = None


def run_flatbuffer_worker(flatbuffer_file_path: str, result_queue: Queue) -> None:
    """
    Runs flatbuffer given as path to flatbuffer file on device.

    Wrapper around runtime `Run` API.

    It is not resistant to segfaults, i.e. some unpredictable errors that can happen.
    Thus it is meant to be used as a worker for a Process which will guard the caller
    from such errors.
    """
    try:
        API.initialize_apis()
        run_instance = API.Run(args={"binary": flatbuffer_file_path})
        return_code, _ = run_instance()

        result_queue.put(RunProcessResult(Status.SUCCESS, return_code))
    except Exception as e:
        result_queue.put(RunProcessResult(Status.ERROR, error=str(e)))


def run_flatbuffer_execution_process(flatbuffer: Binary) -> int:
    """
    Runs `flatbuffer` on device.

    This is a segfault resistant function. It runs the pybound translation pass in a
    separate process, thus protecting the caller of this function from any unpredictable
    (those that cannot be caught with a try-except) errors.

    Returns
    -------
    Return code of `ttrt run flatbuffer.file_path` process.

    Raises
    ------
    RuntimeError if any errors happen.
    """
    result: RunProcessResult = _run_worker_in_separate_process(
        run_flatbuffer_worker, (flatbuffer.file_path,)
    )
    return result.return_code
