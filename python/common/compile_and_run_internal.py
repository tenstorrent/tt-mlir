# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from multiprocessing import queues
from typing import Callable, Tuple

from ttmlir.ir import Module
from ttmlir.passes import (
    stablehlo_to_ttir_pipeline,
    ttir_to_ttmetal_backend_pipeline,
    ttir_to_ttnn_backend_pipeline,
    ttmetal_to_flatbuffer_file,
    ttnn_to_flatbuffer_file,
)
from ttrt.common.api import API
from ttrt.common.util import Binary, FileManager, Logger

from .compile_and_run_utils import *

# ---------- Utility wrappers around compiler passes ----------


def stablehlo_to_ttir_pipeline_worker(
    module_str: str, result_queue: queues.Queue
) -> None:
    """
    Wrapper around `stablehlo_to_ttir_pipeline` pybound pass.

    It is not resistant to segfaults, i.e. some unpredictable errors that can happen
    inside the pybound call. Thus it is meant to be used as a worker for a mp.Process
    which will guard the caller from such errors.
    """
    try:
        module = create_mlir_module_from_string(module_str)

        stablehlo_to_ttir_pipeline(module)

        result_queue.put(CompilationProcessResult.success(str(module)))
    except Exception as e:
        result_queue.put(CompilationProcessResult.error(str(e)))


def ttir_to_ttnn_backend_pipeline_worker(
    module_str: str, system_desc: str, result_queue: queues.Queue
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

        result_queue.put(CompilationProcessResult.success(str(module)))
    except Exception as e:
        result_queue.put(CompilationProcessResult.error(str(e)))


def ttir_to_ttmetal_backend_pipeline_worker(
    module_str: str, system_desc: str, result_queue: queues.Queue
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

        result_queue.put(CompilationProcessResult.success(str(module)))
    except Exception as e:
        result_queue.put(CompilationProcessResult.error(str(e)))


# ---------- Utility wrappers around translation passes ----------


def ttnn_to_flatbuffer_file_worker(
    module_str: str, output_file_name: str, result_queue: queues.Queue
) -> None:
    """
    Wrapper around `ttnn_to_flatbuffer_file` pybound pass.

    It is not resistant to segfaults, i.e. some unpredictable errors that can happen
    inside the pybound call. Thus it is meant to be used as a worker for a Process
    which will guard the caller from such errors.
    """
    try:
        module = create_mlir_module_from_string(module_str)

        ttnn_to_flatbuffer_file(module, output_file_name, {}, {})

        result_queue.put(TranslationProcessResult.success(output_file_name))
    except Exception as e:
        result_queue.put(TranslationProcessResult.error(str(e)))


def ttmetal_to_flatbuffer_file_worker(
    module_str: str, output_file_name: str, result_queue: queues.Queue
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

        result_queue.put(TranslationProcessResult.success(output_file_name))
    except Exception as e:
        result_queue.put(TranslationProcessResult.error(str(e)))


# ---------- Utility wrappers around ttrt ----------


def run_flatbuffer_worker(
    flatbuffer_file_path: str, result_queue: queues.Queue
) -> None:
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

        result_queue.put(RunProcessResult.success(return_code))
    except Exception as e:
        result_queue.put(RunProcessResult.error(str(e)))


# ---------- Public API ----------


def run_compilation_process(
    worker_fn: Callable,
    worker_args_without_queue: Tuple = (),
) -> Module:
    """
    Runs `worker_fn` (function doing some compilation pass) in a separate process,
    returns produced Module if no errors happened, otherwise raises RuntimeError.
    """
    process_manager = get_process_manager()
    result: CompilationProcessResult = process_manager.run(
        worker_fn, worker_args_without_queue
    )
    return create_mlir_module_from_string(result.module_str)


def run_translation_process(
    worker_fn: Callable,
    worker_args_without_queue: Tuple = (),
) -> Binary:
    """
    Runs `worker_fn` (function doing some translation pass) in a separate process,
    returns produced flatbuffer `Binary` instance if no errors happened, otherwise
    raises RuntimeError.
    """

    def create_binary_from_fb_file(fb_file_path: str) -> Binary:
        logger = Logger()
        file_manager = FileManager(logger)
        return Binary(logger, file_manager, fb_file_path)

    process_manager = get_process_manager()
    result: TranslationProcessResult = process_manager.run(
        worker_fn, worker_args_without_queue
    )
    return create_binary_from_fb_file(result.fb_file_path)


def run_flatbuffer_execution_process(
    worker_fn: Callable,
    worker_args_without_queue: Tuple = (),
) -> int:
    """
    Runss `worker_fn` (function doing flatbuffer run on device) in a separate process,
    returns return code of `ttrt run flatbuffer.file_path` process if no errors
    happened, otherwise raises RuntimeError.
    """
    process_manager = get_process_manager()
    result: RunProcessResult = process_manager.run(worker_fn, worker_args_without_queue)
    return result.return_code
