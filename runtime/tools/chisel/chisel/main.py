# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import pathlib
import numpy as np
import torch
import time
import os
import sys
import logging
from typing import Tuple

from ttmlir.ir import Context

from chisel.core.ops import IRModule
from chisel.core.registry import Registry
from chisel.core.golden_executor import GoldenExecutor
from chisel.core.tensors import (
    TensorPool,
    get_torch_tensor,
    update_device_tensor,
    TensorValue,
    get_function_inputs,
)
from chisel.core.enums import ExecutionType
from chisel.utils.metrics import compute_pcc, compute_abs_err, compute_rel_err
from chisel.utils.location import parse_op_location
from chisel.utils.mapping import ttir_dtype_maps

from ttrt.common.api import API as RtApi
from ttrt.common.util import Logger as RtLogger
from ttrt.common.util import Artifacts as RtArtifacts
from ttrt.runtime import (
    DebugHooks,
    get_op_input_refs,
    get_op_output_ref,
    get_op_debug_str,
    get_op_loc_info,
    get_tensor,
    DataType,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("chisel")

import pdb
import traceback

timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
csvfile = f"pccdata_{timestamp}.csv"


class ChiselContext:
    def __init__(
        self,
        input_dir: pathlib.Path,
        output_dir: pathlib.Path,
        op_config: pathlib.Path,
        main_fn: str,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.op_config = op_config
        self.main_fn = main_fn

        self.ttnn_path: pathlib.Path = self.input_dir / "ttnn.mlir"
        self.ttir_path: pathlib.Path = self.input_dir / "ttir.mlir"
        self.flatbuffer_path: pathlib.Path = self.input_dir / "fb.ttnn"

        self.context = Context()
        self.context.load_all_available_dialects()

        logger.debug("Loading IRs...")
        self.device_ir_module = IRModule(
            mlir_path=self.ttnn_path,
            context=self.context,
            execution_type=ExecutionType.DEVICE,
            main_op_name=self.main_fn,
        )
        self.golden_ir_module = IRModule(
            mlir_path=self.ttir_path,
            context=self.context,
            execution_type=ExecutionType.GOLDEN,
            main_op_name=self.main_fn,
        )

        modules = {
            ExecutionType.DEVICE: self.device_ir_module,
            ExecutionType.GOLDEN: self.golden_ir_module,
        }

        self.registry = Registry(modules)
        self.golden_tensor_pool = TensorPool()
        self.device_tensor_pool = TensorPool()
        self.executor = GoldenExecutor(self.registry, self.golden_tensor_pool)
        self.load_inputs()

        logger.debug("Setting up TTRT...")
        args = {
            "binary": str(self.flatbuffer_path),
            "save-artifacts": True,
        }
        self.rt_logger = RtLogger()
        self.rt_artifacts = RtArtifacts(
            logger=self.rt_logger, artifacts_folder_path=str(self.output_dir)
        )
        RtApi.initialize_apis()
        self.rt_api = RtApi.Run(
            args=args, logger=self.rt_logger, artifacts=self.rt_artifacts
        )

        logger.debug("Setting up TTRT hooks...")
        self.setup_ttrt_hooks()

        if not (self.output_dir / "intermediates").exists():
            (self.output_dir / "intermediates").mkdir(parents=True, exist_ok=True)
        if not (self.output_dir / "goldens").exists():
            (self.output_dir / "goldens").mkdir(parents=True, exist_ok=True)

    def generate_inputs(self):
        pass

    def compare_outputs(self, op_location: Tuple[int, int]):
        last_device_op = self.registry.get_last(op_location, ExecutionType.DEVICE)
        last_golden_op = self.registry.get_last(op_location, ExecutionType.GOLDEN)

        if last_device_op is None or last_golden_op is None:
            return

        if len(last_device_op.outputs) == 0 or len(last_golden_op.outputs) == 0:
            return

        device_output_name = last_device_op.outputs[0].name
        golden_output_name = last_golden_op.outputs[0].name

        device_output = self.device_tensor_pool[device_output_name].data
        golden_output = self.golden_tensor_pool[golden_output_name].data

        pcc = compute_pcc(device_output, golden_output)
        abs_err = compute_abs_err(device_output, golden_output)
        rel_err = compute_rel_err(device_output, golden_output)
        print("-" * 100)
        print(f"Comparing outputs for {op_location}")
        print(f"Device output: {device_output_name}")
        print(f"Golden output: {golden_output_name}")
        print(f"PCC: {pcc}, Abs err: {abs_err}, Rel err: {rel_err}")
        print("Golden output: ", golden_output)
        print("Device output: ", device_output)
        print("-" * 100)
        pdb.set_trace()

    def preop(self, binary, programContext, opContext):
        debug_str = get_op_debug_str(opContext)
        op_location = parse_op_location(get_op_loc_info(opContext))

        input_refs = get_op_input_refs(opContext, programContext)
        if len(input_refs) == 0:
            return

        op = self.registry.find_op(op_location, debug_str, ExecutionType.DEVICE)
        if op is None:
            return
        if len(op.inputs) == 0:
            return

        for i, input_ref in enumerate(input_refs):
            input_tensor = get_tensor(programContext, input_ref)
            if input_tensor is None:
                continue
            torch_tensor = get_torch_tensor(input_tensor)

            print(f"Input {i} is {torch_tensor}")
            input_name = op.inputs[i].name
            if input_name not in self.device_tensor_pool:
                continue

            tensor_value = self.device_tensor_pool[input_name]
            if tensor_value.execution_data is not None:
                input_tensor = get_tensor(programContext, input_ref)
                update_device_tensor(
                    programContext, input_ref, input_tensor, tensor_value.execution_data
                )
                tensor_value.tensor_ref = input_ref
                tensor_value.execution_data = None
                tensor_value.tensor = get_tensor(programContext, input_ref)
                tensor_value.data = get_torch_tensor(tensor_value.tensor)

    def load_inputs(self):
        for arg in get_function_inputs(
            self.device_ir_module.get_main_op(), ExecutionType.DEVICE
        ):
            tensor_path = self.input_dir / "tensors" / f"{arg.name}.pt"
            print(tensor_path)
            assert (
                tensor_path.exists()
            ), f"Input tensor {arg.name} not found on path {tensor_path}"
            tensor = torch.load(tensor_path) * 2
            tensor = tensor.to(ttir_dtype_maps[str(arg.dtype)])
            print(
                f"Loaded tensor {arg.name} with shape {tensor.shape} with dtype {tensor.dtype}"
            )
            print(f"Tensor: {tensor}")
            self.device_tensor_pool[arg.name] = TensorValue(
                arg.name, tensor, ExecutionType.DEVICE
            )
            self.golden_tensor_pool[arg.name] = TensorValue(
                arg.name, tensor, ExecutionType.GOLDEN
            )
            self.device_tensor_pool[arg.name].execution_data = self.device_tensor_pool[
                arg.name
            ].data

    def postop(self, binary, programContext, opContext):
        debug_str = get_op_debug_str(opContext)
        op_location = parse_op_location(get_op_loc_info(opContext))
        # save the device output to the device tensor pool

        device_op = self.registry.find_op(op_location, debug_str, ExecutionType.DEVICE)
        if device_op is None:
            return
        if len(device_op.outputs) == 0:
            return

        output_ref = get_op_output_ref(opContext, programContext)
        if output_ref is None:
            return
        output_tensor = get_tensor(programContext, output_ref)

        output_name = device_op.outputs[0].name
        if output_name not in self.device_tensor_pool:
            self.device_tensor_pool[output_name] = TensorValue(
                output_name, None, ExecutionType.DEVICE
            )
        tensor_value = self.device_tensor_pool[output_name]
        tensor_value.tensor_ref = output_ref
        tensor_value.data = get_torch_tensor(output_tensor)
        tensor_value.tensor = output_tensor

        if self.executor.execute_golden(op_location, debug_str):
            self.compare_outputs(op_location)

    def debug_preop(self, binary, programContext, opContext):
        try:
            self.preop(binary, programContext, opContext)
        except Exception as e:
            traceback.print_exc()
            pdb.set_trace()
            raise e

    def debug_postop(self, binary, programContext, opContext):
        try:
            self.postop(binary, programContext, opContext)
        except Exception as e:
            traceback.print_exc()
            pdb.set_trace()
            raise e

    def setup_ttrt_hooks(self):
        callback_env_pre = DebugHooks.get(self.debug_preop, self.debug_postop)

    def run(self):
        # self.run_prerequisites()
        logger.info("Running runtime...")
        result_code, results = self.rt_api()


def main():
    model = "llama_debug"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=pathlib.Path, default=f"/localdev/ndrakulic/chisel/{model}"
    )
    parser.add_argument("--ttnn_path", type=pathlib.Path, default=None)
    parser.add_argument("--ttir_path", type=pathlib.Path, default=None)
    parser.add_argument("--flatbuffer_path", type=pathlib.Path, default=None)
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=f"/localdev/ndrakulic/chisel/{model}/output",
    )
    parser.add_argument("--op_config", type=pathlib.Path)
    parser.add_argument("--main_fn", type=str, default="forward")
    args = parser.parse_args()

    chisel_context = ChiselContext(
        args.input_dir, args.output_dir, args.op_config, args.main_fn
    )
    chisel_context.run()


if __name__ == "__main__":
    main()
