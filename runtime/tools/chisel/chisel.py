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

from core.managers import TTIROpManager, TTNNOpManager, tie_manager_ops
from core.validator import Validator
from core.value import TensorStatus
from core.ttir_executor import TTIRExecutor

from utils.mapping import ttir_dtype_maps

from ttmlir.ir import Context, Module
from ttmlir.dialects import ttir, ttnn


from ttrt.common.api import API as RtApi
from ttrt.common.util import Logger as RtLogger
from ttrt.common.util import Artifacts as RtArtifacts
from ttrt.runtime import (
    DebugHooks,
    get_op_input_refs,
    get_op_output_ref,
    get_op_debug_str,
    get_tensor,
    DataType,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("chisel")


import pdb

timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
csvfile = f"pccdata_{timestamp}.csv"

# sys.excepthook = lambda *args: pdb.pm()


class ChiselContext:
    def __init__(self, args):
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.op_config = args.op_config

        self.ttnn_path: pathlib.Path = (
            self.input_dir / "ttnn.mlir" if not args.ttnn_path else args.ttnn_path
        )
        self.ttir_path: pathlib.Path = (
            self.input_dir / "ttir.mlir" if not args.ttir_path else args.ttir_path
        )

        self.flatbuffer_path: pathlib.Path = (
            self.input_dir / "fb.ttnn"
            if not args.flatbuffer_path
            else args.flatbuffer_path
        )

        self.context = Context()
        # ttir.register_dialect(self.context)
        # ttnn.register_dialect(self.context)
        self.context.load_all_available_dialects()

        logger.debug("Loading IRs...")
        self.load_irs()
        logger.debug("Setting up TTRT...")
        self.setup_ttrt()
        logger.debug("Setting up TTRT hooks...")
        self.setup_ttrt_hooks()

        if not (self.output_dir / "intermediates").exists():
            (self.output_dir / "intermediates").mkdir(parents=True, exist_ok=True)
        if not (self.output_dir / "goldens").exists():
            (self.output_dir / "goldens").mkdir(parents=True, exist_ok=True)

        self.current_ttir_loc = None
        self.current_ttnn_loc = None

        self.current_ttir_op = None

        self.ttir_executor = TTIRExecutor()

        self.validator = Validator()

        self.ttnn_op_idx = 0

        self.pcc_data = []

        self.should_skip = set()

    def generate_inputs(self):
        mlir_inputs = self.ttir_module.body.operations[0].type.inputs
        arg_attrs = self.ttir_module.body.operations[0].arg_attrs
        self.tensor_inputs = {}
        for i, input in enumerate(mlir_inputs):
            # check if exsitst input_dir/tensors/i.pt
            if os.path.exists(self.input_dir / f"tensors/{i}.pt"):
                tensor = torch.load(self.input_dir / f"tensors/{i}.pt")
                self.tensor_inputs[f"%arg{i}"] = tensor
                continue
            attrs = arg_attrs[i]
            # ttir_name_attr = attrs["ttir.name"]
            # if ttir_name_attr is None:
            #     raise ValueError(f"No 'ttir.name' attribute found for input {i}")

            # ttir_name = ttir_name_attr.value  # this is the string value
            data_format = input.element_type.__str__()
            dtype = ttir_dtype_maps[data_format]
            shape = input.shape
            name = f"%arg{i}"
            # Heuristics for principled init
            # Please prefer to load tensors from disk if available
            if dtype in (torch.float32, torch.bfloat16):
                # Assume that 2D float tensors are weights, apply LeCun init
                tensor = torch.randn(shape)
                if len(shape) == 2 or len(shape) == 4:
                    # Assume that 4D float tensors are conv kernels, and 2D linear weights
                    # we don't know which axis is which(conventions differ between torch and jax)
                    # so we assume two largest axes are in and out channels
                    # and apply Glorot init to be agnostic to which is which
                    sorted_shape = sorted(shape)
                    a = sorted_shape[-2]
                    b = sorted_shape[-1]
                    factor = (2 / (a + b)) ** 0.5
                    tensor.mul_(factor)
            else:
                # random ones ought to work good enough for both masks and indices
                tensor = (torch.randn(shape) > 0.5).to(dtype)
            self.tensor_inputs[name] = tensor
        self.ttir_executor.tensor_pool.update(self.tensor_inputs)

    def load_irs(self):
        logger.debug("Opening ttnn...")
        with open(self.ttnn_path, "r") as f:
            ttnn_text = f.read()
        self.ttnn_module = Module.parse(ttnn_text, self.context)
        self.ttnn_manager = TTNNOpManager(self.ttnn_module)

        logger.debug("Opening ttir...")
        with open(self.ttir_path, "r") as f:
            ttir_text = f.read()
        self.ttir_module = Module.parse(ttir_text, self.context)
        self.ttir_manager = TTIROpManager(self.ttir_module)
        logger.debug("Creating manager...")

        logger.debug("Tie manager ops")
        self.op_groups, self.ttnn_ops_list = tie_manager_ops(
            self.ttir_manager, self.ttnn_manager
        )
        for op_group in list(self.op_groups.values()):
            logger.debug(f"-----GROUP-----")
            logger.debug(op_group)

    def setup_ttrt(self):
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

    def setup_ttrt_hooks(self):
        def preop(binary, programContext, opContext):
            logger.debug("===============PREOP============")
            logger.debug(self.ttnn_op_idx)
            if self.ttnn_op_idx >= len(self.ttnn_ops_list):
                return
            tensor_refs = get_op_input_refs(opContext, programContext)
            self.current_ttnn_op = self.ttnn_ops_list[self.ttnn_op_idx]
            self.ttnn_manager.populate_op(self.current_ttnn_op)

            if not self.current_ttnn_op.inputs:
                return
            for i, tensor_ref in enumerate(tensor_refs):
                skipped = False
                tensor = get_tensor(programContext, tensor_ref) if tensor_ref else None
                op_input: TensorValue = self.current_ttnn_op.inputs[i]
                op_input.tensor_ref = tensor_ref
                op_input.tensor = tensor
                input_name = op_input.name
                if (
                    input_name not in self.tensor_inputs
                    and input_name not in self.should_skip
                ):
                    continue
                if input_name in self.tensor_inputs:
                    input_tensor = self.tensor_inputs[input_name]
                if input_name in self.should_skip:
                    # print(self.should_skip)
                    # import pdb; pdb.set_trace()
                    input_tensor = self.ttir_executor.tensor_pool[
                        self.validator.ttnn2ttir_tensor[input_name]
                    ]
                    # It might happen that when skipping. the shapes don't match even tho the data is right
                    # fetch the shape from the TTIR ir op and reshape the tensor
                    skipped = True
                if op_input.status != TensorStatus.NOT_INITIALIZED:
                    continue
                logger.debug(
                    f"Input {i}: {tensor.get_shape() if tensor is not None else None}"
                )
                logger.debug(
                    f"Op input {i}: {input_tensor.shape if input_tensor is not None else None}"
                )
                if not skipped:
                    assert tensor.get_shape() == list(
                        input_tensor.shape
                    ), f"Input {i} shape mismatch. Got {input_tensor.shape}, expected {tensor.get_shape()}"
                else:
                    input_tensor = input_tensor.reshape(tensor.get_shape())
                op_input.set_device_data(input_tensor, programContext)

        def postop(binary, programContext, opContext):
            logger.debug("===============POSTOP============")
            logger.debug(self.current_ttnn_op.name)
            logger.debug(get_op_debug_str(opContext))

            tensor_ref = get_op_output_ref(opContext, programContext)
            tensor = get_tensor(programContext, tensor_ref) if tensor_ref else None
            if tensor is not None and len(self.current_ttnn_op.outputs) > 0:
                self.current_ttnn_op.outputs[0].tensor_ref = tensor_ref
                self.current_ttnn_op.outputs[0].tensor = tensor

            target_groups = []
            # see if there are previous groups with no ttnn ops that need to be recomputed
            missing_groups = []
            for i in range(self.current_ttnn_op.line_no - 1, -1, -1):
                if i not in self.op_groups:
                    break
                group = self.op_groups[i]
                if len(group.ttnn) > 0 or group.computed_ttir:
                    break
                missing_groups.append(group)
            missing_groups.reverse()
            target_groups.extend(missing_groups)

            target_group = self.op_groups[self.current_ttnn_op.line_no]
            target_groups.append(target_group)

            for group in filter(lambda g: not g.computed_ttir, target_groups):
                logger.debug(f"***RUNNING TTIR*** {group.ttir}")
                for op in group.ttir:
                    self.ttir_manager.populate_op(op)
                    self.ttir_executor.execute_op(op, programContext)
                group.computed_ttir = True
            self.validator.validate(self.current_ttnn_op, target_group, self, False)
            self.validator.export_csv(self.output_dir / csvfile)

            for tensor_value in self.current_ttnn_op.outputs:
                if tensor_value.tt_data is not None:
                    data = tensor_value.tt_data
                    if data.dtype == torch.uint32:
                        data = data.clone().float()
                    torch.save(
                        data,
                        self.output_dir
                        / "intermediates"
                        / f"{tensor_value.name[1:]}.pt",
                    )

            # if self.current_ttnn_op.name == "ttnn.add":
            #     # check for dtype of the output tensor
            #     if tensor.get_dtype() in [DataType.Int32]:
            #         self.should_skip.add(self.current_ttnn_op.outputs[0].name)

            # if self.current_ttnn_op.name == "ttnn.conv2d":
            #     self.should_skip.add(self.current_ttnn_op.outputs[0].name)

            self.ttnn_op_idx += 1

        def debug_preop(binary, programContext, opContext):
            try:
                preop(binary, programContext, opContext)
            except Exception as e:
                # print stacktrace
                import traceback

                traceback.print_exc()
                import pdb

                pdb.set_trace()
                raise e

        def debug_postop(binary, programContext, opContext):
            try:
                postop(binary, programContext, opContext)
            except Exception as e:
                import traceback

                traceback.print_exc()
                import pdb

                pdb.set_trace()
                raise e

        callback_env_pre = DebugHooks.get(debug_preop, debug_postop)

    def set_inputs(self, inputs):
        self.tensor_inputs = {f"%arg{idx}": input for idx, input in enumerate(inputs)}
        self.ttir_executor.tensor_pool.update(self.tensor_inputs)

    def run(self):
        # self.run_prerequisites()
        logger.info("Running runtime...")
        result_code, results = self.rt_api()


def create_and_save_arrays(array_specs, arrays_file):
    logger.info("Creating arrays and saving to disk...")
    start_time = time.time()
    arrays = []
    for i, (shape, dtype) in enumerate(array_specs):
        arr = np.empty(shape=shape, dtype=dtype)
        arr.fill(1)
        arrays.append(arr)
    np.savez_compressed(arrays_file, *arrays)
    elapsed = time.time() - start_time
    logger.info(f"Arrays created and saved in {elapsed:.4f} seconds")
    logger.info(f"File size: {os.path.getsize(arrays_file) / (1024*1024):.2f} MB")


def load_tensor_list(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".pt", ".pth"]:
        data = torch.load(file_path)
    elif ext == ".npy":
        data = np.load(file_path, allow_pickle=True)
        data = data.tolist()
    elif ext == ".npz":
        npz_data = np.load(file_path)
        data = [npz_data[key] for key in npz_data.files]
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    tensor_list = [
        torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr for arr in data
    ]

    return tensor_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=pathlib.Path, default=".")
    parser.add_argument("--ttnn_path", type=pathlib.Path, default=None)
    parser.add_argument("--ttir_path", type=pathlib.Path, default=None)
    parser.add_argument("--flatbuffer_path", type=pathlib.Path, default=None)
    parser.add_argument("--output_dir", type=pathlib.Path, default=".")
    parser.add_argument("--op_config", type=pathlib.Path)
    parser.add_argument(
        "--inputs_path", type=pathlib.Path, default="cached_numpy_arrays.npz"
    )
    args = parser.parse_args()

    # inputs_specs = [
    #     ((1, 6), np.int32),
    #     ((1, 6), np.int32),
    #     ((128,), np.float32),
    #     ((128,), np.float32),
    #     ((512, 128), np.float32),
    #     ((2, 128), np.float32),
    #     ((30000, 128), np.float32),
    #     ((768,), np.float32),
    #     ((768,), np.float32),
    #     ((768,), np.float32),
    #     ((768, 768), np.float32),
    #     ((768,), np.float32),
    #     ((768, 768), np.float32),
    #     ((768,), np.float32),
    #     ((768, 768), np.float32),
    #     ((768,), np.float32),
    #     ((768, 768), np.float32),
    #     ((3072,), np.float32),
    #     ((768, 3072), np.float32),
    #     ((768,), np.float32),
    #     ((3072, 768), np.float32),
    #     ((768,), np.float32),
    #     ((768,), np.float32),
    #     ((768,), np.float32),
    #     ((128, 768), np.float32),
    #     ((128,), np.float32),
    #     ((128,), np.float32),
    #     ((30000,), np.float32),
    #     ((128,), np.float32),
    #     ((768, 128), np.float32),
    #     ((1, 6), np.int32),
    # ]
    # arrays_file = args.inputs_path
    # create_and_save_arrays(inputs_specs, arrays_file)
    # inputs = load_tensor_list(arrays_file)

    chisel_context = ChiselContext(args)
    # chisel_context.set_inputs(inputs)
    with Context():
        chisel_context.generate_inputs()
    chisel_context.run()

    logger.debug(chisel_context.pcc_data)

    # chisel_context.liveness_analysis()


if __name__ == "__main__":
    main()
