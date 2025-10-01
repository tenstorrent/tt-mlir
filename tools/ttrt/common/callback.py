# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re
from functools import partial
import csv
import json

from ttrt.common.util import *
from builder.base.builder_golden import get_golden_function

try:
    from ttmlir.dialects import ttir
except ImportError:
    ttir = None

# Mapping from TTNN operation names to corresponding TTIR operation classes
TTNN_TO_TTIR_OP_MAPPING = {}


def _init_ttnn_to_ttir_mapping():
    """Initialize the TTNN to TTIR operation mapping if ttir module is available."""
    if ttir is None:
        return

    # Arithmetic operations
    TTNN_TO_TTIR_OP_MAPPING.update(
        {
            "ttnn.add": ttir.AddOp,
            "ttnn.multiply": ttir.MultiplyOp,
            "ttnn.subtract": ttir.SubtractOp,
            "ttnn.divide": ttir.DivOp,
        }
    )

    # Comparison operations
    TTNN_TO_TTIR_OP_MAPPING.update(
        {
            "ttnn.eq": ttir.EqualOp,
            "ttnn.ne": ttir.NotEqualOp,
            "ttnn.gt": ttir.GreaterThanOp,
            "ttnn.ge": ttir.GreaterEqualOp,
            "ttnn.lt": ttir.LessThanOp,
            "ttnn.le": ttir.LessEqualOp,
        }
    )

    # Logical operations
    TTNN_TO_TTIR_OP_MAPPING.update(
        {
            "ttnn.logical_and": ttir.LogicalAndOp,
            "ttnn.logical_or": ttir.LogicalOrOp,
            "ttnn.logical_xor": ttir.LogicalXorOp,
            "ttnn.logical_not": ttir.LogicalNotOp,
        }
    )

    # Bitwise operations
    TTNN_TO_TTIR_OP_MAPPING.update(
        {
            "ttnn.bitwise_and": ttir.BitwiseAndOp,
            "ttnn.bitwise_or": ttir.BitwiseOrOp,
            "ttnn.bitwise_xor": ttir.BitwiseXorOp,
            "ttnn.bitwise_not": ttir.BitwiseNotOp,
        }
    )

    # Mathematical operations
    TTNN_TO_TTIR_OP_MAPPING.update(
        {
            "ttnn.abs": ttir.AbsOp,
            "ttnn.neg": ttir.NegOp,
            "ttnn.sqrt": ttir.SqrtOp,
            "ttnn.rsqrt": ttir.RsqrtOp,
            "ttnn.reciprocal": ttir.ReciprocalOp,
            "ttnn.exp": ttir.ExpOp,
            "ttnn.log": ttir.LogOp,
            "ttnn.sin": ttir.SinOp,
            "ttnn.cos": ttir.CosOp,
            "ttnn.tan": ttir.TanOp,
            "ttnn.atan": ttir.AtanOp,
            "ttnn.tanh": ttir.TanhOp,
            "ttnn.sigmoid": ttir.SigmoidOp,
            "ttnn.relu": ttir.ReluOp,
            "ttnn.relu6": ttir.Relu6Op,
            "ttnn.gelu": ttir.GeluOp,
            "ttnn.leaky_relu": ttir.LeakyReluOp,
        }
    )

    # Matrix operations
    TTNN_TO_TTIR_OP_MAPPING.update(
        {
            "ttnn.matmul": ttir.MatmulOp,
            "ttnn.dot_general": ttir.DotGeneralOp,
        }
    )

    # Other operations
    TTNN_TO_TTIR_OP_MAPPING.update(
        {
            "ttnn.where": ttir.WhereOp,
            "ttnn.typecast": ttir.TypecastOp,
            "ttnn.to_layout": ttir.ToLayoutOp,
            "ttnn.get_dimension_size": ttir.GetDimensionSizeOp,
        }
    )


# Initialize the mapping
_init_ttnn_to_ttir_mapping()


def get_ttir_op_class(ttnn_op_name):
    """
    Get the corresponding TTIR operation class for a given TTNN operation name.

    Args:
        ttnn_op_name (str): The TTNN operation name (e.g., "ttnn.add", "ttnn.multiply")

    Returns:
        class: The corresponding TTIR operation class (e.g., ttir.AddOp, ttir.MultiplyOp),
               or None if no mapping exists
    """
    return TTNN_TO_TTIR_OP_MAPPING.get(ttnn_op_name, None)


class CallbackRuntimeConfig:
    def __init__(
        self,
        device=None,
        artifact_dir="",
        pcc=0.99,
        atol=1e-08,
        rtol=1e-05,
        save_golden_tensors=False,
        logging=None,
        enable_golden=False,
        enable_memory=False,
        enable_debugger=False,
        golden_report={},
        memory_report={},
    ):
        self.device = device
        self.artifact_dir = artifact_dir
        self.pcc = pcc
        self.atol = atol
        self.rtol = rtol
        self.save_golden_tensors = save_golden_tensors
        self.logging = logging
        self.enable_golden = enable_golden
        self.enable_memory = enable_memory
        self.enable_debugger = enable_debugger
        self.golden_report = golden_report
        self.memory_report = memory_report
        self.counter = -1

    def start_new_callback(self, artifact_dir):
        self.artifact_dir = artifact_dir
        self.counter = -1
        self.golden_report = {}
        self.memory_report = {}

    def callback_counter(self):
        self.counter = self.counter + 1
        return self.counter

    def save_golden_report(self, golden_report_path):
        with open(golden_report_path, "w") as json_file:
            json.dump(self.golden_report, json_file, indent=4)

        self.logging.debug(f"Saved golden report to={golden_report_path}")

    def save_memory_report(self, memory_report_path):
        with open(memory_report_path, "w") as json_file:
            json.dump(self.memory_report, json_file, indent=4)

        self.logging.debug(f"Saved memory report to={memory_report_path}")

    def check_pcc(self):
        for loc, device_data in self.golden_report.items():
            for device_id, golden_data in device_data.items():
                if golden_data["actual_pcc"] < golden_data["expected_pcc"]:
                    raise PCCErrorException(
                        f"Failed: golden comparison failed at loc={loc} for device={device_id}, actual_pcc={golden_data['actual_pcc']} < expected_pcc={golden_data['expected_pcc']}"
                    )

    def check_memory_leak(self):
        num_items = 0
        for key, value in self.memory_report.items():
            num_items += 1

        if num_items == 0:
            self.logging.warning(f"No memory data found")
        else:
            # query initial memory usage
            dram_initial_size_per_device = self.memory_report[0]["dram"]
            l1_initial_size_per_device = self.memory_report[0]["l1"]

            # query final memory usage and ensure no memory leaks
            dram_final_size_per_device = self.memory_report[num_items - 1]["dram"]
            l1_final_size_per_device = self.memory_report[num_items - 1]["l1"]

            for key, value in dram_initial_size_per_device.items():
                dram_initial_size = value["total_bytes_allocated_per_bank"]
                dram_final_size = dram_final_size_per_device[key][
                    "total_bytes_allocated_per_bank"
                ]

                if dram_final_size > dram_initial_size:
                    raise Exception(f"Memory leak detected in DRAM for device={key}")

            for key, value in l1_initial_size_per_device.items():
                l1_initial_size = value["total_bytes_allocated_per_bank"]
                l1_final_size = l1_final_size_per_device[key][
                    "total_bytes_allocated_per_bank"
                ]

                if l1_final_size > l1_initial_size:
                    raise Exception(
                        f"Memory leak detected in L1 cache for device={key}"
                    )


"""
-----------------------GOLDEN CALLBACK-----------------------
"""


def golden(callback_runtime_config, binary, program_context, op_context):
    import torch
    import ttrt.runtime

    logging = callback_runtime_config.logging
    logging.debug("executing golden comparison")

    loc = ttrt.runtime.get_op_loc_info(op_context)
    debug_str = ttrt.runtime.get_op_debug_str(op_context)

    # Extract operation name from debug string (e.g., "ttnn.add" from the full debug string)
    import re

    op_name_match = re.search(r'"([^"]+)"', debug_str)
    op_name = op_name_match.group(1) if op_name_match else "unknown"
    print("Dhruv", op_name)

    # Get input tensors
    op_input_tensor_refs = ttrt.runtime.get_op_input_refs(op_context, program_context)
    op_input_tensors_torch = {}
    for i, tensor_ref in enumerate(op_input_tensor_refs):
        tensor = ttrt.runtime.retrieve_tensor_from_pool(program_context, tensor_ref)
        if tensor:
            # Convert to torch tensor
            rt_buffer = tensor.get_data_buffer()
            dtype = ttrt_datatype_to_torch_dtype(tensor.get_dtype())
            input_tensor_torch = torch.frombuffer(rt_buffer, dtype=dtype).flatten()
            op_input_tensors_torch[i] = input_tensor_torch

    op_golden_tensor_map = binary.get_debug_info_golden(loc)
    if len(op_golden_tensor_map) == 0:
        logging.debug("Golden tensor is None - skipping golden comparison")
        return

    op_output_tensor_map = ttrt.runtime.get_op_output_tensor(
        op_context, program_context
    )
    if len(op_output_tensor_map) == 0:
        logging.debug("Output tensor is empty - skipping golden comparison")
        return

    # loop through all devices and compare golden tensors
    device_results = {}
    for device_id, op_golden_tensor in op_golden_tensor_map.items():
        if device_id not in op_output_tensor_map.keys():
            logging.debug(
                f"Device {device_id} does not have an output tensor - skipping golden comparison"
            )
            continue

        op_output_tensor = op_output_tensor_map[device_id]
        rt_buffer = op_output_tensor.get_data_buffer()
        dtype = ttrt_datatype_to_torch_dtype(op_golden_tensor.dtype)
        # Get the corresponding TTIR operation class and its golden function
        ttir_op_class = get_ttir_op_class(op_name)
        if ttir_op_class:
            golden_func = get_golden_function(ttir_op_class)
            if golden_func:
                golden_tensor_torch = golden_func(
                    *op_input_tensors_torch.values()
                ).flatten()
            else:
                logging.warning(f"No golden function found for operation {op_name}")
                return
        else:
            logging.warning(f"No TTIR operation class found for operation {op_name}")
            return

        output_tensor_torch = torch.frombuffer(rt_buffer, dtype=dtype).flatten()
        print(op_name + " golden comparison op by op")
        print(golden_tensor_torch)
        print(output_tensor_torch)
        if callback_runtime_config.save_golden_tensors:
            golden_tensor_torch_name = get_sanitized_filename(
                f"{loc}_{device_id}_golden.pt"
            )
            device_tensor_torch_name = get_sanitized_filename(
                f"{loc}_{device_id}_device.pt"
            )
            torch.save(
                golden_tensor_torch,
                f"{callback_runtime_config.artifact_dir}/{golden_tensor_torch_name}",
            )
            torch.save(
                output_tensor_torch,
                f"{callback_runtime_config.artifact_dir}/{device_tensor_torch_name}",
            )

        if golden_tensor_torch.shape != output_tensor_torch.shape:
            logging.debug(
                "Golden and output tensor shapes do not match - skipping golden comparison"
            )
            return

        _, _, cal_pcc, output_str = get_atol_rtol_pcc(
            golden_tensor_torch, output_tensor_torch, logging
        )

        # Handle case where tensor has only one element.
        if golden_tensor_torch.numel() == 1:
            cal_pcc = (
                1.0
                if torch.nn.functional.cosine_similarity(
                    golden_tensor_torch.float().unsqueeze(0),
                    output_tensor_torch.float().unsqueeze(0),
                ).item()
                else 0.0
            )

        logging.debug(f"For device {device_id} at loc={loc}, PCC={cal_pcc}")
        logging.debug(output_str)

        results = {}
        results["expected_pcc"] = callback_runtime_config.pcc
        results["actual_pcc"] = cal_pcc
        results["atol"] = callback_runtime_config.atol
        results["rtol"] = callback_runtime_config.rtol
        results["allclose"] = torch.allclose(
            golden_tensor_torch,
            output_tensor_torch,
            atol=callback_runtime_config.atol,
            rtol=callback_runtime_config.rtol,
        )
        results["max"] = torch.max(
            torch.abs(golden_tensor_torch - output_tensor_torch)
        ).item()
        results["mean_absolute_error"] = torch.mean(
            torch.abs(golden_tensor_torch.float() - output_tensor_torch.float())
        ).item()
        results["root_mean_square_error"] = torch.sqrt(
            torch.mean((golden_tensor_torch.float() - output_tensor_torch.float()) ** 2)
        ).item()
        results["cosine_similarity"] = torch.nn.functional.cosine_similarity(
            golden_tensor_torch.float().unsqueeze(0),
            output_tensor_torch.float().unsqueeze(0),
        ).item()

        device_results[device_id] = results

    callback_runtime_config.golden_report[loc] = device_results


"""
-----------------------MEMORY CALLBACK-----------------------
"""


def create_memory_dictionary(memory_view):
    memory_dict = {}
    memory_dict["num_banks"] = memory_view.num_banks
    memory_dict["total_bytes_per_bank"] = memory_view.total_bytes_per_bank
    memory_dict[
        "total_bytes_allocated_per_bank"
    ] = memory_view.total_bytes_allocated_per_bank
    memory_dict["total_bytes_free_per_bank"] = memory_view.total_bytes_free_per_bank
    memory_dict[
        "largest_contiguous_bytes_free_per_bank"
    ] = memory_view.largest_contiguous_bytes_free_per_bank
    memory_dict["block_table"] = memory_view.block_table

    return memory_dict


def memory(callback_runtime_config, binary, program_context, op_context):
    import ttrt.runtime

    device = callback_runtime_config.device
    logging = callback_runtime_config.logging
    logging.debug("executing memory dump")
    loc = ttrt.runtime.get_op_loc_info(op_context)
    debug_str = ttrt.runtime.get_op_debug_str(op_context)

    memory_views = device.get_memory_view()
    dram_memory_view = memory_views[ttrt.runtime.MemoryBufferType.DRAM]
    l1_memory_view = memory_views[ttrt.runtime.MemoryBufferType.L1]
    l1_small_memory_view = memory_views[ttrt.runtime.MemoryBufferType.L1_SMALL]
    trace_memory_view = memory_views[ttrt.runtime.MemoryBufferType.TRACE]

    op_memory_report = {}
    op_memory_report["loc"] = loc
    op_memory_report["debug_str"] = debug_str
    op_memory_report["dram"] = create_memory_dictionary(dram_memory_view)
    op_memory_report["l1"] = create_memory_dictionary(l1_memory_view)
    op_memory_report["l1_small"] = create_memory_dictionary(l1_small_memory_view)
    op_memory_report["trace"] = create_memory_dictionary(trace_memory_view)

    callback_runtime_config.memory_report[
        callback_runtime_config.callback_counter()
    ] = op_memory_report


"""
-----------------------DEBUGGER CALLBACK-----------------------
"""


def debugger(callback_runtime_config, binary, program_context, op_context):
    import pdb
    import ttrt.runtime

    device = callback_runtime_config.device
    logging = callback_runtime_config.logging
    logging.debug("starting pdb debugger")
    pdb.set_trace()


def pre_op_callback(callback_runtime_config, binary, program_context, op_context):
    # Pre_callback logic to be implemented here
    pass


def pre_op_get_callback_fn(callback_runtime_config):
    return partial(pre_op_callback, callback_runtime_config)


def post_op_callback(callback_runtime_config, binary, program_context, op_context):

    if callback_runtime_config.enable_golden:
        golden(callback_runtime_config, binary, program_context, op_context)

    if callback_runtime_config.enable_memory:
        memory(callback_runtime_config, binary, program_context, op_context)

    if callback_runtime_config.enable_debugger:
        debugger(callback_runtime_config, binary, program_context, op_context)


def post_op_get_callback_fn(callback_runtime_config):
    return partial(post_op_callback, callback_runtime_config)
