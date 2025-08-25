# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math
import hashlib
from collections import namedtuple

from .kernel_types import *


# Import ttnn for only the op.py module
try:
    import ttnn
except Exception as e:
    ttnn = e


OpCircularBuffer = namedtuple(
    "OpCircularBuffer", ["cb_type", "cb_format", "cb_descriptor"]
)
OpKernel = namedtuple("OpKernel", ["kernel_type", "kernel_descriptor"])


class PyKernelOp:
    """
    Base class for PyKernel Operations, contains a variety of methods to be overriden.
    """

    def __init__(self):
        """Initialize the PyKernelOp with an empty kernel selection dictionary. Intakes the `ttnn` module to operate."""
        self.kernel_selection = {}
        self.kernel_cache = {}
        self.tensor_accessor_config = 0

        # Keep a mobile statewise reference to the ttnn module
        if isinstance(ttnn, Exception):
            raise ttnn
        self.ttnn = ttnn

    def _tile_bytes_from_ttnn_dtype(self, dtype):
        # Retrieved from: https://github.com/tenstorrent/tt-buda/blob/main/pybuda/csrc/balancer/balancer_utils.hpp
        match dtype:
            case 0 | 6:  # BFP16, U16
                return 32 * 32 * 2
            case 1 | 2 | 7:
                return 32 * 32 * 4
            case 3:
                return 32 * 32 + 64
            case 5:
                return 32 * 32
            case 4:
                return 512 + 64
            case _:
                return "Invalid DataType Processed"

    def _mlir_dtype_from_ttnn_dtype(self, dtype):
        match dtype:
            case 0:
                return "BFloat16"
            case 1:
                return "Float32"
            case 2:
                return "UInt32"
            case 5:
                return "UInt8"
            case 6:
                return "UInt16"
            case 7:
                return "Int32"
            case _:
                return "Invalid"

    def get_core_ranges(self, core_ranges=None):
        if core_ranges is None:
            if not hasattr(self, "defined_core_ranges"):
                raise ValueError(
                    "Trying to retrieve core ranges before function initialization"
                )
            return self.defined_core_ranges
        return core_ranges

    def define_core_ranges(self, tensors, options):
        """Define Core Ranges from tensors and options."""
        # Default implementation - subclasses should override
        # Returns a 0, 0 core range:
        core = self.ttnn.CoreCoord(0, 0)
        return self.ttnn.CoreRangeSet([self.ttnn.CoreRange(core, core)])

    def set_tensor_accessor_config(self, tensors):
        """
        Set the tensor accessor config based on the passed tensors.
        Right now, the only relevant flags are IsDram and Sharded.
        """
        config = TensorAccessorConfig.NONE
        if not tensors:
            raise ValueError("Must provide at least one tensor.")
        if not isinstance(tensors, (list, tuple)):
            tensors = [tensors]
        memory_config = tensors[0].memory_config()

        if memory_config.buffer_type == self.ttnn.BufferType.DRAM:
            config |= TensorAccessorConfig.IsDram
        if memory_config.is_sharded():
            config |= TensorAccessorConfig.Sharded

        self.tensor_accessor_config = config

    def _compute_input_hash(self, tensors, options):
        """Compute a hash of the input tensors and compile-time options."""
        # Simplified implementation - should be enhanced for production
        hash_input = str(
            [(getattr(t, "shape", None), getattr(t, "dtype", None)) for t in tensors]
        )
        hash_input += str(options)
        return hashlib.md5(hash_input.encode()).hexdigest()

    def invoke(self, *tensors, **options):
        # Default Implementation - Returns None and throws error
        return None

    def _config_from_thread_type(self, thread_name):
        match thread_name:
            case "reader_thread":
                return self.ttnn.ReaderConfigDescriptor
            case "writer_thread":
                return self.ttnn.WriterConfigDescriptor
            case "compute_thread":
                return self.ttnn.ComputeConfigDescriptor
            case _:
                raise ValueError(
                    f"Kernel must be decorated with valid thread option. Thread Name Provided: {thread_name}"
                )

    def make_ct_args(self, **ct_args):
        return [CompileTimeValue(k, v) for k, v in ct_args.items()]

    def get_buffer_addr(self, tensor):
        return tensor.buffer_address()

    def create_cb(self, tensor, buffer_index, core_ranges=None, **options):
        # Utility function to create CB given tensor. Returns an OpCircularBuffer Object
        dtype = tensor.dtype
        page_size = self._tile_bytes_from_ttnn_dtype(int(dtype))
        num_tiles = math.ceil(tensor.volume() / 1024)
        total_size = num_tiles * page_size

        # Define core_ranges
        core_ranges = self.get_core_ranges(core_ranges)

        cb_format = self.ttnn.CBFormatDescriptor(
            buffer_index=buffer_index, data_format=dtype, page_size=page_size
        )

        cb_desc = self.ttnn.CBDescriptor(
            total_size=total_size,
            core_ranges=core_ranges,
            format_descriptors=[cb_format],
        )

        cb_type = CircularBuffer(
            buffer_index,
            tuple(tensor.shape),
            dtype=self._mlir_dtype_from_ttnn_dtype(int(dtype)),
        )

        return OpCircularBuffer(cb_type, cb_format, cb_desc)

    def create_rt_args(self, core_ranges=None):
        # Argument_dict is a dict with keys of Core types, and values of list[int]

        # Check for various instances of core types
        # Get the core grid size from the define_core_ranges function
        core_grid = self.get_core_ranges(core_ranges)
        grid_size = core_grid.bounding_box().grid_size()

        result = [[0 for j in range(grid_size.y)] for i in range(grid_size.x)]

        return result

    # Use common runtime args for scalars, otherwise use arg_val for list of lists and template function to resolve this.
    # Who knew function signature theory and resolution order would be relevant one day
    def create_kernel(self, kernel, *args, core_ranges=None, **kwargs):
        # Resolve core_ranges
        core_ranges = self.get_core_ranges(core_ranges)

        # Resolve arguments from list of list structure
        # Check for all arguments

        cb_args = []
        common_rt_args = []
        arg_idx = 0
        common_idx = 0
        rt_args = None
        all_rt_args = []

        for arg in args:
            if isinstance(arg, OpCircularBuffer):
                cb_args.append(arg)
            elif isinstance(arg, int):
                # Scalar Integer, treat as a CommonRuntimeArg
                _arg = Arguments.make_common(arg, common_idx)
                common_rt_args.append(arg)
                all_rt_args.append(_arg)
                common_idx += 1
                arg_idx += 1
            elif isinstance(arg, list):
                # Make sure it's a list of lists
                if not arg:
                    raise IndexError("Empty list provided as positional argument.")
                if not isinstance(arg[0], (list, tuple)):
                    raise TypeError(
                        "Core-Specific RT Args must be formatted as a list of lists spanning the core_range"
                    )
                if not all(
                    all(all(isinstance(x, int) for x in _args) for _args in row)
                    for row in arg
                ):
                    raise TypeError(
                        "Core-Specific RT Args must all be integer values across the 2D Core Grid."
                    )

                # just a placeholder so that the IR parses the RT arg as an int type (uses get_arg_val)
                all_rt_args.append(0)

                # Construct the mega rt_args
                if rt_args is None:
                    # initialize to size of list provided
                    rt_args = arg
                else:
                    # List already initialized, append to each element
                    for i, row in enumerate(rt_args):
                        for j, _list in enumerate(row):
                            _list.extend(arg[i][j])

                arg_idx += 1

        ct_const_args = self.make_ct_args(**kwargs)

        if rt_args is None:
            rt_args = [[[]]]

        # Get the PyKernel type for cb_args
        cb_args = [x.cb_type for x in args if isinstance(x, OpCircularBuffer)]

        kernel_string = kernel(*cb_args, *all_rt_args, *ct_const_args)
        kernel_t = Kernel(kernel.__name__, kernel_string)
        kernel_path = kernel_t.dump_to_file()

        config = self._config_from_thread_type(kernel._decorator_name)
        compile_time_args = [cb.cb_id for cb in cb_args] + [self.tensor_accessor_config]

        kernel_desc_args = {
            "kernel_source": kernel_path,
            "core_ranges": self.get_core_ranges(core_ranges),
            "compile_time_args": compile_time_args,
            "runtime_args": rt_args,
            "config": config(),
        }

        if common_rt_args:
            kernel_desc_args["common_runtime_args"] = common_rt_args

        kernel_desc = self.ttnn.KernelDescriptor(**kernel_desc_args)

        return OpKernel(kernel_t, kernel_desc)

    def create_program(self, kernels, cbs, semaphores=[]):
        return self.ttnn.ProgramDescriptor(
            kernels=[kernel.kernel_descriptor for kernel in kernels],
            cbs=[cb.cb_descriptor for cb in cbs],
            semaphores=semaphores,
        )

    def __call__(self, *tensors, **options):
        """
        Execute the kernel operation on the given tensors with specified options.

        Args:
            tensors: Input tensors
            options: Additional options for kernel execution

        Returns:
            Result tensor(s) from the operation
        """

        # Invoked at "runtime", set the defined_core_ranges from the define_core_ranges function
        self.defined_core_ranges = self.define_core_ranges(tensors, options)

        # Compute hash for kernel selection and caching
        input_hash = self._compute_input_hash(tensors, options)

        # Check if kernel is already cached
        if input_hash not in self.kernel_cache:
            program_descriptor = self.invoke(*tensors, **options)
            if program_descriptor is None:
                raise Exception("invoke method must be defined for PyKernelOp.")

            # Store the compiled kernel in the cache
            self.kernel_cache[input_hash] = {
                "program_descriptor": program_descriptor,
            }

        # Retrieve cached kernel information and execute
        cached_info = self.kernel_cache[input_hash]
        program = cached_info["program_descriptor"]

        return self.ttnn.generic_op(tensors, program)
