# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel.types import *
import hashlib
import inspect
import logging

from collections import namedtuple
import math


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

    def set_common_runtime_args(self, args):
        self.common_runtime_args = args
        return args

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
        return [CompiledValue(k, v) for k, v in ct_args.items()]

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
            tuple(tensor.shape)[1:],
            dtype=self._mlir_dtype_from_ttnn_dtype(int(dtype)),
        )

        return OpCircularBuffer(cb_type, cb_format, cb_desc)

    def set_args(self, arguments: Arguments, core, *args):
        if not (
            all(isinstance(x, int) for x in args)
            or all(isinstance(x, CompiledValue) for x in args)
        ):
            raise TypeError("Arguments must be a list of integers")

        args = list(args)

        if isinstance(core, ttnn.CoreCoord):
            arguments.set_args_at_core(core.x, core.y, args)
        elif isinstance(core, ttnn.CoreRange):
            for x in range(core.start.x, core.end.x):
                for y in range(core.start.y, core.end.y):
                    arguments.set_args_at_core(x, y, args)
        elif isinstance(core, ttnn.CoreRangeSet):
            for rng in core.ranges():
                for x in range(rng.start.x, rng.end.x):
                    for y in range(rng.start.y, rng.end.y):
                        arguments.set_args_at_core(x, y, args)

    def create_rt_args(self, arguments: dict = {}, core_ranges=None):
        # Argument_dict is a dict with keys of Core types, and values of list[int]

        # Check for various instances of core types
        # Get the core grid size from the define_core_ranges function
        core_grid = self.get_core_ranges(core_ranges)
        grid_size = core_grid.bounding_box().grid_size()

        result = Arguments(grid_size.x, grid_size.y)

        # Define for each of the cores specifically
        for core, args in arguments.items():
            self.set_args(result, core, *args)

        return result

    def create_ct_args(self, arguments: dict = {}, core_ranges=None):
        core_grid = self.get_core_ranges(core_ranges)
        grid_size = core_grid.bounding_box().grid_size()

        result = Arguments(grid_size.x, grid_size.y)

        # Specific core definitions
        for core, _args in arguments.items():
            args = [CompiledValue(k, v) for k, v in _args.items()]
            self.set_args(result, core, *args)

        return result

    # Who knew function signature theory and resolution order would be relevant one day
    def create_kernel(
        self, kernel, *args, rt_args=None, ct_args=None, core_ranges=None, **kwargs
    ):
        # Resolve core_ranges
        core_ranges = self.get_core_ranges(core_ranges)

        # Resolve arguments
        _rt_args = [x for x in args if not isinstance(x, OpCircularBuffer)]
        if rt_args is not None and _rt_args:
            raise TypeError("Can't define both rt_args and positional rt_args")
        elif _rt_args:
            # If positional arguments are defined we will override the value of rt_args
            rt_args = self.create_rt_args({core_ranges: _rt_args})
        elif rt_args is None and not _rt_args:
            rt_args = self.create_rt_args()

        if ct_args is not None and kwargs:
            raise TypeError("Can't define both ct_args and keyword ct_args")
        elif kwargs:
            # If kwargs are defined, we will override the value of ct_args
            ct_args = self.create_ct_args({core_ranges: kwargs})
        elif ct_args is None and not kwargs:
            ct_args = self.create_ct_args()

        # Get the PyKernel type for cb_args
        cb_args = [x.cb_type for x in args if isinstance(x, OpCircularBuffer)]

        kernel_string = kernel(*cb_args, *rt_args.get_args(), *ct_args.get_args())
        kernel_t = Kernel(kernel.__name__, kernel_string)
        kernel_path = kernel_t.dump_to_file()

        config = self._config_from_thread_type(kernel._decorator_name)

        kernel_desc_args = {
            "kernel_source": kernel_path,
            "core_ranges": self.get_core_ranges(core_ranges),
            "compile_time_args": [cb.cb_id for cb in cb_args],
            "runtime_args": rt_args.get_all_args(),
            "config": config(),
        }

        print(kernel.__name__, kernel_desc_args)

        if hasattr(self, "common_runtime_args"):
            kernel_desc_args["common_runtime_args"] = self.common_runtime_args

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
