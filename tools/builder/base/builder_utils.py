# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import inspect
import time
import torch
from functools import reduce
import operator
from typing import Callable, List, Optional, Tuple, Union, Literal, Dict
from collections import OrderedDict

from ttmlir.ir import *
from ttmlir.dialects import func, ttcore, ttnn, ttir
from ttmlir.passmanager import PassManager
from ttmlir.passes import (
    tt_populate_argument_types,
    ttir_to_ttnn_backend_pipeline,
    ttnn_to_flatbuffer_file,
    ttir_to_ttmetal_backend_pipeline,
    ttmetal_to_flatbuffer_file,
    translate_to_cpp,
    translate_to_python,
    MLIRModuleLogger,
    stablehlo_pipeline,
    stablehlo_to_ttir_pipeline,
    ttir_to_emitpy_pipeline,
)

from builder.base.builder import *
from builder.ttir.ttir_builder import TTIRBuilder
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.d2m.d2m_builder import D2MBuilder

# Imports for runtime execution
import ttrt.runtime
from ttrt.common.util import (
    Logger,
    FileManager,
    Binary,
    golden_tensor_to_torch,
    ttrt_datatype_to_torch_dtype,
    get_atol_rtol_pcc,
    parse_fabric_config,
)


# ----- Exception Classes -----


class TTBuilderCompileException(Exception):
    """Exception raised when builder compilation fails during compile_ttir_to_flatbuffer."""

    pass


class TTBuilderRuntimeException(Exception):
    """Exception raised when compiled builder code fails during runtime execution.

    This exception is reserved for future use when runtime execution is implemented.
    """

    pass


class TTBuilderGoldenException(Exception):
    """Exception raised when builder output doesn't match expected golden results.

    This exception is reserved for future use when golden verification is implemented.
    """

    pass


class BuilderCompileConfig:
    """
    Holds configuration for building, compiling and executing builder-based pipelines.
    Provides methods that mirror the previous module-level functions, but with most
    arguments captured as instance variables.
    """

    def __init__(
        self,
        system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
        output_root: str = ".",
        target: Literal["ttnn", "ttmetal", "emitc", "emitpy"] = "ttnn",
        module_dump: bool = True,
        mesh_name: str = "mesh",
        mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
        argument_types_string: Optional[str] = None,
        custom_pipeline: Optional[Union[Callable, str]] = None,
        pipeline_options: Optional[List[str]] = None,
        ttir_pipeline_options: Optional[List[str]] = None,
        shlo_pipeline_options: Optional[List[str]] = None,
        shlo_to_ttir_pipeline_options: Optional[List[str]] = None,
        print_ir: Union[bool, str] = False,
        device=None,
        pcc: float = 0.99,
        atol: float = 1e-08,
        rtol: float = 1e-05,
        disable_golden: bool = False,
        skip_exec: bool = False,
        check_atol: bool = False,
        check_rtol: bool = False,
        default_test_base: str = "test",
    ) -> None:
        # Environment
        self.system_desc_path = system_desc_path
        self.output_root = output_root
        self.target = target
        self.module_dump = module_dump
        # Mesh
        self.mesh_name = mesh_name
        self.mesh_dict = mesh_dict
        # Pipeline controls
        self.argument_types_string = argument_types_string
        self.custom_pipeline = custom_pipeline
        self.pipeline_options = pipeline_options if pipeline_options is not None else []
        self.ttir_pipeline_options = (
            ttir_pipeline_options if ttir_pipeline_options is not None else []
        )
        self.shlo_pipeline_options = (
            shlo_pipeline_options if shlo_pipeline_options is not None else []
        )
        self.shlo_to_ttir_pipeline_options = (
            shlo_to_ttir_pipeline_options
            if shlo_to_ttir_pipeline_options is not None
            else []
        )
        self.print_ir = print_ir
        # Execution / golden
        self.device = device
        self.pcc = pcc
        self.atol = atol
        self.rtol = rtol
        self.disable_golden = disable_golden
        self.skip_exec = skip_exec
        self.check_atol = check_atol
        self.check_rtol = check_rtol
        # Naming
        self.default_test_base = default_test_base
        # Tracks the builder/frontend in use for this config's most recent build
        self.frontend: Optional[str] = None

    # ---------- Build ----------
    def build_module(
        self,
        fn: Callable,
        builder_type: Literal["ttir", "stablehlo", "ttnn", "d2m"],
        inputs_shapes: List[Shape],
        inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
        *,
        test_base: Optional[str] = None,
    ):
        # Update both module-level and instance-level frontend tracking
        global frontend
        frontend = builder_type
        self.frontend = builder_type
        ctx = Context()
        # Source location for easier debugging
        try:
            fname = inspect.getfile(fn)
            line_no = inspect.getsourcelines(fn)[1]
            loc = Location.file(fname, line_no, 0, ctx)
        except (OSError, TypeError):
            loc = Location.unknown(ctx)

        encoding_fn = None
        if builder_type == "ttir":
            builder = TTIRBuilder(ctx, loc, self.mesh_name, self.mesh_dict)
        elif builder_type == "stablehlo":
            builder = StableHLOBuilder(ctx, loc, self.mesh_name, self.mesh_dict)
        elif builder_type == "ttnn":
            builder = TTNNBuilder(ctx, loc)
            encoding_fn = builder.create_tensor_encoding
        elif builder_type == "d2m":
            builder = D2MBuilder(ctx, loc, self.mesh_name, self.mesh_dict)
        else:
            raise ValueError(f"Unsupported builder_type: {builder_type}")
        dir_name = builder_type + "-builder-artifacts"
        mlir_suffix = "_" + builder_type + ".mlir"

        # Default to all f32s
        if inputs_types is None:
            inputs_types = [torch.float32] * len(inputs_shapes)
        if len(inputs_shapes) != len(inputs_types):
            raise ValueError(
                f"inputs_shapes and inputs_types must have the same length: "
                f"{len(inputs_shapes)} != {len(inputs_types)}"
            )

        with ctx, loc:
            fn_input_types = [
                builder._create_ranked_tensor_type(
                    shape,
                    builder._get_type_from_torch_dtype(
                        dtype if isinstance(dtype, torch.dtype) else dtype
                    ),
                    encoding_fn(shape, dtype) if encoding_fn else None,
                )
                for (shape, dtype) in zip(inputs_shapes, inputs_types)
            ]

            module = Module.create()
            if builder_type == "stablehlo":
                module.body.append(builder._get_mesh(self.mesh_name))

            with InsertionPoint(module.body):

                @func.func(*fn_input_types, name=fn.__name__)
                def decorated_func(*inputs):
                    input_goldens: Dict[Operand, GoldenMapTensor] = {}
                    for index, (operand, dtype) in enumerate(zip(inputs, inputs_types)):
                        input_goldens[operand] = builder._generate_golden_tensor(
                            operand, dtype
                        )
                    builder._set_goldens(input_goldens)
                    builder._set_input_ordering(inputs)

                    result = fn(*inputs, builder)

                    outputs = result if hasattr(result, "__iter__") else (result,)
                    output_goldens: Dict[Operand, GoldenMapTensor] = {}
                    for op in outputs:
                        output_goldens[op] = builder._get_golden_tensor(op)
                    builder._set_goldens(output_goldens)
                    builder._set_output_ordering(outputs)

                    return _process_multi_return_result(result)

            print(f"`{fn.__name__}` successfully transformed into a MLIR module.")
            base = fn.__name__ if test_base is None else test_base
            filename = _get_target_path(
                self.output_root, dir_name, base + mlir_suffix, builder_type
            )

            if self.module_dump:
                with open(filename, "w") as f:
                    f.write(str(module))
                    print(module)

            return module, builder

    # ---------- Core compile helpers ----------
    def compile_ttir_module_to_flatbuffer(
        self,
        module: Module,
        builder: Builder,
        *,
        test_base: Optional[str] = None,
        builder_dir: str = "ttir-builder-artifacts",
        goldens: Optional[Dict[Operand, "GoldenMapTensor"]] = None,
    ) -> str:
        pipeline_options = self.pipeline_options or []
        custom_pipeline = self.custom_pipeline
        print_ir = self.print_ir

        if isinstance(custom_pipeline, str):
            custom_pipeline = _create_custom_ttir_pipeline_fn(
                custom_pipeline, print_ir=print_ir
            )

        if self.target == "ttnn":
            pipeline_fn = (
                custom_pipeline if custom_pipeline else ttir_to_ttnn_backend_pipeline
            )
            to_target = ttnn_to_flatbuffer_file
            mlir_suffix = "_ttnn.mlir"
            target_extension = "ttnn"
        elif self.target == "ttmetal":
            pipeline_fn = (
                custom_pipeline if custom_pipeline else ttir_to_ttmetal_backend_pipeline
            )
            to_target = ttmetal_to_flatbuffer_file
            mlir_suffix = "_ttm.mlir"
            target_extension = "ttm"
        elif self.target == "emitc":
            ttir_to_ttnn_emitc_pipeline = _create_custom_ttir_pipeline_fn(
                "ttir-to-emitc-pipeline", print_ir=print_ir
            )
            pipeline_fn = (
                custom_pipeline if custom_pipeline else ttir_to_ttnn_emitc_pipeline
            )
            to_target = _emitc_to_executable
            mlir_suffix = "_ttnn.mlir"
            target_extension = "cpp"
        elif self.target == "emitpy":
            pipeline_fn = (
                custom_pipeline if custom_pipeline else ttir_to_emitpy_pipeline
            )
            to_target = _emitpy_to_executable
            mlir_suffix = "_ttnn.mlir"
            target_extension = "py"
        else:
            raise ValueError("Unsupported target: " + self.target)

        output_file_mlir = _get_target_path(
            self.output_root,
            builder_dir,
            (test_base or self.default_test_base) + mlir_suffix,
            self.target,
        )
        output_file_fbb = ".".join([output_file_mlir, target_extension])

        goldens_map = dict(builder.golden_map) if goldens is None else goldens

        # Compile TTIR MLIR -> TT{Metal,NN} MLIR
        try:
            module = _run_ttir_pipeline(
                module,
                pipeline_fn,
                pipeline_options=pipeline_options,
                dump_to_file=self.module_dump,
                output_file_name=output_file_mlir,
                system_desc_path=self.system_desc_path,
                mesh_dict=self.mesh_dict,
                argument_types_string=self.argument_types_string,
            )
        except Exception as e:
            raise TTBuilderCompileException(e)

        print(f"{self.target} pipeline ran successfully.")

        module_logger = MLIRModuleLogger()
        module_logger.attach_context(module.context)

        # Compile TT{Metal,NN} MLIR -> flatbuffer
        try:
            to_target(
                module,
                output_file_fbb,
                goldens_map,
                module_logger.module_log if module_logger.module_log else [],
            )
        except Exception as e:
            raise TTBuilderCompileException(e)

        print(f"{self.target} flatbuffer created successfully at: {output_file_fbb}")

        return output_file_mlir

    def execute_fb(
        self,
        fb_path: str,
        pcc: Optional[float] = None,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        disable_golden: Optional[bool] = None,
        device=None,
        check_atol: Optional[bool] = None,
        check_rtol: Optional[bool] = None,
    ) -> None:
        """
        Execute a flatbuffer and optionally validate against embedded goldens.
        Defaults fall back to this config's settings when not provided.
        """
        pcc = self.pcc if pcc is None else pcc
        atol = self.atol if atol is None else atol
        rtol = self.rtol if rtol is None else rtol
        disable_golden = (
            self.disable_golden if disable_golden is None else disable_golden
        )
        device = self.device if device is None else device
        check_atol = self.check_atol if check_atol is None else check_atol
        check_rtol = self.check_rtol if check_rtol is None else check_rtol

        assert device is not None

        def create_tensor(tensor):
            isEmptyTensor = not all(tensor.shape)
            if isEmptyTensor:
                return ttrt.runtime.create_owned_host_tensor(
                    tensor.data_ptr(),
                    list(tensor.shape),
                    list(tensor.stride()),
                    tensor.element_size(),
                    Binary.Program.to_data_type(tensor.dtype),
                )
            return ttrt.runtime.create_borrowed_host_tensor(
                tensor.data_ptr(),
                list(tensor.shape),
                list(tensor.stride()),
                tensor.element_size(),
                Binary.Program.to_data_type(tensor.dtype),
            )

        def convert_input_layouts(device, inputs, fbb, program_index):
            import ttrt.runtime

            inputs_converted = []
            for input_index in range(len(inputs)):
                input_layout = ttrt.runtime.get_layout(fbb, program_index, input_index)
                inputs_converted.append(
                    ttrt.runtime.to_layout(
                        inputs[input_index], device, input_layout, True
                    )
                )
            return inputs_converted

        logger = Logger()
        logging = logger.get_logger()
        file_manager = FileManager(logger)

        print(f"Begining flatbuffer execution on {fb_path}")

        bin = Binary(logger, file_manager, fb_path)
        logging.info(f"evaluating binary={bin.file_path}")
        program_indices = list(range(bin.get_num_programs()))

        for program_index in program_indices:
            print(f"evaluating program={program_index} for binary={bin.file_path}")
            program = bin.get_program(program_index)

            if program.is_private():
                continue

            # Fetch goldens
            golden_inputs = []
            for i in range(program.num_inputs()):
                golden_tensor = {}
                if not disable_golden:
                    golden_tensor = bin.fbb.get_debug_info_golden(f"input_{i}")
                if len(golden_tensor) != 0:
                    golden_tensor = golden_tensor[0]
                    golden_tensor_torch = golden_tensor_to_torch(golden_tensor)
                    golden_inputs.append(golden_tensor_torch)

            program.populate_inputs(torch.randn, golden_inputs)
            program.populate_outputs(torch.zeros)

            inputs = [create_tensor(i) for i in program.input_tensors]
            outputs = [create_tensor(i) for i in program.output_tensors]

            if not disable_golden:
                golden_outputs_torch = []
                for idx in range(0, len(program.output_tensors)):
                    golden_tensor = {}
                    golden_tensor = bin.fbb.get_debug_info_golden(f"output_{idx}")
                    if len(golden_tensor) != 0:
                        golden_tensor = golden_tensor[0]
                        golden_tensor_torch = golden_tensor_to_torch(golden_tensor)
                        golden_outputs_torch.append(golden_tensor_torch)

            inputs = convert_input_layouts(device, inputs, bin.fbb, program_index)
            logging.debug(f"starting exectution of binary={bin.file_path}")

            start_submit = time.perf_counter_ns()
            try:
                runtime_outputs = ttrt.runtime.submit(
                    device,
                    bin.fbb,
                    program_index,
                    inputs,
                )
                ttrt.runtime.wait(runtime_outputs)
            except Exception as e:
                raise TTBuilderRuntimeException(e)
            end_submit = time.perf_counter_ns()
            e2e_duration_nanoseconds_submit = end_submit - start_submit
            e2e_duration_nanoseconds_output = 0

            for i, runtime_output_tensor in enumerate(runtime_outputs):
                start_get_output = time.perf_counter_ns()
                output_host = ttrt.runtime.to_host(
                    runtime_output_tensor, untilize=True
                )[0]
                end_get_output = time.perf_counter_ns()
                e2e_duration_nanoseconds_output += end_get_output - start_get_output

                ttrt.runtime.memcpy(outputs[i], output_host)
                ttrt.runtime.deallocate_tensor(runtime_output_tensor, force=True)

                output_tensor_torch = None
                if not disable_golden:
                    isEmptyTensor = not all(outputs[i].get_shape())
                    data_buffer = bytearray(outputs[i].get_data_buffer())
                    if isEmptyTensor and len(data_buffer) == 0:
                        output_tensor_torch = torch.empty(
                            outputs[i].get_shape(),
                            dtype=ttrt_datatype_to_torch_dtype(outputs[i].get_dtype()),
                        )
                    elif not isEmptyTensor and len(data_buffer) > 0:
                        output_tensor_torch = torch.frombuffer(
                            data_buffer,
                            dtype=ttrt_datatype_to_torch_dtype(outputs[i].get_dtype()),
                        ).reshape(outputs[i].get_shape())
                    else:
                        raise Exception(
                            f"Failed: Tensor shape=({outputs[i].get_shape()}) and data buffer size={len(data_buffer)} do not match."
                        )

                golden_tensor_torch = None
                if (not disable_golden) and (i < len(golden_outputs_torch)):
                    print(f"executing program level golden comparison for output_{i}")
                    golden_tensor_torch = golden_outputs_torch[i]
                    if golden_tensor_torch.shape != output_tensor_torch.shape:
                        raise TTBuilderGoldenException(
                            f"Failed: program-level output doesn't match golden shape! golden_shape={golden_tensor_torch.shape}, output_shape={output_tensor_torch.shape}"
                        )

                cal_atol, cal_rtol, cal_pcc, _ = get_atol_rtol_pcc(
                    golden_tensor_torch,
                    output_tensor_torch,
                    atol,
                    rtol,
                    logging,
                )

                if cal_pcc < pcc:
                    raise TTBuilderGoldenException(
                        f"Failed: program-level output golden comparison failed, actual_pcc={cal_pcc} < expected_pcc={pcc}"
                    )
                else:
                    print(f"Program level golden for output_{i} matched. pcc={cal_pcc}")

                if check_atol and cal_atol > atol:
                    raise TTBuilderGoldenException(
                        f"Failed: program-level output atol check failed, actual_atol={cal_atol} > expected_atol={atol}"
                    )
                elif check_atol:
                    print(
                        f"Program level atol check for output_{i} passed. atol={cal_atol}"
                    )

                if check_rtol and cal_rtol > rtol:
                    raise TTBuilderGoldenException(
                        f"Failed: program-level output rtol check failed, actual_rtol={cal_rtol} > expected_rtol={rtol}"
                    )
                elif check_rtol:
                    print(
                        f"Program level rtol check for output_{i} passed. rtol={cal_rtol}"
                    )

            print("Adding program results...")
            bin.add_program_results(
                program_index,
                1,
                e2e_duration_nanoseconds_submit,
                e2e_duration_nanoseconds_output,
            )

            print(f"input tensors for program={program_index}")
            for tensor in program.input_tensors:
                logging.debug(f"{tensor}\n")
            print(f"output tensors for program={program_index}")
            for tensor in program.output_tensors:
                logging.debug(f"{tensor}\n")

    # ---------- Compile entrypoints (module creation + backend pipeline) ----------
    def compile_ttir_to_flatbuffer(
        self,
        fn: Callable,
        inputs_shapes: List[Shape],
        inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
        *,
        test_base: Optional[str] = None,
    ) -> str:
        module, builder = self.build_module(
            fn, "ttir", inputs_shapes, inputs_types, test_base=test_base
        )
        return self.compile_ttir_module_to_flatbuffer(
            module=module,
            builder=builder,
            test_base=test_base,
            builder_dir="ttir-builder-artifacts",
        )

    def compile_ttnn_to_flatbuffer(
        self,
        fn: Callable,
        inputs_shapes: List[Shape],
        inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
        *,
        test_base: Optional[str] = None,
    ) -> str:
        module, builder = self.build_module(
            fn, "ttnn", inputs_shapes, inputs_types, test_base=test_base
        )
        return self.compile_ttir_module_to_flatbuffer(
            module=module,
            builder=builder,
            test_base=test_base,
            builder_dir="ttnn-builder-artifacts",
        )

    def compile_d2m_to_flatbuffer(
        self,
        fn: Callable,
        inputs_shapes: List[Shape],
        inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
        *,
        test_base: Optional[str] = None,
    ) -> str:
        module, builder = self.build_module(
            fn, "d2m", inputs_shapes, inputs_types, test_base=test_base
        )
        return self.compile_ttir_module_to_flatbuffer(
            module=module,
            builder=builder,
            test_base=test_base,
            builder_dir="d2m-builder-artifacts",
        )

    def compile_stablehlo_to_flatbuffer(
        self,
        fn: Callable,
        inputs_shapes: List[Shape],
        inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
        *,
        test_base: Optional[str] = None,
    ) -> str:
        # Build StableHLO module
        module, builder = self.build_module(
            fn, "stablehlo", inputs_shapes, inputs_types, test_base=test_base
        )
        goldens = dict(builder.golden_map)

        stablehlo_pipeline(module, " ".join(self.shlo_pipeline_options))
        filename = _get_target_path(
            self.output_root,
            "stablehlo-builder-artifacts",
            (test_base or self.default_test_base) + "_shlo_pipeline.mlir",
            "shlo_pipeline",
        )
        if self.module_dump:
            with open(filename, "w") as f:
                f.write(str(module))

        stablehlo_to_ttir_pipeline(module, " ".join(self.shlo_to_ttir_pipeline_options))
        filename = _get_target_path(
            self.output_root,
            "stablehlo-builder-artifacts",
            (test_base or self.default_test_base) + "_ttir.mlir",
            "ttir",
        )
        if self.module_dump:
            with open(filename, "w") as f:
                f.write(str(module))

        return self.compile_ttir_module_to_flatbuffer(
            module=module,
            builder=builder,
            test_base=test_base,
            builder_dir="stablehlo-builder-artifacts",
            goldens=goldens,
        )

    # ---------- Compile+Execute convenience wrappers ----------
    def _target_extension(self) -> Optional[str]:
        if self.target == "ttnn":
            return "ttnn"
        if self.target == "ttmetal":
            return "ttm"
        return None

    def _maybe_execute(self, mlir_path: str) -> None:
        if self.skip_exec:
            raise TTBuilderRuntimeException("Manually skipped execution")
        ext = self._target_extension()
        if ext is None:
            return
        fb_path = f"{mlir_path}.{ext}"
        self.execute_fb(fb_path)

    def compile_and_execute_ttir(
        self,
        fn: Callable,
        inputs_shapes: List[Shape],
        inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
        *,
        test_base: Optional[str] = None,
    ) -> str:
        mlir_path = self.compile_ttir_to_flatbuffer(
            fn, inputs_shapes, inputs_types, test_base=test_base
        )
        self._maybe_execute(mlir_path)
        return mlir_path

    def compile_and_execute_ttnn(
        self,
        fn: Callable,
        inputs_shapes: List[Shape],
        inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
        *,
        test_base: Optional[str] = None,
    ) -> str:
        mlir_path = self.compile_ttnn_to_flatbuffer(
            fn, inputs_shapes, inputs_types, test_base=test_base
        )
        self._maybe_execute(mlir_path)
        return mlir_path

    def compile_and_execute_d2m(
        self,
        fn: Callable,
        inputs_shapes: List[Shape],
        inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
        *,
        test_base: Optional[str] = None,
    ) -> str:
        mlir_path = self.compile_d2m_to_flatbuffer(
            fn, inputs_shapes, inputs_types, test_base=test_base
        )
        self._maybe_execute(mlir_path)
        return mlir_path

    def compile_and_execute_shlo(
        self,
        fn: Callable,
        inputs_shapes: List[Shape],
        inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
        *,
        test_base: Optional[str] = None,
    ) -> str:
        mlir_path = self.compile_stablehlo_to_flatbuffer(
            fn, inputs_shapes, inputs_types, test_base=test_base
        )
        self._maybe_execute(mlir_path)
        return mlir_path


def get_metal_tensor_layout(
    ctx: Context,
    logical_shape: Shape,
    tiled=False,
    oobVal=ttcore.OOBVal.Undef,
    memorySpace=ttcore.MemorySpace.DeviceL1,
    grid: Optional[Tuple[int, int]] = None,
    index_map: Optional[AffineMap] = None,
    memory_layout: Optional[
        ttcore.TensorMemoryLayout
    ] = ttcore.TensorMemoryLayout.Sharded,
) -> RankedTensorType:
    """
    Create a metal tensor layout.

    This function creates metal tensor layouts for both TTIR and D2M operations.
    Previously duplicated between TTIRBuilder and D2MBuilder.

    Parameters
    ----------
    ctx : Context
        MLIR context
    logical_shape : Shape
        Logical shape of the tensor
    tiled : bool
        Whether to use tiled layout (32x32 tiles)
    oobVal : ttcore.OOBVal
        Out-of-bounds value handling
    memorySpace : ttcore.MemorySpace
        Memory space (L1, DRAM, etc.)
    grid : Optional[Tuple[int, int]]
        Grid shape for sharding
    index_map : Optional[AffineMap]
        Optional affine map for layout transformation

    Returns
    -------
    RankedTensorType
        The metal tensor type with layout
    """
    # Create grid shape by 1s filling logical rank.
    if grid is None:
        original_rank = len(logical_shape)
        grid_shape = [1] * original_rank
    else:
        grid_shape = list(grid)

    # Create layout with original logical shape.
    if index_map is None:
        layout = ttcore.ir.MetalLayoutAttr.get(
            ctx, logical_shape, oobVal, memorySpace, memory_layout
        )
    else:
        layout = ttcore.ir.MetalLayoutAttr.get(
            ctx,
            logical_shape,
            oobVal,
            memorySpace,
            memory_layout,
            index_map,
        )

    shard_shape = []
    for l, g in zip(logical_shape, grid_shape):
        assert l % g == 0, f"Logical shape {l} must be divisible by grid shape {g}"
        shard_shape.append(l // g)

    # Get sharded shape w/ proper collapse & alignment logic.
    typed_layout = ttcore.ir.MetalLayoutAttr.maybe_downcast(layout)
    if typed_layout is None:
        raise RuntimeError("Failed to downcast MetalLayoutAttr")
    device_shape = typed_layout.getDeviceShape(
        grid_shape, [32, 32] if tiled else [1, 1]
    )

    elemType = F32Type.get(ctx)

    # For tiled layouts, ensure the device shape accounts for tiles.
    if tiled:
        elemType = ttcore.ir.TileType.get(ctx, 32, 32, ttcore.DataType.Float32)
        if grid is None or grid == (1, 1):
            # For default 1x1 grid, use exact tile count.
            tile_count_h = (logical_shape[-2] + 31) // 32
            tile_count_w = (logical_shape[-1] + 31) // 32
            device_shape[-2] = tile_count_h
            device_shape[-1] = tile_count_w
        else:
            # For explicit grids, calculate proper sharded tile count.
            shard_h, shard_w = shard_shape[-2], shard_shape[-1]
            tiles_per_shard_h = (shard_h + 31) // 32
            tiles_per_shard_w = (shard_w + 31) // 32
            device_shape[-2] = tiles_per_shard_h
            device_shape[-1] = tiles_per_shard_w

    return RankedTensorType.get(device_shape, elemType, layout, Location.unknown(ctx))


# ----- Private APIs -----


def _compile_and_execute(
    compile_fn: Callable,
    target: Literal["ttnn", "ttmetal", "emitc", "emitpy"],
    pcc: float,
    atol: float,
    rtol: float,
    disable_golden: bool,
    device,
    skip_exec: bool = False,
    check_atol: bool = False,
    check_rtol: bool = False,
    **compile_kwargs,
) -> str:
    """
    Generic function that compiles a builder module to flatbuffer and executes it.

    This is an internal helper that handles the common logic for all compile-and-execute
    entry points.

    Parameters
    ----------
    compile_fn : Callable
        The compilation function to use (e.g., compile_ttir_to_flatbuffer)
    target : Literal["ttnn", "ttmetal", "emitc", "emitpy"]
        Target backend to use
    pcc : float
        PCC threshold for golden comparison
    atol : float
        Absolute tolerance for golden comparison
    rtol : float
        Relative tolerance for golden comparison
    disable_golden : bool
        Whether to disable golden comparison
    device : Optional
        Device to execute on (if None, opens a new device)
    skip_exec: bool
        Whether or not to skip execution in cases of hangs, throwing a `TTBuilderRuntimeException`
    check_atol : bool
        Whether to check absolute tolerance during golden comparison
    check_rtol : bool
        Whether to check relative tolerance during golden comparison
    **compile_kwargs
        All other arguments to pass through to the compile function
    """
    mlir_path = compile_fn(
        target=target,
        **compile_kwargs,
    )

    if skip_exec:
        raise TTBuilderRuntimeException("Manually skipped execution")

    fb_path = mlir_path + "." + ("ttnn" if target == "ttnn" else "ttm")

    # Execute the flatbuffer
    if target in ["ttnn", "ttmetal"]:
        execute_fb(
            fb_path=fb_path,
            pcc=pcc,
            atol=atol,
            rtol=rtol,
            disable_golden=disable_golden,
            device=device,
            check_atol=check_atol,
            check_rtol=check_rtol,
        )

    return mlir_path


def _get_target_path(output_path, builder_dir, filename, target):
    target_dir = os.path.join(output_path, builder_dir, target)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return os.path.join(target_dir, filename)


def _emitc_to_executable(module, filepath: str, golden_map, module_cache):
    py = translate_to_cpp(module)
    with open(filepath, "w") as f:
        f.write(py)


def _emitpy_to_executable(module, filepath: str, golden_map, module_cache):
    cpp = translate_to_python(module)
    with open(filepath, "w") as f:
        f.write(cpp)


def _convert_to_mlir_value(obj):
    if hasattr(obj, "operation") and hasattr(obj.operation, "results"):
        results = obj.operation.results
        if len(results) == 1:
            return results[0]
        else:
            return results
    elif hasattr(obj, "type"):
        return obj
    else:
        return obj


def _process_multi_return_result(result):
    if hasattr(result, "__iter__") and not isinstance(result, str):
        converted_results = []
        for item in result:
            converted = _convert_to_mlir_value(item)
            if hasattr(converted, "__iter__") and not hasattr(converted, "type"):
                converted_results.extend(converted)
            else:
                converted_results.append(converted)
        return tuple(converted_results)
    else:
        return _convert_to_mlir_value(result)


def _create_custom_ttir_pipeline_fn(
    pipeline: str, verify: bool = True, print_ir: Union[bool, str] = False
) -> Callable:
    def wrapper(module, device_register_options):
        register_device = "ttcore-register-device"
        if device_register_options:
            register_device = f"{register_device}{{{device_register_options}}}"

        pipeline_str = f"builtin.module({','.join([register_device, pipeline])})"
        with module.context:
            pm = PassManager.parse(pipeline_str)
            pm.enable_verifier(verify)
            print("Running custom pipeline:", pm)
            if print_ir:
                print_ir_path = print_ir if isinstance(print_ir, str) else None
                pm.enable_ir_printing(tree_printing_dir_path=print_ir_path)
            pm.run(module.operation)

    return wrapper


def _run_ttir_pipeline(
    module,
    pipeline_fn: Callable,
    pipeline_options: Optional[List[str]] = None,
    dump_to_file: bool = True,
    output_file_name: str = "test.mlir",
    system_desc_path: Optional[str] = None,
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    argument_types_string: Optional[str] = None,
):
    if pipeline_options is None:
        pipeline_options = []

    if argument_types_string:
        tt_populate_argument_types(module, argument_types_string)
        pipeline_options.append("enable-const-eval=true")

    # Default to the `SYSTEM_DESC_PATH` envvar.
    if system_desc_path is None:
        system_desc_path = os.getenv("SYSTEM_DESC_PATH", "")
    pipeline_options.append(f"system-desc-path={system_desc_path}")

    mesh_shape = tuple(mesh_dict.values())
    if len(mesh_shape) != 2:
        raise ValueError(f"Mesh shape must be a tuple of length 2, got: {mesh_shape}")

    pipeline_options.append(f"mesh-shape={mesh_shape[0]},{mesh_shape[1]}")

    # Now, pass it through the pipeline. Module gets modified in place.
    pipeline_fn(module, " ".join(pipeline_options))

    # Optionally dump to file.
    if dump_to_file:
        with open(output_file_name, "w") as f:
            f.write(str(module))

    return module


def compile_ttir_module_to_flatbuffer(
    module: Module,
    builder: Builder,
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    test_base: str = "test",
    output_root: str = ".",
    builder_dir: str = "ttir-builder-artifacts",
    target: Literal["ttnn", "ttmetal", "emitc", "emitpy"] = "ttnn",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    module_dump: bool = True,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Optional[Union[Callable, str]] = None,
    pipeline_options: List[str] = None,
    print_ir: Union[bool, str] = False,
    goldens: Dict[Operand, GoldenMapTensor] = None,
):
    """
    Compiles a TTIR MLIR module to flatbuffer format.

    This decorator takes an existing TTIR MLIR module and compiles it through
    the backend pipeline to generate a flatbuffer file. It supports multiple
    targets including TTNN, TTMetal, emitc, and emitpy. It is mainly a wrapper around the following functions, with
    each next function called on the output of the last:

    1. `_run_ttir_pipeline`
    2. `to_target`

    Parameters
    ----------
    module : Module
        The TTIR MLIR module to compile

    builder : *Union[TTIRBuilder, StableHLOBuilder]*
        The builder instance containing golden reference values

    system_desc_path : str, optional
        Path to the system descriptor file

    test_base : str, optional
        The string to be used as the test_base name for dumped files.

    output_root : str, optional
        The path to dump all generated files under

    target : *Literal["ttnn", "ttmetal", "emitpy", "emitc"]*, optional
        The target backend to use. Default is "ttnn"

    mesh_dict : *OrderedDict[str, int]*, optional
        Dictionary that defines the mesh shape.

    module_dump : bool, optional
        Set to True to print out generated MLIR modules. Default is True.

    argument_types_string : *Optional[str]*, optional
        String defining argument types for constant evaluation

    custom_pipeline : *Optional[Union[Callable, str]]*
        Custom pipeline function or string to run instead of default pipeline

    pipeline_options : *List[str]*, optional
        Additional pipeline options to pass to the pipeline

    print_ir : Union[bool, str], optional
        Controls intermediate IR dumping during compilation.
        - True  →  Print IR to stdout after each pass.
                This is convenient for quick inspection or interactive
                debugging (e.g. with breakpoints), but is unreliable if
                the process crashes or aborts as the output may be truncated or
                lost.
        - str (directory path)  →  Write IR after each pass to a separate file
                under the given directory. This is more reliable than stdout,
                since files are flushed incrementally and preserved up to the
                point of failure. It can give hints about where the pipeline
                crashed.
        Notes:
            - For fatal crashes (e.g. MLIR assertions), neither mode guarantees
            a complete dump. Using a directory at least preserves passes run
            before the crash.
            - For stdout mode, you may need to run Python with unbuffered output
            (e.g. `pytest -s` or `python -u`) and/or use pdb to reliably see
            dumps before a crash.
        Default is False (no IR printed).

    goldens : *Optional[Dict[Operand, GoldenMapTensor]]*, optional
        Dictionary of golden tensors to use for comparison. If None, the golden
        tensors will be generated from the builder.
        Default is None.

    Returns
    -------
    str
        The path to the generated target MLIR file

    Raises
    ------
    ValueError
        If an unsupported target is specified
    """

    if pipeline_options is None:
        pipeline_options = []

    if type(custom_pipeline) is str:
        custom_pipeline = _create_custom_ttir_pipeline_fn(
            custom_pipeline, print_ir=print_ir
        )

    pipeline_fn: Callable
    to_target: Callable
    mlir_suffix: str
    target_extension: str

    if target == "ttnn":
        pipeline_fn = (
            custom_pipeline if custom_pipeline else ttir_to_ttnn_backend_pipeline
        )
        to_target = ttnn_to_flatbuffer_file
        mlir_suffix = "_ttnn.mlir"
        target_extension = "ttnn"
    elif target == "ttmetal":
        pipeline_fn = (
            custom_pipeline if custom_pipeline else ttir_to_ttmetal_backend_pipeline
        )
        to_target = ttmetal_to_flatbuffer_file
        mlir_suffix = "_ttm.mlir"
        target_extension = "ttm"
    elif target == "emitc":
        ttir_to_ttnn_emitc_pipeline = _create_custom_ttir_pipeline_fn(
            "ttir-to-emitc-pipeline", print_ir=print_ir
        )
        pipeline_fn = (
            custom_pipeline if custom_pipeline else ttir_to_ttnn_emitc_pipeline
        )
        to_target = _emitc_to_executable
        mlir_suffix = "_ttnn.mlir"
        target_extension = "cpp"
    elif target == "emitpy":
        pipeline_fn = custom_pipeline if custom_pipeline else ttir_to_emitpy_pipeline
        to_target = _emitpy_to_executable
        mlir_suffix = "_ttnn.mlir"
        target_extension = "py"
    else:
        raise ValueError("Unsupported target: " + target)

    output_file_mlir = _get_target_path(
        output_root, builder_dir, test_base + mlir_suffix, target
    )
    output_file_fbb = ".".join([output_file_mlir, target_extension])

    goldens = dict(builder.golden_map) if goldens is None else goldens

    # Compile TTIR MLIR -> TT{Metal,NN} MLIR
    try:
        module = _run_ttir_pipeline(
            module,
            pipeline_fn,
            pipeline_options=pipeline_options,
            dump_to_file=module_dump,
            output_file_name=output_file_mlir,
            system_desc_path=system_desc_path,
            mesh_dict=mesh_dict,
            argument_types_string=argument_types_string,
        )
    except Exception as e:
        raise TTBuilderCompileException(e)

    print(f"{target} pipeline ran successfully.")

    module_logger = MLIRModuleLogger()
    module_logger.attach_context(module.context)

    # Compile TT{Metal,NN} MLIR -> flatbuffer
    try:
        to_target(
            module,
            output_file_fbb,
            goldens,
            module_logger.module_log if module_logger.module_log else [],
        )
    except Exception as e:
        raise TTBuilderCompileException(e)

    print(f"{target} flatbuffer created successfully at: {output_file_fbb}")

    return output_file_mlir


def load_mlir_file(
    mlir_text: str,
    target: Literal["ttir", "ttnn", "d2m", "stablehlo"] = "ttir",
) -> (Module, Builder):
    ctx = Context()
    module = Module.parse(mlir_text, ctx)

    with ctx:
        if target == "ttir":
            builder, module = TTIRBuilder.from_module(ctx, module)
        else:
            raise NotImplementedError(
                "Loading MLIR files is only supported for ttir currently."
            )

    return builder, module


def split_mlir_file(
    module: Module,
    builder: Builder,
    target: Literal["ttir", "ttnn", "d2m", "stablehlo"] = "ttir",
) -> List[Tuple[Module, Builder]]:
    ctx = Context()

    with ctx:
        if target == "ttir":
            modules_and_builders = TTIRBuilder.split_module(ctx, module, builder)
        else:
            raise NotImplementedError(
                "Splitting MLIR files is only supported for ttir currently."
            )

    return modules_and_builders


# ----- Experimental Public APIs -----


def experimental_build_stablehlo_module(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    mesh_name: List[str] = ["mesh"],
    mesh_dict: List[OrderedDict[str, int]] = [OrderedDict([("x", 1), ("y", 1)])],
    module_dump: bool = False,
    base: Optional[str] = None,
    output_root: str = ".",
) -> Tuple[Module, StableHLOBuilder]:
    ctx = Context()

    # Grab the location of the test function in python for later debugging
    try:
        fname = inspect.getfile(fn)
        line_no = inspect.getsourcelines(fn)[1]
        loc = Location.file(fname, line_no, 0, ctx)
    except (OSError, TypeError):
        loc = Location.unknown(ctx)

    # Instantiate builder which is passed as the last argument to
    # `fn` so the user can use it to build ops.
    stablehlo_builder = StableHLOBuilder(ctx, loc, mesh_name, mesh_dict)

    # Default to all f32s
    if inputs_types is None:
        inputs_types = [torch.float32] * len(inputs_shapes)

    if len(inputs_shapes) != len(inputs_types):
        raise ValueError(
            f"inputs_shapes and inputs_types must have the same length: "
            f"{len(inputs_shapes)} != {len(inputs_types)}"
        )

    with ctx, loc:
        fn_input_types = [
            stablehlo_builder._create_ranked_tensor_type(
                shape,
                stablehlo_builder._get_type_from_torch_dtype(
                    dtype if isinstance(dtype, torch.dtype) else dtype
                ),
            )
            for (shape, dtype) in zip(inputs_shapes, inputs_types)
        ]

        # Wrap everything in a mlir module.
        module = Module.create()

        with InsertionPoint(module.body):
            # Wrap everything in a mlir function.
            @func.func(*fn_input_types, name=fn.__name__)
            def decorated_func(*inputs):
                input_goldens: Dict[Operand, GoldenMapTensor] = {}
                for index, (operand, dtype) in enumerate(zip(inputs, inputs_types)):
                    input_goldens[operand] = stablehlo_builder._generate_golden_tensor(
                        operand, dtype
                    )
                stablehlo_builder._set_goldens(input_goldens)
                stablehlo_builder._set_input_ordering(inputs)

                result = fn(*inputs, stablehlo_builder)

                outputs = result if hasattr(result, "__iter__") else (result,)
                output_goldens: Dict[Operand, GoldenMapTensor] = {}
                for op in outputs:
                    output_goldens[op] = stablehlo_builder._get_golden_tensor(op)
                stablehlo_builder._set_goldens(output_goldens)
                stablehlo_builder._set_output_ordering(outputs)

                # Convert OpView objects to MLIR Values for multi-return support
                return _process_multi_return_result(result)

            # Create named meshes and add them to the module
            named_mesh_list = []
            for mesh_name, mesh_dict in zip(mesh_name, mesh_dict):
                named_mesh_attr = stablehlo_builder.experimental_named_mesh_attr(
                    mesh_name,
                    stablehlo_builder._create_mesh_attr_from_ordered_dict(mesh_dict),
                )
                named_mesh_list.append(named_mesh_attr)
            topology_attr = stablehlo_builder.experimental_topology_attr(
                named_mesh_list
            )
            func_op = module.body.operations[-1]
            func_op.attributes["topology"] = topology_attr

        print(f"`{fn.__name__}` sucessfully transformed into a MLIR module.")
        base = fn.__name__ if base is None else base
        filename = _get_target_path(
            output_root, "stablehlo-builder-artifacts", "stablehlo.mlir", base
        )

        if module_dump:
            with open(filename, "w") as f:
                f.write(str(module))
                print(module)

        return module, stablehlo_builder
