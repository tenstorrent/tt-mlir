# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from contextvars import ContextVar
import os
import inspect
import time
import torch
from functools import partial, reduce
import itertools
import operator
from typing import Callable, List, Optional, Tuple, Union, Literal, Dict
from collections import OrderedDict
import json

from ttmlir.ir import *
from ttmlir.dialects import func, ttcore, ttnn, ttir, sdy
from ttmlir.passmanager import PassManager
from ttmlir.passes import (
    tt_populate_argument_types,
    ttir_to_ttnn_backend_pipeline,
    ttir_to_ttmetal_backend_pipeline,
    translate_to_cpp,
    translate_to_python,
    MLIRModuleLogger,
    stablehlo_pipeline,
    stablehlo_to_ttir_pipeline,
    ttir_to_emitpy_pipeline,
    ttnn_to_flatbuffer_bin,
    ttmetal_to_flatbuffer_bin,
    ttnn_to_flatbuffer_file,
    ttmetal_to_flatbuffer_file,
)

from builder.base.builder_enums import *
from builder.base.builder_utils import *
from builder.base.builder_runtime import *
from builder.base.builder import Builder
from builder.ttir.ttir_builder import TTIRBuilder
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.stablehlo.shardy_parallelization_utils import *
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.d2m.d2m_builder import D2MBuilder

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
    check_pcc: bool = True,
    check_atol: bool = False,
    check_rtol: bool = False,
    enable_intermediate_verification: bool = False,
    dump_memory: bool = False,
    **compile_kwargs,
) -> str:
    builder, compiled_bin, input_output_goldens, intermediate_goldens = compile_fn(
        target=target,
        **compile_kwargs,
    )

    if skip_exec:
        raise TTBuilderRuntimeException("Manually skipped execution")

    # Execute the flatbuffer
    if target in ["ttnn", "ttmetal"]:
        execute_fb(
            compiled_bin,
            pcc=pcc,
            atol=atol,
            rtol=rtol,
            disable_golden=disable_golden,
            device=device,
            check_pcc=check_pcc,
            check_atol=check_atol,
            check_rtol=check_rtol,
            input_output_goldens=input_output_goldens,
            intermediate_goldens=intermediate_goldens,
            bypass_ops=builder._bypass_ops,
            enable_intermediate_verification=enable_intermediate_verification,
            save_artifacts=compile_kwargs.get("save_artifacts", False),
            artifact_dir=compile_kwargs.get("artifact_dir", "."),
            dump_memory=dump_memory,
        )

    elif target == "emitpy":
        execute_py(
            compiled_bin,
            pcc=pcc,
            atol=atol,
            rtol=rtol,
            disable_golden=disable_golden,
            check_pcc=check_pcc,
            check_atol=check_atol,
            check_rtol=check_rtol,
            input_output_goldens=input_output_goldens,
            save_artifacts=compile_kwargs.get("save_artifacts", False),
            artifact_dir=compile_kwargs.get("artifact_dir", "."),
        )

    elif target == "emitc":
        cpp_path = os.path.join(
            compile_kwargs.get("artifact_dir", "."), "emitc_compiled.cpp"
        )
        execute_cpp(
            cpp_path,
            pcc=pcc,
            atol=atol,
            rtol=rtol,
            disable_golden=disable_golden,
            device=device,
            check_pcc=check_pcc,
            check_atol=check_atol,
            check_rtol=check_rtol,
            input_output_goldens=input_output_goldens,
            save_artifacts=compile_kwargs.get("save_artifacts", False),
            artifact_dir=compile_kwargs.get("artifact_dir", "."),
        )


def _compile(root_func: Callable, builder: Builder):
    new_module = Module.create()
    builder._root_module_insertion_point = new_module.body
    builder._current_module_insertion_point = new_module.body

    if isinstance(builder, StableHLOBuilder):
        new_module.body.append(builder._get_mesh())

    with InsertionPoint(new_module.body):
        root_func(builder)

    return new_module


# ----- Public APIs -----


def build_module(
    mod: Callable,
    builder_type: Literal["ttir", "stablehlo", "ttnn", "d2m"],
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    save_artifacts: bool = False,
    artifact_dir: str = ".",
) -> Tuple[Module, Union[TTIRBuilder, StableHLOBuilder, TTNNBuilder, D2MBuilder]]:
    """
    Build an MLIR `Module` from a Python emission function using the chosen builder.

    Parameters
    ----------
    mod : Callable
        A Python function that receives a builder instance and emits ops.
        Its body is captured into an MLIR module. Example signature: `def mod(b: TTIRBuilder): ...`.
    builder_type : Literal["ttir", "stablehlo", "ttnn", "d2m"]
        The type of builder to use for emission.
    mesh_name : str
        Mesh symbol name to attach to the module (where applicable).
    mesh_dict : OrderedDict[str, int]
        Mesh shape per dimension. Example: OrderedDict([("x", 1), ("y", 1)]).
    save_artifacts : bool
        When True, writes the emitted module to `artifact_dir`.
    artifact_dir : str
        Directory to write artifacts if `save_artifacts` is True.

    Returns
    -------
    Tuple[Module, Union[TTIRBuilder, StableHLOBuilder, TTNNBuilder, D2MBuilder]]
        The constructed MLIR module and the corresponding builder instance.
    """
    ctx = Context()

    try:
        fname = inspect.getfile(mod)
        line_no = inspect.getsourcelines(mod)[1]
        loc = Location.file(fname, line_no, 0, ctx)
    except (OSError, TypeError):
        loc = Location.unknown(ctx)

    if builder_type == "ttir":
        builder = TTIRBuilder(ctx, loc, mesh_name, mesh_dict)
    elif builder_type == "stablehlo":
        builder = StableHLOBuilder(ctx, loc, mesh_name, mesh_dict)
    elif builder_type == "ttnn":
        builder = TTNNBuilder(ctx, loc, mesh_name, mesh_dict)
    elif builder_type == "d2m":
        builder = D2MBuilder(ctx, loc, mesh_name, mesh_dict)

    with ctx, loc:
        new_module = _compile(mod, builder)

        print(f"`{mod.__name__}` successfully transformed into a MLIR module.")
        print(new_module)

        if save_artifacts:
            os.makedirs(artifact_dir, exist_ok=True)
            filename = os.path.join(artifact_dir, builder_type + "_module.mlir")
            with open(filename, "w") as f:
                f.write(str(new_module))

    return new_module, builder


def compile_and_execute_d2m(
    fn: Callable,
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    test_base: str = "test",
    output_root: str = ".",
    target: Literal["ttnn", "ttmetal", "emitc", "emitpy"] = "ttnn",
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    save_artifacts: bool = False,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Optional[Union[Callable, str]] = None,
    pipeline_options: Optional[List[str]] = None,
    print_ir: Union[bool, str] = False,
    device=None,
    pcc: float = 0.99,
    atol: float = 1e-08,
    rtol: float = 1e-05,
    disable_golden: bool = False,
    skip_exec: bool = False,
    check_pcc: bool = True,
    check_atol: bool = False,
    check_rtol: bool = False,
    enable_intermediate_verification: bool = False,
    dump_memory: bool = False,
) -> str:
    """
    Compiles and executes a D2MBuilder function through the complete pipeline.

    This function:
    1. Builds a D2M MLIR module from the function
    2. Compiles it to a flatbuffer
    3. Executes the flatbuffer on device

    Parameters
    ----------
    fn : Callable
        The D2MBuilder function to compile and execute
    inputs_shapes : List[Shape]
        Shapes of the respective ranked tensor inputs
    inputs_types : Optional[List[Union[torch.dtype, TypeInfo]]]
        The dtypes to use for the inputs
    system_desc_path : str
        Path to the system descriptor file
    test_base : str
        Base name for dumped files
    output_root : str
        Path to dump all generated files
    target : Literal["ttnn", "ttmetal", "emitc", "emitpy"]
        Target backend to use
    mesh_name : str
        Name of the mesh to be used
    mesh_dict : OrderedDict[str, int]
        Dictionary defining the mesh shape
    save_artifacts : bool
        Whether to save generated artifacts to `artifact_dir`
    argument_types_string : Optional[str]
        String defining argument types for constant evaluation
    custom_pipeline : Optional[Union[Callable, str]]
        Custom pipeline function or string
    pipeline_options : Optional[List[str]]
        Additional pipeline options
    print_ir : Union[bool, str]
        Controls intermediate IR dumping
    device : Optional
        Device to execute on (if None, opens a new device)
    pcc : float
        PCC threshold for golden comparison
    atol : float
        Absolute tolerance for golden comparison
    rtol : float
        Relative tolerance for golden comparison
    disable_golden : bool
        Whether to disable golden comparison
    check_atol : bool
        Whether to check absolute tolerance during golden comparison
    check_rtol : bool
        Whether to check relative tolerance during golden comparison
    dump_memory : bool
        Dump a per-op memory report into the artifact_dir.
    """
    artifact_dir = get_artifact_dir(
        output_root, "D2MBuilder", test_base, save_artifacts
    )
    _compile_and_execute(
        compile_fn=compile_d2m_to_flatbuffer,
        fn=fn,
        system_desc_path=system_desc_path,
        artifact_dir=artifact_dir,
        target=target,
        mesh_name=mesh_name,
        mesh_dict=mesh_dict,
        save_artifacts=save_artifacts,
        argument_types_string=argument_types_string,
        custom_pipeline=custom_pipeline,
        pipeline_options=pipeline_options,
        print_ir=print_ir,
        device=device,
        pcc=pcc,
        atol=atol,
        rtol=rtol,
        disable_golden=disable_golden,
        skip_exec=skip_exec,
        check_pcc=check_pcc,
        check_atol=check_atol,
        check_rtol=check_rtol,
        enable_intermediate_verification=enable_intermediate_verification,
        dump_memory=dump_memory,
    )


def compile_and_execute_shlo(
    fn: Callable,
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    test_base: str = "test",
    output_root: str = ".",
    target: Literal["ttnn", "ttmetal", "emitc", "emitpy"] = "ttnn",
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    save_artifacts: bool = False,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Optional[Union[Callable, str]] = None,
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
    check_pcc: bool = True,
    check_atol: bool = False,
    check_rtol: bool = False,
    enable_intermediate_verification: bool = False,
    dump_memory: bool = False,
) -> str:
    """
    Compiles and executes a StableHLO function through the complete pipeline.

    This function:
    1. Builds a StableHLO MLIR module from the function
    2. Compiles it through StableHLO -> TTIR -> TT{Metal,NN} -> Flatbuffer
    3. Executes the flatbuffer on device

    Parameters
    ----------
    fn : Callable
        The StableHLO function to compile and execute
    inputs_shapes : List[Shape]
        Shapes of the respective ranked tensor inputs
    inputs_types : Optional[List[Union[torch.dtype, TypeInfo]]]
        The dtypes to use for the inputs
    system_desc_path : str
        Path to the system descriptor file
    test_base : str
        Base name for dumped files
    output_root : str
        Path to dump all generated files
    target : Literal["ttnn", "ttmetal", "emitc", "emitpy"]
        Target backend to use
    mesh_name : str
        Name of the mesh to be used
    mesh_dict : OrderedDict[str, int]
        Dictionary defining the mesh shape
    save_artifacts : bool
        Whether to save generated artifacts to `artifact_dir`
    argument_types_string : Optional[str]
        String defining argument types for constant evaluation
    custom_pipeline : Optional[Union[Callable, str]]
        Custom pipeline function or string
    ttir_pipeline_options : Optional[List[str]]
        Pipeline options for TTIR pipeline
    shlo_pipeline_options : Optional[List[str]]
        Pipeline options for StableHLO pipeline
    shlo_to_ttir_pipeline_options : Optional[List[str]]
        Pipeline options for StableHLO to TTIR conversion
    print_ir : Union[bool, str]
        Controls intermediate IR dumping
    device : Optional
        Device to execute on (if None, opens a new device)
    pcc : float
        PCC threshold for golden comparison
    atol : float
        Absolute tolerance for golden comparison
    rtol : float
        Relative tolerance for golden comparison
    disable_golden : bool
        Whether to disable golden comparison
    check_atol : bool
        Whether to check absolute tolerance during golden comparison
    check_rtol : bool
        Whether to check relative tolerance during golden comparison
    dump_memory : bool
        Dump a per-op memory report into the artifact_dir.
    """
    artifact_dir = get_artifact_dir(
        output_root, "StableHLOBuilder", test_base, save_artifacts
    )
    _compile_and_execute(
        compile_fn=compile_stablehlo_to_flatbuffer,
        fn=fn,
        system_desc_path=system_desc_path,
        artifact_dir=artifact_dir,
        target=target,
        mesh_name=mesh_name,
        mesh_dict=mesh_dict,
        save_artifacts=save_artifacts,
        argument_types_string=argument_types_string,
        custom_pipeline=custom_pipeline,
        ttir_pipeline_options=ttir_pipeline_options,
        shlo_pipeline_options=shlo_pipeline_options,
        shlo_to_ttir_pipeline_options=shlo_to_ttir_pipeline_options,
        print_ir=print_ir,
        device=device,
        pcc=pcc,
        atol=atol,
        rtol=rtol,
        disable_golden=disable_golden,
        skip_exec=skip_exec,
        check_pcc=check_pcc,
        check_atol=check_atol,
        check_rtol=check_rtol,
        enable_intermediate_verification=enable_intermediate_verification,
        dump_memory=dump_memory,
    )


def compile_and_execute_ttnn(
    fn: Callable,
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    test_base: str = "test",
    output_root: str = ".",
    target: Literal["ttnn", "emitc", "emitpy"] = "ttnn",
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    save_artifacts: bool = False,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Optional[Union[Callable, str]] = None,
    pipeline_options: Optional[List[str]] = None,
    print_ir: Union[bool, str] = False,
    device=None,
    pcc: float = 0.99,
    atol: float = 1e-08,
    rtol: float = 1e-05,
    disable_golden: bool = False,
    skip_exec: bool = False,
    check_pcc: bool = True,
    check_atol: bool = False,
    check_rtol: bool = False,
    enable_intermediate_verification: bool = False,
    dump_memory: bool = False,
) -> str:
    """
    Compiles and executes a TTNNBuilder function through the complete pipeline.
    This function:
    1. Builds a TTNN MLIR module from the function
    2. Compiles it to a flatbuffer
    3. Executes the flatbuffer on device

    Parameters
    ----------
    fn : Callable
        The TTNNBuilder function to compile and execute
    inputs_shapes : List[Shape]
        Shapes of the respective ranked tensor inputs
    inputs_types : Optional[List[Union[torch.dtype, TypeInfo]]]
        The dtypes to use for the inputs
    system_desc_path : str
        Path to the system descriptor file
    test_base : str
        Base name for dumped files
    output_root : str
        Path to dump all generated files
    target : Literal["ttnn", "emitc", "emitpy"]
        Target backend to use
    mesh_name : str
        Name of the mesh to be used
    mesh_dict : OrderedDict[str, int]
        Dictionary defining the mesh shape
    save_artifacts : bool
        Whether to save generated artifacts to `artifact_dir`
    argument_types_string : Optional[str]
        String defining argument types for constant evaluation
    custom_pipeline : Optional[Union[Callable, str]]
        Custom pipeline function or string
    ttir_pipeline_options : Optional[List[str]]
        Pipeline options for TTIR pipeline
    shlo_pipeline_options : Optional[List[str]]
        Pipeline options for StableHLO pipeline
    shlo_to_ttir_pipeline_options : Optional[List[str]]
        Pipeline options for StableHLO to TTIR conversion
    print_ir : Union[bool, str]
        Controls intermediate IR dumping
    device : Optional
        Device to execute on (if None, opens a new device)
    pcc : float
        PCC threshold for golden comparison
    atol : float
        Absolute tolerance for golden comparison
    rtol : float
        Relative tolerance for golden comparison
    disable_golden : bool
        Whether to disable golden comparison
    check_atol : bool
        Whether to check absolute tolerance during golden comparison
    check_rtol : bool
        Whether to check relative tolerance during golden comparison
    dump_memory : bool
        Dump a per-op memory report into the artifact_dir.
    """
    artifact_dir = get_artifact_dir(
        output_root, "TTNNBuilder", test_base, save_artifacts
    )
    _compile_and_execute(
        compile_fn=compile_ttnn_to_flatbuffer,
        fn=fn,
        system_desc_path=system_desc_path,
        artifact_dir=artifact_dir,
        target=target,
        mesh_name=mesh_name,
        mesh_dict=mesh_dict,
        save_artifacts=save_artifacts,
        argument_types_string=argument_types_string,
        custom_pipeline=custom_pipeline,
        pipeline_options=pipeline_options,
        print_ir=print_ir,
        device=device,
        pcc=pcc,
        atol=atol,
        rtol=rtol,
        disable_golden=disable_golden,
        skip_exec=skip_exec,
        check_pcc=check_pcc,
        check_atol=check_atol,
        check_rtol=check_rtol,
        enable_intermediate_verification=enable_intermediate_verification,
        dump_memory=dump_memory,
    )


def compile_and_execute_ttir(
    fn: Callable,
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    test_base: str = "test",
    output_root: str = ".",
    target: Literal["ttnn", "ttmetal", "emitc", "emitpy"] = "ttnn",
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    save_artifacts: bool = False,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Optional[Union[Callable, str]] = None,
    pipeline_options: Optional[List[str]] = None,
    print_ir: Union[bool, str] = False,
    device=None,
    pcc: float = 0.99,
    atol: float = 1e-08,
    rtol: float = 1e-05,
    disable_golden: bool = False,
    skip_exec: bool = False,
    check_pcc: bool = True,
    check_atol: bool = False,
    check_rtol: bool = False,
    enable_intermediate_verification: bool = False,
    dump_memory: bool = False,
) -> str:
    """
    Compiles and executes a TTIR function through the complete pipeline.

    This function:
    1. Builds a TTIR MLIR module from the function
    2. Compiles it to a flatbuffer
    3. Executes the flatbuffer on device

    Parameters
    ----------
    fn : Callable
        The TTIRBuilder function to compile and execute
    inputs_shapes : List[Shape]
        Shapes of the respective ranked tensor inputs
    inputs_types : Optional[List[Union[torch.dtype, TypeInfo]]]
        The dtypes to use for the inputs
    system_desc_path : str
        Path to the system descriptor file
    test_base : str
        Base name for dumped files
    output_root : str
        Path to dump all generated files
    target : Literal["ttnn", "ttmetal", "emitc", "emitpy"]
        Target backend to use
    mesh_name : str
        Name of the mesh to be used
    mesh_dict : OrderedDict[str, int]
        Dictionary defining the mesh shape
    save_artifacts : bool
        Whether to save generated artifacts to `artifact_dir`
    argument_types_string : Optional[str]
        String defining argument types for constant evaluation
    custom_pipeline : Optional[Union[Callable, str]]
        Custom pipeline function or string
    pipeline_options : Optional[List[str]]
        Additional pipeline options
    print_ir : Union[bool, str]
        Controls intermediate IR dumping
    device : Optional
        Device to execute on (if None, opens a new device)
    pcc : float
        PCC threshold for golden comparison
    atol : float
        Absolute tolerance for golden comparison
    rtol : float
        Relative tolerance for golden comparison
    disable_golden : bool
        Whether to disable golden comparison
    check_atol : bool
        Whether to check absolute tolerance during golden comparison
    check_rtol : bool
        Whether to check relative tolerance during golden comparison
    dump_memory : bool
        Dump a per-op memory report into the artifact_dir.
    """
    artifact_dir = get_artifact_dir(
        output_root, "TTIRBuilder", test_base, save_artifacts
    )
    _compile_and_execute(
        compile_fn=compile_ttir_to_flatbuffer,
        fn=fn,
        system_desc_path=system_desc_path,
        artifact_dir=artifact_dir,
        target=target,
        mesh_name=mesh_name,
        mesh_dict=mesh_dict,
        save_artifacts=save_artifacts,
        argument_types_string=argument_types_string,
        custom_pipeline=custom_pipeline,
        pipeline_options=pipeline_options,
        print_ir=print_ir,
        device=device,
        pcc=pcc,
        atol=atol,
        rtol=rtol,
        disable_golden=disable_golden,
        skip_exec=skip_exec,
        check_pcc=check_pcc,
        check_atol=check_atol,
        check_rtol=check_rtol,
        enable_intermediate_verification=enable_intermediate_verification,
        dump_memory=dump_memory,
    )


def compile_ttir_to_flatbuffer(
    fn: Callable,
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    artifact_dir: str = ".",
    target: Literal["ttnn", "ttmetal", "emitc", "emitpy"] = "ttnn",
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    save_artifacts: bool = False,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Optional[Union[Callable, str]] = None,
    pipeline_options: Optional[List[str]] = None,
    print_ir: Union[bool, str] = False,
) -> str:
    """
    Compiles a TTIRBuilder function `fn` to TTIR MLIR -> TT{Metal,NN} MLIR -> Flatbuffer.

    This decorator is mainly a wrapper around the following functions, with
    each next function called on the output of the last:

    1. `build_module`
    2. `run_ttir_pipeline`
    3. `to_target`

    The choice of TTNN vs. TTMetal is controlled by the `target` parameter.

    Parameters
    ----------
    fn : Callable
        The TTIRBuilder function to compile. Must take `builder : TTIRBuilder` as a kwarg.
    inputs_shapes : *List[Shape]*
        Shapes of the respective ranked tensor inputs of the test function.
    inputs_types : *Optional[List[torch.dtype]]*, optional
        The dtypes to use for the inputs to `fn`. Note that if supplied,
        `len(inputs_shapes) == len(inputs_types)` must be true.
        Default is None.
    test_base : str
        The string to be used as the base name for dumped files throughout the
        process. If `None` is provided, then the `__name__` of `fn` will be used.
    output_root : str
        The path to dump all generated arguments under. If this path doesn't
        exist, it will be created.
    target : *Literal["ttnn", "ttmetal", "emitc", "emitpy"]*
        Either "ttnn", "ttmetal", or "emitc". This controls which backend to use.
    mesh_name : *str*, optional
        Name of the mesh to be used in the module. Default is "mesh".
    mesh_dict : *OrderedDict[str, int]*, optional
        Dictionary that defines the mesh shape, e.g. OrderedDict([("x", 1), ("y", 1)]).
    argument_types_string : *Optional[str]*, optional
        String defining argument types for constant evaluation.
    argument_types_string : *Optional[str]*
    custom_pipeline : *Union[Callable, str]*, optional
        Pipeline function to run.
        Can be either:
        - A Callable: custom_pipeline(module, options)
        - A str: "ttir-lower-to-layout,ttir-bufferization-pipeline"
    system_desc_path : str, optional
        Path to the system descriptor file
    mesh_name : *str*
        Name of the mesh to be used in the module. Default is "mesh".
    mesh_dict : *OrderedDict[str, int]*
        Dictionary that defines the mesh shape, e.g. OrderedDict([("x", 1), ("y", 1)]).
    save_artifacts : bool
        Set to True to print out generated TTIR MLIR module.
        Default is False.
    pipeline_options : *Optional[List[str]]*
        Pipeline options to be added to the pass
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

    Returns
    -------
    Tuple[Builder, Any, Dict[int, Dict[str, Dict[int, GoldenMapTensor]]], Dict[str, Dict[int, GoldenMapTensor]]]
        Builder, compiled_bin, input/output goldens, intermediate goldens
    """

    # Compile model to TTIR MLIR
    try:
        module, builder = build_module(
            fn,
            "ttir",
            mesh_name=mesh_name,
            mesh_dict=mesh_dict,
            save_artifacts=save_artifacts,
            artifact_dir=artifact_dir,
        )
    except Exception as e:
        raise TTBuilderCompileException(e)

    return builder, *compile_ttir_module_to_flatbuffer(
        module,
        builder,
        system_desc_path=system_desc_path,
        artifact_dir=artifact_dir,
        target=target,
        mesh_dict=mesh_dict,
        save_artifacts=save_artifacts,
        argument_types_string=argument_types_string,
        custom_pipeline=custom_pipeline,
        pipeline_options=pipeline_options,
        print_ir=print_ir,
    )


def compile_ttnn_to_flatbuffer(
    fn: Callable,
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    artifact_dir: str = ".",
    target: Literal["ttnn", "ttmetal", "emitc", "emitpy"] = "ttnn",
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    save_artifacts: bool = False,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Optional[Union[Callable, str]] = None,
    pipeline_options: Optional[List[str]] = None,
    print_ir: Union[bool, str] = False,
) -> str:
    """
    Compiles a TTNN function to flatbuffer format.

    This helper function generates a TTNN mlir module runs the compilation
    pipeline using ttir-to-ttnn-backend-pipeline and finally generates a flatbuffer.

    Parameters
    ----------
    fn : Callable
        The TTNN function to compile
    inputs_shapes : *List[Shape]*
        Shapes of the respective ranked tensor inputs of the test function
    inputs_types : *Optional[List[Union[torch.dtype, TypeInfo]]]*, optional
        The dtypes to use for the inputs to `fn`
    system_desc_path : str, optional
        Path to the system descriptor file
    test_base : str, optional
        The string to be used as the test_base name for dumped files
    output_root : str, optional
        The path to dump all generated files under
    target : *Literal["ttnn", "ttmetal", "emitc"]*, optional
        The target backend to use. Default is "ttnn"
    mesh_name : str, optional
        Name of the mesh to be used in the module
    mesh_dict : *OrderedDict[str, int]*, optional
        Dictionary that defines the mesh shape
    pipeline_options: *List[str]*
        Additional pipeline options to pass to the pipeline
    save_artifacts : bool
        Set to True to print out generated TTIR MLIR module.
        Default is False.

    Returns
    -------
    Tuple[Builder, Any, Dict[int, Dict[str, Dict[int, GoldenMapTensor]]], Dict[str, Dict[int, GoldenMapTensor]]]
        Builder, compiled_bin, input/output goldens, intermediate goldens

    Raises
    ------
    ValueError
        If inputs_shapes and inputs_types have different lengths
    TTBuilderCompileException
        If compilation fails at any stage
    """

    # Create module containing TTNN ops
    try:
        module, builder = build_module(
            fn,
            "ttnn",
            mesh_name=mesh_name,
            mesh_dict=mesh_dict,
            save_artifacts=save_artifacts,
            artifact_dir=artifact_dir,
        )
    except Exception as e:
        raise TTBuilderCompileException(e)

    return builder, *compile_ttir_module_to_flatbuffer(
        module,
        builder,
        system_desc_path=system_desc_path,
        artifact_dir=artifact_dir,
        target=target,
        mesh_dict=mesh_dict,
        pipeline_options=pipeline_options,
        print_ir=print_ir,
        save_artifacts=save_artifacts,
        argument_types_string=argument_types_string,
        custom_pipeline=custom_pipeline,
    )


def compile_d2m_to_flatbuffer(
    fn: Callable,
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    artifact_dir: str = ".",
    target: Literal["ttnn", "ttmetal", "emitc", "emitpy"] = "ttnn",
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    save_artifacts: bool = False,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Optional[Union[Callable, str]] = None,
    pipeline_options: Optional[List[str]] = None,
    print_ir: Union[bool, str] = False,
    device=None,
) -> str:
    """
    Compiles a D2MBuilder function `fn` to D2M MLIR -> TTMetal MLIR -> Flatbuffer.

    This decorator is a wrapper around:

    1. `build_module`
    2. `run_ttir_pipeline`
    3. `to_target`

    The choice of TTNN vs. TTMetal is controlled by the `target` parameter.

    Parameters
    ----------
    fn : Callable
        The D2MBuilder function to compile. Must take `builder : D2MBuilder` as a kwarg.
    inputs_shapes : *List[Shape]*
        Shapes of the respective ranked tensor inputs of the test function.
    inputs_types : *Optional[List[torch.dtype]]*, optional
        The dtypes to use for the inputs to `fn`. Note that if supplied,
        `len(inputs_shapes) == len(inputs_types)` must be true.
        Default is None.
    test_base : str
        The string to be used as the base name for dumped files throughout the
        process. If `None` is provided, then the `__name__` of `fn` will be used.
    output_root : str
        The path to dump all generated arguments under. If this path doesn't
        exist, it will be created.
    target : *Literal["ttnn", "ttmetal", "emitc", "emitpy"]*
        Either "ttnn", "ttmetal", or "emitc". This controls which backend to use.
    mesh_name : *str*, optional
        Name of the mesh to be used in the module. Default is "mesh".
    mesh_dict : *OrderedDict[str, int]*, optional
        Dictionary that defines the mesh shape, e.g. OrderedDict([("x", 1), ("y", 1)]).
    argument_types_string : *Optional[str]*, optional
        String defining argument types for constant evaluation.
    custom_pipeline : *Union[Callable, str]*, optional
        Pipeline function to run.
        Can be either:
        - A Callable: custom_pipeline(module, options)
        - A str: "ttir-lower-to-layout,ttir-bufferization-pipeline"
    system_desc_path : str, optional
        Path to the system descriptor file
    save_artifacts : bool
        Set to True to print out generated D2M MLIR module.
        Default is False.
    pipeline_options : *Optional[List[str]]*
        Pipeline options to be added to the pass
    print_ir : *Union[bool, str]*, optional
        Set to True to print IR to stdout. Set to dir path to print IR after
        each pass to its own file under that directory.
        Default is False.

    Returns
    -------
    Tuple[Builder, Any, Dict[int, Dict[str, Dict[int, GoldenMapTensor]]], Dict[str, Dict[int, GoldenMapTensor]]]
        Builder, compiled_bin, input/output goldens, intermediate goldens
    """

    # Compile model to D2M MLIR
    try:
        module, builder = build_module(
            fn,
            "d2m",
            mesh_name=mesh_name,
            mesh_dict=mesh_dict,
            save_artifacts=save_artifacts,
            artifact_dir=artifact_dir,
        )
    except Exception as e:
        raise TTBuilderCompileException(e)

    return builder, *compile_ttir_module_to_flatbuffer(
        module,
        builder,
        system_desc_path=system_desc_path,
        artifact_dir=artifact_dir,
        target=target,
        mesh_dict=mesh_dict,
        save_artifacts=save_artifacts,
        argument_types_string=argument_types_string,
        custom_pipeline=custom_pipeline,
        pipeline_options=pipeline_options,
        print_ir=print_ir,
    )


def compile_stablehlo_to_flatbuffer(
    fn: Callable,
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    artifact_dir: str = ".",
    target: Literal["ttnn", "ttmetal", "emitc", "emitpy"] = "ttnn",
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    save_artifacts: bool = False,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Optional[Union[Callable, str]] = None,
    ttir_pipeline_options: List[str] = [],
    shlo_pipeline_options: Optional[List[str]] = None,
    shlo_to_ttir_pipeline_options: Optional[List[str]] = None,
    print_ir: Union[bool, str] = False,
) -> str:
    """
    Compiles a StableHLO function to flatbuffer format.

    This function compiles a StableHLO function through the complete pipeline:
    StableHLO -> TTIR -> TT{Metal,NN} -> Flatbuffer. It first builds a StableHLO
    module, runs the stablehlo pipeline and conversion to TTIR, then compiles
    the TTIR module to the target flatbuffer format.

    Parameters
    ----------
    fn : Callable
        The StableHLO function to compile
    inputs_shapes : *List[Shape]*
        Shapes of the respective ranked tensor inputs of the test function
    inputs_types : *Optional[List[Union[torch.dtype, TypeInfo]]]*, optional
        The dtypes to use for the inputs to `fn`
    system_desc_path : str, optional
        Path to the system descriptor file
    test_base : str, optional
        The string to be used as the test_base name for dumped files
    output_root : str, optional
        The path to dump all generated files under
    target : *Literal["ttnn", "ttmetal", "emitpy", "emitc"]*, optional
        The target backend to use. Default is "ttnn"
    mesh_name : str, optional
        Name of the mesh to be used in the module
    mesh_dict : *OrderedDict[str, int]*, optional
        Dictionary that defines the mesh shape
    save_artifacts : bool, optional
        Set to True to print out generated MLIR modules
        Default is True.
    argument_types_string : *Optional[str]*
        String defining argument types for constant evaluation
    custom_pipeline : *Optional[Union[Callable, str]]*
        Custom pipeline function or string to run instead of default pipeline
    ttir_pipeline_options : *List[str]*
        Additional pipeline options to pass to the TTIR pipeline
    shlo_pipeline_options : *List[str]*
        Additional pipeline options to pass to the StableHLO pipeline
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

    Returns
    -------
    Tuple[Builder, Any, Dict[int, Dict[str, Dict[int, GoldenMapTensor]]], Dict[str, Dict[int, GoldenMapTensor]]]
        Builder, compiled_bin, input/output goldens, intermediate goldens

    Raises
    ------
    ValueError
        If inputs_shapes and inputs_types have different lengths
    """
    if shlo_pipeline_options is None:
        shlo_pipeline_options = []

    if shlo_to_ttir_pipeline_options is None:
        shlo_to_ttir_pipeline_options = []

    # Compile model to StableHLO and run stablehlo pipeline to TTIR MLIR
    try:
        module, builder = build_module(
            fn,
            "stablehlo",
            mesh_name=mesh_name,
            mesh_dict=mesh_dict,
            save_artifacts=save_artifacts,
            artifact_dir=artifact_dir,
        )
    except Exception as e:
        raise TTBuilderCompileException(e)

    # We need to generate golden dictionary before pipeline run because pipeline run modifies the graph in place.
    input_output_goldens, intermediate_goldens = builder.golden_map

    stablehlo_pipeline(module, " ".join(shlo_pipeline_options), print_ir=print_ir)
    print(f"`{fn.__name__}` successfully ran stablehlo-pipeline.")
    print(module)

    if save_artifacts:
        filename = os.path.join(artifact_dir, "stablehlo_compiled.mlir")
        with open(filename, "w") as f:
            f.write(str(module))

    stablehlo_to_ttir_pipeline(
        module, " ".join(shlo_to_ttir_pipeline_options), print_ir=print_ir
    )
    print(f"`{fn.__name__}` successfully transformed into a TTIR MLIR module.")
    print(module)

    if save_artifacts:
        filename = os.path.join(artifact_dir, "ttir_compiled.mlir")
        with open(filename, "w") as f:
            f.write(str(module))

    return builder, *compile_ttir_module_to_flatbuffer(
        module,
        builder,
        system_desc_path=system_desc_path,
        artifact_dir=artifact_dir,
        target=target,
        mesh_dict=mesh_dict,
        save_artifacts=save_artifacts,
        argument_types_string=argument_types_string,
        custom_pipeline=custom_pipeline,
        pipeline_options=ttir_pipeline_options,
        print_ir=print_ir,
        input_output_goldens=input_output_goldens,
        intermediate_goldens=intermediate_goldens,
    )


def compile_ttir_module_to_flatbuffer(
    module: Module,
    builder: Builder,
    system_desc_path: str = "ttrt-artifacts/system_desc.ttsys",
    artifact_dir: str = ".",
    target: Literal["ttnn", "ttmetal", "emitc", "emitpy"] = "ttnn",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    save_artifacts: bool = False,
    argument_types_string: Optional[str] = None,
    custom_pipeline: Optional[Union[Callable, str]] = None,
    pipeline_options: List[str] = None,
    print_ir: Union[bool, str] = False,
    input_output_goldens: Optional[
        Dict[int, Dict[str, Dict[int, GoldenMapTensor]]]
    ] = None,
    intermediate_goldens: Optional[Dict[str, Dict[int, GoldenMapTensor]]] = None,
):
    """
    Compiles a TTIR MLIR module to flatbuffer format.

    This decorator takes an existing TTIR MLIR module and compiles it through
    the backend pipeline to generate a flatbuffer file. It supports multiple
    targets including TTNN, TTMetal, emitc, and emitpy. It is mainly a wrapper around the following functions, with
    each next function called on the output of the last:

    1. `run_ttir_pipeline`
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
    save_artifacts : bool, optional
        When True, writes artifacts (intermediate MLIR, compiled outputs) to `artifact_dir`.
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
        custom_pipeline = create_custom_ttir_pipeline_fn(
            custom_pipeline, print_ir=print_ir
        )

    pipeline_fn: Callable
    to_target: Callable
    target_extension: str

    # Helper to wrap default pipelines with print_ir flag
    def wrap_pipeline_with_print_ir(pipeline):
        if print_ir:
            return partial(pipeline, print_ir=True)
        return pipeline

    if target == "ttnn":
        pipeline_fn = (
            custom_pipeline
            if custom_pipeline
            else wrap_pipeline_with_print_ir(ttir_to_ttnn_backend_pipeline)
        )
        to_target = ttnn_to_flatbuffer_bin
        to_file = ttnn_to_flatbuffer_file
        target_extension = "ttnn"
    elif target == "ttmetal":
        pipeline_fn = (
            custom_pipeline
            if custom_pipeline
            else wrap_pipeline_with_print_ir(ttir_to_ttmetal_backend_pipeline)
        )
        to_target = ttmetal_to_flatbuffer_bin
        to_file = ttmetal_to_flatbuffer_file
        target_extension = "ttm"
    elif target == "emitc":
        ttir_to_ttnn_emitc_pipeline = create_custom_ttir_pipeline_fn(
            "ttir-to-emitc-pipeline", print_ir=print_ir
        )
        pipeline_fn = (
            custom_pipeline if custom_pipeline else ttir_to_ttnn_emitc_pipeline
        )
        to_target = emitc_to_executable
        target_extension = "cpp"
    elif target == "emitpy":
        pipeline_fn = (
            custom_pipeline
            if custom_pipeline
            else wrap_pipeline_with_print_ir(ttir_to_emitpy_pipeline)
        )
        to_target = emitpy_to_executable
        target_extension = "py"
    else:
        raise ValueError("Unsupported target: " + target)

    output_file_name = os.path.join(artifact_dir, target + "_compiled.mlir")

    # We need to generate golden dictionary before pipeline run because pipeline run modifies the graph in place.
    if input_output_goldens is None or intermediate_goldens is None:
        input_output_goldens, intermediate_goldens = builder.golden_map

    # Compile TTIR MLIR -> TT{Metal,NN} MLIR
    try:
        module = run_ttir_pipeline(
            module,
            pipeline_fn,
            pipeline_options=pipeline_options,
            save_artifacts=save_artifacts,
            output_file_name=output_file_name,
            system_desc_path=system_desc_path,
            mesh_dict=mesh_dict,
            argument_types_string=argument_types_string,
        )
    except Exception as e:
        raise TTBuilderCompileException(e)

    print(f"{target} pipeline ran successfully.")

    # Compile TT{Metal,NN} MLIR -> flatbuffer
    try:
        compiled_bin = to_target(module)
    except Exception as e:
        raise TTBuilderCompileException(e)

    output_file_bin = os.path.join(
        artifact_dir, target + "_compiled." + target_extension
    )
    if target == "emitc":
        os.makedirs(artifact_dir, exist_ok=True)
        with open(output_file_bin, "w") as f:
            f.write(compiled_bin)
    if save_artifacts:
        print(f"Writing compiled flatbuffer to {output_file_bin}")
        os.makedirs(artifact_dir, exist_ok=True)
        if target == "emitpy":
            with open(output_file_bin, "w") as f:
                f.write(compiled_bin)
        elif target in ["ttnn", "ttmetal"]:
            to_file(module, output_file_bin, {}, [])

    return compiled_bin, input_output_goldens, intermediate_goldens


def load_mlir_file(
    mlir_text: str,
    golden_inputs: Dict[str, List[torch.tensor]] = None,
    target: Literal["ttir", "ttnn", "d2m", "stablehlo"] = "ttir",
) -> (Module, Builder):
    """
    Load an MLIR module text into a `Module` and reconstruct the corresponding builder.

    Parameters
    ----------
    mlir_text : str
        The MLIR textual IR to parse.
    golden_inputs : Dict[str, List[torch.tensor]], optional
        Optional map of input names to lists of torch tensors used to seed builder goldens.
    target : Literal["ttir", "ttnn", "d2m", "stablehlo"]
        Which dialect to interpret the module as, controls which builder is reconstructed.

    Returns
    -------
    Tuple[Module, Builder]
        Parsed MLIR module and a builder instance initialized from it.
    """
    ctx = Context()

    if target == "ttir":
        module, builder = TTIRBuilder.from_module(ctx, mlir_text, golden_inputs)
    elif target == "stablehlo":
        module, builder = StableHLOBuilder.from_module(ctx, mlir_text, golden_inputs)
    elif target == "ttnn":
        module, builder = TTNNBuilder.from_module(ctx, mlir_text, golden_inputs)
    else:
        raise NotImplementedError(
            "Loading MLIR files is only supported for ttir, stablehlo and ttnn currently."
        )

    return module, builder


def split_mlir_file(
    module: Module,
    builder: Builder,
    target: Literal["ttir", "ttnn", "d2m", "stablehlo"] = "ttir",
) -> List[Tuple[Module, Builder]]:
    """
    Split a MLIR module into multiple per-program modules with matching builders.

    Parameters
    ----------
    module : Module
        The MLIR module to split.
    builder : Builder
        The builder associated with the module, used to propagate goldens and metadata.
    target : Literal["ttir", "ttnn", "d2m", "stablehlo"]
        Dialect type to select the appropriate split logic.

    Returns
    -------
    List[Tuple[Module, Builder]]
        A list of (module, builder) pairs for each split program.
    """
    if target == "ttir":
        modules_and_builders = TTIRBuilder.split_module(module, builder)
    elif target == "stablehlo":
        modules_and_builders = StableHLOBuilder.split_module(module, builder)
    elif target == "ttnn":
        modules_and_builders = TTNNBuilder.split_module(module, builder)
    else:
        raise NotImplementedError(
            "Splitting MLIR files is only supported for ttir, stablehlo and ttnn currently."
        )

    return modules_and_builders


def generate_all_module_permutations(mlir_text: str, num_devices: int) -> List[Module]:
    """
    Generate all valid module permutations across device meshes from MLIR text.

    Parameters
    ----------
    mlir_text : str
        The source MLIR text to parse and permute.
    num_devices : int
        Number of available devices to consider when generating permutations.

    Returns
    -------
    List[Module]
        All discovered module permutations suitable for the given device count.
    """
    ctx = Context()
    loc = Location.unknown(ctx)

    with ctx, loc:
        module = Module.parse(mlir_text)
        return find_module_permutations(module, num_devices)


def get_optimal_module_least_num_collectives(
    module_permutations: List[Module],
) -> List[Module]:
    """
    Select the optimal module(s) minimizing the total number of collective ops.

    Parameters
    ----------
    module_permutations : List[Module]
        Candidate modules to evaluate.

    Returns
    -------
    List[Module]
        The module(s) determined optimal under the least-collectives heuristic.
    """
    copied_modules = []

    for module in module_permutations:
        copied_modules.append(Module.parse(str(module), module.context))

    return optimal_module_least_num_collectives(copied_modules)


# ----- Experimental Public APIs -----


def experimental_build_stablehlo_module(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    mesh_name: List[str] = ["mesh"],
    mesh_dict: List[OrderedDict[str, int]] = [OrderedDict([("x", 1), ("y", 1)])],
    save_artifacts: bool = False,
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
        stablehlo_builder._root_module_insertion_point = module.body
        stablehlo_builder._current_module_insertion_point = module.body

        ordered_inputs = []
        ordered_outputs = []

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
                ordered_inputs.extend(inputs)

                result = fn(*inputs, stablehlo_builder)

                outputs = result if hasattr(result, "__iter__") else [result]
                output_goldens: Dict[Operand, GoldenMapTensor] = {}
                for op in outputs:
                    output_goldens[op] = stablehlo_builder._get_golden_tensor(op)
                stablehlo_builder._set_goldens(output_goldens)
                ordered_outputs.extend(outputs)

                # Convert OpView objects to MLIR Values for multi-return support
                return process_multi_return_result(result)

            new_func_op = decorated_func.func_op
            stablehlo_builder._func_ops_generated[new_func_op] = [
                ordered_inputs,
                ordered_outputs,
            ]

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
        filename = get_target_path(
            output_root, "stablehlo-builder-artifacts", "stablehlo.mlir", base
        )

        if save_artifacts:
            with open(filename, "w") as f:
                f.write(str(module))
                print(module)

        return module, stablehlo_builder
