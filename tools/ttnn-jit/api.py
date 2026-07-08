# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect

import ttnn
from ttnn_jit._src.jit import JitFunction
from ttnn_jit._src.shard_advisor import ShardAdvisor


def jit(
    compile_only: bool = False,
    debug: bool = False,
    enable_cache: bool = False,
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
    memory_config: ttnn.MemoryConfig = None,
    fallback: bool = False,
    extra_pipeline_options: str = "",
):
    """
    Sets up the decorated function to be JIT compiled through D2M.

    Args:
        compile_only: If True, only compile the function to a flatbuffer.
        debug: If True, print debug information during compilation and execution.
        enable_cache: If True, enables caching for compiled JIT graphs.
        math_fidelity: Set the math fidelity level for computations. Options are "LoFi", "HiFi2", "HiFi3", and "HiFi4".
        memory_config: Output memory configuration for the function. If specified, the output tensor will use this exact layout.
                      If unspecified (None), the output will use a maximally L1 block sharded layout.
        fallback: If True, falls back to running the original function directly
                 through ttnn when JIT compilation or execution fails.
                 Cannot be used together with compile_only=True.
        extra_pipeline_options: Additional pipeline options passed verbatim to
                 the D2M compilation pipeline. For advanced use only.

    Returns:
        A wrapped version of the function that when invoked, will JIT compile through D2M and execute the resulting flatbuffer.
    """

    def _decorator(f):
        jit_func = JitFunction(
            f,
            compile_only,
            debug,
            enable_cache,
            math_fidelity,
            memory_config,
            fallback,
            extra_pipeline_options,
        )

        if inspect.ismethod(f):
            return staticmethod(jit_func)
        return jit_func

    return _decorator


def shard_advisor(
    optimization_level: int = 2,
    debug: bool = False,
    out_dir: str = None,
    extra_pipeline_options: str = "",
    pipeline: str = "scoped",
):
    """Decorate a ttnn function to run the greedy L1-sharding advisor.

    The decorated callable, when invoked with live ttnn.Tensor arguments,
    traces the function to TTIR, runs the greedy memory-layout optimizer, and
    returns an AdvisorReport (printing the text report as a side effect). It
    does not compile to a flatbuffer or execute on device.

    Args:
        optimization_level: Optimizer level. >=2 enables L1 sharding analysis
            and the L1 spill pass; 1 runs greedy without L1 sharding.
        debug: Print the pipeline options and enable verbose tracing.
        out_dir: Directory for the decision-trace JSON and saved system desc.
            Defaults to generated/ttnn-jit/<func>/advisor.
        extra_pipeline_options: Extra options appended verbatim to the
            ttir-to-ttnn runtime pipeline.
        pipeline: Which TTIR-to-TTNN pipeline to run the advice over. Either
            "scoped" (default; the 1:1 ttir-to-ttnn-l1-advisor pipeline with
            no fusion/decomposition) or "full" (the runtime pipeline).
    """

    def _decorator(f):
        advisor = ShardAdvisor(
            f,
            optimization_level=optimization_level,
            debug=debug,
            out_dir=out_dir,
            extra_pipeline_options=extra_pipeline_options,
            pipeline=pipeline,
        )

        def _wrapped(*args, **kwargs):
            return advisor.run(*args, **kwargs)

        _wrapped.advisor = advisor
        return _wrapped

    return _decorator
