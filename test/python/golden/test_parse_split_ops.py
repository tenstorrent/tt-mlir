# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple, Union
from collections import OrderedDict
from functools import reduce
import operator
import os

from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import (
    build_module,
    load_mlir_file,
    split_mlir_file,
    create_custom_ttir_pipeline_fn,
    compile_ttir_module_to_flatbuffer,
)
from builder.base.builder_runtime import *
from builder.base.builder_enums import *

pytestmark = pytest.mark.frontend("ttir")

ttir_mlir_snippets = {}
skip_split_ttir_tests = [
    "ttir_reduce_scatter.mlir",
    "ttir_all_to_all.mlir",
    "ttir_collective_permute.mlir",
    "ttir_all_gather.mlir",
    "ttir_collective_broadcast.mlir",
    "ttir_device_module_nested_func.mlir",
    "ttir_nested_funcs.mlir",
]
ttir_snippets_dir_path = os.path.join(os.path.dirname(__file__), "mlir_snippets/ttir")
for filename in os.listdir(ttir_snippets_dir_path):
    if filename.endswith(".mlir"):
        file_path = os.path.join(ttir_snippets_dir_path, filename)
        with open(file_path, "r") as f:
            mlir_ir_string = f.read()
            ttir_mlir_snippets[filename] = mlir_ir_string

sdy_mlir_snippets = {}
sdy_snippets_dir_path = os.path.join(os.path.dirname(__file__), "mlir_snippets/sdy")
for filename in os.listdir(sdy_snippets_dir_path):
    if filename.endswith(".mlir"):
        file_path = os.path.join(sdy_snippets_dir_path, filename)
        with open(file_path, "r") as f:
            mlir_ir_string = f.read()
            sdy_mlir_snippets[filename] = mlir_ir_string

stablehlo_mlir_snippets = {}
stablehlo_snippets_dir_path = os.path.join(
    os.path.dirname(__file__), "mlir_snippets/stablehlo"
)
for filename in os.listdir(stablehlo_snippets_dir_path):
    if filename.endswith(".mlir"):
        file_path = os.path.join(stablehlo_snippets_dir_path, filename)
        with open(file_path, "r") as f:
            mlir_ir_string = f.read()
            stablehlo_mlir_snippets[filename] = mlir_ir_string

ttnn_mlir_snippets = {}
ttnn_snippets_dir_path = os.path.join(os.path.dirname(__file__), "mlir_snippets/ttnn")
for filename in os.listdir(ttnn_snippets_dir_path):
    if filename.endswith(".mlir"):
        file_path = os.path.join(ttnn_snippets_dir_path, filename)
        with open(file_path, "r") as f:
            mlir_ir_string = f.read()
            ttnn_mlir_snippets[filename] = mlir_ir_string


@pytest.mark.parametrize("mlir_snippet", ttir_mlir_snippets.keys())
def test_ttir_parsing_splitting_ops(mlir_snippet, request, device):
    mlir_ir_string = ttir_mlir_snippets[mlir_snippet]
    mlir_module, builder = load_mlir_file(mlir_ir_string, target="ttir")

    if mlir_snippet not in skip_split_ttir_tests:
        split_modules = split_mlir_file(mlir_module, builder)


@pytest.mark.parametrize("mlir_snippet", sdy_mlir_snippets.keys())
def test_sdy_parsing_ops(mlir_snippet, request, device):
    mlir_ir_string = sdy_mlir_snippets[mlir_snippet]
    mlir_module, builder = load_mlir_file(mlir_ir_string, target="stablehlo")


@pytest.mark.parametrize("mlir_snippet", stablehlo_mlir_snippets.keys())
def test_stablehlo_parsing_splitting_ops(mlir_snippet, request, device):
    mlir_ir_string = stablehlo_mlir_snippets[mlir_snippet]
    mlir_module, builder = load_mlir_file(mlir_ir_string, target="stablehlo")
    split_modules = split_mlir_file(mlir_module, builder, target="stablehlo")


skip_split_ttnn_tests = [
    "ttnn_all_gather.mlir",
    "ttnn_reduce_scatter.mlir",
]


@pytest.mark.parametrize("mlir_snippet", ttnn_mlir_snippets.keys())
def test_ttnn_parsing_splitting_ops(mlir_snippet, request, device):
    mlir_ir_string = ttnn_mlir_snippets[mlir_snippet]
    mlir_module, builder = load_mlir_file(mlir_ir_string, target="ttnn")

    if mlir_snippet == "ttnn_rand.mlir":
        split_modules = split_mlir_file(mlir_module, builder, target="ttnn")
        for m in split_modules:
            print(m[0])
        input_output_goldens, intermediate_goldens = builder.golden_map
        custom_pipeline = create_custom_ttir_pipeline_fn("")
        (
            mlir_path,
            input_output_goldens,
            intermediate_goldens,
        ) = compile_ttir_module_to_flatbuffer(
            mlir_module,
            builder,
            input_output_goldens=input_output_goldens,
            intermediate_goldens=intermediate_goldens,
            custom_pipeline=custom_pipeline,
            # bypass_ops=builder._bypass_ops,
        )
        execute_fb(
            mlir_path,
            input_output_goldens=input_output_goldens,
            intermediate_goldens=intermediate_goldens,
            device=device,
            bypass_ops=builder._bypass_ops,
        )
    elif mlir_snippet not in skip_split_ttnn_tests:
        split_modules = split_mlir_file(mlir_module, builder, target="ttnn")


"""
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple, Union
from collections import OrderedDict
from functools import reduce
import operator
import os

from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import (
    build_module,
    load_mlir_file,
    split_mlir_file,
    compile_ttir_module_to_flatbuffer,
)
from builder.base.builder_runtime import *
from builder.base.builder_enums import *

pytestmark = pytest.mark.frontend("ttir")

ttir_mlir_snippets = {}
skip_split_ttir_tests = [
    "ttir_reduce_scatter.mlir",
    "ttir_all_to_all.mlir",
    "ttir_collective_permute.mlir",
    "ttir_all_gather.mlir",
    "ttir_collective_broadcast.mlir",
    "ttir_device_module_nested_func.mlir",
    "ttir_nested_funcs.mlir",
]
ttir_snippets_dir_path = os.path.join(os.path.dirname(__file__), "mlir_snippets/ttir")
for filename in os.listdir(ttir_snippets_dir_path):
    if filename.endswith(".mlir"):
        file_path = os.path.join(ttir_snippets_dir_path, filename)
        with open(file_path, "r") as f:
            mlir_ir_string = f.read()
            ttir_mlir_snippets[filename] = mlir_ir_string

sdy_mlir_snippets = {}
sdy_snippets_dir_path = os.path.join(os.path.dirname(__file__), "mlir_snippets/sdy")
for filename in os.listdir(sdy_snippets_dir_path):
    if filename.endswith(".mlir"):
        file_path = os.path.join(sdy_snippets_dir_path, filename)
        with open(file_path, "r") as f:
            mlir_ir_string = f.read()
            sdy_mlir_snippets[filename] = mlir_ir_string

stablehlo_mlir_snippets = {}
stablehlo_snippets_dir_path = os.path.join(
    os.path.dirname(__file__), "mlir_snippets/stablehlo"
)
for filename in os.listdir(stablehlo_snippets_dir_path):
    if filename.endswith(".mlir"):
        file_path = os.path.join(stablehlo_snippets_dir_path, filename)
        with open(file_path, "r") as f:
            mlir_ir_string = f.read()
            stablehlo_mlir_snippets[filename] = mlir_ir_string

ttnn_mlir_snippets = {}
ttnn_snippets_dir_path = os.path.join(os.path.dirname(__file__), "mlir_snippets/ttnn")
for filename in os.listdir(ttnn_snippets_dir_path):
    if filename.endswith(".mlir"):
        file_path = os.path.join(ttnn_snippets_dir_path, filename)
        with open(file_path, "r") as f:
            mlir_ir_string = f.read()
            ttnn_mlir_snippets[filename] = mlir_ir_string


@pytest.mark.parametrize("mlir_snippet", ttir_mlir_snippets.keys())
def test_ttir_parsing_splitting_ops(mlir_snippet, request, device):
    mlir_ir_string = ttir_mlir_snippets[mlir_snippet]
    mlir_module, builder = load_mlir_file(mlir_ir_string, target="ttir")

    if mlir_snippet not in skip_split_ttir_tests:
        split_modules = split_mlir_file(mlir_module, builder)


@pytest.mark.parametrize("mlir_snippet", sdy_mlir_snippets.keys())
def test_sdy_parsing_ops(mlir_snippet, request, device):
    mlir_ir_string = sdy_mlir_snippets[mlir_snippet]
    mlir_module, builder = load_mlir_file(mlir_ir_string, target="stablehlo")


@pytest.mark.parametrize("mlir_snippet", stablehlo_mlir_snippets.keys())
def test_stablehlo_parsing_splitting_ops(mlir_snippet, request, device):
    mlir_ir_string = stablehlo_mlir_snippets[mlir_snippet]
    mlir_module, builder = load_mlir_file(mlir_ir_string, target="stablehlo")
    split_modules = split_mlir_file(mlir_module, builder, target="stablehlo")


skip_split_ttnn_tests = [
    "ttnn_all_gather.mlir",
    "ttnn_reduce_scatter.mlir",
]


@pytest.mark.parametrize("mlir_snippet", ttnn_mlir_snippets.keys())
def test_ttnn_parsing_splitting_ops(mlir_snippet, request, device):
    mlir_ir_string = ttnn_mlir_snippets[mlir_snippet]
    mlir_module, builder = load_mlir_file(mlir_ir_string, target="ttnn")
    split_modules = split_mlir_file(mlir_module, builder, target="ttnn")
    for m in split_modules:
        print(m[0])
    if mlir_ir_string == "ttnn_add.mlir":
        return
    input_output_goldens, intermediate_goldens = builder.golden_map
    print(mlir_module)
    custom_pipeline = create_custom_pipeline_fn("")
    (
        mlir_path,
        input_output_goldens,
        intermediate_goldens,
    ) = compile_ttir_module_to_flatbuffer(
        mlir_module,
        builder,
        input_output_goldens=input_output_goldens,
        intermediate_goldens=intermediate_goldens,
        module_dump=True,
        custom_pipeline=custom_pipeline,
    )
    print(mlir_path)
    fb_path = mlir_path + ".ttnn"
    execute_fb(
        fb_path=fb_path,
        input_output_goldens=input_output_goldens,
        intermediate_goldens=intermediate_goldens,
        device=device,
        bypass_ops=builder._bypass_ops,
    )

"""
