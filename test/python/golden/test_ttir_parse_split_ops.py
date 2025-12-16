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
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.base.builder_utils import run_ttir_pipeline
from builder.base.builder_apis import (
    build_module,
    load_mlir_file,
    split_mlir_file,
    compile_ttir_module_to_flatbuffer,
)
from builder.base.builder_runtime import *
from ttmlir.passes import ttir_to_ttnn_backend_pipeline

pytestmark = pytest.mark.frontend("ttir")

ttir_mlir_snippets = {}
ttir_snippets_dir_path = os.path.join(os.path.dirname(__file__), "mlir_snippets/ttir")
for filename in os.listdir(ttir_snippets_dir_path):
    if filename.endswith(".mlir"):
        file_path = os.path.join(ttir_snippets_dir_path, filename)
        with open(file_path, "r") as f:
            mlir_ir_string = f.read()
            ttir_mlir_snippets[filename] = mlir_ir_string

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

ttir_to_ttnn_mlir_snippets = {}
ttir_to_ttnn_snippets_dir_path = os.path.join(
    os.path.dirname(__file__), "mlir_snippets/ttir_to_ttnn"
)
for filename in os.listdir(ttir_to_ttnn_snippets_dir_path):
    if filename.endswith(".mlir"):
        file_path = os.path.join(ttir_to_ttnn_snippets_dir_path, filename)
        with open(file_path, "r") as f:
            mlir_ir_string = f.read()
            ttir_to_ttnn_mlir_snippets[filename] = mlir_ir_string


@pytest.mark.parametrize("mlir_snippet", ttir_mlir_snippets.keys())
def test_ttir_parsing_splitting_ops(mlir_snippet, request, device):
    mlir_ir_string = ttir_mlir_snippets[mlir_snippet]
    mlir_module, builder = load_mlir_file(mlir_ir_string, target="ttir")
    split_modules = split_mlir_file(mlir_module, builder)


@pytest.mark.parametrize("mlir_snippet", stablehlo_mlir_snippets.keys())
def test_stablehlo_parsing_splitting_ops(mlir_snippet, request, device):
    mlir_ir_string = stablehlo_mlir_snippets[mlir_snippet]
    mlir_module, builder = load_mlir_file(mlir_ir_string, target="stablehlo")
    split_modules = split_mlir_file(mlir_module, builder)


@pytest.mark.parametrize("mlir_snippet", ttnn_mlir_snippets.keys())
def test_ttnn_parsing_splitting_ops(mlir_snippet, request, device):
    mlir_ir_string = ttnn_mlir_snippets[mlir_snippet]
    mlir_module, builder = load_mlir_file(mlir_ir_string, target="ttnn")
    split_modules = split_mlir_file(mlir_module, builder)


@pytest.mark.parametrize("mlir_snippet", ttir_to_ttnn_mlir_snippets.keys())
def test_ttir_to_ttnn_parsing_splitting_ops(mlir_snippet, request, device):
    mlir_ir_string = ttir_to_ttnn_mlir_snippets[mlir_snippet]
    mlir_module, builder = load_mlir_file(mlir_ir_string, target="ttir")
    ttnn_compiled_path = mlir_snippet.replace("ttir_to_ttnn", "ttnn_compiled")
    run_ttir_pipeline(
        mlir_module,
        ttir_to_ttnn_backend_pipeline,
        output_file_name=ttnn_compiled_path,
        system_desc_path=request.config.getoption("--sys-desc"),
    )
    with open(ttnn_compiled_path, "r") as f:
        compiled_mlir_ir_string = f.read()
    compiled_mlir_module, compiled_builder = load_mlir_file(
        compiled_mlir_ir_string, target="ttnn"
    )
    split_modules = split_mlir_file(compiled_mlir_module, compiled_builder)
    module = split_modules[0][0]
    builder = split_modules[0][1]
    goldens = dict(builder.golden_map)
    mlir_path, goldens = compile_ttir_module_to_flatbuffer(
        module, builder, goldens=goldens
    )
    fb_path = mlir_path + ".ttnn"
    execute_fb(
        fb_path=fb_path,
        goldens=goldens,
        device=device,
    )
