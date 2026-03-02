# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import ast
import os
from typing import Dict, Any

from ttmlir.ir import *
from ttmlir.dialects import func, scf
from ttmlir.dialects import _d2m_ops_gen as d2m

from d2m_jit._src.mlir_generator import generate_ir

class AstCompiler:
    def __init__(self, func, grid, compile_only, debug, math_fidelity):
        self.func = func
        self.grid = grid
        self.compile_only = compile_only
        self.debug = debug
        self.math_fidelity = math_fidelity
        self.out_dir = os.path.join("generated", "d2m_jit", func.__name__)
        os.makedirs(self.out_dir, exist_ok=True)
        
    def _get_source_and_parse(self):
        source = inspect.getsource(self.func)
        
        # Strip decorators
        lines = source.split('\n')
        while lines and lines[0].strip().startswith('@'):
            lines.pop(0)
            
        source = '\n'.join(lines)
        
        # Python ast doesn't like the pseudo-code "Tensor a"
        # We replace it with "a: Tensor"
        import re
        source = re.sub(r'Tensor\s+([a-zA-Z_]\w*)', r'\1: Tensor', source)
        
        tree = ast.parse(source)
        return tree, source
        
    def compile_and_run(self, *args, **kwargs):
        tree, source = self._get_source_and_parse()
        
        if self.debug:
            print(f"--- Parsed Source for {self.func.__name__} ---")
            print(source)
            if hasattr(ast, "unparse"):
                print("--- Unparsed AST ---")
                print(ast.unparse(tree))
            else:
                print(ast.dump(tree, indent=4))
            
        # Parse tensor shapes and dtypes from args
        sig = inspect.signature(self.func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        tensor_metadata = {}
        for name, value in bound_args.arguments.items():
            if hasattr(value, "shape"): # Relaxed check since we're using a stub class in tests
                tensor_metadata[name] = {
                    "shape": list(value.shape),
                    "dtype": value.dtype,
                    "is_output": name == "out" # Hack for now: assume param named 'out' is the output
                }
                
        # Generate MLIR
        module = generate_ir(tree.body[0], self.grid, tensor_metadata, self.debug)
        
        if self.debug:
            print("--- Generated MLIR ---")
            print(module)
            
        if self.compile_only:
            # TODO: Add lowering to flatbuffer here
            return module
            
        # TODO: Add runtime execution
        raise NotImplementedError("Execution not yet implemented")