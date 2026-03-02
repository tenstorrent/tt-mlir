import sys
from ttmlir.ir import *
from ttmlir.dialects import func, affine, arith

with Context() as ctx, Location.unknown(ctx):
    ctx.allow_unregistered_dialects = True
    module = Module.create()
    with InsertionPoint(module.body):
        # Let's try to create affine.for
        # affine.for %arg = 0 to 128 step 32 { ... }
        
        # affine.for takes lower_bound, upper_bound, step as attributes or affine maps usually, but in python bindings it might be different.
        try:
            # Let's see the signature of affine.ForOp
            print(help(affine.ForOp.__init__))
        except Exception as e:
            print("Error getting help:", e)
            
EOF
