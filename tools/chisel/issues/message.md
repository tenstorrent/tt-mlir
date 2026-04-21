Hey Julia! Thanks for reaching out and for all the work on your branch - I went through `jgrim/runtime-goldens` and there's a lot of overlap with what we're building in Chisel v2.


- CMake and build changes - we need the same ones for Chisel v2
- We would also need The postExecutionCallback mechanism.
- Runtime golden integration into builder - enable_runtime_goldens can be implemented in the same way, only diff will be that we will add the callback on the Python side.
- We will not use regex-based attribute parsing from the flatbuffer - the flatbuffer already contains the full MLIR source (binary.mlir.source), and we can parse it into a MLIR Module using ttmlir Module.parse(). This gives us structured access to op attributes, operands, and types directly from the IR. So we can get all the op attributes the same way builder does it.
- No registerCallback from runtime.cpp - in Chisel v2 we will support importing both ttrt and torch_xla since they point to the same MLIR runtime .so, so we use DebugHooks.get() directly from Python rather than going through a separate registration path.
- One thing that is left unclear is how to generally call the golden_op, as we can access the op object the key to the dict is not a problem, but passing the right attributes can be tricky. I hope that it would be possible to do: `golden_fn(*[pool[o.get_name()] for o in op.operands], **{name: value for name, value in op.attributes})`. Do you think this will work?
