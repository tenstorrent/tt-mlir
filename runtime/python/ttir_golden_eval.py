import ttrt
import torch
from ttmlir.ir import *
from ttmlir.dialects import ttir, tt, tensor, func

class GoldenContext:
    def __init__(self, source, inputs):
        self.mlir_ctx = Context()
        self.mlir_ctx.allow_unregistered_dialects = True
        self.module = Module.parse(source, self.mlir_ctx)
        self.entry = None
        for op in self.module.body:
            if isinstance(op, func.FuncOp):
                assert self.entry is None, "multiple entry points"
                self.entry = op
        assert self.entry is not None, "no entry point"
        self.loc_to_op = {}
        self.values = {}
        for arg in self.entry.regions[0].blocks[0].arguments:
            self.values[arg] = inputs[arg.arg_number]
        for op in self.entry.regions[0].blocks[0]:
            self.loc_to_op[str(op.location)] = op

    def eval(self, binary, program_context, op_context):
        loc = ttrt.runtime.get_op_loc_info(op_context)
        op = self.loc_to_op.get(loc, None)
        if op is None:
            return
        if op.result in self.values:
            # Already evaluated this op, but this needs way more thought
            # Many decomps will explode multiple ops with the same location.
            # Probably heuristic , last op wins?  Maybe compiler tags the
            # locations with a special marked for golden.
            return

        print(f"golden evaling op: {op.name} {loc}")
        op_output_tensor = ttrt.runtime.get_op_output_tensor(op_context, program_context)
        assert op_output_tensor
        device_tensor = self.to_torch(op_output_tensor)
        golden_tensor = self.eval_golden_tensor(op, program_context, op_context)
        print(f"device: {device_tensor}")
        print(f"golden: {golden_tensor}")
        print(f"error: {torch.max(torch.abs(device_tensor - golden_tensor))}")

    def to_torch(self, rt_tensor):
        if type(rt_tensor) == torch.Tensor:
            return rt_tensor
        dtype = ttrt.common.util.ttrt_datatype_to_torch_dtype(rt_tensor.get_dtype())
        return torch.frombuffer(rt_tensor.get_data_buffer(), dtype=dtype).flatten()

    def eval_golden_tensor(self, op, program_context, op_context):
        attr = op.name.replace(".", "_")
        if not hasattr(self, attr):
            print(f"No golden eval implemented for op: {op.name}")
            return None
        operands = []
        for operand in op.operands:
            if operand in self.values:
                operands.append(self.to_torch(self.values[operand]))
            elif operand.owner.name == "ttir.empty":
                continue
            else:
                raise NotImplementedError(
                    f"Operand {operand} not found")
        golden = getattr(self, attr)(op, operands)
        self.values[op.result] = golden
        return golden

    def ttir_abs(self, op, operands):
        assert len(operands) == 1
        return torch.abs(operands[0])
