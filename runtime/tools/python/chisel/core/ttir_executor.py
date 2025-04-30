# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pprint import pprint
from utils.mapping import ttir_to_torch_mapping


class TTIRExecutor:
    def __init__(self):
        self.tensor_pool = {}

    def execute_op(self, op, programContext):
        pprint(f"Executing op: {op.name}")

        if op.name not in ttir_to_torch_mapping:
            raise ValueError(f"Unknown op: {op.name}")

        mapping = ttir_to_torch_mapping[op.name]
        outputs = []
        if "outputs" in dir(op.ir_op):
            outputs = [x.get_name() for x in op.ir_op.outputs]
        if "output" in dir(op.ir_op):
            outputs.append(op.ir_op.output.get_name())

        inputs = [
            self.tensor_pool.get(input.name)
            for input in op.inputs
            if input.name not in outputs
        ]
        pprint(
            f"Input shapes: {[(inp.name, x.shape if x is not None else None) for inp, x in zip(op.inputs, inputs)]}"
        )
        op_result = mapping(op.ir_op, inputs)
        if op.name == "func.return":
            return op_result

        for attr_name in ["output", "result"]:
            if attr_name not in dir(op.ir_op):
                continue
            tensor_name = getattr(op.ir_op, attr_name).get_name()
            if len(op.outputs) > 0:
                op.outputs[0].set_cpu_data(op_result, programContext)
                pprint(
                    f"Output shape: {tensor_name} = {op_result.shape if op_result is not None else None}"
                )
            self.tensor_pool[tensor_name] = op_result

        return op_result

    def execute_ops(self, ops):
        for op in ops:
            self.execute_op(op)
