# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from core.value import TensorValue
from core.op import Op, OpGroup
from abc import abstractmethod


class OpManager:
    def __init__(self, module) -> None:
        self.ops = []
        self.current_op = 0
        self.module = module

    @abstractmethod
    def parse_module(self):
        pass

    def populate_op(self, op):
        if op.populated:
            return
        for input in op.ir_op.operands:
            tensor_value = TensorValue(input.get_name())
            op.inputs.append(tensor_value)
        for output in op.ir_op.results:
            tensor_value = TensorValue(output.get_name())
            op.outputs.append(tensor_value)
        op.populated = True

    def current_op(self):
        return self.ops[self.current_op]

    def get_op(self):
        if self.current_op >= len(self.ops):
            return None
        op = self.ops[self.current_op]
        self.current_op += 1
        return op

    def reset(self):
        self.current_op = 0

    def __len__(self):
        return len(self.ops)


class TTNNOpManager(OpManager):
    def __init__(self, module):
        super().__init__(module)
        self.parse_module()

    def parse_module(self):
        device_module = self.module.body.operations[0]
        inner_module = device_module.regions[0].blocks[0].operations[0]
        func_op = inner_module.regions[0].blocks[0].operations[1]

        for ttnn_op in func_op.regions[0].blocks[0].operations:
            op = Op(ttnn_op.name, ttnn_op.location, ttnn_op)
            self.ops.append(op)


class TTIROpManager(OpManager):
    def __init__(self, module):
        super().__init__(module)
        self.parse_module()

    def parse_module(self):
        func_op = self.module.body.operations[0]
        region = func_op.regions[0]
        block = region.blocks[0]

        for ttir_op in block.operations:
            op = Op(ttir_op.name, ttir_op.location, ttir_op)
            self.ops.append(op)


def tie_manager_ops(ttir_manager: TTIROpManager, ttnn_manager: TTNNOpManager):
    ttir_manager.reset()
    ttnn_manager.reset()

    line_dicts = {}
    list_ttnn_ops = []

    for op in ttir_manager.ops:
        if op.line_no not in line_dicts:
            line_dicts[op.line_no] = OpGroup()
        line_dicts[op.line_no].add_ttir_op(op)
        line_dicts[op.line_no].line_no = op.line_no
    for op in ttnn_manager.ops:
        if op.line_no not in line_dicts:
            line_dicts[op.line_no] = OpGroup()
        line_dicts[op.line_no].add_ttnn_op(op)
        line_dicts[op.line_no].line_no = op.line_no
        list_ttnn_ops.append(op)

    return line_dicts, list_ttnn_ops
