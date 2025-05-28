# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import functools


class Op:
    def __init__(self, name, loc, ir_op=None):
        self.name = name
        self.loc = loc
        self.str_loc = str(loc)[4:]
        self.str_loc = self.str_loc[:-1]
        self.line_no = None if loc is None else int(self.str_loc.split(":")[1])
        self.ir_op = ir_op
        self.inputs = []
        self.outputs = []

        self.pair_op_start = None
        self.pair_op_end = None
        self.populated = False

    def __repr__(self):
        return f"Op({self.name=}, {self.line_no=})"

    def __str__(self):
        return self.__repr__()


class OpGroup:
    def __init__(self):
        self._ttir = []
        self._ttnn = []
        self.status = []
        self.computed_ttir = False
        self.line_no = -1

    def __repr__(self):
        return f"OpGroup(\n\t{self._ttir=},\n\t{self._ttnn=}\n)"

    def __str__(self):
        return self.__repr__()

    @property
    def ttir(self):
        return self._ttir

    @property
    def ttnn(self):
        return self._ttnn

    @ttir.setter
    def ttir(self, value):
        raise AttributeError("Cannot set ttir")

    @ttnn.setter
    def ttnn(self, value):
        raise AttributeError("Cannot set ttnn")

    def add_ttir_op(self, op):
        self.get_last_ttnn_op.cache_clear()
        self._ttir.append(op)

    def add_ttnn_op(self, op):
        self.get_last_ttir_op.cache_clear()
        self._ttnn.append(op)

    @functools.cache
    def get_last_ttir_op(self):
        if len(self._ttir) == 0:
            return None
        return self._ttir[-1]

    @functools.cache
    def get_last_ttnn_op(self, with_output=False):
        if len(self._ttnn) == 0:
            return None

        if not with_output:
            return self._ttnn[-1]

        for op in self._ttnn[::-1]:
            ir_op = op.ir_op
            if hasattr(ir_op, "results") and len(ir_op.results) > 0:
                return op
        return None
