# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import csv
from dataclasses import dataclass
from utils.metrics import compute_pcc, compute_abs_err


@dataclass
class ValidatorInfo:
    ttir_op: str
    ttnn_op: str
    pcc: float
    abs_err: float
    info: str
    line_no: int
    lig: bool  # TTNN last with output in its group; used as an easy way to filter the output log

    def __repr__(self) -> str:
        return f"ValidatorInfo({self.ttir_op=}, {self.ttnn_op=}, {self.pcc=}, {self.abs_err=}, {self.info=}, {self.line_no=}, {self.lig=})"


class Validator:
    def __init__(self):
        self.pcc_data = {}
        self._first_export = True
        self._exported_count = 0
        self.ttir2ttnn_map = {}
        self.ttnn2ttir_tensor = {}

    def validate(self, ttnn_op, op_group, intermediate=False):
        if op_group.line_no not in self.pcc_data:
            self.pcc_data[op_group.line_no] = []
        self.compare_group(ttnn_op, op_group, intermediate)

    def export_csv(self, filename):
        mode = "w" if self._first_export else "a"
        total_items = sum(len(data) for data in self.pcc_data.values())

        if not (self._first_export or total_items > self._exported_count):
            return

        with open(filename, mode, newline="") as f:
            writer = csv.writer(f)

            if self._first_export:
                writer.writerow(
                    ["TTIR Line", "TTIR Op", "TTNN Op", "PCC", "Abs Err", "Info", "LIG"]
                )
                self._first_export = False

            all_items = [
                (line_no, item)
                for line_no, data in self.pcc_data.items()
                for item in data
            ]

            for line_no, item in all_items[self._exported_count :]:
                writer.writerow(
                    [
                        line_no,
                        item.ttir_op,
                        item.ttnn_op,
                        item.pcc,
                        item.abs_err,
                        item.info,
                        item.lig,
                    ]
                )

            self._exported_count = total_items

    def compare_group(self, ttnn_op, op_group, intermediate=False):
        last_ttir_result = None
        last_ttir_op = None
        # find if ttnn op doesn't have an output, just set None everywhere
        if len(op_group.ttir) == 0:
            return
        if len(ttnn_op.outputs) == 0 or ttnn_op.outputs[0].tt_data is None:
            validator_info = ValidatorInfo(
                ttir_op=str(op_group.ttir[0].ir_op),
                ttnn_op=str(ttnn_op.ir_op),
                pcc=None,
                abs_err=None,
                info="No output",
                line_no=op_group.line_no,
                lig=(ttnn_op == op_group.get_last_ttnn_op(with_output=True)),
            )
            op_group.status.append(validator_info)
            self.pcc_data[op_group.line_no].append(validator_info)
            return
        last_ttnn_result = ttnn_op.outputs[0].tt_data

        pccs = []
        max_pcc = 0.0
        min_abs_err = float("inf")
        # will be reworked

        for op in op_group.ttir[::-1]:
            if not op.outputs:
                continue
            output = op.outputs[0]
            if output.cpu_data is None:
                continue
            last_ttir_result = output.cpu_data
            pcc = compute_pcc(last_ttir_result, last_ttnn_result)
            abs_err = compute_abs_err(last_ttir_result, last_ttnn_result)
            if pcc is None:
                continue
            # import pdb; pdb.set_trace()
            self.ttir2ttnn_map[str(op.ir_op.result.get_name())] = str(
                ttnn_op.ir_op.result.get_name()
            )
            self.ttnn2ttir_tensor[str(ttnn_op.ir_op.result.get_name())] = str(
                op.ir_op.result.get_name()
            )
            pccs.append(
                ValidatorInfo(
                    ttir_op=str(op.ir_op),
                    ttnn_op=str(ttnn_op.ir_op),
                    pcc=pcc,
                    abs_err=abs_err,
                    info="",
                    line_no=op_group.line_no,
                    lig=(ttnn_op == op_group.get_last_ttnn_op(with_output=True)),
                )
            )
            max_pcc = max(max_pcc, pcc)
            min_abs_err = min(min_abs_err, abs_err)

        self.ttir2ttnn_map[str(op.ir_op.result.get_name())] = str(
            ttnn_op.ir_op.result.get_name()
        )
        self.ttnn2ttir_tensor[str(ttnn_op.ir_op.result.get_name())] = str(
            op.ir_op.result.get_name()
        )
        validator_info = ValidatorInfo(
            ttir_op=str(op.ir_op),
            ttnn_op=str(ttnn_op.ir_op),
            pcc=max_pcc,
            abs_err=min_abs_err,
            info=f"{last_ttir_result}, {last_ttnn_result}",
            line_no=op_group.line_no,
            lig=(ttnn_op == op_group.get_last_ttnn_op(with_output=True)),
        )

        op_group.status.append(validator_info)
        self.pcc_data[op_group.line_no].append(validator_info)

        if intermediate:
            op_group.status[
                -1
            ].info = f"pcc={op_group.status[-1].pcc}, abs_err={op_group.status[-1].abs_err}, info={op_group.status[-1].info}"
            op_group.status[-1].pcc = None
            op_group.status[-1].abs_err = None

            pcc_data_entry = self.pcc_data[op_group.line_no][-1]
            pcc_data_entry.info = f"pcc={pcc_data_entry.pcc}, abs_err={pcc_data_entry.abs_err}, info={pcc_data_entry.info}"
            pcc_data_entry.pcc = None
            pcc_data_entry.abs_err = None
