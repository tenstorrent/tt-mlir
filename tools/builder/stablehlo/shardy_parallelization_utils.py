# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import itertools
from typing import Callable, List, Optional, Tuple, Union, Literal, Dict
from collections import OrderedDict

from ttmlir.ir import *
from ttmlir.dialects import func, ttcore, ttnn, ttir, sdy
from ttmlir.passmanager import PassManager
from ttmlir.passes import (
    stablehlo_pipeline,
)


def find_module_permutations(module, num_devices) -> List[Module]:
    module_permutations: List[Module] = []
    mesh_shapes = find_mesh_permutations(num_devices)
    argument_tensor_sharding_attrs: List[List[sdy.TensorShardingAttr]] = []

    for func_op in module.body.operations:
        if not isinstance(func_op, func.FuncOp):
            continue

        for arg in func_op.type.inputs:
            argument_tensor_sharding_attrs.append(
                generate_sharding_permutations(arg, "mesh", ["x", "y"])
            )

    combinations = list(itertools.product(*argument_tensor_sharding_attrs))

    for mesh_shape in mesh_shapes:
        for arg_attrs_combination in combinations:
            module_clone = Module.parse(str(module), module.context)

            mesh_axes = []
            for axis_name, axis_size in mesh_shape.items():
                mesh_axis = sdy.MeshAxisAttr.get(axis_name, axis_size)
                mesh_axes.append(mesh_axis)

            mesh_attr = sdy.MeshAttr.get(mesh_axes)
            module_clone.body.append(sdy.MeshOp("mesh", mesh_attr))

            for func_op in module_clone.body.operations:
                if not isinstance(func_op, func.FuncOp):
                    continue

                arg_attr_list = func_op.arg_attrs
                new_arg_attr_list = []
                for arg_number, arg_attrs in enumerate(arg_attr_list):
                    new_arg_attr = {}
                    for attr in arg_attrs:
                        new_arg_attr[attr.name.value] = attr
                    new_arg_attr["sdy.sharding"] = arg_attrs_combination[arg_number]
                    new_arg_attr_list.append(DictAttr.get(new_arg_attr))

                func_op.arg_attrs = ArrayAttr.get(new_arg_attr_list)
                module_permutations.append(module_clone)

    return module_permutations


def find_mesh_permutations(num_devices: int) -> List[OrderedDict[str, int]]:
    permutations: List[OrderedDict[str, int]] = []

    for x in range(1, int(num_devices**0.5) + 1):
        if num_devices % x == 0:
            y = num_devices // x
            permutations.append(OrderedDict([("x", x), ("y", y)]))

            if x != y:
                permutations.append(OrderedDict([("x", y), ("y", x)]))

    return permutations


def generate_sharding_permutations(
    ranked_tensor_type: RankedTensorType, mesh_name: str, axes_list: List[str]
) -> List[sdy.TensorShardingAttr]:
    sharding_permutations: List[sdy.TensorShardingAttr] = []

    rank = ranked_tensor_type.rank
    states = [i for i in range(rank)]
    all_combinations = []

    for k in range(1, len(axes_list) + 1):
        for state_subset in itertools.combinations(states, k):
            for value_perm in itertools.permutations(axes_list, k):
                assignment = {s: None for s in states}
                for s, v in zip(state_subset, value_perm):
                    assignment[s] = v
                all_combinations.append(assignment)

    for assignment in all_combinations:
        dimension_shardings = []
        for i in range(rank):
            axes = []
            if assignment[i] != None:
                axes.append(sdy.AxisRefAttr.get(assignment[i]))
            dimension_sharding_attr = sdy.DimensionShardingAttr.get(axes, False)
            dimension_shardings.append(dimension_sharding_attr)

        sharding_attr = sdy.TensorShardingAttr.get(mesh_name, dimension_shardings)
        sharding_permutations.append(sharding_attr)

    return sharding_permutations


def optimal_module_least_num_collectives(
    module_permutations: List[Module],
) -> List[Module]:
    min_collectives = float("inf")
    optimal_module = []

    for module in module_permutations:
        try:
            num_collectives_found = 0
            stablehlo_pipeline(module)

            for func_op in module.body.operations:
                if not isinstance(func_op, func.FuncOp):
                    continue

                for block in func_op.body:
                    for op in block.operations:
                        if not isinstance(op, sdy.ManualComputationOp):
                            continue

                        for block in op.body:
                            for inner_op in block.operations:
                                if (
                                    isinstance(inner_op, sdy.AllGatherOp)
                                    or isinstance(inner_op, sdy.AllReduceOp)
                                    or isinstance(inner_op, sdy.AllSliceOp)
                                    or isinstance(inner_op, sdy.AllToAllOp)
                                    or isinstance(inner_op, sdy.CollectivePermuteOp)
                                    or isinstance(inner_op, sdy.ReduceScatterOp)
                                ):
                                    num_collectives_found += 1

            if num_collectives_found <= min_collectives:
                min_collectives = num_collectives_found
                optimal_module.append(module)
        except:
            continue

    return optimal_module
