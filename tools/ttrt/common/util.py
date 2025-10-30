# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import importlib.machinery
import importlib.util
import json
import os
import shutil
from pprint import pprint
import re

import torch
from pkg_resources import get_distribution

# environment tweaks
if "LOGGER_LEVEL" not in os.environ:
    os.environ["LOGGER_LEVEL"] = "FATAL"
if "TT_METAL_LOGGER_LEVEL" not in os.environ:
    os.environ["TT_METAL_LOGGER_LEVEL"] = "FATAL"


def ttrt_datatype_to_torch_dtype(dtype) -> torch.dtype:
    """Converts a PyBound `::tt::target::DataType` into a `torch.dtype`.

    Currently, only `float32`, `uint32`, `uint16`, & `uint8` are supported for
    this conversion

    Arguments
    ---------

    dtype : DataType
        A datatype from the PyBound `DataType` enum from ttrt

    Returns
    -------

    A `torch.dtype` corresponding to `dtype`

    Throws
    ------

    A `ValueError` if `dtype` is not one of `Float32`, `UInt32`, `UInt16`, or `UInt8`

    """
    from ttrt.runtime import DataType

    if dtype == DataType.Float32:
        return torch.float32
    elif dtype == DataType.UInt32:
        return torch.uint32
    elif dtype == DataType.UInt16:
        return torch.uint16
    elif dtype == DataType.UInt8:
        return torch.uint8
    elif dtype == DataType.BFloat16:
        return torch.bfloat16
    elif dtype == DataType.Int32:
        return torch.int32
    else:
        raise ValueError(
            "Only F32, BF16, and unsigned integers are supported in the runtime"
        )


def get_ttrt_metal_home_path():
    package_name = "ttrt"
    spec = importlib.util.find_spec(package_name)
    package_path = os.path.dirname(spec.origin)
    tt_metal_home = f"{package_path}/runtime"
    return tt_metal_home


def mask_torch_inf_nan(tensor):
    import torch

    tensor[
        torch.logical_or(
            torch.isnan(tensor),
            torch.logical_or(torch.isinf(tensor), torch.isneginf(tensor)),
        )
    ] = 0
    return tensor


def get_atol_rtol_pcc(golden, calculated, atol, rtol, logging):
    import numpy as np
    import torch

    # abs() and masked_fill() don't support unsigned integers
    if not torch.is_floating_point(golden):
        golden = golden.to(torch.float64)
    if not torch.is_floating_point(calculated):
        calculated = calculated.to(torch.float64)

    # Calculate atol and rtol
    cal_atol = torch.max(torch.abs(golden - calculated)).item()
    cal_rtol = torch.max(torch.abs((golden - calculated) / calculated)).item()

    # Calculate PCC
    def get_pcc(golden, calculated):
        # Both tensors are nan
        if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
            logging.debug("Both tensors are 'nan'")
            return 1.0
        # Test if either is completely zero
        elif torch.any(golden.bool()) != torch.any(calculated.bool()):
            return 0.0
        # One tensor is all nan, the other is not
        elif torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
            logging.debug("One tensor is all nan, the other is not.")
            return 0.0
        else:
            # For now, mask all infs and nans so that we check the rest... TODO
            golden = mask_torch_inf_nan(golden)
            calculated = mask_torch_inf_nan(calculated)

            if torch.equal(golden, calculated):
                return 1.0

            if golden.dtype == torch.bfloat16:
                golden = golden.type(torch.float32)
                calculated = calculated.type(torch.float32)

            # Single element case
            if golden.numel() == 1:
                return float(torch.isclose(golden, calculated, atol=atol, rtol=rtol))

            # If both tensors are contant
            if torch.max(golden) == torch.min(golden) and torch.max(
                calculated
            ) == torch.min(calculated):
                return torch.isclose(torch.max(golden), torch.max(calculated)).item()

            cal_pcc = np.ma.corrcoef(
                np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
                np.ma.masked_invalid(
                    torch.squeeze(calculated).detach().numpy()
                ).flatten(),
            )
            # Remove correlation coefficient with self (typically always 1.0)
            mask = np.ones(cal_pcc.shape, dtype=bool)
            np.fill_diagonal(mask, 0)
            cal_pcc = np.min(cal_pcc[mask])

            if isinstance(cal_pcc, np.ma.core.MaskedConstant):
                return 1.0

            return cal_pcc

    cal_pcc = get_pcc(golden, calculated)

    return (
        cal_atol,
        cal_rtol,
        cal_pcc,
        f"Max ATOL Delta: {cal_atol}, Max RTOL Delta: {cal_rtol}, PCC: {cal_pcc}",
    )


# Given two torch tensors, return a list of the top k absolute/relative differences.
# Result format: [(v_golden, v_output, abs_diff/rel_diff, index), ...].
def get_topk_diff(golden, calculated, top_k, relative=False):
    import torch

    # Store original dtypes to preserve integer formatting later.
    golden_is_int = not torch.is_floating_point(golden)
    calculated_is_int = not torch.is_floating_point(calculated)

    if not torch.is_floating_point(golden):
        golden = golden.to(torch.float64)
    if not torch.is_floating_point(calculated):
        calculated = calculated.to(torch.float64)

    diff = torch.abs(golden - calculated)
    if relative:
        diff = torch.abs(diff / golden)
        # In case of division by zero
        diff_nz = torch.abs((calculated + 1.0) / (golden + 1.0)) - 1.0
        diff = torch.where(torch.isfinite(diff), diff, diff_nz)

    top_k = min(top_k, diff.numel())
    top_values, top_indices = torch.topk(diff.flatten(), top_k)

    golden_shape = golden.shape
    results = []
    for i in range(top_k):
        flat_idx = top_indices[i].item()
        multi_idx = torch.unravel_index(torch.tensor(flat_idx), golden_shape)
        v_golden = golden[multi_idx].item()
        v_output = calculated[multi_idx].item()
        v_diff = top_values[i].item()
        results.append(
            (
                v_golden,
                v_output,
                v_diff,
                tuple(i.item() for i in multi_idx),
                golden_is_int and calculated_is_int,
            )
        )
    return results


def golden_tensor_to_torch(golden_tensor: "ttrt.binary.GoldenTensor"):
    dtype = ttrt_datatype_to_torch_dtype(golden_tensor.dtype)
    torch_tensor = torch.frombuffer(
        golden_tensor.get_data_buffer(), dtype=dtype
    ).reshape(golden_tensor.shape)
    return torch_tensor


def parse_fabric_config(fabric_config_str: str):
    import ttrt.runtime

    key = fabric_config_str.strip().lower().replace("-", "_")
    if key == "disabled":
        return ttrt.runtime.FabricConfig.DISABLED
    elif key == "fabric_1d":
        return ttrt.runtime.FabricConfig.FABRIC_1D
    elif key == "fabric_1d_ring":
        return ttrt.runtime.FabricConfig.FABRIC_1D_RING
    elif key == "fabric_2d":
        return ttrt.runtime.FabricConfig.FABRIC_2D
    elif key == "fabric_2d_torus":
        return ttrt.runtime.FabricConfig.FABRIC_2D_TORUS
    elif key == "fabric_2d_dynamic":
        return ttrt.runtime.FabricConfig.FABRIC_2D_DYNAMIC
    elif key == "custom":
        return ttrt.runtime.FabricConfig.CUSTOM
    else:
        raise ValueError(f"unknown fabric config '{fabric_config_str}'.")


def convert_input_layouts(device, inputs, fbb, program_index):
    import ttrt.runtime

    inputs_converted = []
    for input_index in range(len(inputs)):
        input_layout = ttrt.runtime.get_layout(fbb, program_index, input_index)
        inputs_converted.append(
            ttrt.runtime.to_layout(inputs[input_index], device, input_layout, True)
        )
    return inputs_converted


def create_tensor(tensor):
    import ttrt.runtime

    # Empty tensor if any of the dim is zero.
    isEmptyTensor = not all(tensor.shape)

    # Create 'owned tensor' in case of empty tensor;
    if isEmptyTensor:
        return ttrt.runtime.create_owned_host_tensor(
            tensor.data_ptr(),
            list(tensor.shape),
            list(tensor.stride()),
            tensor.element_size(),
            Binary.Program.to_data_type(tensor.dtype),
        )

    # Create 'borrowed tensor'.
    return ttrt.runtime.create_borrowed_host_tensor(
        tensor.data_ptr(),
        list(tensor.shape),
        list(tensor.stride()),
        tensor.element_size(),
        Binary.Program.to_data_type(tensor.dtype),
    )


def convert_runtime_to_torch_tensor(runtime_tensor):
    torch_tensor = None
    isEmptyTensor = not all(runtime_tensor.get_shape())
    data_buffer = bytearray(runtime_tensor.get_data_buffer())
    if isEmptyTensor and len(data_buffer) == 0:
        # Create empty tensor.
        torch_tensor = torch.empty(
            runtime_tensor.get_shape(),
            dtype=ttrt_datatype_to_torch_dtype(runtime_tensor.get_dtype()),
        )
    elif not isEmptyTensor and len(data_buffer) > 0:
        # Create regular tensor.
        torch_tensor = torch.frombuffer(
            data_buffer,
            dtype=ttrt_datatype_to_torch_dtype(runtime_tensor.get_dtype()),
        ).reshape(runtime_tensor.get_shape())
    else:
        raise Exception(
            f"Failed: Tensor shape=({runtime_tensor.get_shape()}) and data buffer size={len(data_buffer)} do not match."
        )

    return torch_tensor


class Logger:
    def __init__(self, file_name=""):
        import logging

        self.logging = logging
        self.file_name = file_name
        LEVEL = self.logging.INFO

        if "TTRT_LOGGER_LEVEL" in os.environ:
            if os.environ["TTRT_LOGGER_LEVEL"] == "CRITICAL":
                LEVEL = self.logging.CRITICAL
            elif os.environ["TTRT_LOGGER_LEVEL"] == "ERROR":
                LEVEL = self.logging.ERROR
            elif os.environ["TTRT_LOGGER_LEVEL"] == "WARNING":
                LEVEL = self.logging.WARNING
            elif os.environ["TTRT_LOGGER_LEVEL"] == "DEBUG":
                LEVEL = self.logging.DEBUG

        self.logging.basicConfig(
            filename=self.file_name,
            filemode="w",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=LEVEL,
        )

        self.logging.info(f"set log file={self.file_name}")

    def get_logger(self):
        return self.logging


class Globals:
    def __init__(self, logger):
        self.logger = logger
        self.logging = self.logger.get_logger()

    def add_global_env(self, key, value):
        self.logging.debug(f"adding (key, value)=({key}, {value}) to environment")

        env_vars = dict(os.environ)
        env_vars[key] = value

    def remove_global_env(self, key):
        self.logging.debug(f"removing (key)=({key}) from environment")

        del os.environ[key]

    def get_globals(self):
        return dict(os.environ)

    def print_globals(self):
        env_vars = dict(os.environ)
        for var_name, value in env_vars.items():
            self.logging.info(f"{var_name}: {value}")

    def check_global_exists(self, key):
        env_vars = dict(os.environ)
        if key in env_vars.keys():
            return True

        return False

    def get_ld_path(self, path):
        current_path = os.getenv("LD_LIBRARY_PATH", "")
        updated_path = f"{path}:{current_path}"
        return updated_path

    @staticmethod
    def get_ttmetal_home_path():
        return os.environ.get(
            "TT_METAL_RUNTIME_ROOT", "third_party/tt-metal/src/tt-metal"
        )


class FileManager:
    def __init__(self, logger):
        self.logger = logger
        self.logging = self.logger.get_logger()

    def create_file(self, file_path):
        self.logging.debug(f"creating file={file_path}")

        try:
            if not self.check_directory_exists(os.path.dirname(file_path)):
                self.create_directory(os.path.dirname(file_path))

            with open(file_path, "w") as file:
                file.write("")
        except OSError as e:
            raise OSError(f"error creating file: {e}")
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    def create_directory(self, directory_path):
        self.logging.debug(f"creating directory={directory_path}")

        try:
            os.makedirs(directory_path)
        except FileExistsError as e:
            self.logging.warning(f"directory '{directory_path}' already exists")
        except OSError as e:
            raise OSError(f"error creating directory: {e}")
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    def remove_file(self, file_path):
        self.logging.debug(f"removing file={file_path}")

        try:
            os.remove(file_path)
        except FileNotFoundError:
            self.logging.warning(f"file '{file_path}' not found - cannot remove")
        except PermissionError:
            raise PermissionError(
                f"insufficient permissions to remove file '{file_path}'"
            )
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    def remove_directory(self, directory_path):
        self.logging.debug(f"removing directory={directory_path}")

        try:
            shutil.rmtree(directory_path)
        except FileNotFoundError:
            self.logging.warning(
                f"directory '{directory_path}' not found - cannot remove"
            )
        except PermissionError:
            raise PermissionError(
                f"insufficient permissions to remove directory '{directory_path}'"
            )
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    def copy_file(self, dest_file_path, src_file_path):
        self.logging.debug(f"copying file from={src_file_path} to={dest_file_path}")

        try:
            shutil.copy2(src_file_path, dest_file_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"the source file does not exist: '{src_file_path}'"
            )
        except PermissionError as e:
            raise PermissionError(
                f"permission denied: '{src_file_path}' or '{dest_file_path}'"
            )
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    def copy_directory(self, dest_directory_path, src_directory_path):
        self.logging.debug(
            f"copying directory from={src_directory_path} to={dest_directory_path}"
        )

        try:
            shutil.copytree(src_directory_path, dest_directory_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"source directory does not exist: '{src_directory_path}"
            )
        except FileExistsError as e:
            raise FileExistsError(
                f"destination directory already exists: '{dest_directory_path}'"
            )
        except PermissionError as e:
            raise PermissionError(
                f"permission denied: '{src_directory_path}' or '{dest_directory_path}'"
            ) from e
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    def check_file_exists(self, file_path):
        self.logging.debug(f"checking if file={file_path} exists")
        exists = False

        try:
            if os.path.exists(file_path):
                exists = True
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return exists

    def check_directory_exists(self, directory_path):
        self.logging.debug(f"checking if directory={directory_path} exists")
        exists = False

        try:
            if os.path.isdir(directory_path):
                exists = True
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return exists

    def is_file(self, file_path):
        is_file = False

        try:
            if os.path.isfile(file_path):
                is_file = True
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return is_file

    def is_directory(self, directory_path):
        is_directory = False

        try:
            if os.path.isdir(directory_path):
                is_directory = True
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return is_directory

    def get_file_name(self, file_path):
        self.logging.debug(f"getting file name of={file_path}")
        file_name = ""

        try:
            file_name = os.path.basename(file_path)
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return file_name

    def get_file_extension(self, file_path):
        self.logging.debug(f"getting file extension of={file_path}")
        file_extension = ""

        try:
            _, file_extension = os.path.splitext(file_path)
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return file_extension

    def find_file_paths(self, path, extension):
        self.logging.debug(f"finding all {extension} files from={path}")
        found_files = []

        if self.is_file(path):
            if self.check_file_exists(path):
                if self.get_file_extension(path) == extension:
                    found_files.append(path)
                    self.logging.debug(f"found file={path}")
            else:
                self.logging.info(f"file '{path}' not found - skipping")
        else:
            self.check_directory_exists(path)
            try:
                for root, _, files in os.walk(path):
                    for file in files:
                        if self.get_file_extension(file) == extension:
                            found_files.append(os.path.join(root, file))
                            self.logging.debug(f"found file={os.path.join(root, file)}")
            except Exception as e:
                raise Exception(f"an unexpected error occurred: {e}")

        # Sort files alphabetically to ensure consistent ordering.
        found_files.sort()
        return found_files

    def find_ttnn_binary_paths(self, path):
        return self.find_file_paths(path, Flatbuffer.get_ttnn_file_extension())

    def find_ttmetal_binary_paths(self, path):
        return self.find_file_paths(path, Flatbuffer.get_ttmetal_file_extension())

    def find_ttsys_binary_paths(self, path):
        return self.find_file_paths(path, Flatbuffer.get_ttsys_file_extension())

    def find_emitpy_dylib_paths(self, path):
        return self.find_file_paths(path, EmitPyDylib.get_py_file_extension())

    def find_emitc_dylib_paths(self, path):
        return self.find_file_paths(path, EmitCDylib.get_so_file_extension())

    def find_py_corresponding_ttnn_in_directory(self, path, ttnn_directory):
        filename = self.get_file_name(path)
        ttnn_filename = filename.replace(EmitPyDylib.get_py_file_extension(), ".ttnn")
        found_paths = self.find_file_paths(ttnn_directory, ".ttnn")

        for file_path in found_paths:
            file_basename = self.get_file_name(file_path)
            if file_basename == ttnn_filename:
                return file_path
        return None

    def find_so_corresponding_ttnn_in_directory(self, path, ttnn_directory):
        filename = self.get_file_name(path)
        ttnn_filename = filename.replace(EmitCDylib.get_so_file_extension(), ".ttnn")
        found_paths = self.find_file_paths(ttnn_directory, ".ttnn")

        for file_path in found_paths:
            file_basename = self.get_file_name(file_path)
            if file_basename == ttnn_filename:
                return file_path
        return None

    def load_tensors_from_artifacts(self, bin, key, artifacts_path):
        """
        Open directory, loop through subdirectories, load all .pt files into torch tensors, and save them according to their respective program.
        """
        program_tensors = {}
        program_names = [d for d in os.listdir(artifacts_path)]
        self.logging.debug(f"Loading .pt tensors from directory: {artifacts_path}")
        for program in program_names:
            program_dir = os.path.join(artifacts_path, program)
            files = sorted(
                [d for d in os.listdir(program_dir)],
                key=lambda x: int(re.search(r"_(\d+)\.pt$", x).group(1))
                if re.search(r"_(\d+)\.pt$", x)
                else 0,
            )
            tensors = []
            for file in files:
                file = os.path.join(program_dir, file)
                if file.endswith(".pt") and key in file:
                    try:
                        tensors.append(torch.load(file, weights_only=True))
                        self.logging.debug(f"Loading tensor from file: {file}")
                    except Exception as e:
                        raise Exception(
                            f"Error loading tensor from file {file}: {str(e)}"
                        )

            program_tensors[program] = tensors

        return program_tensors


class Artifacts:
    def __init__(self, logger, file_manager=None, artifacts_folder_path=""):
        self.logger = logger
        self.logging = self.logger.get_logger()
        self.file_manager = (
            file_manager if file_manager != None else FileManager(self.logger)
        )
        self.artifacts_folder_path = (
            artifacts_folder_path
            if artifacts_folder_path != ""
            else f"{os.getcwd()}/ttrt-artifacts"
        )

        self.logging.info(f"setting artifacts folder path={self.artifacts_folder_path}")

    def get_artifacts_folder_path(self):
        return self.artifacts_folder_path

    def get_binary_folder_path(self, binary):
        return f"{self.get_artifacts_folder_path()}/{binary.name}"

    def get_binary_perf_folder_path(self, binary):
        return f"{self.get_artifacts_folder_path()}/{binary.name}/perf"

    def get_binary_run_folder_path(self, binary):
        return f"{self.get_artifacts_folder_path()}/{binary.name}/run"

    def get_emitpy_dylib_folder_path(self, dylib):
        return f"{self.get_artifacts_folder_path()}/{dylib.name}/emitpy"

    def get_emitc_dylib_folder_path(self, dylib):
        return f"{self.get_artifacts_folder_path()}/{dylib.name}/emitc"

    def create_artifacts(self):
        self.file_manager.create_directory(self.get_artifacts_folder_path())

    def clean_artifacts(self):
        self.file_manager.remove_directory(self.get_artifacts_folder_path())

    def clean_binary_artifacts(self, binary):
        self.file_manager.remove_directory(self.get_binary_folder_path(binary))

    def create_binary_artifacts_folder(self, binary):
        binary_folder = self.get_binary_folder_path(binary)
        self.file_manager.create_directory(binary_folder)
        self.file_manager.create_directory(f"{binary_folder}/run")
        self.file_manager.create_directory(f"{binary_folder}/perf")

        for program in binary.programs:
            program_folder = f"{binary_folder}/run/program_{program.index}"
            self.file_manager.create_directory(program_folder)

    def save_binary(self, binary, query=None):
        binary_folder = self.get_binary_folder_path(binary)

        self.logging.info(
            f"saving binary={binary.file_path} to binary_folder={binary_folder}"
        )
        self.file_manager.copy_file(f"{binary_folder}", binary.file_path)

        for program in binary.programs:
            program_folder = f"{binary_folder}/run/program_{program.index}"

            self.logging.info(
                f"saving program={program.index} for binary={binary.file_path} to program_folder={program_folder}"
            )

            for i in range(len(program.input_tensors)):
                self.save_torch_tensor(
                    program_folder,
                    program.input_tensors[i],
                    f"input_{i}.pt",
                )

            for i in range(len(program.output_tensors)):
                self.save_torch_tensor(
                    program_folder,
                    program.output_tensors[i],
                    f"device_output_{i}.pt",
                )

        if query != None:
            self.save_system_desc(query.system_desc, f"{binary_folder}")

    def save_emitpy_dylib(self, dylib, query=None):
        dylib_folder = self.get_emitpy_dylib_folder_path(dylib)

        self.logging.info(
            f"saving dylib={dylib.file_path} to dylib_folder={dylib_folder}"
        )
        self.file_manager.copy_file(f"{dylib_folder}", dylib.file_path)

    def save_emitc_dylib(self, dylib, query=None):
        dylib_folder = self.get_emitc_dylib_folder_path(dylib)

        self.logging.info(
            f"saving dylib={dylib.file_path} to dylib_folder={dylib_folder}"
        )
        self.file_manager.copy_file(f"{dylib_folder}", dylib.file_path)

    def save_torch_tensor(self, folder_path, torch_tensor, torch_tensor_name):
        import torch

        torch_tensor_name = get_sanitized_filename(torch_tensor_name)
        self.logging.info(
            f"saving torch tensor={torch_tensor_name} to folder_path={folder_path}"
        )

        if not self.file_manager.check_directory_exists(folder_path):
            self.file_manager.create_directory(folder_path)
        try:
            torch.save(torch_tensor, f"{folder_path}/{torch_tensor_name}")
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    def save_system_desc(
        self, system_desc, system_desc_folder="", system_desc_name="system_desc.ttsys"
    ):
        system_desc_folder = (
            f"{self.get_artifacts_folder_path()}"
            if system_desc_folder == ""
            else system_desc_folder
        )
        try:
            self.logging.info(
                f"saving system_desc={system_desc_name} to system_desc_folder={system_desc_folder}"
            )
            system_desc.store(f"{system_desc_folder}/{system_desc_name}")
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")


class Flatbuffer:
    ttnn_file_extension = ".ttnn"
    ttmetal_file_extension = ".ttm"
    ttsys_file_extension = ".ttsys"

    def __init__(self, logger, file_manager, file_path, capsule=None):
        import ttrt.binary

        self.logger = logger
        self.logging = self.logger.get_logger()
        self.file_manager = file_manager
        self.file_path = file_path if file_path != None else "<binary-from-capsule>"
        self.name = self.file_manager.get_file_name(file_path)
        self.extension = self.file_manager.get_file_extension(file_path)
        self.version = None

        # temporary state value to check if test failed
        self.test_result = "pass"

    def check_version(self, ignore: bool = False):
        raise UnimplementedError

    @staticmethod
    def get_ttnn_file_extension():
        return Flatbuffer.ttnn_file_extension

    @staticmethod
    def get_ttmetal_file_extension():
        return Flatbuffer.ttmetal_file_extension

    @staticmethod
    def get_ttsys_file_extension():
        return Flatbuffer.ttsys_file_extension


class Binary(Flatbuffer):
    def __init__(self, logger, file_manager, file_path, capsule=None):
        super().__init__(logger, file_manager, file_path, capsule=capsule)

        import torch
        import ttrt.binary

        if not capsule:
            self.fbb = ttrt.binary.load_binary_from_path(file_path)
        else:
            self.fbb = ttrt.binary.load_binary_from_capsule(capsule)
        self.system_desc_dict = ttrt.binary.system_desc_as_dict(self.fbb)
        self.version = self.fbb.version
        self.program_indices = range(self.fbb.get_num_programs())
        self.programs = []
        self.program_results = {}

        for i in self.program_indices:
            program = Binary.Program(i, self.fbb)
            self.programs.append(program)

    def check_version(self, ignore: bool = False):
        if not ignore and not self.fbb.check_schema_hash():
            raise Exception(
                "Binary schema mismatch, please recompile the binary with the compiler at the same schema version"
            )
        return True

    def check_system_desc(self, query, ignore: bool = False):
        import ttrt.binary

        try:
            fbb_system_desc = self.system_desc_dict
            device_system_desc = query.get_system_desc_as_dict()["system_desc"]

            if fbb_system_desc != device_system_desc:
                # Serialize to JSON with pretty printing and split into lines
                import json
                import difflib

                fbb_json = json.dumps(
                    fbb_system_desc, indent=2, sort_keys=True
                ).splitlines()
                device_json = json.dumps(
                    device_system_desc, indent=2, sort_keys=True
                ).splitlines()

                # Generate a unified diff
                diff = list(
                    difflib.unified_diff(
                        fbb_json,
                        device_json,
                        fromfile="flatbuffer_system_desc",
                        tofile="device_system_desc",
                        lineterm="",
                    )
                )

                # Log the detailed diff
                diff_text = "\n".join(diff)
                error_msg = f"system desc for device did not match flatbuffer: {self.file_path}\nDiff details:\n{diff_text}"

                # You might want to log this before raising the exception
                # self.logger.error(error_msg)  # Uncomment and add proper logger if available

                if not ignore:
                    raise Exception(error_msg)
                return False

        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return True

    def get_num_programs(self):
        return self.fbb.get_num_programs()

    def check_program_index_exists(self, program_index):
        if program_index >= self.get_num_programs():
            return False

        return True

    def get_program(self, program_index):
        if program_index > self.get_num_programs():
            raise Exception(
                f"program index={program_index} is greater than number of programs availabe={self.get_num_programs()}!"
            )

        return self.programs[program_index]

    def save(self, artifacts, query=None):
        artifacts.save_binary(self, query)

    def print(self):
        print(f"Flatbuffer {self.name}:")

        for i, p in enumerate(self.programs):
            print(f"\nProgram {i+1} operations:")
            pprint(p.fbb_to_dict())

        print()

    def add_program_results(
        self,
        program_index,
        loop,
        total_submit_duration_ns,
        total_get_outputs_duration_ns,
        total_ttnn_api_duration_ns=None,
        total_device_kernel_duration_ns=None,
    ):
        program_key = f"program_index_{program_index}"
        if program_key not in self.program_results.keys():
            self.program_results[program_key] = {}

        loop_key = f"loop_{loop}"
        if loop_key not in self.program_results[program_key].keys():
            self.program_results[program_key][loop_key] = {}

        self.program_results[program_key][loop_key][
            "total_submit_duration_ns"
        ] = total_submit_duration_ns
        self.program_results[program_key][loop_key][
            "total_get_outputs_duration_ns"
        ] = total_get_outputs_duration_ns
        self.program_results[program_key][loop_key][
            "total_submit_plus_get_outputs_duration_ns"
        ] = (total_submit_duration_ns + total_get_outputs_duration_ns)
        self.program_results[program_key][loop_key][
            "total_ttnn_api_duration_ns"
        ] = total_ttnn_api_duration_ns
        self.program_results[program_key][loop_key][
            "total_device_kernel_duration_ns"
        ] = total_device_kernel_duration_ns

    def update_total_ttnn_api_duration_ns(
        self, program_index, loop, total_ttnn_api_duration_ns
    ):
        self.program_results[f"program_index_{program_index}"][f"loop_{loop}"][
            "total_ttnn_api_duration_ns"
        ] = total_ttnn_api_duration_ns

    def update_total_device_kernel_duration_ns(
        self, program_index, loop, total_device_kernel_duration_ns
    ):
        self.program_results[f"program_index_{program_index}"][f"loop_{loop}"][
            "total_device_kernel_duration_ns"
        ] = total_device_kernel_duration_ns

    class Program:
        def __init__(self, index, binary):
            import ttrt.binary

            self.fbb = binary
            self.index = index
            self.name = self.fbb.get_program_name(self.index)
            self.inputs = ttrt.binary.program_inputs_as_dict(self.fbb, self.index)
            self.outputs = ttrt.binary.program_outputs_as_dict(self.fbb, self.index)
            self.mesh_shape = self.fbb.get_program_mesh_shape(self.index)
            self.input_tensors = []
            self.output_tensors = []

        def num_inputs(self):
            return len(self.inputs)

        def num_outputs(self):
            return len(self.outputs)

        def populate_inputs(self, init_fn, golden_inputs=[]):
            if len(golden_inputs) > 0:
                assert len(golden_inputs) == len(self.inputs)
                for index, input_fb in enumerate(self.inputs):
                    reshaped = torch.reshape(
                        golden_inputs[index], input_fb["desc"]["shape"]
                    )
                    self.input_tensors.append(reshaped)
            else:
                for i in self.inputs:
                    torch_tensor = init_fn(
                        i["desc"]["shape"],
                        dtype=Binary.Program.from_data_type(
                            i["desc"]["layout"]["memory_desc"]["data_type"]
                        ),
                    )
                    self.input_tensors.append(torch_tensor)

        def populate_outputs(self, init_fn):
            for i in self.outputs:
                torch_tensor = init_fn(
                    i["desc"]["shape"],
                    dtype=Binary.Program.from_data_type(
                        i["desc"]["layout"]["memory_desc"]["data_type"]
                    ),
                )
                self.output_tensors.append(torch_tensor)

        def is_private(self):
            import ttrt.binary

            return self.fbb.is_program_private(self.index)

        def to_dict(self) -> dict:
            return {
                i: op["debug_info"]
                for i, op in enumerate(
                    self.ttrt.binary.program_ops_as_dict(self.fbb, i)
                )
            }

        @staticmethod
        def to_data_type(dtype):
            import torch
            import ttrt.runtime

            if dtype == torch.float32:
                return ttrt.runtime.DataType.Float32
            if dtype == torch.float16:
                return ttrt.runtime.DataType.Float16
            if dtype == torch.bfloat16:
                return ttrt.runtime.DataType.BFloat16
            if dtype == torch.uint32:
                return ttrt.runtime.DataType.UInt32
            if dtype == torch.uint16:
                return ttrt.runtime.DataType.UInt16
            if dtype == torch.uint8:
                return ttrt.runtime.DataType.UInt8
            if dtype == torch.int32:
                return ttrt.runtime.DataType.Int32
            # Data types which are unsupported on ttnn
            if dtype == torch.float64:
                return ttrt.runtime.DataType.Float64
            if dtype == torch.int64:
                return ttrt.runtime.DataType.Int64
            if dtype == torch.uint64:
                return ttrt.runtime.DataType.UInt64
            if dtype == torch.int16:
                return ttrt.runtime.DataType.Int16
            if dtype == torch.int8:
                return ttrt.runtime.DataType.Int8
            if dtype == torch.bool:
                return ttrt.runtime.DataType.Bool
            raise ValueError(f"Torch dtype: {dtype} has no runtime DataType equivalent")

        @staticmethod
        def from_data_type(dtype):
            import torch

            if dtype == "Float32":
                return torch.float32
            if dtype == "Float16":
                return torch.float16
            if dtype == "BFloat16":
                return torch.bfloat16
            if dtype == "UInt32":
                return torch.uint32
            if dtype == "UInt16":
                return torch.uint16
            if dtype == "UInt8":
                return torch.uint8
            if dtype == "Int32":
                return torch.int32
            # Data types which are unsupported on ttnn
            if dtype == "Float64":
                return torch.float64
            if dtype == "Int64":
                return torch.int64
            if dtype == "UInt64":
                return torch.uint64
            if dtype == "Int16":
                return torch.int16
            if dtype == "Int8":
                return torch.int8
            if dtype == "Bool":
                return torch.bool

            raise ValueError(f"unsupported dtype: {dtype}")


class SystemDesc(Flatbuffer):
    def __init__(self, logger, file_manager, file_path):
        super().__init__(logger, file_manager, file_path)

        import ttrt.binary

        self.fbb = ttrt.binary.load_system_desc_from_path(file_path)
        self.fbb_dict = ttrt.binary.fbb_as_dict(self.fbb)
        self.version = self.fbb.version

        # temporary state value to check if test failed
        self.test_result = "pass"

    def check_version(self, ignore: bool = False):
        if not ignore and not self.fbb.check_schema_hash():
            raise Exception(
                "Binary schema mismatch, please recompile the binary with the compiler at the same schema version"
            )
        return True


class EmitPyDylib:
    def __init__(self, logger, file_manager, file_path, capsule=None):
        self.logger = logger
        self.logging = self.logger.get_logger()
        self.file_manager = file_manager
        self.file_path = file_path if file_path != None else "<dylib-from-capsule>"
        self.name = self.file_manager.get_file_name(file_path)

        # temporary state value to check if test failed
        self.test_result = "pass"

    @staticmethod
    def get_py_file_extension():
        return ".py"


class EmitCDylib:
    def __init__(self, logger, file_manager, file_path, capsule=None):
        self.logger = logger
        self.logging = self.logger.get_logger()
        self.file_manager = file_manager
        self.file_path = file_path if file_path != None else "<dylib-from-capsule>"
        self.name = self.file_manager.get_file_name(file_path)

        # temporary state value to check if test failed
        self.test_result = "pass"

    @staticmethod
    def get_so_file_extension():
        return ".so"


class TTRTTestException(Exception):
    """ "Base class for all "Test Specific" Errors in TTRT"""

    pass


class PCCErrorException(TTRTTestException):
    """Class to store PCC Comparison Errors"""

    pass


class AllCloseErrorException(TTRTTestException):
    """Class to store AllClose Comparison Errors"""

    pass


# Define a constant TTRT_TEST_ERROR_RETURN_CODE
TTRT_TEST_EXCEPTION_RETURN_CODE = 42


class Results:
    def __init__(self, logger, file_manager):
        self.logger = logger
        self.logging = self.logger.get_logger()
        self.file_manager = file_manager if file_manager != None else file_manager
        self.results = []

    def add_result(self, result):
        self.results.append(result)

    def save_results(self, file_name="results.json"):
        with open(file_name, "w") as file:
            json.dump(self.results, file, indent=2)

        self.logging.info(f"results saved to={file_name}")

        # count total tests, skips and failures
        with open(file_name, "r") as file:
            data = json.load(file)

        import xml.etree.ElementTree as ET

        total_tests = len(data)
        failures = sum(1 for item in data if item.get("result", "") != "pass")
        skipped = sum(1 for item in data if item.get("result", "") == "skipped")

        testsuites = ET.Element("testsuites")
        testsuites.set("name", "TTRT")
        testsuites.set("tests", str(total_tests))
        testsuites.set("failures", str(failures))
        testsuites.set("skipped", str(skipped))

        testsuite = ET.SubElement(testsuites, "testsuite")
        testsuite.set("name", "TTRT")
        testsuite.set("tests", str(total_tests))
        testsuite.set("failures", str(failures))
        testsuite.set("skipped", str(skipped))

        for item in data:
            testcase = ET.SubElement(testsuite, "testcase")
            testcase.set("name", item.get("file_path", ""))
            testcase.set("file_path", item.get("file_path", ""))
            testcase.set("result", item.get("result", ""))
            testcase.set("exception", item.get("exception", ""))
            testcase.set("log_file", item.get("log_file", ""))
            testcase.set("artifacts", item.get("artifacts", ""))

        tree = ET.ElementTree(testsuites)
        xml_file_path = "ttrt_report.xml"
        tree.write(xml_file_path, encoding="utf-8", xml_declaration=True)

    def get_result_code(self):
        return_code = 0
        for entry in self.results:
            res = entry.get("result")
            if entry.get("result") != "pass":
                if res == "test_error":
                    return_code = TTRT_TEST_EXCEPTION_RETURN_CODE
                else:
                    # Prioritize severity of return_code 1 if any non-test errors are encountered
                    return 1

        return return_code

    def get_results(self):
        return self.results


def get_sanitized_filename(name: str, replacement: str = "_") -> str:
    # make string safe for file name
    forbidden = ':"/\\|?*\0'
    s = re.sub(f"[{re.escape(forbidden)}\\x00-\\x1F]", replacement, name)
    if not s:
        s = "untitled"
    return s.strip()
