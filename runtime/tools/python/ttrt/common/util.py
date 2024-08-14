# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttrt.binary
import os
import json
import importlib.machinery
import sys
import signal
import os
import io
import subprocess
import time
import socket
from pkg_resources import get_distribution
import sys
import shutil

#######################################################################################
#######################################**UTILS**#######################################
#######################################################################################
class Logger:
    def __init__(self, file_name):
        import logging
        self.logging = logging
        self.file_name = file_name

        self.logging.basicConfig(
            filename=self.file_name,
            filemode='w',        
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=self.logging.NOTSET
        )

    def get_logger(self):
        return self.logging

class Globals:
    def __init__(self, logging):
        self.logging = logging

        if self.check_global_exists("LOGGER_LEVEL"):
            self.add_global_env("LOGGER_LEVEL", "FATAL")
        if self.check_global_exists("TT_METAL_LOGGER_LEVEL"):
            self.add_global_env("TT_METAL_LOGGER_LEVEL", "FATAL")

    def add_global_env(self, key, value):
        env_vars = dict(os.environ)
        env_vars[key] = value

    def remove_global_env(self, key):
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

    @staticmethod
    def get_ttmlir_home_path():
        return os.environ.get("TT_MLIR_HOME", f"{os.getcwd()}")

    @staticmethod
    def get_ttmlir_venv_path():
        return os.environ.get("TTMLIR_VENV_DIR", "/opt/ttmlir-toolchain/venv")

    @staticmethod
    def get_ttmetal_home_path():
        return os.environ.get("TT_METAL_HOME", "third_party/tt-metal/src/tt-metal")

class FileManager:
    def __init__(self, logging):
        self.logging = logging

    def file_directory_constraint_check(self, path):
        if not isinstance(path, str):
            raise TypeError(f"the path '{path}' must be a string")

        if not path:
            raise ValueError(f"the path '{path}' cannot be an empty string")

    def create_file(self, file_path):
        self.file_directory_constraint_check(file_path)

        try:
            if not self.check_directory_exists(os.path.dirname(file_path)):
                self.create_directory(os.path.dirname(file_path))

            with open(file_path, 'w') as file:
                file.write('')
        except OSError as e:
            raise OSError(f"error creating file: {e}")
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    def create_directory(self, directory_path):
        self.file_directory_constraint_check(directory_path)

        try:
            os.makedirs(directory_path)
        except FileExistsError as e:
            self.logging.warning(f"directory '{directory_path}' already exists")
        except OSError as e:
            raise OSError(f"error creating directory: {e}")
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    def remove_file(self, file_path):
        self.file_directory_constraint_check(file_path)

        try:
            os.remove(file_path)
        except FileNotFoundError:
            self.logging.warning(f"file '{file_path}' not found")
        except PermissionError:
            raise PermissionError(f"insufficient permissions to remove file '{file_path}'")
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    def remove_directory(self, directory_path):
        self.file_directory_constraint_check(directory_path)

        try:
            shutil.rmtree(directory_path)
        except FileNotFoundError:
            self.logging.warning(f"directory '{directory_path}' not found")
        except PermissionError:
            raise PermissionError(f"insufficient permissions to remove directory '{directory_path}'")
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    def copy_file(self, dest_file_path, src_file_path):
        self.file_directory_constraint_check(dest_file_path)
        self.file_directory_constraint_check(src_file_path)

        try:
            shutil.copy2(src_file_path, dest_file_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"the source file does not exist: '{src_file_path}'")
        except PermissionError as e:
            raise PermissionError(f"permission denied: '{src_file_path}' or '{dest_file_path}'")
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    def copy_directory(self, dest_directory_path, src_directory_path):
        self.file_directory_constraint_check(dest_directory_path)
        self.file_directory_constraint_check(src_directory_path)

        try:
            shutil.copytree(src_directory_path, dest_directory_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"source directory does not exist: '{src_directory_path}")
        except FileExistsError as e:
            raise FileExistsError(f"destination directory already exists: '{dest_directory_path}'")
        except PermissionError as e:
            raise PermissionError(f"permission denied: '{src_directory_path}' or '{dest_directory_path}'") from e
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")
    
    def check_file_exists(self, file_path):
        self.file_directory_constraint_check(file_path)
        exists = False

        try:
            if os.path.exists(file_path):
                exists = True
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return exists

    def check_directory_exists(self, directory_path):
        self.file_directory_constraint_check(directory_path)
        exists = False

        try:
            if os.path.isdir(directory_path):
                exists = True
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return exists

    def is_file(self, file_path):
        self.file_directory_constraint_check(file_path)
        is_file = False

        try:
            if os.path.exists(file_path):
                is_file = True
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return is_file

    def is_directory(self, directory_path):
        self.file_directory_constraint_check(directory_path)
        is_directory = False

        try:
            if os.path.isdir(directory_path):
                is_directory = True
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return is_directory

    def get_file_name(self, file_path):
        self.file_directory_constraint_check(file_path)
        file_name = ""

        try:
            file_name = os.path.basename(file_path)
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return file_name

    def get_file_extension(self, file_path):
        self.file_directory_constraint_check(file_path)
        file_extension = ""

        try:
            _, file_extension = os.path.splitext(filepath)
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return file_extension

class Artifacts:
    def __init__(self, logging, file_manager, artifacts_folder_path=""):
      self.logging = logging
      self.file_manager = file_manager
      self.artifacts_folder_path = artifacts_folder_path if artifacts_folder_path != "" else f"{Globals.get_ttmlir_home_path()}/ttrt-artifacts"

    def get_artifacts_folder_path(self):
        return self.artifacts_folder_path

    def get_binary_folder_path(self, binary):
        return f"{self.get_artifacts_folder_path()}/{binary.name}_{binary.extension}"

    def get_binary_perf_folder_path(self, binary):
        return f"{self.get_artifacts_folder_path()}/{binary.name}_{binary.extension}/perf"

    def create_artifacts(self):
        self.file_manager.create_directory(self.get_artifacts_folder_path())

    def clean_artifacts(self):
        self.file_manager.remove_directory(self.get_artifacts_folder_path())

    def clean_binary_artifacts(self, binary):
        self.file_manager.remove_directory(self.get_binary_folder_path(binary))

    def save_binary(self, binary, query):
        binary_folder = self.get_binary_folder_path(binary)
        self.file_manager.create_directory(binary_folder)
        self.file_manager.create_directory(f"{binary_folder}/run")
        self.file_manager.create_directory(f"{binary_folder}/perf")

        self.file_manager.copy_file(f"{binary_folder}", binary.file_path)

        for program in binary.programs():
            program_folder = f"{binary_folder}/run/program_{program.index}"
            self.file_manager.create_directory(program_folder)

            for i in range(len(program.get_inputs())):
                self.save_torch_tensor(program_folder, program.get_inputs()[i], f"program_{program.index}_input_{i}.pt")

            for i in range(len(program.get_outputs())):
                self.save_torch_tensor(program_folder, program.get_outputs()[i], f"program_{program.index}_output_{i}.pt")

        if query != None:
            self.save_system_desc(query.system_desc, f"{binary_folder}")

    def save_torch_tensor(self, folder_path, torch_tensor, torch_tensor_name):
        import torch

        try:
            torch.save(torch_tensor, f"{folder_path}/{torch_tensor_name}")
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

    def save_system_desc(self, system_desc, system_desc_folder="", system_desc_name="system_desc.ttsys"):
        system_desc_folder = f"{self.get_artifacts_folder_path()}" if system_desc_folder == "" else system_desc_folder
        try:
            system_desc.store(f"{system_desc_folder}/{system_desc_name}")
        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

class Binary:
    def __init__(self, file_path):
        import ttrt.binary

        self.file_path = file_path
        self.name = FileManager.get_file_name(file_path)
        self.extension = FileManager.get_file_extension(file_path)
        self.fbb = ttrt.binary.load_from_path(file_path)
        self.fbb_dict = ttrt.binary.as_dict(self.fbb)
        self.programs = []

        for i in range(len(self.fbb_dict["programs"])):
            program = Program(i, self.fbb_dict["programs"][i])
            self.programs.append(program)

    def check_version(self):
        package_name = "ttrt"

        try:
            package_version = get_distribution(package_name).version
        except Exception as e:
            raise Exception(f"error retrieving version: {e} for {package_name}")

        if package_version != self.fbb.version:
            logging.info(f"{package_name}: v{package_version} does not match flatbuffer: v{self.fbb.version} for flatbuffer: {self.file_path} - skipping this test")
            return False
        
        return True

    def check_system_desc(self, query):
        import ttrt.binary

        try:
            if self.fbb_dict["system_desc"] != query.get_system_desc_as_dict()["system_desc"]:
                logging.info(f"system desc for device did not match flatbuffer: {self.file_path} - skipping this test")
                return False

        except Exception as e:
            raise Exception(f"an unexpected error occurred: {e}")

        return True

    def get_num_programs(self):
        return len(self.programs)

    def check_program_index_exists(self, program_index):
        if program_index > self.get_num_programs():
            return False
        
        return True

    def get_program(self, program_index):
        if program_index > self.get_num_programs():
            raise Exception(f"program index={program_index} is greater than number of programs availabe={self.get_num_programs()}!")

        return self.programs[program_index]

    def save(self, artifacts, query=None):
        artifacts.save_binary(self, query)

    class Program:
        def __init__(self, index, program):
            self.index = index
            self.program = program
            self.input_tensors = []
            self.output_tensors = []

        def get_inputs(self):
            return self.input_tensors

        def get_outputs(self):
            return self.output_tensors

        def populate_inputs(self, init_fn):
            input_dict = self.program["inputs"][self.index]
            torch_tensor = init_fn(input_dict["desc"]["shape"], dtype=fromDataType(input_dict["desc"]["layout"]["memory_desc"]["data_type"]))
            self.input_tensors.append(torch_tensor)
          
        def populate_outputs(self, init_fn):
            output_dict = self.program["outputs"][self.index]
            torch_tensor = init_fn(output_dict["desc"]["shape"], dtype=fromDataType(output_dict["desc"]["layout"]["memory_desc"]["data_type"]))
            self.output_tensors.append(torch_tensor)

class BinaryTTNN(Binary):
    file_extension = ".ttnn"

    def __int__(self, file_path):
        super().__init__(file_path)

    @staticmethod
    def get_file_extension():
        return BinaryTTNN.file_extension

    @staticmethod
    def prepare_binary(file_path):
        return BinaryTTNN(file_path)

    @staticmethod
    def find_ttnn_binary_paths(path):
        files = []

        if FileManager.is_file(path):
            if FileManager.check_file_exists(path):
                files.append(path)
            else:
                logging.info(f"file '{path}' not found - skipping")
        else:
            FileManager.check_directory_exists(path)
            try: 
                for root, _, files in os.walk(path):
                    for file in files:
                        if FileManager.get_file_extension(file) == BinaryTTNN.get_file_extension():
                            files.append(os.path.join(root, file))
            except Exception as e:
                raise Exception(f"an unexpected error occurred: {e}")

        return files

class BinaryTTMetal(Binary):
    file_extension = ".ttb"

    def __int__(self, file_path):
        super().__init__(file_path)

    @staticmethod
    def get_file_extension():
        return BinaryTTMetal.file_extension

    @staticmethod
    def prepare_binary(file_path):
        return BinaryTTMetal(file_path)

    @staticmethod
    def find_ttmetal_binary_paths(directory_path):
        files = []

        if FileManager.is_file(path):
            if FileManager.check_file_exists(path):
                files.append(path)
            else:
                logging.info(f"file '{path}' not found - skipping")
        else:
            FileManager.check_directory_exists(path)
            try: 
                for root, _, files in os.walk(path):
                    for file in files:
                        if FileManager.get_file_extension(file) == BinaryTTMetal.get_file_extension():
                            files.append(os.path.join(root, file))
            except Exception as e:
                raise Exception(f"an unexpected error occurred: {e}")

        return files
