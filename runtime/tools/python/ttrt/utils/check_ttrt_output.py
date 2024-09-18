from dataclasses import dataclass

import numpy as np
from debug_tensors import device_input_tensors, device_output_tensors

np.set_printoptions(suppress=True, precision=3, threshold=10000, linewidth=500)


@dataclass(frozen=True)
class Tile:
    data: np.array
    index: tuple

    @staticmethod
    def shape():
        return (32, 32)

    def __str__(self):
        return f"Tile {self.index}:\n{self.data}\n"


class Tensor:
    def __init__(self, array):
        self.data = array
        self.tilized = self.__tilize()

        self.__iter_tile_index = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def num_tiles(self):
        return len(self.tilized)

    def get_next_tile(self):
        if self.__iter_tile_index < self.num_tiles:
            tile = self.tilized[self.__iter_tile_index]
            self.__iter_tile_index += 1
            return tile
        else:
            raise StopIteration("No more tiles available")

    def __tilize(self) -> list:
        tiles = []

        r, c = self.shape[0], self.shape[1]
        tile_height, tile_width = Tile.shape()

        # Walk a window [tile_height, tile_width] over tensor/np.array and collect
        # tiles.
        for i in range(0, r, tile_height):
            for j in range(0, c, tile_width):
                t = self.data[i : i + tile_height, j : j + tile_width]
                tile = Tile(t, (i // tile_height, j // tile_width))
                tiles.append(tile)

        return tiles


class Comparator:
    def __init__(
        self, golden_function, comparison_function, comparison_function_additional_args=None
    ):
        self.golden_function = golden_function
        self.comparison_function = comparison_function
        self.comparison_function_additional_args = comparison_function_additional_args

    def compare(self):
        print("Running comparison...\n")
        return self.__compare_tile_by_tile()

    def __compare_tile_by_tile(self):
        assert len(device_output_tensors) == 1, "Expected only one output tensor!"

        inputs = [Tensor(input_tensor) for input_tensor in device_input_tensors]
        output = Tensor(device_output_tensors[0])
        # Execute golden on underlying np.arrays and wrap returned np.array into a Tensor.
        golden = Tensor(self.golden_function(*device_input_tensors))

        for input in inputs:
            assert (
                input.num_tiles == output.num_tiles == golden.num_tiles
            ), "Expected same number of tiles in input, output and golden tensors for eltwise functions!"

        comparison_passed_for_all_tiles = True
        mismatched_tiles = []

        for _ in range(output.num_tiles):
            output_tile = output.get_next_tile()
            golden_tile = golden.get_next_tile()

            if comparison_function_additional_args:
                comparison = self.comparison_function(
                    output_tile.data, golden_tile.data, **self.comparison_function_additional_args
                )
            else:
                comparison = self.comparison_function(output_tile.data, golden_tile.data)

            if comparison == False:
                comparison_passed_for_all_tiles = False
                mismatched_tiles.append(output_tile.index)

            print(f"Tile:\n {output_tile.index}")
            print(f"Device output:\n {output_tile}")
            print(f"Golden output:\n {golden_tile}")
            print(f"Comparison:\n {comparison}\n")

        return comparison_passed_for_all_tiles, mismatched_tiles


if __name__ == "__main__":
    # ---- Set these to your desire ----
    golden_function = np.add
    comparison_function = np.allclose
    comparison_function_additional_args = {"rtol": 1e-1, "atol": 1e-1}
    # ----------------------------------

    comp = Comparator(golden_function, comparison_function, comparison_function_additional_args)
    passed, mismatched_tiles = comp.compare()

    if passed:
        print(f"Device output matches golden for all input tiles!")
    else:
        print(f"Device output does not match golden on following tiles: {mismatched_tiles}")
