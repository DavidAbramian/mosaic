import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.spatial import distance

EXTENSIONS = [".jpg", ".jpeg", ".png"]


def load_source_tiles(source_tile_dir: Path) -> NDArray:
    print(f"Loading source tiles from {source_tile_dir.absolute()}")
    # tile_files = list(source_tile_dir.glob("*.png"))
    tile_files = [
        file for file in source_tile_dir.iterdir() if file.suffix in EXTENSIONS
    ]
    n_tiles = len(tile_files)

    source_tiles = np.zeros((n_tiles, TILE_ROWS, TILE_COLUMNS, 3))
    for i, file in enumerate(tile_files):
        image = imageio.imread(file)
        image = Image.fromarray(image).resize((TILE_COLUMNS, TILE_ROWS))
        image = np.array(image)

        if len(image.shape) == 2:
            source_tiles[i] = image[:, :, None] / 255.0
        elif len(image.shape) == 3:
            source_tiles[i] = image[:, :, :3] / 255.0
        else:
            raise ValueError

    return source_tiles


def load_target_image(target_image_path: Path) -> NDArray:
    print(f"Loading target image {target_image_path.absolute()}")
    image = imageio.imread(target_image_path)
    image = Image.fromarray(image).resize((TARGET_COLUMNS, TARGET_ROWS))
    image = np.array(image) / 255.0
    return image


def split_target_image_into_tiles(target_image: NDArray) -> NDArray:
    print("Splitting target image into tiles")
    n_tiles_x = TARGET_COLUMNS // TILE_COLUMNS
    n_tiles_y = TARGET_ROWS // TILE_ROWS
    n_target_tiles = n_tiles_x * n_tiles_y

    target_tiles = np.zeros((n_target_tiles, TILE_ROWS, TILE_COLUMNS, 3))
    for row in range(n_tiles_y):
        for column in range(n_tiles_x):
            target_tiles[n_tiles_x * row + column] = target_image[
                TILE_ROWS * row : TILE_ROWS * (row + 1),
                TILE_COLUMNS * column : TILE_COLUMNS * (column + 1),
            ]
    return target_tiles


def calculate_L2_distances(source_tiles: NDArray, target_tiles: NDArray) -> NDArray:
    print("Calculating distance between source and target tiles")
    n_source_tiles = source_tiles.shape[0]
    n_target_tiles = target_tiles.shape[0]
    L2_distances = distance.cdist(
        source_tiles.reshape((n_source_tiles, -1)),
        target_tiles.reshape((n_target_tiles, -1)),
        "euclidean",
    )
    return L2_distances


def build_canvas(source_tiles: NDArray, L2_distances: NDArray) -> NDArray:
    print("Building canvas from closest source tiles")
    closest_tiles = L2_distances.argmin(axis=0)

    n_tiles_x = TARGET_COLUMNS // TILE_COLUMNS
    n_tiles_y = TARGET_ROWS // TILE_ROWS

    canvas = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]) + (3,))
    for row in range(n_tiles_y):
        for column in range(n_tiles_x):
            canvas[
                TILE_ROWS * row : TILE_ROWS * (row + 1),
                TILE_COLUMNS * column : TILE_COLUMNS * (column + 1),
            ] = source_tiles[closest_tiles[n_tiles_x * row + column]]

    return canvas


def save_canvas(canvas: NDArray, out_file_name: str) -> None:
    print(f"Saving canvas as {out_file_name}")
    imageio.imwrite(out_file_name, (canvas * 256).astype(np.uint8))


def parse_size(size_string: str) -> tuple[int, int]:
    size_string = size_string.strip("()")
    split_size_string = size_string.split(",", 1)
    size_tuple = (int(split_size_string[0]), int(split_size_string[1]))
    return size_tuple


parser = argparse.ArgumentParser()
parser.add_argument("source_tile_dir", type=str)
parser.add_argument("target_image_path", type=str)
parser.add_argument("out_file_name", type=str)
parser.add_argument("--tile-size", type=str, default="(20, 50)")
parser.add_argument("--target-image-size", type=str, default="(1000, 1000)")

args = parser.parse_args()
source_tile_dir = Path(args.source_tile_dir)
target_image_path = Path(args.target_image_path)
out_file_name = args.out_file_name
TILE_SIZE = parse_size(args.tile_size)
TARGET_SIZE = parse_size(args.target_image_size)

# source_tile_dir = Path(".\covers")
# target_image_path = Path(r".\fragment.jpg")
# out_file_name = "abcde.jpg"
# TILE_SIZE = parse_size("(40,40)")
# TARGET_SIZE = parse_size("(1000,1000)")

TILE_COLUMNS, TILE_ROWS = TILE_SIZE
TARGET_COLUMNS, TARGET_ROWS = TARGET_SIZE

source_tiles = load_source_tiles(source_tile_dir)
target_image = load_target_image(target_image_path)
target_image_tiles = split_target_image_into_tiles(target_image)
L2_distances = calculate_L2_distances(source_tiles, target_image_tiles)
canvas = build_canvas(source_tiles, L2_distances)
save_canvas(canvas, out_file_name)
print("Done")
