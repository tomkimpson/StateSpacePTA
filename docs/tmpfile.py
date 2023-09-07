import pathlib
import sys

print(pathlib.Path(__file__).parents[2].resolve().as_posix())
