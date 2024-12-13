#!/usr/bin/env python3

import os
from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple, List

import subprocess
subprocess.check_call(["cargo", "build", "--release"])
import pyo


def read_file(path: os.PathLike) -> Tuple[List[int], List[int]]:
    with open(path, 'r') as content:
        lists = list(zip(*[
            [int(loc) for loc in line.strip().split('   ', 1)]
            for line in content.readlines()
        ]))
    list1, list2 = [pyo.rust_sort(l) for l in lists]
    return list1, list2

def part_one(path: os.PathLike) -> int:
    return pyo.distance_sum(*read_file(path))

def part_two(path: os.PathLike) -> int:
    return pyo.part2(*read_file(path))

def both(path: os.PathLike) -> Tuple[int, int]:
    prepared = read_file(path)
    return pyo.distance_sum(*prepared), pyo.part2(*prepared)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("input_file", type=Path)
    args = p.parse_args()
    print(both(args.input_file))
