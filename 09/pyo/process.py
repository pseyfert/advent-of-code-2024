#!/usr/bin/env python3

import os
from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple

import subprocess
subprocess.check_call(["cargo", "build", "--release"])
import pyo


def read_file(path: os.PathLike) -> Tuple[int, int]:
    with open(path, 'r') as content:
        for line in content.readlines():
            digits = [int(char) for char in line.strip()]
            break
    return (pyo.checksum(digits), pyo.defrag(digits))


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("input_file", type=Path)
    args = p.parse_args()
    print(read_file(args.input_file))
