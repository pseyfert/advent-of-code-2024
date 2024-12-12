#!/usr/bin/env python3

import os
from pathlib import Path
from argparse import ArgumentParser

import subprocess
subprocess.check_call(["cargo", "build", "--release"])
import pyo

def read_file(path: os.PathLike):
    counter = 0
    counter_2 = 0
    with open(path, 'r') as content:
        for line in content.readlines():
            test, operands = line.strip().split(': ', 1)
            operands_int = [int(o) for o in operands.split(' ')]
            test_int = int(test)
            if pyo.check_operators(test_int, operands_int) > 0:
                counter += test_int
            if pyo.check_again(test_int, operands_int) > 0:
                counter_2 += test_int
    print(counter)
    print(counter_2)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("input_file", type=Path)
    args = p.parse_args()
    read_file(args.input_file)
