#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import os
from pathlib import Path
from argparse import ArgumentParser
import argcomplete
from typing import Tuple
import logging

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
    p.add_argument("--log", choices=logging.getLevelNamesMapping().keys(), default="WARN")
    p.add_argument("input_file", type=Path)
    argcomplete.autocomplete(p)
    args = p.parse_args()
    logging.getLogger().setLevel(args.log)
    # some log needed here. it seems
    logging.debug("running with debug. it seems")
    print(read_file(args.input_file))
