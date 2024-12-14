#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import os
from pathlib import Path
from argparse import ArgumentParser
import argcomplete
from typing import List
import logging

import subprocess
subprocess.check_call(["cargo", "build", "--release"])
import pyo


def read_file(path: os.PathLike) -> List[List[int]]:
    with open(path, 'r') as content:
        reports = [[int(level) for level in report.strip().split(' ')]
                   for report in content.readlines()]
    return reports


def part_one(data: List[List[int]]) -> int:
    return pyo.part_one(data)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "--log", choices=logging.getLevelNamesMapping().keys(), default="WARN")
    p.add_argument("input_file", type=Path)
    argcomplete.autocomplete(p)
    args = p.parse_args()
    logging.getLogger().setLevel(args.log)
    # some log needed here. it seems
    logging.debug("running with debug. it seems")
    print(part_one(read_file(args.input_file)))
