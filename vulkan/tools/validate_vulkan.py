#!/usr/bin/env python3
"""Run a native test with core/synchronization validation; errors fail the run.

Numerical tests may exit zero even when the Vulkan layer reports an API error.
This wrapper makes those diagnostics part of the test result. Requires the
Khronos validation layer to be installed on the test machine.
"""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


def has_validation_error(output: str) -> bool:
    return bool(re.search(r"validation error|sync-hazard|failed to find layer|layer.*not found",
                          output, re.IGNORECASE))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("provide the native test command after --")
    env = os.environ.copy()
    layers = [layer for layer in env.get("VK_INSTANCE_LAYERS", "").split(os.pathsep) if layer]
    if "VK_LAYER_KHRONOS_validation" not in layers:
        layers.append("VK_LAYER_KHRONOS_validation")
    env["VK_INSTANCE_LAYERS"] = os.pathsep.join(layers)
    env["VK_LAYER_VALIDATE_SYNC"] = "1"
    result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace")
    sys.stdout.write(result.stdout)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(result.stdout, encoding="utf-8")
    if has_validation_error(result.stdout):
        print("FAIL: Vulkan validation reported an error", file=sys.stderr)
        return 1
    if result.returncode:
        return result.returncode if result.returncode > 0 else 1
    print("PASS: command succeeded without Vulkan validation errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
