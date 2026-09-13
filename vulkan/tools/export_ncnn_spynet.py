#!/usr/bin/env python3
"""Export the six learned SPyNet refinement stages to one ncnn graph."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def shapes(height: int, width: int) -> str:
    return ",".join(f"[1,8,{height},{width}]" for _ in range(6))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--pnnx", type=Path, required=True)
    parser.add_argument("--height", type=int, default=13)
    parser.add_argument("--width", type=int, default=19)
    parser.add_argument("--alternate-height", type=int, default=24)
    parser.add_argument("--alternate-width", type=int, default=32)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[2]
    exporter = Path(__file__).with_name("export_spynet_modules.py")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    parameter_path = args.output_directory / "spynet.param"
    weight_path = args.output_directory / "spynet.bin"

    with tempfile.TemporaryDirectory(prefix="realviformer-spynet-ncnn-") as temporary:
        temporary_directory = Path(temporary)
        torchscript_path = temporary_directory / "spynet.pt"
        subprocess.run(
            [
                sys.executable,
                str(exporter),
                str(args.checkpoint),
                str(torchscript_path),
                "--height",
                str(args.height),
                "--width",
                str(args.width),
            ],
            cwd=repository,
            check=True,
        )
        subprocess.run(
            [
                str(args.pnnx),
                str(torchscript_path),
                f"inputshape={shapes(args.height, args.width)}",
                "inputshape2="
                f"{shapes(args.alternate_height, args.alternate_width)}",
                "fp16=0",
                "device=cpu",
                f"pnnxparam={temporary_directory / 'spynet.pnnx.param'}",
                f"pnnxbin={temporary_directory / 'spynet.pnnx.bin'}",
                f"pnnxpy={temporary_directory / 'spynet_pnnx.py'}",
                f"pnnxonnx={temporary_directory / 'spynet.pnnx.onnx'}",
                f"ncnnparam={parameter_path}",
                f"ncnnbin={weight_path}",
                f"ncnnpy={temporary_directory / 'spynet_ncnn.py'}",
            ],
            cwd=repository,
            check=True,
        )

    layer_types = [
        line.split(maxsplit=1)[0]
        for line in parameter_path.read_text(encoding="utf-8").splitlines()[2:]
        if line
    ]
    unsupported = sorted({"GridSample", "MatMul"}.intersection(layer_types))
    if unsupported:
        raise RuntimeError(f"unsupported operators remain: {', '.join(unsupported)}")
    if layer_types.count("Convolution") != 30:
        raise RuntimeError("unexpected SPyNet convolution graph structure")

    print(f"wrote {parameter_path}")
    print(f"wrote {weight_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
