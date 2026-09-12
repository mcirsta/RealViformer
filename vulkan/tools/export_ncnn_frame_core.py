#!/usr/bin/env python3
"""Export the recurrent RealViformer frame core to a validated ncnn graph."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


MODULE_OPERATOR = "archs.realviformer_arch.AttentionMaskDescriptor"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--pnnx", type=Path, required=True)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--alternate-height", type=int, default=64)
    parser.add_argument("--alternate-width", type=int, default=64)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[2]
    exporter = Path(__file__).with_name("export_frame_core.py")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    parameter_path = args.output_directory / "frame_core.param"
    weight_path = args.output_directory / "frame_core.bin"

    with tempfile.TemporaryDirectory(prefix="realviformer-ncnn-") as temporary:
        temporary_directory = Path(temporary)
        torchscript_path = temporary_directory / "frame_core.pt"
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
                f"inputshape=[1,3,{args.height},{args.width}],[1,48,{args.height},{args.width}]",
                "inputshape2="
                f"[1,3,{args.alternate_height},{args.alternate_width}],"
                f"[1,48,{args.alternate_height},{args.alternate_width}]",
                "fp16=0",
                "device=cpu",
                f"moduleop={MODULE_OPERATOR}",
                f"pnnxparam={temporary_directory / 'frame_core.pnnx.param'}",
                f"pnnxbin={temporary_directory / 'frame_core.pnnx.bin'}",
                f"pnnxpy={temporary_directory / 'frame_core_pnnx.py'}",
                f"pnnxonnx={temporary_directory / 'frame_core.pnnx.onnx'}",
                f"ncnnparam={parameter_path}",
                f"ncnnbin={weight_path}",
                f"ncnnpy={temporary_directory / 'frame_core_ncnn.py'}",
            ],
            cwd=repository,
            check=True,
        )

    layer_types = [
        line.split(maxsplit=1)[0]
        for line in parameter_path.read_text(encoding="utf-8").splitlines()[2:]
        if line
    ]
    forbidden = sorted({"MatMul", "Pooling1D", "GridSample"}.intersection(layer_types))
    if forbidden:
        raise RuntimeError(f"unsupported operators remain: {', '.join(forbidden)}")
    if layer_types.count(MODULE_OPERATOR) != 15 or layer_types.count("SDPA") != 17:
        raise RuntimeError("unexpected RealViformer attention graph structure")

    print(f"wrote {parameter_path}")
    print(f"wrote {weight_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
