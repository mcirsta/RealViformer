#!/usr/bin/env python3
"""Independent PyTorch references for warp rounding, padding, and unit axes."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from archs.arch_util import flow_warp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    for width, height, channels in [(19, 13, 5), (1, 13, 4), (19, 1, 4),
                                     (1, 1, 1), (108, 76, 5), (640, 480, 5)]:
        feature = torch.rand(1, channels, height, width) * 2 - 1
        flow = torch.randn(1, height, width, 2) * 3
        # Huge finite displacements must not overflow an integer conversion.
        flow[0, 0, 0] = torch.tensor([1e30, -1e30])
        directory = args.output / f"{width}x{height}"
        directory.mkdir(parents=True, exist_ok=True)
        tensors = {"feature": feature, "flow": flow.permute(0, 3, 1, 2)}
        for padding in ("zeros", "border"):
            tensors[padding] = flow_warp(feature, flow, padding_mode=padding)
        for name, tensor in tensors.items():
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"nonfinite reference: {directory}/{name}")
            tensor.contiguous().numpy().astype("<f4").tofile(directory / f"{name}.f32")
        (directory / "shape.txt").write_text(f"{width} {height} {channels}\n")
    print(f"wrote six PyTorch warp cases to {args.output}")


if __name__ == "__main__":
    main()
