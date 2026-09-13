#!/usr/bin/env python3
"""Trace the six learned SPyNet refinement modules for PNNX conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from export_frame_core import build_model


class SpyNetModules(nn.Module):
    """The learned portion of SPyNet, separated from pyramid orchestration."""

    def __init__(self, modules: nn.ModuleList):
        super().__init__()
        self.modules_by_level = modules

    def forward(
        self,
        level0: torch.Tensor,
        level1: torch.Tensor,
        level2: torch.Tensor,
        level3: torch.Tensor,
        level4: torch.Tensor,
        level5: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        inputs = (level0, level1, level2, level3, level4, level5)
        return tuple(
            module(tensor)
            for module, tensor in zip(self.modules_by_level, inputs)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--height", type=int, default=13)
    parser.add_argument("--width", type=int, default=19)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        help="also write deterministic FP32 inputs and reference outputs",
    )
    args = parser.parse_args()
    if args.height < 2 or args.width < 2:
        raise ValueError("height and width must be at least two")

    torch.manual_seed(1)
    spynet = build_model(args.checkpoint).spynet
    modules = SpyNetModules(spynet.basic_module).eval()
    inputs = tuple(
        torch.rand(1, 8, args.height, args.width) * 2 - 1 for _ in range(6)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        traced = torch.jit.trace(modules, inputs, check_trace=False, strict=False)
        expected = modules(*inputs)
        actual = traced(*inputs)
    maximum_error = max(
        (left - right).abs().max().item()
        for left, right in zip(expected, actual)
    )
    if maximum_error > 1e-6:
        raise RuntimeError(f"trace validation failed: max error {maximum_error}")
    traced.save(str(args.output))

    if args.fixture_dir:
        args.fixture_dir.mkdir(parents=True, exist_ok=True)
        metadata: dict[str, dict[str, object]] = {}
        for prefix, tensors in (("in", inputs), ("out", expected)):
            for index, tensor in enumerate(tensors):
                name = f"{prefix}{index}"
                array = tensor.detach().contiguous().cpu().numpy()
                array.tofile(args.fixture_dir / f"{name}.f32")
                metadata[name] = {
                    "shape": list(array.shape),
                    "dtype": "float32",
                    "file": f"{name}.f32",
                }
        (args.fixture_dir / "fixture.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    print(
        f"wrote {args.output} for six {args.height}x{args.width} inputs; "
        f"trace max error {maximum_error:.3g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
