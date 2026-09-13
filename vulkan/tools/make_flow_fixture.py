#!/usr/bin/env python3
"""Create a full SPyNet reference from synthetic motion or two video frames."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from export_frame_core import build_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=95)
    parser.add_argument("--height", type=int, default=79)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--start-frame", type=int, default=120)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.width, args.height, args.threads) < 1 or args.start_frame < 0:
        parser.error("dimensions/threads must be positive and start frame nonnegative")
    torch.set_num_threads(args.threads)
    torch.manual_seed(123)

    if args.video:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-threads", "1", "-i", str(args.video),
             "-vf", f"select=gte(n\\,{args.start_frame}),scale={args.width}:{args.height}",
             "-frames:v", "2", "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            check=True, stdout=subprocess.PIPE,
        )
        frames = np.frombuffer(result.stdout, np.uint8).copy().reshape(2, args.height, args.width, 3)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        support, reference = frames[0:1], frames[1:2]
    else:
        support = torch.rand(1, 3, args.height, args.width)
        # A translated, slightly noisy pair exercises nonzero motion and borders.
        reference = (torch.roll(support, shifts=(1, -2), dims=(2, 3))
                     * 0.98 + torch.rand_like(support) * 0.02)

    model = build_model(args.checkpoint).spynet.eval()
    with torch.inference_mode():
        flow = model(reference, support)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, tensor in (("ref", reference), ("supp", support), ("flow", flow)):
        tensor.contiguous().numpy().astype("<f4").tofile(args.output / f"{name}.f32")
    (args.output / "shape.txt").write_text(f"{args.width} {args.height}\n", encoding="utf-8")
    (args.output / "fixture.json").write_text(json.dumps({
        "width": args.width, "height": args.height, "dtype": "float32-le",
        "layout": "CHW", "flow_direction": "current (ref) to previous (supp)",
        "source": str(args.video) if args.video else "seeded translated random image",
        "start_frame": args.start_frame if args.video else None,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.width}x{args.height} flow fixture to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
