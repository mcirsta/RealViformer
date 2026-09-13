#!/usr/bin/env python3
"""Write full-model outputs and recurrent states for a short video sequence."""

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
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--start-frame", type=int, default=120)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if (min(args.width, args.height) < 4 or args.width % 4 or args.height % 4
            or args.frames < 2 or args.threads < 1 or args.start_frame < 0):
        parser.error("dimensions must be positive multiples of four; frames >= 2, threads > 0, start >= 0")
    torch.set_num_threads(args.threads)
    torch.manual_seed(321)
    if args.video:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-threads", "1", "-i", str(args.video),
             "-vf", f"select=gte(n\\,{args.start_frame}),scale={args.width}:{args.height}",
             "-frames:v", str(args.frames), "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            check=True, stdout=subprocess.PIPE,
        )
        array = np.frombuffer(result.stdout, np.uint8).copy().reshape(args.frames, args.height, args.width, 3)
        frames = torch.from_numpy(array).permute(0, 3, 1, 2).float() / 255.0
    else:
        base = torch.rand(3, args.height, args.width)
        frames = torch.stack([torch.roll(base, (i, -i), (1, 2)) for i in range(args.frames)])

    model = build_model(args.checkpoint)
    states = []
    handle = model.compress.register_forward_hook(lambda _m, _i, out: states.append(out.detach().clone()))
    with torch.inference_mode():
        outputs = model(frames.unsqueeze(0))[0]
    handle.remove()
    if len(states) != args.frames:
        raise RuntimeError("unexpected number of recurrent feature states")
    args.output.mkdir(parents=True, exist_ok=True)
    for index in range(args.frames):
        for name, tensor in (("in", frames[index]), ("out", outputs[index]), ("state", states[index])):
            tensor.contiguous().numpy().astype("<f4").tofile(args.output / f"{name}{index}.f32")
    (args.output / "shape.txt").write_text(f"{args.width} {args.height} {args.frames}\n", encoding="utf-8")
    (args.output / "fixture.json").write_text(json.dumps({
        "width": args.width, "height": args.height, "frames": args.frames,
        "dtype": "float32-le", "layout": "CHW", "scale": 4, "state_channels": 48,
        "source": str(args.video) if args.video else "seeded translated random image",
        "start_frame": args.start_frame if args.video else None,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.frames} full-model frame/state references at {args.width}x{args.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
