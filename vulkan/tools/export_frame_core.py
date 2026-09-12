#!/usr/bin/env python3
"""Trace the recurrent RealViformer frame core for PNNX/ncnn conversion."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))

from archs.realviformer_arch import RealViformer  # noqa: E402


class FrameCore(nn.Module):
    """One recurrent frame after optical-flow alignment.

    Optical flow and flow_warp remain separate so that the converted core can
    stay entirely on Vulkan while a custom portable GridSample kernel is added.
    """

    def __init__(self, model: RealViformer):
        super().__init__()
        self.shallow_extraction = model.shallow_extraction
        self.attn_merge = model.attn_merge
        self.encoder_level1 = model.encoder_level1
        self.down1_2 = model.down1_2
        self.encoder_level2 = model.encoder_level2
        self.down2_3 = model.down2_3
        self.latent = model.latent
        self.up3_2 = model.up3_2
        self.reduce_chan_level2 = model.reduce_chan_level2
        self.decoder_level2 = model.decoder_level2
        self.up2_1 = model.up2_1
        self.decoder_level1 = model.decoder_level1
        self.refinement = model.refinement
        self.compress = model.compress
        self.upconv1 = model.upconv1
        self.upconv2 = model.upconv2
        self.conv_hr = model.conv_hr
        self.conv_last = model.conv_last
        self.pixel_shuffle = model.pixel_shuffle
        self.lrelu = model.lrelu

    def forward(
        self, current_frame: torch.Tensor, aligned_previous: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shallow = self.shallow_extraction(current_frame)
        propagated = self.attn_merge(shallow, aligned_previous)

        encoded_1 = self.encoder_level1(propagated)
        encoded_2 = self.encoder_level2(self.down1_2(encoded_1))
        latent = self.latent(self.down2_3(encoded_2))

        decoded_2 = self.up3_2(latent)
        decoded_2 = torch.cat([decoded_2, encoded_2], dim=1)
        decoded_2 = self.decoder_level2(self.reduce_chan_level2(decoded_2))

        decoded_1 = self.up2_1(decoded_2)
        decoded_1 = torch.cat([decoded_1, encoded_1], dim=1)
        decoded_1 = self.decoder_level1(decoded_1)

        refined = self.refinement(decoded_1)
        next_state = self.compress(refined)
        output = self.lrelu(self.pixel_shuffle(self.upconv1(refined)))
        output = self.lrelu(self.pixel_shuffle(self.upconv2(output)))
        output = self.lrelu(self.conv_hr(output))
        output = self.conv_last(output)
        base = F.interpolate(
            current_frame, scale_factor=4, mode="bilinear", align_corners=False
        )
        return output + base, next_state


def build_model(checkpoint_path: Path) -> RealViformer:
    model = RealViformer(
        num_feat=48,
        num_blocks=[2, 3, 4, 1],
        spynet_path=None,
        heads=[1, 2, 4],
        ffn_expansion_factor=2.66,
        merge_head=2,
        bias=False,
        LayerNorm_type="BiasFree",
        ch_compress=True,
        squeeze_factor=[4, 4, 4],
        masked=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    incompatible = model.load_state_dict(checkpoint["params"], strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f"missing checkpoint keys: {incompatible.missing_keys}")
    if incompatible.unexpected_keys != ["attn_merge.attn.masktemp"]:
        raise RuntimeError(
            f"unexpected checkpoint keys: {incompatible.unexpected_keys}"
        )
    return model.eval()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        help="also write deterministic FP32 inputs and reference outputs",
    )
    args = parser.parse_args()
    if args.height < 4 or args.width < 4 or args.height % 4 or args.width % 4:
        raise ValueError("height and width must be positive multiples of four")

    torch.manual_seed(0)
    core = FrameCore(build_model(args.checkpoint)).eval()
    current = torch.rand(1, 3, args.height, args.width)
    previous = torch.rand(1, 48, args.height, args.width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        traced = torch.jit.trace(
            core, (current, previous), check_trace=False, strict=False
        )
        eager_outputs = core(current, previous)
        traced_outputs = traced(current, previous)
    maximum_error = max(
        (expected - actual).abs().max().item()
        for expected, actual in zip(eager_outputs, traced_outputs)
    )
    if maximum_error > 1e-6:
        raise RuntimeError(f"trace validation failed: max error {maximum_error}")
    traced.save(str(args.output))
    if args.fixture_dir:
        args.fixture_dir.mkdir(parents=True, exist_ok=True)
        tensors = {
            "in0": current,
            "in1": previous,
            "out0": eager_outputs[0],
            "out1": eager_outputs[1],
        }
        metadata = {}
        for name, tensor in tensors.items():
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
        f"wrote {args.output} for {args.height}x{args.width}; "
        f"trace max error {maximum_error:.3g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
