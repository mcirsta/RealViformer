# Pretrained model

The official RealViformer pretrained checkpoint is stored at:

```text
pretrained_model/weights.pth
```

- Size: 46,766,389 bytes (44.60 MiB)
- SHA-256: `25d49a0128b1ecf218d8717b8a4041748b7fde25d8defb4ce82e53f120de9804`
- Architecture: RealViformer x4
- Provenance: checkpoint published by the upstream RealViformer authors

The native Vulkan branch also contains a deterministic inference copy:

```text
pretrained_model/weights.rvf
```

- Size: 23,301,184 bytes (22.22 MiB)
- SHA-256: `ab1252bb347936b55dfdd4e3cb679b7f4c52b7d907bfb1ce5c9450abe5bec671`
- Contents: the 295 FP32 tensors from the checkpoint's `params` state dictionary

Regenerate it with `vulkan/tools/export_weights.py`. The native container uses
sorted tensor names, explicit shapes and offsets, and 64-byte-aligned data so
the runtime can upload the payload without loading Python or PyTorch.

The first ncnn deployment graph is also bundled:

```text
pretrained_model/ncnn/frame_core.param
pretrained_model/ncnn/frame_core.bin
```

- Parameter SHA-256: `912ddc0d80451327eb59d15a6c48728194bd0fdd1669c6b90239b6c407cd031c`
- Weight SHA-256: `2d87a8992b7952b2b89db92ce9b2e816566ce69cf0f038bffffac9e244abcefd`
- Contents: one x4 recurrent frame pass after optical-flow alignment
- Precision: FP32 parameters and FP32 arithmetic

The graph uses ncnn's Vulkan SDPA implementation and portable custom Vulkan
operators for the learned attention-mask descriptor and erf-form GELU. The
GELU override is required because the pinned ncnn Vulkan operator uses the tanh
approximation. The checkpoint and graph weights are unchanged.

The six learned SPyNet refinement stages are bundled separately:

```text
pretrained_model/ncnn/spynet.param
pretrained_model/ncnn/spynet.bin
```

- Parameter SHA-256: `f7000033e8980b8f56e1473ccb71871d2521ce58910666a85337ece1f721fe53`
- Weight SHA-256: `9245abc02c8a5b1bd826c0556cbc5acd501fdf013eb51bcdd095af3c1a4eba42`
- Weight size: 5,761,320 bytes
- Contents: 30 FP32 convolutions extracted from the same checkpoint

`rvf::SpyNet` supplies the full pyramid and flow-warp orchestration around these
stages. `vulkan/tools/export_ncnn_spynet.py` reproduces the graph using the pinned
PNNX exporter. No artifact requires a Google Drive download.

The upstream project contains an MIT license but does not state a separate
license specifically for the checkpoint. This fork does not assert additional
rights over the pretrained parameters.
