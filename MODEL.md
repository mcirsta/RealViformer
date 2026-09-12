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

The upstream project contains an MIT license but does not state a separate
license specifically for the checkpoint. This fork does not assert additional
rights over the pretrained parameters.
