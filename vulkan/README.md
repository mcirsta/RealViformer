# Portable Vulkan backend

This directory contains the cross-vendor Vulkan inference backend. The baseline
uses Vulkan 1.1 compute and SPIR-V without AMD- or NVIDIA-specific APIs.
Optional paths may use features such as FP16 arithmetic only after runtime
capability detection.

The first milestone provides:

- compute-device enumeration and selection;
- `VK_EXT_memory_budget` live VRAM accounting;
- a dynamic allocation ceiling with a configurable safety reserve;
- FP16 storage/arithmetic capability reporting;
- a device-local transfer, compute, synchronization, and readback smoke test.
- a deterministic native weight container and verified VRAM upload path.
- a numerically validated ncnn frame-core graph;
- native Vulkan scaled-dot-product attention and mask-descriptor execution.

The frame core runs on Vulkan now. Optical flow, feature warping, tiled x4
reconstruction, video I/O, and the hard dynamic-memory enforcement path remain
under development. The PyTorch path is the correctness reference.

## Build

```sh
cmake -S vulkan -B build/vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/vulkan
```

The default build uses the exact ncnn revision pinned in the Git submodule.
Clone this repository with `--recurse-submodules`, or run:

```sh
git submodule update --init --recursive
```

## Run

```sh
./build/vulkan/rvf-vulkan --validation
```

Validate the bundled frame core against deterministic PyTorch fixtures:

```sh
python vulkan/tools/export_frame_core.py \
    pretrained_model/weights.pth /tmp/realviformer-frame-core.pt \
    --fixture-dir /tmp/realviformer-frame-core-fixture

./build/vulkan/rvf-frame-core-test \
    pretrained_model/ncnn/frame_core.param \
    pretrained_model/ncnn/frame_core.bin \
    /tmp/realviformer-frame-core-fixture

./build/vulkan/rvf-frame-core-test \
    pretrained_model/ncnn/frame_core.param \
    pretrained_model/ncnn/frame_core.bin \
    /tmp/realviformer-frame-core-fixture --vulkan
```

The ncnn graph is exported with PNNX while preserving
`archs.realviformer_arch.AttentionMaskDescriptor` as a module operator. This
removes generic matrix multiplication from the deployed graph and keeps the
entire frame core on Vulkan.

To reproduce the checked-in graph after building PNNX from the pinned ncnn
submodule:

```sh
python vulkan/tools/export_ncnn_frame_core.py \
    pretrained_model/weights.pth pretrained_model/ncnn \
    --pnnx /path/to/pnnx
```

Generate and validate the native model file:

```sh
python vulkan/tools/export_weights.py \
    pretrained_model/weights.pth pretrained_model/weights.rvf
./build/vulkan/rvf-vulkan --validation \
    --model pretrained_model/weights.rvf
```

By default, the ceiling is the driver's live device-local memory budget minus
a 192 MiB safety reserve. On the initial RX 550 system this targets about
2.6--2.7 GiB while KDE is active. It does not preallocate that amount.

Useful overrides:

```sh
./build/vulkan/rvf-vulkan --list
./build/vulkan/rvf-vulkan --device "RX 550"
./build/vulkan/rvf-vulkan --vram-limit-gib 2.65 --reserve-mib 192
```
