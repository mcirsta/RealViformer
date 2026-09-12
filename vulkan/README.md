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

Model inference kernels are not implemented yet. The PyTorch path remains the
reference used to validate each Vulkan operation as it is added.

## Build

```sh
cmake -S vulkan -B build/vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/vulkan
```

## Run

```sh
./build/vulkan/rvf-vulkan --validation
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

