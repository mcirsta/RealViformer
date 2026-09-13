# Portable Vulkan backend

This directory contains the cross-vendor Vulkan inference backend. The baseline
uses Vulkan 1.1 compute and SPIR-V without AMD- or NVIDIA-specific APIs.
Optional paths may use features such as FP16 arithmetic only after runtime
capability detection.

Implemented and validated:

- compute-device enumeration and selection;
- `VK_EXT_memory_budget` live VRAM accounting;
- a dynamic allocation ceiling with a configurable safety reserve;
- FP16 storage/arithmetic capability reporting;
- a device-local transfer, compute, synchronization, and readback smoke test.
- a deterministic native weight container and verified VRAM upload path.
- a numerically validated ncnn frame-core graph;
- native Vulkan scaled-dot-product attention and mask-descriptor execution.
- portable Vulkan bilinear flow warping with zero and border padding;
- a dynamic ncnn graph containing all six learned SPyNet refinement stages.
- the complete SPyNet pyramid, normalization, flow resizing, and refinement loop;
- streaming recurrent restoration with GPU-resident previous-frame features;
- the erf-form GELU required to match the original PyTorch model;
- sequence reset and multi-frame image/state parity tests.
- bounded-dispatch sum/mean reductions, including >65,535 output elements;
- core and synchronization validation with explicit failure on API errors.

Complete recurrent restoration runs on Vulkan for bounded test inputs. The
SPyNet flow path has also been validated at 640x480. Full-resolution restoration
on small GPUs still needs tiled x4 reconstruction and enforcement of the live
memory ceiling. Video I/O and RAM spill remain under development. The current
sequence test and native Vulkan restoration API deliberately limit inputs to
128x128 until those memory controls are connected; this size guard is not a
VRAM guarantee. The device probe's ceiling is not yet enforced by ncnn inference.
The PyTorch path remains the correctness reference.

## Build

```sh
cmake -S vulkan -B build/vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/vulkan -j 2
ctest --test-dir build/vulkan --output-on-failure
```

The default build uses the exact ncnn revision pinned in the Git submodule.
It compiles narrowly patched build-directory copies of three ncnn source files
to fix subgroup-stage validation, constant-buffer copy usage, and pooled-buffer
synchronization. The submodule stays clean; configure fails if the patch anchors
change. `RVF_USE_SYSTEM_NCNN` bypasses these compatibility fixes and requires
independent validation of the installed ncnn version.
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

Validate the flow-warp primitive on CPU and Vulkan:

```sh
./build/vulkan/rvf-flow-warp-test
./build/vulkan/rvf-flow-warp-test --vulkan
```

Independent PyTorch warp references cover borders, extreme finite motion,
one-pixel axes, odd channel padding, and 640x480 coordinates:

```sh
python vulkan/tools/make_warp_fixtures.py build/fixtures/warp
./build/vulkan/rvf-flow-warp-test build/fixtures/warp
./build/vulkan/rvf-flow-warp-test --vulkan build/fixtures/warp
./build/vulkan/rvf-reduction-test
./build/vulkan/rvf-reduction-test --vulkan
```

Run native GPU tests through the validation wrapper (requires the installed
Khronos validation layer). It enables core and synchronization checks and fails
on validation errors even if the native numerical test exits zero:

```sh
python vulkan/tools/validate_vulkan.py --log build/validation.log -- \
    ./build/vulkan/rvf-sequence-test pretrained_model/ncnn \
    build/fixtures/sequence --vulkan
```

`VK_LAYER_SYNCVAL_SHADER_ACCESSES_HEURISTIC=1` additionally enables static shader
access checks on validation-layer versions that support it; those checks can
produce false positives. `RVF_WARP_TEST_DUMP=DIRECTORY` saves warp-test outputs
for numerical diagnosis.

Validate the erf and tanh GELU paths against a double-precision reference:

```sh
./build/vulkan/rvf-gelu-test
./build/vulkan/rvf-gelu-test --vulkan
```

The pinned ncnn revision's Vulkan GELU always uses a tanh approximation, even
when the exported graph requests the erf form. `rvf::GeluLayer` overrides it
without modifying ncnn. The erf approximation has roughly 1.5e-7 absolute error;
the operator test sweeps negative and positive tails and uses a 2e-6 GELU
tolerance. This matters for recurrence: the old approximation produced growing
image/state differences over successive frames.

Validate the six learned SPyNet stages:

```sh
python vulkan/tools/export_spynet_modules.py \
    pretrained_model/weights.pth /tmp/realviformer-spynet.pt \
    --fixture-dir /tmp/realviformer-spynet-fixture

./build/vulkan/rvf-spynet-test \
    pretrained_model/ncnn/spynet.param \
    pretrained_model/ncnn/spynet.bin \
    /tmp/realviformer-spynet-fixture

./build/vulkan/rvf-spynet-test \
    pretrained_model/ncnn/spynet.param \
    pretrained_model/ncnn/spynet.bin \
    /tmp/realviformer-spynet-fixture --vulkan
```

Validate complete optical flow (odd dimensions exercise border replication):

```sh
python vulkan/tools/make_flow_fixture.py pretrained_model/weights.pth \
    build/fixtures/flow-95x79 --width 95 --height 79
./build/vulkan/rvf-spynet-flow-test pretrained_model/ncnn/spynet.param \
    pretrained_model/ncnn/spynet.bin build/fixtures/flow-95x79
./build/vulkan/rvf-spynet-flow-test pretrained_model/ncnn/spynet.param \
    pretrained_model/ncnn/spynet.bin build/fixtures/flow-95x79 --vulkan
```

Also test 32x32 (the upstream five-level special case), and a 640x480 video pair:

```sh
python vulkan/tools/make_flow_fixture.py pretrained_model/weights.pth \
    build/fixtures/flow-video --width 640 --height 480 \
    --video /path/to/source.avi --start-frame 120
./build/vulkan/rvf-spynet-flow-test pretrained_model/ncnn/spynet.param \
    pretrained_model/ncnn/spynet.bin build/fixtures/flow-video --vulkan
```

Validate consecutive restored frames and recurrent states against the complete
PyTorch model, including a reset to a new sequence:

```sh
python vulkan/tools/make_sequence_fixture.py pretrained_model/weights.pth \
    build/fixtures/sequence --width 128 --height 96 --frames 4 \
    --video /path/to/source.avi --start-frame 120
./build/vulkan/rvf-sequence-test pretrained_model/ncnn build/fixtures/sequence
./build/vulkan/rvf-sequence-test pretrained_model/ncnn build/fixtures/sequence \
    --vulkan --device 0
```

Omit `--video` for deterministic translated synthetic frames. Fixture generation
uses PyTorch on two CPU threads by default; the native tests do not need Python
once fixtures exist. `--save DIRECTORY` on the sequence test writes planar FP32
x4 outputs for inspection. Both tests support `--device INDEX` for another GPU.
Image/state tolerances are 1e-4 / 2e-4. All comparisons reject nonfinite values.
These are strict diagnostic thresholds, not a claim of bitwise equivalence:
the 16-frame 108x76 video test currently exceeds the state threshold near its
end despite image differences below 5e-6. This remains an open regression;
the tolerance has not been loosened. See [the audit](AUDIT.md) for details.

On the RX 550 (RADV POLARIS12), a four-frame 128x96 video test produced maximum
image error of 2.8e-6 and state error of 5.2e-5. Frames took about 0.50--0.73 s
including transfers and test-only state readback. The 640x480 flow-only test took
about 1.2--1.3 s with maximum error 3.5e-5 pixels. These are correctness measurements
while a separate CPU render was active, not full-resolution movie benchmarks.

### Native API

`rvf::SpyNet` records the whole optical-flow pipeline into an ncnn `VkCompute`
command buffer, with no CPU readback between pyramid levels. `rvf::RecurrentRestorer`
loads the bundled graphs once and accepts planar FP32 RGB frames through
`process()`, retaining the previous RGB frame and 48-channel state on the GPU.
It downloads only the x4 RGB output unless state readback is explicitly requested.
`reset()` starts a new sequence; dimensions cannot change without a reset. Inputs
must be divisible by four. The upstream short-wide SPyNet limitation is reported
as an error (width > 32 with height <= 32).

The caller owns the ncnn GPU instance and must keep it alive until all restorers
are destroyed. Each restorer owns its allocators and must be used serially.
There is currently no automatic tile scheduler or RAM fallback in this API;
the Vulkan API rejects dimensions above 128 until memory-budget enforcement is
complete. Invalid/nonfinite inputs are rejected and failed readbacks do not
advance the stored frame/state.

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

python vulkan/tools/export_ncnn_spynet.py \
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
