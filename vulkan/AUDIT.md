# Vulkan accuracy and performance audit — 2026-09-13

Scope: the bundled FP32 graphs and native backend, pinned ncnn
`c0abf4830d8f8637efa5d34fce3985ba3d8642bd`, RX 550 / RADV POLARIS12.
No checkpoint weights or the separate PyTorch movie-rendering job were changed.

## Fixed

1. **Warp coordinates and edge cases.** The original model normalizes pixel
   coordinates to [-1,1] and unnormalizes them inside `grid_sample`. Cancelling
   that FP32 round trip changed sample positions and gave incorrect results
   for one-pixel axes. Ordinary GPU division introduced additional rounding
   differences. The kernel now preserves the round trip using a small-integer
   correctly rounded division path for normal FP32 coordinates; it needs no
   FP64 hardware. Border clamping and outside rejection happen before integer
   conversion, avoiding overflow for huge finite displacements. NaN coordinates
   propagate without converting NaN to an integer. Independent PyTorch fixtures
   cover 1x1, 1x13, 19x1, 19x13, 108x76 and 640x480, both padding modes, and
   padded channel layouts. CPU/GPU maximum error is 1.79e-7.
   GLSL permits approximate division, so `precise` alone is insufficient; see
   [the precision rules](https://docs.vulkan.org/glsl/latest/chapters/variables.html).

2. **Oversized reduction dispatch and wasted lanes.** ncnn's generic Vulkan
   reduction dispatches `out_total` workgroups along Y. Per-pixel normalization
   at 640x480 needs 307,200 outputs, beyond the portable 65,535 group bound.
   The model-specific sum/mean override spreads output rows over Y and Z,
   rejects unsupported modes, uses 64 lanes for short channel reductions, and
   chooses up to 256 lanes for long spatial sums according to device limits.
   CPU accumulation uses double precision. Tests cover 140,002 output rows and
   307,200-element reductions with less than 2e-6 relative GPU error. No large
   invalid dispatch was submitted to reproduce the old behavior.

3. **Unsupported subgroup-size request.** Extension/feature presence alone
   does not permit a required subgroup size for every shader stage. RX 550
   reports no supported stages for that request. The build compatibility fix
   additionally checks the compute-stage bit. This removes
   `VUID-VkPipelineShaderStageCreateInfo-pNext-02755`, as required by
   [the pipeline rules](https://docs.vulkan.org/spec/latest/chapters/pipelines.html#VUID-VkPipelineShaderStageCreateInfo-pNext-02755).

4. **Missing weight-buffer copy-source usage.** Constants returned by
   `MemoryData` can be cloned before an in-place operation. The weight allocator
   created these buffers without `TRANSFER_SRC`. Adding that usage removes
   `VUID-vkCmdCopyBuffer-srcBuffer-00118`. See
   [the copy-source requirement](https://docs.vulkan.org/spec/latest/chapters/copies.html#VUID-vkCmdCopyBuffer-srcBuffer-00118).

5. **Recycled-buffer synchronization.** Synchronization validation reported
   write-after-write hazards in buffer copies. Pool reuse loses the old access
   metadata, and the clone path did not barrier its destination. The build
   compatibility patch now treats unknown pooled-buffer state conservatively
   and barriers copy destinations. It handles both immediate push-descriptor
   recording and the delayed-recording path; the latter remains untested on
   hardware here. Core/synchronization validation passes the four-frame video
   test. A synthetic sequence also passes with shader-access heuristics enabled.

6. **False-positive tests and failure handling.** Warp and SPyNet-stage
   comparisons now reject nonfinite values. Warp tests preserve pack1 for
   four-channel fixtures. Attention pipelines are cleaned up after partial
   initialization failures. Input/output type checks and single-frame batch
   checks are stricter; invalid inputs and failed readbacks do not advance the
   recurrent state. `validate_vulkan.py` fails on API/synchronization errors
   even if a native test prints PASS and exits zero.

The ncnn changes are maintained in `cmake/ncnn_compat.cmake`. CMake compiles
patched build-directory copies, checks every replacement anchor, and leaves
the submodule unchanged. An installed system ncnn bypasses those patches and
must be validated separately.

## Measurements and remaining issues

- Four real frames at 128x96: GPU image max error 3.04e-6 and state max error
  6.14e-5 versus the full PyTorch model. Reset and input-buffer reuse tests pass.
- Full optical flow at 640x480: GPU max error 3.03e-5 pixels, about 1.09 s for
  the tested pair. This is optical flow only, not full-resolution restoration.
- Sixteen real frames at 108x76: average frames 1–15 decreased from 370.0 ms
  before this audit to 336.4 ms after the fixes (about 9.1% less time).
  These are indicative measurements with a separate CPU render running,
  not a controlled benchmark or an estimate for the entire movie.
- **Open numerical regression:** the same 16-frame test still fails its
  unchanged 2e-4 absolute state threshold at frame 15: max state error
  2.43425e-4, versus image max error 4.05312e-6. The native CPU path also exceeds
  the state threshold (~2.47e-4). The state has magnitude up to roughly 5, but
  that does not establish the cause or justify silently relaxing the test.
  Further operator/teacher-forced checks and longer sequences are needed.
- **Open memory blocker:** the probe's 2.65 GiB ceiling is not enforced by
  ncnn inference. The native Vulkan API now enforces the same 128x128 test-size
  guard as the CLI, but that is not a memory guarantee on every GPU. Full-size
  restoration remains disabled. Next: enforce allocation budgets, split out
  and tile only the x4 convolutional reconstruction head (preserving global
  attention), then add RAM spill and video I/O. Do not tile the attention body
  as though that were mathematically equivalent.

Reproduce the strict long-sequence diagnostic (currently expected to fail):

```sh
python vulkan/tools/make_sequence_fixture.py pretrained_model/weights.pth \
    build/fixtures/sequence-long --width 108 --height 76 --frames 16 \
    --video /path/to/source.avi --start-frame 120
./build/vulkan/rvf-sequence-test pretrained_model/ncnn \
    build/fixtures/sequence-long --vulkan
```

The measurements used a private source clip; its pixels and local path are not
committed. Synthetic fixture recipes and all deployment model weights remain
available in this GitHub repository.
