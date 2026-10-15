# Running NPU unit tests

NPU unit tests depend on vLLM and vLLM-Ascend. Run them in an Ascend development
environment that uses the versions pinned by this repository.

From the repository root, run:

```bash
python3 -m pytest -q tests/unit -m "not gpu and not vllm_runtime"
```

The suite covers the NPU worker and model-runner contracts, CAM connector
behavior, and module isolation. Tests that require unavailable NPU dependencies
are skipped in CPU-only CI, so a passing CPU run does not replace this check.

`tests/conftest.py` establishes the vLLM-Ascend import order required by the
test suite. No manual module preloading is needed.

When diagnosing a failure, run the failing pytest node by itself first. A test
that passes alone but fails in the full suite usually indicates leaked module
or monkeypatch state. Test fixtures and mocks must also follow the concrete
vLLM types and signatures used by the pinned runtime.

## Layered W4A8 validation

The layered routed-expert operators are `gmm_swiglu_quant_v2_layered` (W13) and
`grouped_matmul_layered` (W2). Both select their layer on device, and their
`all_*` lists hold one element per layer rather than one per expert.

Run the CPU contract and path-selection tests first:

```bash
pytest -q tests/unit/test_envs.py \
  tests/unit/model_executor/test_async_cam_w4a8.py
```

In an environment with matching CANN and torch-npu versions, rebuild the AFD
extension and run the device suites. All three need one device and issue no
collective:

```bash
SOC_VERSION=910c pytest -q tests/unit/compat/npu/test_gmm_layered.py
AFD_RUN_ASCEND_OP_RUNTIME=1 SOC_VERSION=910c LOCAL_RANK=0 \
  pytest -o addopts= -v -ra -s \
  tests/npu/test_async_cam_layered_w4a8.py \
  tests/npu/test_gmm_swiglu_quant_v2_layered_precision.py \
  tests/npu/test_gmm_layered_device_group_list.py
```

The three device suites answer different questions and none is redundant:

| Suite | Operator | Covers |
| --- | --- | --- |
| `test_async_cam_layered_w4a8.py` | W13 then W2 | The `AsyncCAMW4A8Executor` chain, 16 cases |
| `test_gmm_swiglu_quant_v2_layered_precision.py` | W13 | Both quantization scenarios against a reference, 11 cases |
| `test_gmm_layered_device_group_list.py` | W2 | A device-produced group list, both scenarios, 4 cases |

`test_async_cam_layered_w4a8.py` uses two layers with different nonzero INT4
weights, nonzero compensation, per-channel and per-group scales, interleaved
nonconsecutive layer IDs, and zero, single-row, uneven, and full-capacity expert
counts. It compares valid rows with an FP32 reference and perturbs the capacity
tail. It exercises only the `swiglu_limit=0` operator chain; nonzero limits are
temporarily ignored by the layered path and are not precision-equivalent to the
model. It does not check distributed CAM completion or compare checkpoint
results against the legacy path.

The two precision suites are described under "Layered operator precision" in
[`tests/npu/README.md`](../../tests/npu/README.md), including why A8W4 cannot use
the fused built-in as its reference and why each attribute combination needs its
own activation shape. Read that section before changing a tolerance or adding a
case: the tiling cache is keyed on shape and tiling key, and the two attributes
reach the tiling only as data.

For performance, use
[`tests/npu/gmm_layered_performance.py`](../../tests/npu/gmm_layered_performance.py)
rather than an end-to-end comparison when the question is about the operators
themselves. It measures both launch paths - eager and ACL graph - and reports
the fusion contribution and the graph contribution separately, which an
end-to-end run cannot separate. Anchors measured on Ascend 910C at
`M = 98304`, `C = 262144`: W13 reaches about 1.80x the concatenated baseline on
the steady-state per-layer figure, `chain` about 1.46x, and W2 sits at parity
once the graph removes its launch overhead. Re-measure rather than reuse these
numbers; they are a starting point for a regression check, not a claim about a
different shape, topology, or CANN version.

For end-to-end validation, toggle `AFD_ASYNC_CAM_LAYERED_GMM=0/1` with the same
W4A8 checkpoint and deployment configuration. Record the processed parameter
shapes, dtypes, and formats; software versions; actual-path startup logs;
accuracy; peak memory; and profiler traces. Cover completion on empty ranks,
multiple chunks of one layer, interleaved DP work, and async CAM ubatching.
Check that no layer-ID or count metadata D2H occurs between receive and send,
and measure the retained worker group synchronization separately.

The device Tensor `group_list` interface for the second GMM was merged in
PR #384. The W13 layered GMM op_api also accepts the logical 3D view with 5D NZ
storage produced by the W4A8 loader; the kernel math and PyTorch binding are
unchanged. `layer_index` is validated by element count rather than by storage
shape, because a rank-1 one-element int64 tensor can arrive normalized to rank 2
and the observed rank is not stable across runs.
