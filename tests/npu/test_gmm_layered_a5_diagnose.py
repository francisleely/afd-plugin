# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""TEMPORARY A5 diagnosis for the ``grouped_matmul_layered`` A8W8 mismatch.

Run::

    SOC_VERSION=950 AFD_RUN_ASCEND_OP_RUNTIME=1 LOCAL_RANK=1 \\
        python -m pytest -o addopts= -v -ra -s \\
        tests/npu/test_gmm_layered_a5_diagnose.py

No hand-written golden. A reference derived by reading the kernel would only
re-encode the same assumptions, so the checks that need a reference use the
built-in ``npu_grouped_matmul``, which is an independent implementation.

Delete this file once the mismatch is resolved.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.npu

E, M, K, N, LAYERS = 4, 128, 256, 256, 2

# Numeric contract for the A8W8 int8/int8/int32/bfloat16/float column.
X_RANGE = 8  # x, weight in [-8, 8)
SCALE_LO, SCALE_HI = 5e-3, 20e-3
PTS = 0.01
INT8_MAX = 255  # 2**8 - 1
BIAS_UNIT = 1024

# Largest value a correct result can reach. The bias joins the accumulator before
# dequantization - y = (x*w + bias) * scale * per_token_scale - so the peaks add
# first and the scales multiply after. 326400 < 2**31, so int32 cannot overflow.
OUT_BOUND = (K * INT8_MAX + BIAS_UNIT * INT8_MAX) * PTS * SCALE_HI

# bf16 has 8 significand bits, so its ULP is 2**-8 = 3.906e-03 and two values
# rounded independently can differ by one ULP. Override with AFD_GMM_TOLERANCE=.
TOLERANCE = float(os.environ.get("AFD_GMM_TOLERANCE", 2**-8))


def _op(torch, x, weights, biases, scales, layer, group_list, pts, gtype):
    return torch.ops.afd_ascend.grouped_matmul_layered(
        x=[x],
        all_weight=weights,
        all_bias=biases,
        all_scale=scales,
        layer_index=torch.full((1,), layer, device="npu", dtype=torch.int64),
        group_list=group_list,
        per_token_scale=pts,
        group_list_type=gtype,
        split_item=3,
        output_dtype=torch.bfloat16,
    )[0]


def _builtin(torch, torch_npu, x, weight, bias, scale, pts, counts):
    return torch_npu.npu_grouped_matmul(
        x=[x],
        weight=[weight],
        bias=[bias],
        scale=[scale],
        per_token_scale=[pts],
        group_list=counts,
        split_item=3,
        output_dtype=torch.bfloat16,
        group_type=0,
        group_list_type=1,
    )[0]


def _counts(torch, experts):
    return torch.full((experts,), M // experts, device="npu", dtype=torch.int64)


def _group_list(counts, gtype):
    return counts.cumsum(0) if gtype == 0 else counts


def _random_inputs(torch):
    generator = torch.Generator(device="cpu").manual_seed(381)
    x = torch.randint(
        -X_RANGE, X_RANGE, (M, K), generator=generator, dtype=torch.int8
    ).npu()
    weights = [
        torch.randint(
            -X_RANGE, X_RANGE, (E, K, N), generator=generator, dtype=torch.int8
        ).npu()
        for _ in range(LAYERS)
    ]
    scales = [
        (torch.rand((E, N), generator=generator) * (SCALE_HI - SCALE_LO) + SCALE_LO)
        .to(torch.bfloat16)
        .npu()
        for _ in range(LAYERS)
    ]
    biases = [
        torch.randint(
            -BIAS_UNIT * INT8_MAX,
            BIAS_UNIT * INT8_MAX,
            (E, N),
            generator=generator,
            dtype=torch.int64,
        )
        .to(torch.int32)
        .npu()
        for _ in range(LAYERS)
    ]
    pts = torch.full((M,), PTS, dtype=torch.float32, device="npu")
    return x, weights, scales, biases, pts


# Each rung adds one feature. Inputs are powers of two, so every expected value is
# exact arithmetic and the comparison is bitwise - this needs no reference at all,
# and the first failing rung names the stage.
LADDER = [
    ("E=1 L=1 scale=1   pts=1   bias=0", 1, 1, 1.0, 1.0, 0, 256.0),
    ("E=1 L=1 +bias=256               ", 1, 1, 1.0, 1.0, 256, 512.0),
    ("E=1 L=1 +per-channel scale=2    ", 1, 1, 2.0, 1.0, 256, 1024.0),
    ("E=1 L=1 +per-token scale=0.5    ", 1, 1, 2.0, 0.5, 256, 512.0),
    ("E=4     +expert addressing      ", 4, 1, 2.0, 0.5, 256, 512.0),
    ("E=4 L=2 +layer selection        ", 4, 2, 2.0, 0.5, 256, 512.0),
]


@pytest.mark.parametrize("gtype", [0, 1])
def test_a5_a8w8_diagnose(gtype: int) -> None:
    if os.environ.get("AFD_RUN_ASCEND_OP_RUNTIME") != "1":
        pytest.skip("requires opt-in NPU runtime")

    torch = pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    from afd_plugin.compat.npu.ops import ensure_afd_ascend_ops_loaded

    ensure_afd_ascend_ops_loaded()
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    print(f"\n===== group_list_type={gtype} M={M} K={K} N={N} E={E}")

    failures = []

    print("\n--- 1. exact ladder (bitwise; the first failure names the stage) ---")
    for label, experts, layers, scale, pts, bias, expected in LADDER:
        got = (
            _op(
                torch,
                torch.ones((M, K), dtype=torch.int8).npu(),
                [
                    torch.full((experts, K, N), 1, dtype=torch.int8).npu()
                    for _ in range(layers)
                ],
                [
                    torch.full((experts, N), bias, dtype=torch.int32).npu()
                    for _ in range(layers)
                ],
                [
                    torch.full((experts, N), scale, dtype=torch.bfloat16).npu()
                    for _ in range(layers)
                ],
                0,
                _group_list(_counts(torch, experts), gtype),
                torch.full((M,), pts, dtype=torch.float32, device="npu"),
                gtype,
            )
            .cpu()
            .float()
        )
        ok = bool(torch.equal(got, torch.full_like(got, expected)))
        print(
            f"  [{'OK  ' if ok else 'FAIL'}] {label}"
            f" expected={expected:<8.4g} observed={got.abs().max().item():<12.4g}"
        )
        if not ok:
            failures.append(
                f"{label.strip()}: expected {expected:g} got {got.abs().max().item():g}"
            )

    print(
        "\n--- 2. determinism (separates uninitialized memory from a wrong value) ---"
    )
    x, weights, scales, biases, pts = _random_inputs(torch)
    counts = _counts(torch, E)
    gl = _group_list(counts, gtype)
    first = _op(torch, x.clone(), weights, biases, scales, 0, gl, pts, gtype)
    second = _op(torch, x.clone(), weights, biases, scales, 0, gl, pts, gtype)
    if torch.equal(first, second):
        print("  [OK  ] identical calls agree bitwise")
    else:
        print("  [FAIL] identical calls disagree")
        failures.append("two identical calls disagree")

    print("\n--- 3. random data vs the built-in implementation ---")
    print(f"  correct-result bound {OUT_BOUND:.4g}; rtol=atol={TOLERANCE:g}")
    for layer in range(LAYERS):
        ours = (
            _op(torch, x.clone(), weights, biases, scales, layer, gl, pts, gtype)
            .cpu()
            .float()
        )
        builtin = (
            _builtin(
                torch,
                torch_npu,
                x.clone(),
                weights[layer],
                biases[layer],
                scales[layer],
                pts,
                counts,
            )
            .cpu()
            .float()
        )
        diff = (ours - builtin).abs()
        peak = ours.abs().max().item()
        outside = int((diff > TOLERANCE + TOLERANCE * builtin.abs()).sum().item())
        total = diff.numel()
        print(
            f"  layer {layer}: |ours|={peak:<11.4g}"
            f" |builtin|={builtin.abs().max().item():<11.4g}"
            f" maxdiff={diff.max().item():<11.4g}"
        )
        outside_pct = 100.0 * outside / total
        print(
            f"      outside tolerance: {outside}/{total} ({outside_pct:.2f}%)"
            f"  ok={torch.allclose(ours, builtin, rtol=TOLERANCE, atol=TOLERANCE)}"
        )
        # The gate runs before the comparison: beyond this bound the result is an
        # addressing or uninitialized-memory fault, not rounding, so no tolerance
        # applies. A missing dequant alone would only reach the accumulator peak.
        if peak > OUT_BOUND:
            failures.append(
                f"layer {layer}: |ours|={peak:.4g} exceeds the bound {OUT_BOUND:.4g}"
                " -> addressing or uninitialized memory"
            )
        elif not torch.allclose(ours, builtin, rtol=TOLERANCE, atol=TOLERANCE):
            failures.append(
                f"layer {layer}: {outside}/{total} elements outside tolerance,"
                f" maxdiff {diff.max().item():.4g}"
            )

    if failures:
        pytest.fail(f"gtype={gtype}: " + " ;; ".join(failures))
