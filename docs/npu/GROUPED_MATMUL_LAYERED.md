# Layered grouped-matmul operators (NPU)

The `_layered` grouped-matmul family under `csrc/npu/ascend_kernels/`:

| Operator | Directory | Op type | PyTorch schema |
| --- | --- | --- | --- |
| Layered A8W4/A8W8 grouped matmul | `grouped_matmul_layered` | `GroupedMatmulLayered` | `torch.ops.afd_ascend.grouped_matmul_layered` |
| Layered grouped matmul + SwiGLU + quant | `grouped_matmul_swiglu_quant_v2_layered` | `GroupedMatmulSwigluQuantV2Layered` | `torch.ops.afd_ascend.grouped_matmul_swiglu_quant_v2_layered` |

Both are vendored from the cam_async `comm_operator` tree, whose origin is the
`ops-transformer-v9.0.1` `gmm/grouped_matmul` implementation. They are only
built for `ascend910_93`, plus the A8W8 variants of `grouped_matmul_layered`
on `ascend950` (A5). The sibling `grouped_matmul_swiglu_quant_v2_layered` is
910_93-only.

This document is a **hand-off and re-apply specification**. If the AFD pinned
baseline moves (upstream `ops-transformer` or cam_async), use it together with
the new upstream source to re-derive the layered variants without re-reading the
whole history. Every section records *what changed*, *why*, and *how to check
it*. Section [Re-applying on a new baseline](#re-applying-on-a-new-baseline) is
the mechanical procedure; the sections before it are the rationale each step
depends on.

## Why the operator has to be plugin-owned

The CANN built-in `npu_grouped_matmul` addresses weights by **expert**: one
weight list element per expert group. The `_layered` operators address weights
by **layer**: every `all_*` list element is one layer's complete expert set
(all `E` experts of that layer in a single tensor), and a device-side
`layer_index` input selects which element the kernel consumes.

That is the whole point of the family: a model forward that loops over layers
can launch **one** kernel whose layer selection happens on device, instead of a
host-side loop that re-launches per layer. The built-in cannot express it, so
the operator must be built from source and is plugin-owned.

List-length semantics, stated once because most of the port's complexity follows
from it:

- `all_weight`, `all_bias`, `all_scale` are each `Tensor[]` of length `L`
  (the layer count). All three lengths must be equal.
- Element `i` is **all experts of layer `i`**, shape `[E, K, N]` (or the packed
  / NZ equivalents). Element `i` is *not* expert `i`.
- `layer_index` is `int64 [1]` on device and selects `i`.
- Within a selected layer, experts are addressed by element offsets of that one
  tensor — i.e. the operator is always in "single weight tensor" mode. This is
  why `isSingleWeight_` is forced to `true` in tiling
  (`op_host/op_tiling/grouped_matmul_tiling.cpp:221`) regardless of `L`.

## Provenance (record this whenever it changes)

| Item | Value used for this port |
| --- | --- |
| Upstream operator source | `ops-transformer-v9.0.1` `gmm/grouped_matmul` (132 files in `op_host/` + `op_kernel/`) |
| `grouped_matmul_layered` vendoring source | cam_async `src/comm_operator/ascend_kernels/grouped_matmul_layered` |
| cam_async revision | `8b89f28` (`master`, 2026-09-22) |
| `grouped_matmul_swiglu_quant_v2_layered` original source | umdk `cam_feature` @ `0281a98a`, `src/cam/comm_operator/ascend_kernels/` (vendored in the sibling's own PR) |
| Sibling reference copy | cam_async `src/comm_operator/ascend_kernels/grouped_matmul_swiglu_quant_v2_layered` (holds the newer bounded signature) |
| Shared header origin | umdk `cam_feature` @ `0281a98a`; upgraded to the bounded form in cam_async `utils/op_kernel/layered/layered_tensor_addr.h` |
| AFD revision of the port itself | merged upstream as `79b5e8b` ("Vendor the layered `grouped_matmul_layered` CAM operator", #376) |
| Target SOC | `ascend910_93` (DAV_2201), CANN 9.0.0 (sibling JSON fix validated on 9.0.1) |
| Verified scenario | A8W4 (x `INT8`, weight packed `INT4`) |

**The two operators do not share a vendoring revision,** and this matters when
you diff:

- `grouped_matmul_layered` started as a **clean** vendor of the cam_async copy
  (no logic differed), and the A5 port has since added real deltas on top. A
  `diff -rq` against cam_async is therefore expected to be non-empty in both
  directions; see [Ascend 950 (A5 / DAV_3510) support](#ascend-950-a5--dav_3510-support)
  for what the A5 port changed. The baseline inventory was:

  | Category | Count | Detail |
  | --- | --- | --- |
  | Files absent from AFD at port time | 37 | `op_host/op_tiling/arch35/` (6) + `op_kernel/arch35/` (30) + `op_kernel/gmm_infra/status.hpp` — **the two `arch35` trees have since been vendored for A5** |
  | Files differing, but only in whitespace/comment formatting | 10 | `grouped_matmul.h`, `grouped_matmul_tiling_key.h`, `grouped_matmul_antiquant.h`, `grouped_matmul_utils.h`, `grouped_matmul_quant_mixcore.h`, `grouped_matmul_antiquant_a16w8_msd.h`, `gmm_infra/coord.hpp`, `gmm_infra/gemm/block/block_swizzle.hpp`, `gmm_infra/gemm/tile/copy_gm_to_l1.hpp`, `op_host/grouped_matmul_def.cpp` |
  | Files differing in code | 4 | see below |

  (cam_async had 118 files under this directory, AFD 81 at port time.)

  The four with real code deltas at port time:

  | File | Delta |
  | --- | --- |
  | `op_host/op_tiling/grouped_matmul_tiling.cpp` | the `DAV_3510` dispatch block and its three includes were dropped — **since restored by the A5 port** |
  | `op_api/aclnn_grouped_matmul_layered.cpp` | spelling corrections inside log / `CHECK_COND` message strings only (7 occurrences across "split", "tensor" and "equal") — no executable change; the A5 port has since rewired the four `DAV_3510` branches |
  | `op_host/grouped_matmul_infershape.cpp` | the same class of message-string and comment spelling corrections |
  | `op_host/grouped_matmul_infershape_quant_checker.cpp` | one message-string spelling correction |

  A re-apply that reproduces only the spelling corrections is functionally
  correct. Treat those three files as cosmetic, but keep the file count
  expectation above so an unexpected entry is not dismissed. If you do re-apply
  them, note the corrected spellings come from AFD's own cleanup and are not
  upstream behaviour to preserve.

- `grouped_matmul_swiglu_quant_v2_layered` is **not** a clean vendor of anything.
  It was taken from umdk `cam_feature` @ `0281a98a` in its own PR, which already
  applied AFD-specific adaptations carrying `# ### PATCH START/END` markers and
  host-side changes (`op_api/grouped_matmul_swiglu_quant_utils.h` and
  `op_api/aclnn_grouped_matmul_swiglu_quant_v2_layered_l0op.cpp` are the notable
  ones). This change then adapted 13 further files to the bounded header. So a
  whitespace-stripped diff of the AFD sibling against **either** umdk
  (`14` files) or cam_async (`25` files) is expected to be non-empty, and is
  **not** a clean correctness signal. Diff only the 13 files listed in
  [Sibling adaptation](#sibling-adaptation-13-files), and treat a delta there as
  a real finding.

## Interface contract

### Operator inputs

`op_host/grouped_matmul_def.cpp` is the single source of truth, and its input
names drive the generated `DTYPE_*` / `ORIG_DTYPE_*` macros that the kernel uses
for compile-time dispatch. **Renaming an input here renames macros across the
whole kernel tree** — see [The rename table](#the-rename-table).

| Slot | Input | ParamType | Dtypes (13 columns) |
| --- | --- | --- | --- |
| 0 | `x` | DYNAMIC | `INT8` only |
| 1 | `all_weight` | DYNAMIC | `INT8`, `INT4` |
| 2 | `all_bias` | DYNAMIC | `INT32`, `FLOAT`, `BF16` |
| 3 | `all_scale` | DYNAMIC | `UINT64`, `FLOAT`, `BF16` |
| 4 | `layer_index` | REQUIRED | `INT64` |
| 5 | `group_list` | OPTIONAL | `INT64` |
| 6 | `per_token_scale` | OPTIONAL | `FLOAT` |
| out 0 | `y` | DYNAMIC | `INT8`, `BF16`, `FP16`, `INT32` |

Upstream declares **26 dtype columns**; this port trims to **13**
(`grouped_matmul_def.cpp:19-23` records the rationale). The trim is what makes
the operator A8W4/A8W8-only on 910_93. It is also load-bearing for correctness
of the port — several upstream branches become unreachable *only* because `x`
can never be `BF16`/`FP16`. See
[Latent hazards in unreachable paths](#latent-hazards-in-unreachable-paths).

Dropped relative to upstream:

- `offset`, `antiquant_scale`, `antiquant_offset` inputs — symmetric quant only;
  the asymmetric-quant and A16-antiquant modes that use them are out of scope.
- The `aclnnGroupedMatmul{,V2,V3,V4,V5,WeightNz}*` entry points. Only
  `aclnnGroupedMatmulLayered{,GetWorkspaceSize}` survives, in a **new hybrid
  signature** (11 params: upstream's 10 plus `layerIndex`, minus the offset
  pair). It routes internally to `GMMApiVersion::V5`
  (`op_api/aclnn_grouped_matmul_layered.cpp:2536`).
- The six `*950_checker.{h,cpp}` DAV_3510 validation classes. **The A5 port has
  since vendored them** (as `aclnn_*_950_checker.cpp`) and rewired the four call
  sites; see
  [Ascend 950 (A5 / DAV_3510) support](#ascend-950-a5--dav_3510-support).

`GMMActType` moved from the deleted `aclnn_grouped_matmul_v4.h` into
`op_api/aclnn_grouped_matmul_layered.h:18-25`.

### PyTorch binding

`csrc/npu/pybind/torch_binding_grouped_matmul.cpp`, registered in
`csrc/npu/pybind/CMakeLists.txt`. It needs a local alias that the operator files
do not:

```cpp
using tensor_list = std::vector<at::Tensor>;
```

```
grouped_matmul_layered(Tensor[] x, Tensor[] all_weight, Tensor[] all_bias,
                       Tensor[] all_scale, Tensor layer_index,
                       Tensor? per_token_scale=None, int[]? group_list=None,
                       int group_list_type=0, int split_item=0,
                       ScalarType? output_dtype=None) -> Tensor[]
```

Three dispatcher registrations, matching the other AFD Ascend bindings:
`PrivateUse1` and `AutogradPrivateUse1` both instantiate the `EXECUTE_NPU=true`
template (inference-only, enforced even under grad mode), and `Meta`
instantiates `EXECUTE_NPU=false` so shape/dtype contracts are testable without a
device.

Three behaviours in the binding are worth preserving because each fixes a real
failure:

1. **Device pinning.** `const c10::OptionalDeviceGuard device_guard(at::device_of(x[0]))`
   runs before the first device operation. Without it the current device stays
   at the process default, and since `EXEC_NPU_CMD` derives its stream from the
   current device, the kernel launches in one device's context while holding
   another's addresses — an MTE DDR address error.
2. **`group_list` normalization.** `group_list_type=0` (cumsum) is converted to
   counts and normalized to `1` before the launch, so only one form reaches the
   op_api.
3. **`output_dtype` derivation.** An explicit `output_dtype` wins; otherwise
   `int8 x` + `int32 packed weight` implies `bfloat16` (A8W4 antiquant), and the
   quantized A8W8 family keeps the activation dtype. Never inherit the dtype
   from the packed weight tensor — that dtype describes the storage, not the
   result.

**Known contract limitation, tracked as issue #381.** `group_list` is
`int[]?` — a host integer list, not a tensor. The binding therefore requires a
D2H read of device-produced routing metadata and then a H2D upload of the same
values. The sibling `gmm_swiglu_quant_v2_layered` already takes `Tensor
group_list`, and ACLNN already consumes a device tensor, so this is a binding
restriction rather than an operator requirement. The asynchronous dispatch→GMM
path is interrupted by the forced `.cpu().tolist()`. See
[Issue #381](#issue-381--group_list-should-accept-a-device-tensor) for the
requested fix and the implementation consequence for `alloc_outputs`. Treat the
current schema as subject to change, not as the intended end state.

## The layered transform

This is the semantic delta from upstream `GroupedMatmul` to
`GroupedMatmulLayered`. It is the part that must be re-derived (not copied)
on a new baseline.

### The rename table

Every level of the operator identity is renamed. Upstream contains **zero**
occurrences of the layered names; the rename is one-directional.

| Location | Upstream | Layered |
| --- | --- | --- |
| Directory | `gmm/grouped_matmul` | `grouped_matmul_layered` |
| OpDef class | `class GroupedMatmul : public OpDef` | `class GroupedMatmulLayered` |
| Registration | `OP_ADD(GroupedMatmul)` | `OP_ADD(GroupedMatmulLayered)` |
| Infershape | `IMPL_OP_INFERSHAPE(GroupedMatmul)` | `IMPL_OP_INFERSHAPE(GroupedMatmulLayered)` |
| Tiling | `IMPL_OP_OPTILING(GroupedMatmul)` | `IMPL_OP_OPTILING(GroupedMatmulLayered)` |
| Tiling data class | `REGISTER_TILING_DATA_CLASS(GroupedMatmul, GMMTilingData)` | `...GroupedMatmulLayered, GMMTilingData)` |
| l0op | `OP_TYPE_REGISTER(GroupedMatmul)` | `OP_TYPE_REGISTER(GroupedMatmulLayered)` |
| ACLNN | `aclnnGroupedMatmul{,GetWorkspaceSize}` | `aclnnGroupedMatmulLayered{,GetWorkspaceSize}` |
| Kernel entry | `grouped_matmul(...)` | `grouped_matmul_layered(...)` |
| Input names | `weight`, `bias`, `scale` | `all_weight`, `all_bias`, `all_scale` |

The last row propagates into the kernel's compile-time type system, because the
`DTYPE_*` macros are generated from input names:

| Upstream macro | Layered macro |
| --- | --- |
| `DTYPE_WEIGHT` | `DTYPE_ALL_WEIGHT` |
| `DTYPE_BIAS` | `DTYPE_ALL_BIAS` |
| `DTYPE_SCALE` | `DTYPE_ALL_SCALE` |
| `FORMAT_WEIGHT` | `FORMAT_ALL_WEIGHT` |
| `ORIG_DTYPE_WEIGHT` | `ORIG_DTYPE_ALL_WEIGHT` |
| `ORIG_DTYPE_SCALE` | `ORIG_DTYPE_ALL_SCALE` |

These are concentrated in `op_kernel/grouped_matmul_utils.h`. If a re-apply
renames the inputs, expect compile errors naming the old macros until every
occurrence is updated — that is the intended forcing function.

**One rename was deliberately not done:** `ASCENDC_TPL_ARGS_DECL(GroupedMatmul, ...)`
at `op_kernel/grouped_matmul_tiling_key.h:121` keeps the upstream name. The
sibling operator *did* rename its equivalent. This is harmless while nothing
else in the linked package declares the same template name, and it currently
links, but it is an inconsistency to be aware of — if a future baseline links a
second op declaring `ASCENDC_TPL_ARGS_DECL(GroupedMatmul, ...)`, expect a
duplicate-symbol clash.

### Input slot re-indexing

Inserting `layer_index` at slot 4 shifts the trailing slots. Both the tiling and
the infershape index tables were rewritten:

`op_host/grouped_matmul_host_util.h:20-27`:

| Constant | Upstream | Layered |
| --- | --- | --- |
| `X_INDEX` | 0 | 0 |
| `WEIGHT_INDEX` | 1 | 1 |
| `BIAS_INDEX` | 2 | 2 |
| `SCALE_INDEX` | 3 | 3 |
| `LAYER_INDEX` | — (was `OFFSET_INDEX = 4`) | **4** |
| `GROUPLIST_INDEX` | 7 | **5** |
| `PER_TOKEN_SCALE_INDEX` | 8 | **6** |

`op_host/grouped_matmul_infershape_common_util.h:18-32` mirrors this. The three
dropped indices (`OFFSET`, `ANTIQUANT_SCALE`, `ANTIQUANT_OFFSET`) are kept as
out-of-range sentinels (`99UL`) so the retained A16-antiquant checkers still
compile rather than being deleted wholesale.

`GMM_INDEX_IN_LAYER` is declared but **never used** — the tiling validation
reads the raw `LAYER_INDEX` instead. Dead constant; harmless.

The kernel entry signature loses the dropped inputs and gains `layerIndex` at
the matching position (12 args → 10):

```cpp
// op_kernel/grouped_matmul_layered.cpp:355
__global__ __aicore__ void grouped_matmul_layered(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                                                 GM_ADDR scale, GM_ADDR layerIndex,
                                                 GM_ADDR groupList, GM_ADDR perTokenScale,
                                                 GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
```

**No `extern "C"`.** Unlike the other operators in `csrc/npu/`, this entry is a
template, and C++ forbids `extern "C"` on a template. Upstream's
`grouped_matmul.cpp` omits it for the same reason. Do not "fix" this.

### Device-side layer resolution

One read per core, at kernel Init, shared by both layered kernel entries via
`LayeredReadLayerIndex()`:

```cpp
// op_kernel/grouped_matmul_layered.cpp:366
int64_t curLayer = LayeredReadLayerIndex(layerIndex);
```

`curLayer` is then threaded as an explicit parameter through every
initializer — ten macro bodies plus the autotiling constructor, the fixaxismove
interface, and the A8W4 pre-quantization pass. Upstream functions that gained a
parameter (signature changes that must be documented per `AGENTS.md`):

| File | Added parameter |
| --- | --- |
| `op_kernel/grouped_matmul.h` | `int64_t curLayer` on `Init` (decl + def) |
| `op_kernel/grouped_matmul_quant_mixcore.h` | `int64_t curLayer` on `Init` |
| `op_kernel/grouped_matmul_antiquant_a8w4.h` | `int64_t curLayer` on `Init` |
| `op_kernel/grouped_matmul_antiquant_a8w4_msd.h` | `int64_t curLayer` on `Init` |
| `op_kernel/grouped_matmul_antiquant_a8w4_msd_new.h` | `int64_t curLayer` on `Init` |
| `op_kernel/grouped_matmul_antiquant_a8w4_pre.h` | `GM_ADDR scale`, `int64_t curLayer`, `uint32_t layerNum` |
| `op_kernel/grouped_matmul_autotiling_a8w4.h` | `int64_t curLayer` ctor param, `curLayer_` member |
| `op_kernel/grouped_matmul_fixaxismove_interface.cpp` | `int64_t curLayer`, `uint32_t layerNum` |
| `op_kernel/grouped_matmul_fixaxismove_regular.h` | `int64_t curLayer_`, `uint32_t layerNum_` on `Params` |

Note the `antiquant_a8w4_pre` row: upstream had a **declaration/definition
mismatch** (`scale` present in the definition, absent from the declaration).
Adding `curLayer` is a good moment to reconcile it.

There are **20 `GetLayerTensorAddr<T>(...)` call sites** in 8 kernel files,
replacing the upstream element-0 / `singleWeight` branches:

| File | Sites | Replaces upstream |
| --- | --- | --- |
| `op_kernel/grouped_matmul.h` | 3 | bias / weight / scale `singleWeight==0` branches |
| `op_kernel/grouped_matmul_antiquant_a8w4_msd.h` | 3 | element-0 weight/bias/scale |
| `op_kernel/grouped_matmul_antiquant_a8w4_msd_new.h` | 3 | element-0 weight/bias/scale |
| `op_kernel/grouped_matmul_autotiling_a8w4.h` | 3 | `GetTensorAddr<uint8_t>(0, ...)` |
| `op_kernel/grouped_matmul_antiquant_a8w4.h` | 2 | `GetTensorAddr<...>(0, scale)` |
| `op_kernel/grouped_matmul_antiquant_a8w4_pre.h` | 2 | element-0 weight/scale |
| `op_kernel/grouped_matmul_fixaxismove_regular.h` | 2 | element-0 `ptrB` / `ptrScale` |
| `op_kernel/grouped_matmul_quant_mixcore.h` | 2 | `scaleGm` / `biasGm` branches |

The mechanical rule for a re-apply: **every** upstream
`GetTensorAddr<T>(0, ptr)` or `singleWeight == 0 ? ... : GetTensorAddr(0, ...)`
becomes `GetLayerTensorAddr<T>(curLayer, layerNum, ptr)`.

### `layerNum` plumbing (tiling → kernel)

`GetLayerTensorAddr` bounds-checks the device index against the list length, so
the length has to travel from host tiling to the kernel. This is the plumbing
added at three host sites and consumed from two tiling structs:

```cpp
// op_host/op_tiling/grouped_matmul_tiling.cpp
:459   tilingData.gmmBaseParams.set_layerNum(weightListLen);      // Init() path
:1874  tilingDataA8W4.hpTilingData.set_layerNum(weightListLen);  // autotiling path
:1972  tilingDataA8W4.gmmBaseParams.set_layerNum(weightListLen); // MSD/pergroup path
```

```cpp
// op_host/op_tiling/grouped_matmul_tiling.h
:53  TILING_DATA_FIELD_DEF(uint32_t, layerNum);  // GMMBaseParams
:87  TILING_DATA_FIELD_DEF(uint32_t, layerNum);  // A8W4HPTiling
```

Both fields are needed because the two A8W4 paths serialize different structs:
the autotiling kernel reads `hpTilingData`, the MSD/pergroup kernels read
`gmmBaseParams`. Shipping only one is defect #1 below.

On the kernel side, `grouped_matmul.h` gains `int64_t curLayer_` and
`uint32_t layerNum_` members and reads `layerNum_ = gmmBaseParams->layerNum` in
`Init`.

### Ascend 950 (A5 / DAV_3510) support

**A5 is supported for the A8W8 quant path only.** The 910_93 path is the
`grouped_matmul_layered.cpp` entry; A5 gets its own entry,
`op_kernel/grouped_matmul_layered_apt.cpp`, selected by `opFile.value` in the
`ascend950` AICore config. The two entries differ in template parameter list,
which is why upstream also ships two files:

| | 910_93 entry | A5 entry |
| --- | --- | --- |
| file | `grouped_matmul_layered.cpp` | `grouped_matmul_layered_apt.cpp` |
| guard | `__CCE_AICORE__ == 220 \|\| __NPU_ARCH__ == 3003/3113` | `__CCE_AICORE__ == 310` |
| template args | `D_T_A, D_T_B, D_T_Y, TRANS_A, TRANS_B, ...` | `QUANT_B_TRANS, QUANT_A_TRANS, KERNEL_TYPE` |
| dispatch | `grouped_matmul_tiling_key.h` | `arch35/.../gqmm_tiling_key.h` |
| kernel body | `op_kernel/*.h` (Catlass / MSD) | `op_kernel/arch35/quant_adaptive_sliding_window_templates/*` |

`A5` here means `__CCE_AICORE__ == 310`, the device-side spelling of the host
enum `NpuArch::DAV_3510`. The directory name `arch35` is the upstream
convention for that generation.

#### What the port changes relative to upstream `grouped_matmul_apt.cpp`

1. **The entry is layered.** It takes `layerIndex` between `scale` and
   `groupList`, matching the 910_93 entry's input order in
   `op_host/grouped_matmul_def.cpp`.
2. **Only `V310_GMM_QUANT` (A8W8) is carried.** AFD's op_def drops the
   `offset` / `antiquant_scale` / `antiquant_offset` inputs, so the upstream
   `V310_GMM_ANTI_QUANT` (A8W4/A16W4) and no-quant branches cannot be driven
   through the registered interface.
3. **`cgmct` is not needed.** The `V310_GMM_QUANT_PERTILE` branch and the
   `non_quant` kernel both include a `cgmct/` tree that is not vendored. For
   INT8 x + INT8 w the macro logic in `grouped_matmul_utils.h` never selects
   PERTILE (`V310_GMM_QUANT_PERTILE` additionally requires
   `ORIG_DTYPE_X != DT_INT8`), so the missing tree is out of the compiled set.

#### Where the layer is resolved

The arch35 kernels address experts as *offsets inside one weight tensor*
(`singleW == 1`), which is exactly the layered shape: element `i` of `all_*` is
all experts of layer `i`. So the layer is resolved **once per core at Init** and
substituted into every per-layer descriptor dereference:

```cpp
layerNum_ = gmmQuantParams_->layerNum;
curLayer_ = LayeredReadLayerIndex(layerIndex);
```

Nine call sites changed from `GetTensorAddr<T>(0, ptr)` to
`GetLayerTensorAddr<T>(curLayer_, layerNum_, ptr)`:

- `arch35/.../gqmm_cube_on_the_fly.h` — 5 (weight, bias, 3 scale forms)
- `arch35/.../gqmm_mix_online_dynamic.h` — 4 (weight, bias, 2 scale forms)

`x`, `y`, `groupList` and `perTokenScale` stay at element 0: they are per-call,
not per-layer. `GetLayerTensorAddr` is at **global** scope (not in
`GROUPED_MATMUL::`), so the calls are written `::GetLayerTensorAddr<...>`.

Because the resolution happens before the kernel body, the arch35 kernels need
no other change: the selected layer's `[E, K, N]` element has the same layout the
kernels already expect from a single weight tensor.

#### The layer count reaches the kernel through `reserved`

`GroupedMatmulTilingData::GMMQuantParams` has no spare field, so its unused
`reserved` slot was renamed to `layerNum` (same `uint16_t`, same offset — the
struct layout is unchanged). The host tiling counts the `all_*` list length into
`layerCount`, and `GroupedQbmmTiling::DoOpTiling` ships it. The kernel clamps
`layer_index` against it in `GetLayerTensorAddr`.

#### Two host-side invariants the A5 tiling must also honour

- **`isSingleW` must be forced to `true`.** Upstream derives it from the list
  length, which is always > 1 in the layered contract, and the kernel would then
  index the descriptor by expert instead of by layer. The 910_93 tiling already
  pins `isSingleWeight_ = true` for the same reason.
- **The arch35 quant tiling registers under this operator's own name.**
  Upstream writes `REGISTER_OPS_TILING_TEMPLATE(GroupedMatmul, ...)`, but
  `TilingRegistry::DoTilingImpl` resolves by `context->GetNodeType()`, which is
  `GroupedMatmulLayered`. With the upstream spelling the lookup misses and
  returns `GRAPH_FAILED`; the registration is therefore
  `REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulLayered, GroupedQbmmTiling, 0)`.

#### The DAV_3510 dispatch is restored

`op_host/op_tiling/grouped_matmul_tiling.cpp` carries the upstream 20-line
`if (npuArch == NpuArch::DAV_3510)` block (quant -> `TilingRegistry`, mixed
dtype -> `GroupedWeightQuantBatchMatmulTiling`, no-quant ->
`GroupedNoQuantMatmulTiling`) and the includes it needs, including
`tiling_base/tiling_templates_registry.h` and its
`using namespace Ops::Transformer::OpTiling;`.

**The `npuArch != DAV_3510` platform sanity check stays** in
`TilingPrepareForGMM`. Do not remove it.

#### The op_api gates now call the upstream checkers

The four DAV_3510 branches in `op_api/aclnn_grouped_matmul_layered.cpp`
previously refused every A5 call with "not shipped in the cam_async layered
build". They now delegate to the ported upstream validators:

| site | caller | delegates to |
| --- | --- | --- |
| `CheckFunctionParams` | quant (A8W8) | `AclnnGroupedMatmulDAV3510Checker<aclTensorList>` |
| `CheckFunctionParams` | weight-quant | `AclnnGroupedMatmulWeightQuantDAV3510Checker` |
| `CheckGroupSize` | no-quant | `AclnnGroupedMatmulNoQuantDAV3510Checker` |
| `IsPerTileQuantMode` | quant | `AclnnGroupedMatmulDAV3510Checker::IsPerTileQuantMode` |

The six `grouped_matmul*_950_checker.{h,cpp}` files are vendored from
ops-transformer. **The `.cpp` files are named `aclnn_*.cpp` in AFD**, because the
op_host build globs only `aclnn*.cpp` into `cust_opapi`; those files hold the
explicit template instantiations
(`template class AclnnGroupedMatmulDAV3510Checker<aclTensorList>`), so under
their upstream names they would be silently dropped and the link would fail.

#### Files that are no longer excluded

`op_kernel/arch35/` (30 files) and `op_host/op_tiling/arch35/` (6 files) are now
vendored, and `op_kernel/grouped_matmul_layered_apt.cpp` is new. Still **not**
vendored, and still out of scope:

- `op_kernel/grouped_matmul_apt.cpp` — replaced by the layered entry above
- `op_kernel/fallback_grouped_matmul.cpp` — the `IMPL_OP(...).OpExecuteFunc`
  aclnn fallback, upstream-only
- `op_host/op_api/aclnn_grouped_matmul_v{2,3,4,5}.h`,
  `aclnn_grouped_matmul_weight_nz.h` — other aclnn entry points
- the `cgmct/` tree — only reachable from PERTILE and non-quant, neither of
  which INT8 x + INT8 w selects
- `op_kernel/gmm_infra/status.hpp` — dead upstream (nothing includes it)

Also dropped: `op_host/op_api/` is **relocated** to the operator's top-level
`op_api/`, and `op_host/CMakeLists.txt` / `tests/` / `docs/` / `examples/` are
not vendored.

### Shared `layered_tensor_addr.h`

`csrc/npu/ascend_kernels/utils/op_kernel/layered/layered_tensor_addr.h` (56
lines). Shared by both layered operators so the descriptor decoding and the
`layer_index` read have exactly one definition.

This header was **changed for AFD** relative to the vendoring tree: it gained the
`allTensorLen` bound and `LayeredReadLayerIndex()`, and the sibling operator was
adapted to the new signature (see below). The cam_async tree has since picked up
the same signature.

```cpp
__aicore__ inline int64_t LayeredReadLayerIndex(GM_ADDR layerIndexPtr);

template <typename T>
__aicore__ inline __gm__ T* GetLayerTensorAddr(int64_t layerIndex, uint32_t allTensorLen,
                                               GM_ADDR allTensorPtr);
```

Descriptor layout (`ListTensorDesc` / `GetTensorAddr`): the first `uint64` holds
the byte offset from the descriptor head to the data-pointer array; element `i`
of that array is tensor `i`'s GM address. **Index semantics here is the LAYER
number**, not the expert number.

**The guard clamps rather than traps, and that is deliberate.**
`ASCENDC_ASSERT` compiles to a no-op in device translation units (see
`platform_ascendc.h`), so an assert would silently do nothing. A clamp at least
turns an out-of-range index into a wrong-but-observable layer 0 result instead
of a wild pointer dereference into the descriptor's pointer array:

```cpp
if (unlikely(layerIndex < 0 || layerIndex >= static_cast<int64_t>(allTensorLen))) {
    layerIndex = 0;
}
```

Cache maintenance: **none is needed.** The header comment records a negative
result — an A/B test of `Barrier + DataCacheCleanAndInvalid + dsb` showed no
behavioural difference, and both the ops-transformer original and the CANN
built-in read these descriptors bare. Do not re-investigate this from scratch;
the earlier cam_async commits that added and then removed a flush helper
(`ce1c058`, `a010769`, `2ff0db1`) are already covered by that note.

Because `compile_ascend_proj.sh:44-45` flattens `utils/op_kernel/*` into
`<op>/op_kernel/` at build time, the include is the bare `"layered/layered_tensor_addr.h"`.
Only two files include it: `op_kernel/grouped_matmul_utils.h:27` and
`op_kernel/grouped_matmul_fixaxismove_regular.h:19`.

### Host-side validation added for the layered contract

All of the following is **new code with no upstream counterpart**:

- `op_host/op_tiling/grouped_matmul_tiling.cpp:428-460` — validates
  `layer_index` dtype (`INT64`), rank and shape (`1-D`, size 1), counts all three
  `all_*` list lengths by walking `GetDynamicInputTensor`, requires them equal and
  non-zero, and ships `layerNum`.
- `op_host/op_tiling/grouped_matmul_tiling.cpp:1772-1776` — a **second**
  `weightListLen` count inside `A8W4Tiling`, because that path returns before
  `Init()` runs and therefore never executes the block above.
- `op_host/grouped_matmul_infershape.cpp:409-413` — requires `all_weight[0]` to be
  3-dim `(e, k, n)`, and takes the expert count from `dim(0)`. Upstream could
  take the group count from the tensor-list length; that reading is wrong here,
  where the list length is the *layer* count.
- `op_host/grouped_matmul_infershape.cpp:1063-1071` — per-`x` K-axis loop with a
  `GMM_MAX_INNER_AXIS` bound, replacing `CheckShapeSameLengthTensorList`.
- `op_host/grouped_matmul_infershape.cpp:1080-1088` — cross-layer expert-count
  consistency: every `all_weight[i]` must have the same `dim(0)`.
- `op_host/grouped_matmul_infershape.cpp:1100` — `CheckCaseSplitM` collapses the
  upstream four-branch `split-m` dispatch to a single call into
  `SplitMSingleXSingleWeightSingleY`, mirroring tiling's `isSingleWeight_ = true`.
  Three upstream helpers were deleted with it, with tombstone comments:
  `CheckShapeDiffLengthTensorList`, `SplitMSingleXSeparatedWeightSingleY`,
  `SplitMSeparatedXSeparatedWeightSingleY`.
- `op_api/aclnn_grouped_matmul_layered.cpp:2428-2454` — the **list-shrink
  validation shim**, the most intricate added logic in the port. The upstream
  shape checks were written for per-expert separation and would misread the layer
  count as expert separation. The fix builds 1-element `aclCreateTensorList`
  views (layer 0 as representative — all layers are shape-identical by contract),
  swaps them into `gmmParams`, runs `CheckParam`, and restores the full lists for
  the launch.
- `op_api/aclnn_grouped_matmul_layered.cpp:2504-2511` — the int32→int4
  `UnpackB32ToB4` unpack (behaviour copied from upstream V5; the rationale comment
  is new). torch_npu expresses int4 as `int32` with eight nibbles per word.
- Two new `scaleDimNum` branches accepting **2-D per-channel `[e, n]` scale** in
  addition to upstream's 3-D per-group `[e, g, n]`:
  `grouped_matmul_tiling.cpp:1163-1179` and `:1910-1926`. Upstream read
  `quantGroupNum = scale.GetDim(1)` unconditionally; both sites now derive `n`
  and `quantGroupNum` from the rank, and the `A8W4noMsdSpace_` arithmetic
  (`:1183`) uses the derived `n`.

### Two defects fixed during bring-up

Both are in the vendored history rather than newly introduced here, and both are
recorded because they are exactly the failure mode this family invites. **If a
re-apply "simplifies" the `layerNum` plumbing or the pre-quantization addressing,
these come back.**

**Defect 1 — every layer silently resolved to layer 0.**
The A8W4 path shipped `layerNum` only on the autotiling struct
(`hpTilingData`), while the MSD/pergroup kernels read it from `gmmBaseParams`.
`gmmBaseParams` was only filled by `Init()` — a function the A8W4 path returns
before. The field therefore defaulted to `0`, and because the new range guard
clamps rather than traps, **every** layer index was clamped to `0`. The operator
produced correct-looking output for layer 0 and wrong output for all others,
with no error anywhere.

*Fix:* ship the length on **both** structs — `tilingDataA8W4.hpTilingData.set_layerNum`
(:1874) and `tilingDataA8W4.gmmBaseParams.set_layerNum` (:1972).

**Defect 2 — the pre-quantization pass read layer 0's weights.**
The A8W4 pre-quantization pass read `all_weight` / `all_scale` via
`GetTensorAddr(0, ...)`. That is correct for a single-tensor list but wrong once
the list holds one element per layer. It cannot be corrected downstream either:
the pass writes into a workspace that the main compute consumes as a single
tensor, so a wrong layer there is baked into the arithmetic.

*Fix:* resolve through `GetLayerTensorAddr<int4b_t>(curLayer, layerNum, weight)`
and `GetLayerTensorAddr<int64_t>(curLayer, layerNum, scale)` at
`op_kernel/grouped_matmul_antiquant_a8w4_pre.h:67,71`. The comment there —
*"a wrong layer here cannot be corrected downstream"* — is the highest-value
comment in the port; keep it.

### Sibling adaptation (13 files)

The sibling `grouped_matmul_swiglu_quant_v2_layered` had **no `layerNum` at all**
before this change (confirmed against its `0281a98a` source: zero occurrences,
and its `GetLayerTensorAddr` calls were the 2-argument form). Adopting the shared
bounded header meant carrying the list length from tiling to kernel there too.
This is the same shape as the cam_async change that introduced the signature.

| File | Change |
| --- | --- |
| `op_host/op_tiling/..._base_tiling.h` | `uint32_t layerNum_ = 1;` member |
| `op_host/op_tiling/..._base_tiling.cpp` | set `layerNum_ = weightListLen`; `set_layerNum` in `FillTilingData` |
| `op_host/op_tiling/..._fusion_tiling.h` | `uint32_t layerNum_ = 1;` member |
| `op_host/op_tiling/..._fusion_tiling.cpp` | count `WEIGHT_INDEX` list; `set_layerNum` |
| `op_host/op_tiling/..._tiling.h` | `TILING_DATA_FIELD_DEF(uint32_t, layerNum)` on **both** serialized structs |
| `op_kernel/..._utils.h` | `uint32_t layerNum = 1;` on `GMAddrParams` |
| `op_kernel/..._v2_layered.cpp` | `LayeredReadLayerIndex(layerIndex)` replaces the inline GM read |
| `op_kernel/..._a4w4_pipeline.h` | `gmAddrParams.layerNum = ...->layerNum` |
| `op_kernel/..._a8w4_msd_pipeline.h` | `gmAddrParams.layerNum = ...->layerNum` |
| `op_kernel/..._a4w4_mid.h` | bound at the two `GetLayerTensorAddr` sites |
| `op_kernel/..._a8w4_msd_mid.h` | bound at the two sites |
| `op_kernel/..._a8w4_msd_post.h` | bound at the assist-matrix site |
| `op_kernel/..._spilit_fusion.h` | bound at the two sites |

Note the two tiling structs and the two *tiling classes*: the base and fusion
paths serialize different structs, so — exactly as in defect 1 — the field must
be added to both, and `layerNum` must be populated on both paths.

These 13 files are the complete set this change touches in the sibling; no other
sibling file is modified, so a re-apply should not need to touch others either.
Both operators must keep building together afterwards, and the `afd_async_*`
operators must be rebuilt to confirm no regression from the shared-header change.

## Post-merge follow-ups (read before re-applying)

The port was merged upstream as `79b5e8b` (#376). Four things landed around it
that a re-apply must reproduce, and one of them is a **compile blocker that the
original commit shipped without**. This section is the reason to re-read upstream
history rather than trusting this port's own commit as the finished state.

### The binding does not compile as originally committed (upstream `f679b77`, #378)

`torch_binding_grouped_matmul.cpp` uses `tensor_list` in `alloc_outputs` and in
the `grouped_matmul_layered` template, but **the original commit never declared
it**. The build fails with `'tensor_list' does not name a type`. The fix is one
line in the anonymous namespace:

```cpp
using tensor_list = std::vector<at::Tensor>;
```

The sibling binding (`torch_binding_gmm_swiglu_quant.cpp:19`) has always had this
alias, which is why the pattern was assumed present. **This is the highest-value
lesson in this document:** the original commit's `build_aclnn.sh` run proves the
*operator* builds, but `build_aclnn.sh` does **not** compile the pybind extension
— that happens later in `setup.py`'s `CMakeExtension("afd_plugin._C_ascend",
"csrc/npu/pybind")` step. A green operator build says nothing about the binding.

Practical consequence for any re-apply or new binding: build through the full
package path, not just the operator path, before believing the change compiles:

```bash
SOC_VERSION=910c AFD_BUILD_ASCEND_OPS=1 python -m pip install -e . -v --no-build-isolation
```

Upstream's re-verification of the fix recorded **62 passed, no skips** with real
NPU calls for both layered GMM operators, on Ascend910_9382 / CANN 9.0.1 /
Python 3.12.13 / torch 2.10.0 + torch-npu 2.10.0.post2.

### ACLNN JSON kernel-config name lookup (upstream `50fef29`, #373; issue #372)

Affects the **sibling**, and is a trap for this operator too.

`aclnnGroupedMatmulSwigluQuantV2Layered` failed on CANN 9.0.1 before kernel
launch with `Parse dynamic kernel config fail`. The build installs
`grouped_matmul_swiglu_quant_v2_layered.json`, but the runtime derives a
different name from the op type: `V2Layered` gets split on the digit, so it looks
for `grouped_matmul_swiglu_quant_v_2_layered.json` (or
`grouped_matmul_swiglu_quant_v2layered.json`). Neither exists.

The fix registers the real filename explicitly, once per process, through a
thread-safe function-local static at the top of the l0op entry:

```cpp
#include "opdev/op_def.h"
...
[[maybe_unused]] static const bool configRegistered = []() {
    op::BinConfigJsonDict::UpdateConfigJsonPath(
        GroupedMatmulSwigluQuantV2LayeredOpTypeId(),
        "grouped_matmul_swiglu_quant_v2_layered.json");
    return true;
}();
```

**Why `grouped_matmul_layered` does not need this, and when that changes.** The
name-mangling trigger is a **digit in the op type**: `GroupedMatmulSwigluQuantV2Layered`
contains `V2`, so the runtime splits it; `GroupedMatmulLayered` contains no digit,
so the default conversion produces the correct `grouped_matmul_layered.json`.

This is not merely inferred. The `f679b77` re-verification ran the suite with
`AFD_RUN_ASCEND_OP_RUNTIME=1`, which is exactly the gate that makes
`test_gmm_layered_runtime_multi_layer` launch the real kernel
(`tests/unit/compat/npu/test_gmm_layered.py:216`), and reported **62 passed, no
skips** on Ascend910_9382 / CANN 9.0.1. "No skips" means that device test
executed. A wrong config-name lookup would have failed it with the same
`Parse dynamic kernel config fail` the sibling hit, so the default resolution is
**device-verified for this operator on CANN 9.0.1**.

Treat it as a latent coupling rather than a permanent fact: **if the op type is
ever renamed to include a digit** (for example `GroupedMatmulV2Layered`), or a
future CANN changes the conversion, this operator will hit the same
`Parse dynamic kernel config fail` and needs the same one-line registration.
Check the installed JSON filename against the runtime's expected name whenever
the CANN version or the op type moves.

Upstream did not version-gate that fix and documented it in `csrc/npu/README.md`
instead of adding JSON aliases, on the grounds that the explicit filename is the
package contract. Follow that precedent rather than installing alias files.

### Issue #381 — `group_list` should accept a device tensor

Open upstream, **not yet fixed**, and it directly targets this operator's
binding. Recording it here because it is the known next change to
`torch_binding_grouped_matmul.cpp`, and a re-apply should not treat the current
host-list contract as final.

Current schema: `int[]? group_list=None`. The binding takes
`c10::optional<std::vector<int64_t>>`, optionally converts cumulative offsets to
counts **on the host**, builds a CPU int64 tensor via `at::from_blob(...).clone()`,
and calls `.to(device)` — a device→host→device round trip.

Impact, in the issue's own terms: in an async MoE path where expert token counts
are produced on the NPU, the host-list contract forces
`group_list_device.cpu().tolist()` before the call, and the binding then copies
the values back to the device. That interrupts the asynchronous
dispatch→GMM path with an unnecessary D2H synchronization.

Note the asymmetry that makes this clearly a binding-level defect rather than an
operator requirement: the sibling `gmm_swiglu_quant_v2_layered` already declares
`Tensor group_list`, and the ACLNN entry already consumes a device
`group_list_tensor`. The restriction exists only in this Python-facing schema.

Requested fix (paraphrasing the issue's acceptance list, since it defines the
contract any re-apply should aim at):

1. change the schema and C++ binding to accept a device `group_list` Tensor, with
   documented dtype/device/shape semantics and optional behaviour;
2. pass count-style device tensors straight to ACLNN, keeping any supported
   cumsum→count conversion **on device**;
3. remove host reads of group-list values from output allocation and the
   execution path; update callers and tests;
4. add a real-NPU regression passing device-produced counts, including repeated
   calls and each supported group-list representation;
5. verify numerical agreement with the built-in reference using identical inputs
   and fresh clones where inputs are modified in place;
6. attach a profiler trace showing no group-list-induced D2H/host sync and no
   D2H→H2D round trip in the operator path (test-only reference synchronization
   stays outside the measured path).

Two implementation notes worth carrying forward, because they explain why the
current shape exists and what breaks when it changes:

- **`alloc_outputs` depends on the host values.** With per-group `x` it sums
  `group_rows` to size `y`. Moving to a device tensor means output allocation has
  to use shape metadata instead of reading group counts — which is exactly what
  the issue asks for in its third point. This is the non-mechanical part of the
  change; do not start it by only swapping the schema type.
- **The cumsum→count conversion is named in the current comment as host-side on
  purpose** ("Every compiled A8W4 variant is count-style, so a cumsum-style list
  is normalized to counts on the host; the same counts drive the y row
  allocation. The H2D upload rides the current stream and nothing
  synchronizes."). That comment becomes wrong once the input is a device tensor
  and must be rewritten, not left in place.

### New harnesses do not cover these operators (`8f506a9`, #374)

`tests/npu/` gained HCCL precision/performance harnesses
(`async_cam_common.py`, `async_cam_precision.py`, `async_cam_performance.py`)
plus three `tests/unit/compat/npu/test_async_cam_*.py` files. They cover the
**routed async CAM operators only** — `rg` finds no `grouped_matmul` reference
anywhere under `tests/npu/`.

So the layered operators still have no harness-based numerical coverage, and the
accuracy procedure in this document (copied from cam_async) remains the only
numerical gate. If a future change adopts that harness pattern, the GMM operators
would be a good candidate, but do not assume they are covered today.

Also unrelated to these operators but in the same window: `5ca9528` (#371,
DSV4 async CAM shared-expert fusion) and `874e2ce` (#383, DBO microbatch device
binding). Neither touches `csrc/npu/`.

## AFD integration outside the operator directory

| File | Change |
| --- | --- |
| `csrc/npu/ascend_kernels/operator_registry.json` | add `grouped_matmul_layered` to `soc_versions.ascend910_93` (**last**, order is asserted) and `operator_meta` (`requires_shmem: false`) |
| `csrc/npu/pybind/CMakeLists.txt` | add `torch_binding_grouped_matmul.cpp` to the `afd_plugin_ascend_C` sources |
| `csrc/npu/README.md` | add the operator to the `ascend_kernels/` tree listing |
| `tests/unit/package/test_ascend_build_files.py` | extend the expected `ascend910_93` operator list in `test_select_ops_resolves_registry_order` |

No build-file change was needed inside `cmake_files/`: this operator keeps its
tiling implementation in `op_host/op_tiling/`, which the recursive host-source
collection (introduced with the sibling operator) already covers. That collection
is load-bearing — a top-level-only `aux_source_directory` would silently drop the
`IMPL_OP_OPTILING` registration from the run package without failing any
registry or Meta test.

## Build

Two build gates, and **both** are needed. This is the single most important
practical lesson from the post-merge follow-ups: the original commit passed the
operator build and still shipped a binding that did not compile.

```bash
# Gate 1 — operator build (CANN npu_op_* system; does NOT compile the extension)
bash csrc/npu/build_aclnn.sh "$PWD" 910c

# Gate 2 — full package build (compiles csrc/npu/pybind via setup.py's
# CMakeExtension("afd_plugin._C_ascend", ...)); this is what catches binding errors
SOC_VERSION=910c AFD_BUILD_ASCEND_OPS=1 pip install -e . -v --no-build-isolation
```

`build_aclnn.sh` only drives the ACLNN run package. Anything wrong in
`csrc/npu/pybind/**` is invisible to it. Gate 2 is mandatory after touching any
binding file, including a one-line alias.

Gate 1 success criteria:

- The `Selected operators for ascend910_93:` line lists all eight, including
  `grouped_matmul_layered`.
- `[100%] Built target GroupedMatmulLayered_ascend910_93`, `0 errors`.
- The tiling registration lands in `libcust_optiling.so`.
- The sibling and the four `afd_async_*` operators still build (the
  shared-header change must not regress them).

Gate 2 success criteria: the extension links and imports
(`afd_plugin._C_ascend`), and the ops load through
`ensure_afd_ascend_ops_loaded()`. A missing type alias or a signature mismatch
surfaces here and nowhere earlier.

For a plugin-owned operator, a clean `build_aclnn.sh` is **not** evidence that
the change compiles. State which gate each piece of build evidence came from.

Line endings matter when diffing against upstream: **upstream files are CRLF,
vendored files are LF.** Always use `diff --strip-trailing-cr` / `diff -rq
--strip-trailing-cr`, or a one-line change reads as a whole-file rewrite (the
tiling-key header reports 515+/533− raw, but its true delta is one line).

## Testing

### CPU-safe (no Ascend toolchain; CI-runnable)

```bash
pytest tests/unit/package/
pytest tests/unit/compat/npu/test_gmm_layered.py
```

`tests/unit/package/` covers registry ordering, `select_ops.py` resolution, build
staging and the recursive `op_host` collection. The second file's Meta tests run
wherever the extension is importable and skip otherwise; they validate the
dispatcher registrations (`PrivateUse1` / `AutogradPrivateUse1` / `Meta`), the
Meta shape and dtype contract, multi-layer list handling, and the
argument-validation rejections. Torch imports live inside the fixture so CPU
collection stays independent of torch/torch_npu.

### Device (910C + AFD CANN run package)

```bash
SOC_VERSION=910c AFD_RUN_ASCEND_OP_RUNTIME=1 \
    pytest tests/unit/compat/npu/test_gmm_layered.py
```

`test_gmm_layered_runtime_multi_layer` is the only test that reaches the op_api
validation and the kernel. It runs the A8W4 path for two layers and checks output
geometry — enough to prove the lists are accepted and the device path executes,
but **it asserts no numerical correctness** (all tensors are zeros). Numerical
evidence has to come from the check below.

Recorded device evidence for the operator, on Ascend910_9382 / CANN-HCCL 9.0.1 /
Python 3.12.13 / torch 2.10.0+cpu / torch-npu 2.10.0.post2 (from the `f679b77`
re-verification, which ran the gate above with `AFD_RUN_ASCEND_OP_RUNTIME=1`):
**62 passed, no skips across the CAM and both layered-GMM suites**, i.e. the
two-layer A8W4 invocation really launched. That establishes invocation, output
geometry, and correct kernel-config resolution on 9.0.1 — not numerical accuracy.

The canonical invocation command, including the surrounding suites so a binding
change is exercised against the sibling too:

```bash
SOC_VERSION=910c AFD_RUN_ASCEND_OP_RUNTIME=1 python -m pytest -o addopts= -q -ra \
  tests/unit/compat/npu/test_async_cam_harness.py \
  tests/unit/compat/npu/test_async_cam_performance.py \
  tests/unit/compat/npu/test_async_cam_golden.py \
  tests/unit/compat/npu/test_cam_async_ops.py \
  tests/unit/compat/npu/test_gmm_layered.py \
  tests/unit/compat/npu/test_gmm_swiglu_quant_v2_layered.py
```

Upstream routinely passes `-o addopts=` to drop the repository's default
`addopts = "-q"` (`pyproject.toml:76`) so the explicit flags fully control
verbosity. It is harmless either way — there is no conflicting default option to
override — but keep it when copying a command verbatim from a PR description.

### Accuracy / numerical validation

The numerical check is **not in this repository** — it lives in the cam_async
tree as `test/test_gmm_multi_layer.py` (288 lines). To use it here, copy it and
change the operator lookup from `torch.ops.umdk_cam_op_lib.gmm_layered` to
`torch.ops.afd_ascend.grouped_matmul_layered`; the argument names and shapes
already match the AFD schema. The rest of this section describes what it does, so
the check can be re-derived if the file is lost.

**Golden reference.** The CANN built-in non-layered `torch_npu.npu_grouped_matmul`
run on the *selected layer's* weight/scale/bias. The layered op must reproduce it
for **every** `layer_index`. Because a wrong layer selection shows up as a large
deviation immediately, the per-layer comparison is the primary accuracy gate.

**Two phases, run in order.** Phase 1 is the migrated-operator precision check
with the layer-selection path disabled; if it fails, nothing downstream can be
trusted.

```bash
LAYERS=1 python3 test_gmm_multi_layer.py   # phase 1: single layer, no selection
LAYERS=4 python3 test_gmm_multi_layer.py   # phase 2: per-layer sweep + cross-layer
```

Phase 2 additionally asserts a **cross-layer difference**: `mean |out[0] - out[1]|`
must exceed `1e-3`. That is the guard against a silently dead selection path — a
dead path would make all layers identical and still "pass" a per-layer comparison
if the reference were computed the same way. This is the check that catches
defect 1 above, so do not drop it.

**Shape constraints** (enforced by the script, driven by the NZ layout):
`k % 16 == 0`, `n % 64 == 0`, `m >= e`.

**Input generation** (`gen_tensors`), all random per run:

| Tensor | Shape / dtype | Notes |
| --- | --- | --- |
| `x` | `int8 [m, k]` | per-token quantized activation |
| `per_token_scale` | `float32 [m]` | in `[5e-3, 2e-2]` |
| `expert_tokens` | `int64 [e]` | positive counts summing to `m`, shuffled — `group_list_type=1` |
| `all_weight[l]` | `int32 [e, k, n/8]`, FRACTAL_NZ | 8 int4 nibbles per word, nibbles in `[-8, 7]`, packed little-endian |
| `all_scale[l]` | `int64 [e, n]` | two int32 halves of small float32 values; the aclnn reinterprets to `uint64` |
| `all_bias[l]` | `float32 [e, n]` | in `[-0.5, 0.5]` |

Both the layered op and the golden consume the **identical** tensors, so the
distribution only needs to keep the dequantified arithmetic well conditioned.

**Two traps encoded in the script — preserve both:**

1. **`x` must be cloned per call.** The A8W4 MSD pre-quantization pass rewrites
   the activation in place (the kernel binds the same buffer as both input and
   output). Without a private copy, whichever op runs first changes what the
   second sees, and each further layer re-quantizes the previous result.
2. **The golden needs an `unsqueeze(1)` on the scale.** The built-in A8W4 tiling
   requires a 3-D per-group scale `[e, g, n]` (it reads `dim(1)` as the K-group
   count unconditionally). Our per-channel `[e, n]` is memory-identical to
   `[e, 1, n]`, so the `unsqueeze(1)` view feeds the built-in without changing the
   payload.
3. **The generator must be seeded explicitly.** A freshly constructed
   `torch.Generator` does *not* inherit `torch.manual_seed`; without the explicit
   seed every run produces byte-identical tensors and `--seed`/`--loops` silently
   do nothing. The script also fingerprints `x` in the log so a silently-constant
   generator is visible.

**Tolerance.** `torch.allclose(rtol=2e-2, atol=2e-2)` on float32 casts, plus a
finite check (NaN/Inf fails) and a shape check. Reported statistics: max/mean/p99
absolute difference and max/mean relative difference.

**Invocation matrix.** Sweep with fresh random data per round:

```bash
python3 test_gmm_multi_layer.py --loops 5                 # fresh data per loop
LAYERS=4 M=256 K=1024 N=2048 E=8 SEED=42 python3 test_gmm_multi_layer.py
DEVICE=npu:7 python3 test_gmm_multi_layer.py             # device selection
```

Run with `torch_npu.npu.config.allow_internal_format = True` (the script sets it).

### What is not covered

- The `arch35` / ascend950 path on real hardware: it is vendored and compiled
  for `ascend950`, but no device test has been run against it from this
  repository yet. Only compilation and the A8W8 static gate are covered.
- The Meta tests prove registration, shapes, and the allocation contract; they
  perform no communication and no arithmetic.
- The device unit test proves execution, geometry, and kernel-config resolution,
  not numerics. Its inputs are all zeros, so no arithmetic result is asserted.
- No harness covers these operators. `tests/npu/` (added in `8f506a9`, #374)
  covers the routed async CAM operators only; `rg grouped_matmul tests/npu/`
  returns nothing. Numerical evidence remains the cam_async-derived check above,
  which is not in this repository.
- Connector/model wiring that would consume the operator is out of scope; the
  operator is not yet on a model path.
- Issue #381 (`group_list` as a device tensor) is open upstream and **not**
  implemented here. The current binding performs a host round trip by design;
  do not present the operator as free of group-list-induced synchronization.

## Latent hazards in unreachable paths

While vendoring, upstream code whose inputs this port dropped was stubbed rather
than deleted, so it still compiles. Two kinds of stub exist and they are **not**
equivalent:

**Correct stubs** — the null pointer is null-checked before use:

- `GetWithOffset` (`grouped_matmul_tiling.cpp:1687-1697`) sets `offset = nullptr`
  and guards with `if (offset != nullptr)`. This one is fine and returns 0, which
  is the right answer for a symmetric-only port.

**Fragile stubs** — a null pointer is dereferenced:

- `GetPerGroupNum` (`:526-530`) sets `antiquantScale = nullptr`, then calls
  `OP_CHECK_NULL_WITH_CONTEXT` (which does **not** return) and dereferences it.
- `CheckA16W4MsdEnable` (`:1651-1652`) and `A16W4MsdTiling` (`:1705`, `:1727`)
  dereference `antiquantScaleDropped` / `antiquantScaleDropped2`.
- `CheckGroupedMatmulAntiQuantForDtype` (`grouped_matmul_infershape.cpp:1484-1492`)
  and `CheckFunctionQuantParams` (`:1477-1479`) pass the sentinel index `99UL`
  into `GetDynamicInputDataType` / `CheckTensorListDataType`.

**These are currently unreachable**, and the reason is the dtype trim, not the
guard: the op_def declares `x` as `INT8` only, so

- `A16W4MsdTiling` is gated on `xDType == DT_BF16` (`:2154`) — never true;
- `GetPerGroupNum` is called only under `xDType_ == DT_BF16 || DT_FLOAT16`
  (`:707`) — never true;
- the antiquant infershape path is gated on `xDType == BF16/FP16` — never true.

That argument was established by reading the dispatch conditions, **not by
executing those paths**. Treat it as a standing constraint rather than a settled
fact: **if a future baseline widens the op_def dtype columns to reintroduce A16
antiquant or asymmetric quant, these four sites become live null dereferences**
and must be given real inputs (or real guards) first. Re-check them as part of
any dtype-table change.

## Re-applying on a new baseline

The procedure, in order. Steps 1–3 are re-derivation; 4–8 are mechanical; 9–11
are the gates.

1. **Locate the new upstream source.** The operator is
   `<ops-transformer>/gmm/grouped_matmul` (confirm by locating
   `OP_ADD(GroupedMatmul)` in `op_host/grouped_matmul_def.cpp`). Do **not**
   confuse it with `gmm/grouped_matmul_swiglu_quant_v2`, which is a different
   operator backing the sibling.
2. **Census the upstream tree** before copying: `find <op>/op_host <op>/op_kernel
   -type f`. Compare against the file table in
   [Ascend 950 (A5 / DAV_3510) support](#ascend-950-a5--dav_3510-support) and
   note anything new it does not mention. The A5 entry, both `arch35` trees and
   the `*950_checker.*` files **are** vendored now; re-derive the layered deltas
   listed in that section rather than re-adding the upstream forms.
3. **Copy, excluding** `grouped_matmul_apt.cpp` (replaced by the layered A5
   entry), `fallback_grouped_matmul.cpp`, the
   `aclnn_grouped_matmul_v{2,3,4,5}.h` / `_weight_nz.h` headers, `status.hpp`,
   `CMakeLists.txt`, `tests/`, `docs/`, `examples/`. Normalize CRLF → LF.
   Relocate `op_host/op_api/` → `op_api/`.
4. **Apply the rename table** (section [The rename table](#the-rename-table)) —
   directory, OpDef class, `OP_ADD`, both `IMPL_OP_*` registrations,
   `OP_TYPE_REGISTER`, `REGISTER_TILING_DATA_CLASS`, the ACLNN symbol, the kernel
   entry, the l0op function, and the three input names plus their propagated
   `DTYPE_*` / `ORIG_DTYPE_*` / `FORMAT_*` macros. Decide explicitly whether to
   also rename `ASCENDC_TPL_ARGS_DECL(GroupedMatmul, ...)` (see the note in that
   section) instead of leaving it silently inconsistent.
5. **Add `layer_index`** at slot 4, drop the three unused inputs, and rewrite
   both index tables (`grouped_matmul_host_util.h`,
   `grouped_matmul_infershape_common_util.h`). Keep the dropped indices as
   sentinels if the retained checkers still reference them. Trim the op_def dtype
   columns to the A8W4/A8W8 set and re-read
   [Latent hazards](#latent-hazards-in-unreachable-paths) before deciding the trim
   is safe.
6. **Thread the layer index**: the single `LayeredReadLayerIndex` read at the
   kernel entry; `curLayer` through every initializer; `GetLayerTensorAddr` at all
   20 sites replacing upstream's element-0 addressing; `layerNum` on **both**
   tiling structs and **both** tiling paths; `layerNum` on `GMAddrParams` and the
   autotiling/fixaxismove members.
7. **Re-apply the added host validation**: the `layer_index` checks and list-length
   equality, the 3-dim `all_weight[0]` requirement, the expert-count consistency
   loop, the `CheckCaseSplitM` collapse, the `isSingleWeight_ = true` force, the
   2-D per-channel scale branches, and the op_api list-shrink shim. Then re-stub
   or delete whatever `DAV_3510` dispatch the new baseline has, and confirm the
   `arch35` tiling templates still register under the upstream name (argument 3
   above) — if upstream renamed them, that argument evaporates and the block has
   to be reconsidered rather than commented out.
8. **Re-apply the shared header change** and the sibling's 13-file adaptation,
   then the AFD integration files (registry, pybind CMakeLists, README, build
   test) from [AFD integration](#afd-integration-outside-the-operator-directory).
   Re-read [Post-merge follow-ups](#post-merge-follow-ups-read-before-re-applying)
   and fold in whatever of it the new baseline does not already carry — in
   particular the `tensor_list` alias, which is easy to lose when the binding is
   re-derived from upstream.
9. **Build for 910C through both gates** in [Build](#build), not just
   `build_aclnn.sh`; check that the sibling and `afd_async_*` operators still
   build too, and that the extension links and imports.
10. **Run the CPU-safe tests**, then the device gate with
    `AFD_RUN_ASCEND_OP_RUNTIME=1`.
11. **Run the numerical validation** (both phases, then the sweep). Phase 1 must
    pass before phase 2 means anything, and the phase-2 cross-layer difference
    check must not be skipped. Record the exact CANN, torch-npu, PyTorch, driver
    and device versions with the results.

### Diffing a re-apply against this port

The two operators need different diff strategies, because they have different
provenance (see [Provenance](#provenance-record-this-whenever-it-changes)).

**`grouped_matmul_layered`** has no logic differences from the cam_async copy
apart from the `DAV_3510` removal, so a full-tree diff is a usable signal — but
expect the categories in [Provenance](#provenance-record-this-whenever-it-changes),
not an empty diff:

```bash
diff -rq --strip-trailing-cr <new-vendored> \
  <cam_async>/src/comm_operator/ascend_kernels/grouped_matmul_layered
```

Expected: 37 "Only in cam_async" entries (the `arch35` trees and `status.hpp`),
10 whitespace/comment-only "differ" entries, and exactly 4 with code deltas — one
intentional (`grouped_matmul_tiling.cpp`, the dispatch block) and three
string-typo-only. Any entry outside those sets is new and must be explained. To
separate formatting from logic, strip comments and whitespace before comparing
rather than relying on `diff -w`, which still reports reformatted lines.

**`grouped_matmul_swiglu_quant_v2_layered`** is *not* a clean vendor of either
tree, so a full-tree diff will be noisy and must not be used as a gate. Scope the
diff to the 13 files in [Sibling adaptation](#sibling-adaptation-13-files); a
delta in any of them is a real finding, a delta elsewhere is pre-existing.

Always pass `--strip-trailing-cr`: upstream files are CRLF and vendored files are
LF, so without it a one-line change reads as a whole-file rewrite (the tiling-key
header reports 515+/533− raw, while its true delta is one line).

## Checklist for a re-apply

- [ ] Upstream revision and cam_async revision recorded (table in
      [Provenance](#provenance-record-this-whenever-it-changes)).
- [ ] Rename applied at every level, including the propagated `DTYPE_*` macros.
- [ ] `layer_index` at slot 4; both index tables rewritten.
- [ ] `layerNum` on both tiling structs **and** both tiling paths.
- [ ] All 20 `GetLayerTensorAddr` sites carry the bound.
- [ ] Both defects' fixes present and their comments retained.
- [ ] `arch35` reasoning re-validated, not assumed.
- [ ] Latent-hazard stubs re-checked against the new dtype table.
- [ ] Registry, pybind CMakeLists, README, build test updated.
- [ ] **Binding declares `tensor_list`** (`using tensor_list =
      std::vector<at::Tensor>;`) — see the `f679b77` note; the original commit
      omitted it.
- [ ] **Gate 2 build run**, not just `build_aclnn.sh`; a green operator build does
      not prove the binding compiles.
- [ ] Op type still free of digits, or the JSON config name registered
      (`50fef29` precedent); check the installed JSON filename against the
      runtime's expectation if the CANN version or op type moved.
- [ ] Sibling 13-file adaptation re-applied; sibling + `afd_async_*` still build.
- [ ] `pytest tests/unit/package/` and the Meta tests pass.
- [ ] Device gate run; numerical validation phase 1 then phase 2 (with the
      cross-layer difference check) pass, with versions recorded.
- [ ] Issue #381 status re-checked before claiming the binding contract is final:
      if the device `group_list` tensor change has landed upstream, this document's
      `group_list` description and the `alloc_outputs` reasoning are stale.
