/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_layered_apt.cpp
 * \brief Ascend 950 (A5, DAV_3510 / __CCE_AICORE__ == 310) kernel entry for the
 *        layered grouped matmul. Selected by `opFile.value` in the ascend950
 *        AICore config; the 910_93 entry stays in grouped_matmul_layered.cpp.
 *
 * This is the AFD layered counterpart of the upstream ops-transformer
 * `grouped_matmul_apt.cpp`. Two deltas from upstream:
 *
 *  1. The entry takes `layerIndex` (AFD's layered contract) between `scale` and
 *     `groupList`, matching the 910_93 entry's input order in
 *     `op_host/grouped_matmul_def.cpp`. The selected layer is resolved once per
 *     core and handed to the arch35 kernels, which use it to address the
 *     per-layer element of the `all_*` TensorList descriptors.
 *  2. Only the `V310_GMM_QUANT` (A8W8) branch is carried. AFD's op_def drops the
 *     `offset` / `antiquant_scale` / `antiquant_offset` inputs, so the upstream
 *     `V310_GMM_ANTI_QUANT` (A8W4/A16W4) and no-quant branches cannot be driven
 *     through the registered interface; enabling them requires the A5
 *     antiquant_scale representation first (see docs/npu/GROUPED_MATMUL_LAYERED.md).
 *
 * The `V310_GMM_*` macros are defined in `grouped_matmul_utils.h` under the
 * `__CCE_AICORE__ == 310` guard, so this file compiles to nothing on 910_93.
 */
#include "grouped_matmul_utils.h"
#include "arch35/grouped_matmul_tiling_data_apt.h"

using GMMWeightQuantTilingData = GroupedMatmulTilingData::GMMWeightQuantTilingData;
using GMMNoQuantTilingData = GroupedMatmulTilingData::GMMNoQuantTilingData;
using GMMQuantTilingData = GroupedMatmulTilingData::GMMQuantTilingData;

#include "kernel_operator.h"
#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 310) && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113)

#include "arch35/quant_adaptive_sliding_window_templates/gqmm_tiling_key.h"
#if defined(V310_GMM_QUANT_MX) || defined(V310_GMM_QUANT_CUBE) || defined(V310_GMM_QUANT_PERTENSOR_CUBE)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_cube_on_the_fly.h"
#endif
#if defined(V310_GMM_QUANT_MX) || defined(V310_GMM_QUANT_PERTENSOR_CUBE)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_init_output.h"
#endif
#if defined(V310_GMM_QUANT_MIX)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_mix_online_dynamic.h"
#endif

using namespace AscendC;
using namespace matmul;
using namespace GROUPED_MATMUL;

#ifndef FORMAT_FRACTAL_NZ
    #define FORMAT_FRACTAL_NZ
#endif

namespace {
#if defined(FORMAT_ALL_WEIGHT) && FORMAT_ALL_WEIGHT == FORMAT_FRACTAL_NZ
constexpr CubeFormat wFormat = CubeFormat::NZ;
#else
constexpr CubeFormat wFormat = CubeFormat::ND;
#endif
}

// Same IMPL helpers as the upstream A5 entry, with `layerIndex` threaded
// through to the kernel so it can resolve the per-layer descriptor element.
#define GMM_QUANT_IMPL_CLASS(transposeX1, transposeX2, templateClass)                                                  \
    do {                                                                                                               \
        templateClass<DTYPE_X, DTYPE_ALL_WEIGHT, DTYPE_ALL_BIAS, DTYPE_ALL_SCALE, DTYPE_Y, wFormat,                    \
                      transposeX1, transposeX2>                                                                        \
            op;                                                                                                        \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling);                           \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling);                               \
        GET_TILING_DATA_MEMBER_ADDR(GMMQuantTilingData, gmmArray, gmmArrayAddr_, tiling);                              \
        op.Init(x, weight, bias, scale, layerIndex, groupList, perTokenScale, y, user1, &gmmQuantParams_,              \
                &mmTilingData_, gmmArrayAddr_, &tPipe);                                                                \
        op.Process();                                                                                                  \
    } while (0)

#define GMM_QUANT_MIX_IMPL_CLASS(transposeX1, transposeX2, templateClass)                                              \
    do {                                                                                                               \
        templateClass<DTYPE_X, DTYPE_ALL_WEIGHT, DTYPE_ALL_BIAS, DTYPE_ALL_SCALE, float, DTYPE_Y, wFormat,             \
                      transposeX1, transposeX2, DTYPE_L0C_LOCAL>                                                       \
            op;                                                                                                        \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling);                           \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling);                               \
        GET_TILING_DATA_MEMBER_ADDR(GMMQuantTilingData, gmmArray, gmmArrayAddr_, tiling);                              \
        op.Init(x, weight, bias, scale, layerIndex, groupList, perTokenScale, y, user1, &gmmQuantParams_,              \
                &mmTilingData_, gmmArrayAddr_, &tPipe);                                                                \
        op.Process();                                                                                                  \
    } while (0)

#if defined(V310_GMM_QUANT)
template <int8_t QUANT_B_TRANS, int8_t QUANT_A_TRANS, int8_t KERNEL_TYPE>
#else
template <int8_t NO_QUANT_B_TRANS, int8_t NO_QUANT_A_TRANS>
#endif
__global__ __aicore__ void grouped_matmul_layered(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                                                  GM_ADDR layerIndex, GM_ADDR groupList, GM_ADDR perTokenScale,
                                                  GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe tPipe;
    AscendCUtils::SetOverflow(1);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    GM_ADDR user1 = GetUserWorkspace(workspace);
    (void)user1;
    (void)layerIndex;

#ifndef __CCE_KT_TEST__
#if defined(V310_GMM_QUANT) // Quant: A8W8
    REGISTER_TILING_DEFAULT(GMMQuantTilingData);
#if defined(V310_GMM_QUANT_MX) // mxfp
    if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_NO_TRANS
        && KERNEL_TYPE == GMM_DEQUANT_FIXP) {
        GMM_QUANT_IMPL_CLASS(false, false, GmmASWKernel);
    } else if constexpr (QUANT_B_TRANS == GMM_TRANS && QUANT_A_TRANS == GMM_NO_TRANS
        && KERNEL_TYPE == GMM_DEQUANT_FIXP) {
        GMM_QUANT_IMPL_CLASS(false, true, GmmASWKernel);
    }
#endif
#if defined(V310_GMM_QUANT_CUBE) || defined(V310_GMM_QUANT_PERTENSOR_CUBE) // scale64/perTensor
    if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_NO_TRANS
        && KERNEL_TYPE == GMM_DEQUANT_FIXP) {
        GMM_QUANT_IMPL_CLASS(false, false, GmmASWKernel);
    } else if constexpr (QUANT_B_TRANS == GMM_TRANS && QUANT_A_TRANS == GMM_NO_TRANS
        && KERNEL_TYPE == GMM_DEQUANT_FIXP) {
        GMM_QUANT_IMPL_CLASS(false, true, GmmASWKernel);
    }
#endif
#if defined(V310_GMM_QUANT_MIX) // perToken/SPLIT_K/scale bf16/fp32
    if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_NO_TRANS
        && KERNEL_TYPE == GMM_DEQUANT_VECTOR) {
        GMM_QUANT_MIX_IMPL_CLASS(false, false, GQmmMixRegbaseKernel);
    } else if constexpr (QUANT_B_TRANS == GMM_TRANS && QUANT_A_TRANS == GMM_NO_TRANS
        && KERNEL_TYPE == GMM_DEQUANT_VECTOR) {
        GMM_QUANT_MIX_IMPL_CLASS(false, true, GQmmMixRegbaseKernel);
    } else if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_TRANS
        && KERNEL_TYPE == GMM_DEQUANT_VECTOR) {
        GMM_QUANT_MIX_IMPL_CLASS(true, false, GQmmMixRegbaseKernel);
    }
#endif
#else
    // V310_GMM_ANTI_QUANT and the no-quant branch are not carried: the AFD
    // op_def does not register the inputs they require on ascend950.
    REGISTER_TILING_DEFAULT(GMMNoQuantTilingData);
#endif
#endif
}

#endif // __CCE_AICORE__ == 310 (Ascend 950 / DAV_3510)
