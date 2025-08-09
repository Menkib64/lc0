/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2024 The LCZero Authors
  Copyright (C) 2023 Intel Corporation

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
   
  SPDX-License-Identifier:GNU General Public License v3.0 or later
*/

#include <sycl/sycl.hpp>
#include "dpct/dpct.hpp"
#include "sycl_common.h"
#include "neural/backends/shared/activation.h"

// Allow building on an old architecture.
#if DPCT_COMPATIBILITY_TEMP < 530
#define SKIP_FP16_BITS 1
#endif
#include "winograd_helper.h"

namespace lczero {
namespace sycldnn_backend {

/////////////////////////////////////////////////////////////////////////////
//          fp16-specific kernels used by certain layers                   //
/////////////////////////////////////////////////////////////////////////////

// SE layer implementation using single fused kernel.

// Get board for this thread from shared memory.
// We are just using shared memory to store local thread data in this kernel to
// help reduce some register pressure and spills to local memory.
#define BOARD(y, x) shboard[(y)*8 + (x)]

// input is in transformed space (HWNC layout) --- output of GEMM
// output is also in transformed space (HWNC layout) --- input to GEMM (for
// next layer)
// 'C' threads per block
// 'N' blocks
// Every thread generates an entire board/plane (8x8 elements).
template <ActivationFunction activation, bool use_bias, bool use_skip>
/*
DPCT1110:35: The total declared local variable size in device function
OutputInputTransformKernel_fp16_shmem_board exceeds 128 bytes and may cause high
register pressure. Consult with your hardware vendor to find the total register
size available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void OutputInputTransformKernel_fp16_shmem_board(
    int N, int C, int se_K, sycl::half* output, const sycl::half* input,
    sycl::half* skip, const sycl::half* bias, const sycl::half* w1,
    const sycl::half* b1, const sycl::half* w2, const sycl::half* b2,
    const sycl::nd_item<3>& item_ct1, uint8_t* dpct_local, float* shared_data,
    sycl::local_accessor<float, 2> shared_sums) {
#if DPCT_COMPATIBILITY_TEMP >= 530
  int k = item_ct1.get_local_id(2);
  int n = item_ct1.get_group(2);

  auto _sboard = (sycl::half*)dpct_local;
  sycl::half* shboard = &_sboard[k * 72];  // 72 instead of 64 to reduce shared
                                           // memory bank conflicts.
  sycl::half b = bias[k];

#pragma unroll
  for (int hStart = 0; hStart < 8; hStart += 4)
#pragma unroll
    for (int wStart = 0; wStart < 8; wStart += 4) {
      //  i) read to per thread registers (for doing output transform)
      int shln = n * 4 + (hStart / 4) * 2 + (wStart / 4);
      sycl::half outElTransformed[6][6];
#pragma unroll
      for (int y = 0; y < 6; y++)
#pragma unroll
        for (int x = 0; x < 6; x++)
          outElTransformed[y][x] = input[TEMP_INDEX_HWNC(y, x, shln, k)];

      // ii) transform it
      sycl::half outEl[4][4];
      OutputTransform4x4(&outEl[0][0], &outElTransformed[0][0]);

#pragma unroll
      for (int y = 0; y < 4; y++)
        copyAs<sycl::uint2>(&BOARD(hStart + y, wStart), &outEl[y][0]);
    }

  // Add bias, and compute the average for SE.
  float S = 0;
  float B = 0;

#pragma unroll
  for (int y = 0; y < 8; y++) {
    sycl::half boardRow[8];
    copyAs<sycl::uint4>(&boardRow, &BOARD(y, 0));
#pragma unroll
    for (int x = 0; x < 8; x++) {
      if (use_bias) boardRow[x] += b;
      S += (float)boardRow[x];
    }
    if (use_bias) copyAs<sycl::uint4>(&BOARD(y, 0), &boardRow);
  }

  float avg = S / 64;
  shared_data[k] = avg;

  int lane = k & 0x1F;
  int warp = k >> 5;
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // First fully-connected layer for SE

  // As se_K << C, we want to loop over se_K instead of C
  // even if it means taking the sum across threads

    // per-warp sums

  for (int i = 0; i < se_K; i++) {
    float val = shared_data[k] * float(readw1(k, i));
    val = warpReduce(val, item_ct1);
    if (lane == 0) shared_sums[warp][i] = val;
  }
  item_ct1.barrier(sycl::access::fence_space::local_space);
  if (k < se_K) {
    S = 0;
    for (int i = 0; i < C / 32; i++) S += shared_sums[i][k];

    S += (float)b1[k];
    S = activate(S, activation);
    shared_data[k] = S;
  }

  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Second fully-connected layer for SE
  S = 0;
  for (int i = 0; i < se_K; i++) {
    float val = shared_data[i];
    S += val * float(readw2(i, k));
    B += val * float(readw2(i, k + C));
  }
  S += (float)b2[k];
  B += (float)b2[k + C];

  // Sigmoid (only on the scale part).
  S = 1.0f / (1.0f + sycl::exp(-S));

  // Scale/bias, add skip connection, perform activation, and write to output.
  for (int h = 0; h < 8; h++) {
    sycl::half boardRow[8];
    copyAs<sycl::uint4>(&boardRow[0], &BOARD(h, 0));

#pragma unroll
    for (int w = 0; w < 8; w++) {
      boardRow[w] = (sycl::half)(float(boardRow[w]) * S + B);
    }

    // residual add
    if (use_skip) {
      sycl::half skipInp[8];
      copyAs<sycl::uint4>(&skipInp[0], &skip[INDEX_NHCW(n, k, h, 0)]);
#pragma unroll
      for (int w = 0; w < 8; w++) boardRow[w] += skipInp[w];
    }

    if (activation != ACTIVATION_NONE) {
#pragma unroll
      for (int w = 0; w < 8; w++)
        boardRow[w] = (sycl::half)activate((float)boardRow[w], activation);
    }

    // write un-transformed output to 'skip' if required
    if (use_skip) {
      copyAs<sycl::uint4>(&skip[INDEX_NHCW(n, k, h, 0)], &boardRow[0]);
    }

    copyAs<sycl::uint4>(&BOARD(h, 0), &boardRow);
  }

  // Perform input transform.

  int c = k;
  // top-left
  {
    sycl::half inEl[6][6] = {};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i + 1][j + 1] = BOARD(i, j);

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 0, c)] = inEl[y][x];
  }

  // top-right
  {
    sycl::half inEl[6][6] = {};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i + 1][j] = BOARD(i, j + 3);

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 1, c)] = inEl[y][x];
  }

  // bottom-left
  {
    sycl::half inEl[6][6] = {};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i][j + 1] = BOARD(i + 3, j);

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 2, c)] = inEl[y][x];
  }

  // bottom-right
  {
    sycl::half inEl[6][6] = {};

#pragma unroll
    for (int i = 0; i < 5; i++)
#pragma unroll
      for (int j = 0; j < 5; j++) inEl[i][j] = BOARD(i + 3, j + 3);

    InputTransform4x4(&inEl[0][0], &inEl[0][0]);

#pragma unroll
    for (int y = 0; y < 6; y++)
#pragma unroll
      for (int x = 0; x < 6; x++)
        output[TEMP_INDEX_HWNC(y, x, n * 4 + 3, c)] = inEl[y][x];
  }
#endif
}

template <typename T = sycl::half, bool use_se, ActivationFunction activation,
          bool use_bias, bool use_skip>
void OutputInputTransform(int N, int C, int se_K, T* output, const T* input,
                          const T* skip, const T* bias, const T* w1,
                          const T* b1, const T* w2, const T* b2, sycl::queue &sycl_queue) {
  // Each thread processes entire chess board.
  if (use_se == false) {
    sycl::range<3> grid_dim(1, N, DivUp(C, kOpInpTransformBlockSize));
    {
      
      sycl_queue.parallel_for(
          sycl::nd_range<3>(
              grid_dim * sycl::range<3>(1, 1, kOpInpTransformBlockSize),
              sycl::range<3>(1, 1, kOpInpTransformBlockSize)),
          [=](sycl::nd_item<3> item_ct1) {
            OutputTransform_relu_InputTransform_kernel<sycl::half, activation,
                                                       use_bias, use_skip>(
                N, C, output, input, (sycl::half*)skip, bias, item_ct1);
          });
    }
  } else if (C > kMaxResBlockFusingChannels) {
    // Use special kernel with reduced register pressure - only works on Ampere,
    // and only for fp16.
    if (C <= kMaxResBlockFusingSeKFp16Ampere) {
      //cudaFuncSetAttribute(
        //  OutputInputTransformKernel_fp16_shmem_board<activation, use_bias,
                                                    //  use_skip>,
         // cudaFuncAttributeMaxDynamicSharedMemorySize,
         // 72 * C * sizeof(sycl::half));
      /*
      DPCT1049:36: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      {
        
        
        sycl_queue.submit([&](sycl::handler& cgh) {
          /*
          DPCT1083:124: The size of local memory in the migrated code may be
          different from the original code. Check that the allocated memory
          size in the migrated code is correct.
          */
          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(72 * C * sizeof(sycl::half)), cgh);
          /*
          DPCT1101:125: 'kMaxResBlockFusingSeKFp16Ampere' expression was
          replaced with a value. Modify the code to use the original
          expression, provided in comments, if it is correct.
          */
          sycl::local_accessor<float, 1> shared_data_acc_ct1(
              sycl::range<1>(512 /*kMaxResBlockFusingSeKFp16Ampere*/), cgh);
          /*
          DPCT1101:126: 'kMaxResBlockFusingSeKFp16Ampere / 32' expression was
          replaced with a value. Modify the code to use the original
          expression, provided in comments, if it is correct.
          */
          /*
          DPCT1101:127: 'kMaxResBlockFusingSeK' expression was replaced with a
          value. Modify the code to use the original expression, provided in
          comments, if it is correct.
          */
          sycl::local_accessor<float, 2> shared_sums_acc_ct1(
              sycl::range<2>(16 /*kMaxResBlockFusingSeKFp16Ampere / 32*/,
                             128 /*kMaxResBlockFusingSeK*/),
              cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(
                  sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
                  sycl::range<3>(1, 1, C)),
              [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                  SYCL_SUB_GROUP_SIZE)]] {
                OutputInputTransformKernel_fp16_shmem_board<activation,
                                                            use_bias, use_skip>(
                    N, C, se_K, (sycl::half*)output, (const sycl::half*)input,
                    (sycl::half*)skip, (sycl::half*)bias, (sycl::half*)w1,
                    (sycl::half*)b1, (sycl::half*)w2, (sycl::half*)b2, item_ct1,
                    dpct_local_acc_ct1.get_pointer(),
                    shared_data_acc_ct1.get_pointer(), shared_sums_acc_ct1);
              });
        });
      }
    } else {
      throw Exception(
          "res block fusing opt not supported for the given data type and no "
          "of filters\n");
    }
  } else {
    /*
    DPCT1049:37: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    
    sycl_queue.submit([&](sycl::handler& cgh) {
      /*
      DPCT1101:128: 'kMaxResBlockFusingChannels' expression was replaced
      with a value. Modify the code to use the original expression, provided
      in comments, if it is correct.
      */
      sycl::local_accessor<float, 1> shared_data_acc_ct1(
          sycl::range<1>(384 /*kMaxResBlockFusingChannels*/), cgh);
      /*
      DPCT1101:129: 'kMaxResBlockFusingChannels / 32' expression was
      replaced with a value. Modify the code to use the original expression,
      provided in comments, if it is correct.
      */
      /*
      DPCT1101:130: 'kMaxResBlockFusingSeK' expression was replaced with a
      value. Modify the code to use the original expression, provided in
      comments, if it is correct.
      */
      sycl::local_accessor<float, 2> shared_sums_acc_ct1(
          sycl::range<2>(12 /*kMaxResBlockFusingChannels / 32*/,
                         128 /*kMaxResBlockFusingSeK*/),
          cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, C),
                            sycl::range<3>(1, 1, C)),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(SYCL_SUB_GROUP_SIZE)]] {
            OutputTransform_SE_relu_InputTransform_kernel<
                sycl::half, activation, use_bias, use_skip>(
                N, C, se_K, output, input, (sycl::half*)skip, bias, w1, b1, w2,
                b2, item_ct1, shared_data_acc_ct1.get_pointer(),
                shared_sums_acc_ct1);
          });
    });
  }
}

template void FilterTransform<sycl::half>(int N, int C, sycl::half* transformedFilter,
                                    const sycl::half* filter, sycl::queue &sycl_queue);

template void InputTransform<sycl::half, true>(int N, int C, sycl::half* transformed_input,
                                         const sycl::half* input, sycl::queue &sycl_queue);

template void InputTransform<sycl::half, false>(int N, int C, sycl::half* transformed_input,
                                          const sycl::half* input, sycl::queue &sycl_queue);

template void OutputTransform<sycl::half, true, ACTIVATION_RELU, true, true, false,
                              false>(int N, int C, int se_K, sycl::half* output,
                                     const sycl::half* input, const sycl::half* skip,
                                     const sycl::half* bias, const sycl::half* w1,
                                     const sycl::half* b1, const sycl::half* w2,
                                     const sycl::half* b2, sycl::queue &sycl_queue);

template void OutputTransform<sycl::half, false, ACTIVATION_RELU, true, true, false,
                              false>(int N, int C, int se_K, sycl::half* output,
                                     const sycl::half* input, const sycl::half* skip,
                                     const sycl::half* bias, const sycl::half* w1,
                                     const sycl::half* b1, const sycl::half* w2,
                                     const sycl::half* b2, sycl::queue &sycl_queue);

template void OutputTransform<sycl::half, true, ACTIVATION_RELU, true, true, true,
                              false>(int N, int C, int se_K, sycl::half* output,
                                     const sycl::half* input, const sycl::half* skip,
                                     const sycl::half* bias, const sycl::half* w1,
                                     const sycl::half* b1, const sycl::half* w2,
                                     const sycl::half* b2, sycl::queue &sycl_queue);

template void OutputTransform<sycl::half, false, ACTIVATION_RELU, true, true, true,
                              false>(int N, int C, int se_K, sycl::half* output,
                                     const sycl::half* input, const sycl::half* skip,
                                     const sycl::half* bias, const sycl::half* w1,
                                     const sycl::half* b1, const sycl::half* w2,
                                     const sycl::half* b2, sycl::queue &sycl_queue);

template void OutputTransform<sycl::half, false, ACTIVATION_RELU, true, false, false,
                              false>(int N, int C, int se_K, sycl::half* output,
                                     const sycl::half* input, const sycl::half* skip,
                                     const sycl::half* bias, const sycl::half* w1,
                                     const sycl::half* b1, const sycl::half* w2,
                                     const sycl::half* b2, sycl::queue &sycl_queue);

template void OutputTransform<sycl::half, false, ACTIVATION_RELU, true, false, false,
                              true>(int N, int C, int se_K, sycl::half* output,
                                    const sycl::half* input, const sycl::half* skip,
                                    const sycl::half* bias, const sycl::half* w1,
                                    const sycl::half* b1, const sycl::half* w2,
                                    const sycl::half* b2, sycl::queue &sycl_queue);

template void OutputTransform<sycl::half, true, ACTIVATION_RELU, true, true, true,
                              true>(int N, int C, int se_K, sycl::half* output,
                                    const sycl::half* input, const sycl::half* skip,
                                    const sycl::half* bias, const sycl::half* w1,
                                    const sycl::half* b1, const sycl::half* w2,
                                    const sycl::half* b2, sycl::queue &sycl_queue);

template void OutputTransform<sycl::half, true, ACTIVATION_MISH, true, true, false,
                              false>(int N, int C, int se_K, sycl::half* output,
                                     const sycl::half* input, const sycl::half* skip,
                                     const sycl::half* bias, const sycl::half* w1,
                                     const sycl::half* b1, const sycl::half* w2,
                                     const sycl::half* b2, sycl::queue &sycl_queue);

template void OutputTransform<sycl::half, false, ACTIVATION_MISH, true, true, false,
                              false>(int N, int C, int se_K, sycl::half* output,
                                     const sycl::half* input, const sycl::half* skip,
                                     const sycl::half* bias, const sycl::half* w1,
                                     const sycl::half* b1, const sycl::half* w2,
                                     const sycl::half* b2, sycl::queue &sycl_queue);

template void OutputTransform<sycl::half, true, ACTIVATION_MISH, true, true, true,
                              false>(int N, int C, int se_K, sycl::half* output,
                                     const sycl::half* input, const sycl::half* skip,
                                     const sycl::half* bias, const sycl::half* w1,
                                     const sycl::half* b1, const sycl::half* w2,
                                     const sycl::half* b2, sycl::queue &sycl_queue);

template void OutputTransform<sycl::half, false, ACTIVATION_MISH, true, true, true,
                              false>(int N, int C, int se_K, sycl::half* output,
                                     const sycl::half* input, const sycl::half* skip,
                                     const sycl::half* bias, const sycl::half* w1,
                                     const sycl::half* b1, const sycl::half* w2,
                                     const sycl::half* b2, sycl::queue &sycl_queue);

template void OutputTransform<sycl::half, false, ACTIVATION_MISH, true, false, false,
                              false>(int N, int C, int se_K, sycl::half* output,
                                     const sycl::half* input, const sycl::half* skip,
                                     const sycl::half* bias, const sycl::half* w1,
                                     const sycl::half* b1, const sycl::half* w2,
                                     const sycl::half* b2, sycl::queue &sycl_queue);

template void OutputTransform<sycl::half, false, ACTIVATION_MISH, true, false, false,
                              true>(int N, int C, int se_K, sycl::half* output,
                                    const sycl::half* input, const sycl::half* skip,
                                    const sycl::half* bias, const sycl::half* w1,
                                    const sycl::half* b1, const sycl::half* w2,
                                    const sycl::half* b2, sycl::queue &sycl_queue);

template void OutputTransform<sycl::half, true, ACTIVATION_MISH, true, true, true,
                              true>(int N, int C, int se_K, sycl::half* output,
                                    const sycl::half* input, const sycl::half* skip,
                                    const sycl::half* bias, const sycl::half* w1,
                                    const sycl::half* b1, const sycl::half* w2,
                                    const sycl::half* b2, sycl::queue &sycl_queue);

template void OutputTransform<sycl::half, false, ACTIVATION_NONE, true, false, false,
                              false>(int N, int C, int se_K, sycl::half* output,
                                     const sycl::half* input, const sycl::half* skip,
                                     const sycl::half* bias, const sycl::half* w1,
                                     const sycl::half* b1, const sycl::half* w2,
                                     const sycl::half* b2, sycl::queue &sycl_queue);

template void OutputInputTransform<sycl::half, true, ACTIVATION_RELU, true, true>(
    int N, int C, int se_K, sycl::half* output, const sycl::half* input, const sycl::half* skip,
    const sycl::half* bias, const sycl::half* w1, const sycl::half* b1, const sycl::half* w2,
    const sycl::half* b2, sycl::queue &sycl_queue);

template void OutputInputTransform<sycl::half, false, ACTIVATION_RELU, true, true>(
    int N, int C, int se_K, sycl::half* output, const sycl::half* input, const sycl::half* skip,
    const sycl::half* bias, const sycl::half* w1, const sycl::half* b1, const sycl::half* w2,
    const sycl::half* b2, sycl::queue &sycl_queue);

template void OutputInputTransform<sycl::half, false, ACTIVATION_RELU, true, false>(
    int N, int C, int se_K, sycl::half* output, const sycl::half* input, const sycl::half* skip,
    const sycl::half* bias, const sycl::half* w1, const sycl::half* b1, const sycl::half* w2,
    const sycl::half* b2, sycl::queue &sycl_queue);

template void OutputInputTransform<sycl::half, true, ACTIVATION_MISH, true, true>(
    int N, int C, int se_K, sycl::half* output, const sycl::half* input, const sycl::half* skip,
    const sycl::half* bias, const sycl::half* w1, const sycl::half* b1, const sycl::half* w2,
    const sycl::half* b2, sycl::queue &sycl_queue);

template void OutputInputTransform<sycl::half, false, ACTIVATION_MISH, true, true>(
    int N, int C, int se_K, sycl::half* output, const sycl::half* input, const sycl::half* skip,
    const sycl::half* bias, const sycl::half* w1, const sycl::half* b1, const sycl::half* w2,
    const sycl::half* b2, sycl::queue &sycl_queue);

template void OutputInputTransform<sycl::half, false, ACTIVATION_MISH, true, false>(
    int N, int C, int se_K, sycl::half* output, const sycl::half* input, const sycl::half* skip,
    const sycl::half* bias, const sycl::half* w1, const sycl::half* b1, const sycl::half* w2,
    const sycl::half* b2, sycl::queue &sycl_queue);

}  // namespace sycldnn_backend
}  // namespace lczero
