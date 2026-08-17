/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Copyright (c) 2026 Hygon Information Technology Co., Ltd.
 * SPDX-License-Identifier: MIT
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/impl/IVFUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>
#include <faiss/gpu/utils/Tensor.cuh>

//
// This kernel is split into a separate compilation unit to cut down
// on compile time
//

namespace faiss {
namespace gpu {

template <
        typename IndexT,
        int ThreadsPerBlock,
        int NumWarpQ,
        int NumThreadQ,
        bool Dir>
__global__ __launch_bounds__(1024) void pass1SelectLists(
        Tensor<idx_t, 2, true> prefixSumOffsets,
        Tensor<float, 1, true> distance,
        int nprobe,
        int k,
        Tensor<float, 3, true> heapDistances,
        Tensor<idx_t, 3, true> heapIndices) {
    if constexpr ((NumWarpQ == 1 && NumThreadQ == 1) || NumWarpQ >= kWarpSize) {
        constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

        __shared__ float smemK[kNumWarps * NumWarpQ];
        __shared__ IndexT smemV[kNumWarps * NumWarpQ];

        for (IndexT queryId = blockIdx.y; queryId < prefixSumOffsets.getSize(0);
             queryId += gridDim.y) {
            constexpr auto kInit = Dir ? kFloatMin : kFloatMax;
            BlockSelect<
                    float,
                    IndexT,
                    Dir,
                    Comparator<float>,
                    NumWarpQ,
                    NumThreadQ,
                    ThreadsPerBlock>
                    heap(kInit, -1, smemK, smemV, k);

            auto sliceId = blockIdx.x;
            auto numSlices = gridDim.x;

            IndexT sliceSize = (nprobe / numSlices);
            IndexT sliceStart = sliceSize * sliceId;
            IndexT sliceEnd = sliceId == (numSlices - 1)
                    ? nprobe
                    : sliceStart + sliceSize;
            auto offsets = prefixSumOffsets[queryId].data();

            // We ensure that before the array (at offset -1), there is a 0
            // value
            auto start = *(&offsets[sliceStart] - 1);
            auto end = offsets[sliceEnd - 1];

            auto num = end - start;
            auto limit = utils::roundDown(num, (IndexT)kWarpSize);

            IndexT i = threadIdx.x;
            auto distanceStart = distance[start].data();

            // BlockSelect add cannot be used in a warp divergent circumstance;
            // we handle the remainder warp below
            for (; i < limit; i += blockDim.x) {
                heap.add(distanceStart[i], IndexT(start + i));
            }

            // Handle the remainder if any separately (warp is divergent)
            if (i < num) {
                heap.addThreadQ(distanceStart[i], IndexT(start + i));
            }

            // Merge all final results
            heap.reduce();

            // Write out the final k-selected values; they should be all
            // together
            for (int i = threadIdx.x; i < k; i += blockDim.x) {
                heapDistances[queryId][sliceId][i] = smemK[i];
                heapIndices[queryId][sliceId][i] = idx_t(smemV[i]);
            }
        }
    }
}

using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
template <size_t len>
union f32x4_vec {
    floatx4   vec[len/4];
    float     data[len];
};
template <
        typename IndexT,
        int ThreadsPerBlock,
        int NumWarpQ,
        int NumThreadQ,
        bool Dir>
__global__   __launch_bounds__(512) void pass1SelectLists2(
        Tensor<idx_t, 2, true> prefixSumOffsets,
        Tensor<float, 1, true> distance,
        int nprobe,
        int k,
        Tensor<float, 3, true> heapDistances,
        Tensor<idx_t, 3, true> heapIndices) {
    if constexpr ((NumWarpQ == 1 && NumThreadQ == 1) || NumWarpQ >= kWarpSize) {
        constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

        __shared__ float smemK[kNumWarps * NumWarpQ];
        __shared__ IndexT smemV[kNumWarps * NumWarpQ];

        for (IndexT queryId = blockIdx.y; queryId < prefixSumOffsets.getSize(0);
             queryId += gridDim.y) {
            constexpr auto kInit = Dir ? kFloatMin : kFloatMax;
            BlockSelect<
                    float,
                    IndexT,
                    Dir,
                    Comparator<float>,
                    NumWarpQ,
                    NumThreadQ,
                    ThreadsPerBlock>
                    heap(kInit, -1, smemK, smemV, k);

            auto sliceId = blockIdx.x;
            auto numSlices = gridDim.x;

            IndexT sliceSize = (nprobe / numSlices);
            IndexT sliceStart = sliceSize * sliceId;
            IndexT sliceEnd = sliceId == (numSlices - 1)
                    ? nprobe
                    : sliceStart + sliceSize;
            auto offsets = prefixSumOffsets[queryId].data();

            // We ensure that before the array (at offset -1), there is a 0
            // value
            auto start = *(&offsets[sliceStart] - 1);
            auto end = offsets[sliceEnd - 1];

            auto num = end - start;
            auto limit = utils::roundDown(num, (IndexT)4 * ThreadsPerBlock);

            IndexT i = threadIdx.x * 4;
            auto distanceStart = distance[start].data();

            // BlockSelect add cannot be used in a warp divergent circumstance;
            // we handle the remainder warp below
            f32x4_vec<4> distance_data;
            for (; i < limit; i += blockDim.x * 4) {
                distance_data.vec[0] = *(floatx4*)(distanceStart + i);
                for(int j = 0; j < 4; j++ ){
                    heap.add(*(float*)(&(distance_data.data[j])), IndexT(start + i + j));
                }
            }

            //TEST_ORIGINAL
            auto limit2 = utils::roundDown((num-limit), (IndexT)kWarpSize);
            IndexT i2 = limit + threadIdx.x;
            //auto distanceStart = distance[start].data();
            for (; i2 < (limit +limit2); i2 += blockDim.x) {
                heap.add(distanceStart[i2], IndexT(start + i2));
            }

            //auto limit2 = utils::roundDown(num, (IndexT)kWarpSize);
            //IndexT i2 = threadIdx.x;
            //auto distanceStart = distance[start].data();
            //for (; i2 < limit2; i2 += blockDim.x) {
                //heap.add(distanceStart[i2], IndexT(start + i2));
            //}
            // Handle the remainder if any separately (warp is divergent)
            if(i2<num){
                heap.addThreadQ(distanceStart[i2],IndexT(start + i2));
            }
            // Merge all final results in lds
            heap.reduce();

            // Write out the final k-selected values; they should be all
            // together
            for (int i = threadIdx.x; i < k; i += blockDim.x) {
                heapDistances[queryId][sliceId][i] = smemK[i];
                heapIndices[queryId][sliceId][i] = idx_t(smemV[i]);
            }
        }
    }
}

void runPass1SelectLists(
        Tensor<idx_t, 2, true>& prefixSumOffsets,
        Tensor<float, 1, true>& distance,
        int nprobe,
        int k,
        bool use64BitSelection,
        bool chooseLargest,
        Tensor<float, 3, true>& heapDistances,
        Tensor<idx_t, 3, true>& heapIndices,
        cudaStream_t stream,
        bool keyconfig) {
    // This is also caught at a higher level
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);

    auto grid =
            dim3(heapDistances.getSize(1),
                 std::min(
                         prefixSumOffsets.getSize(0),
                         (idx_t)getMaxGridCurrentDevice().y));

#define RUN_PASS(INDEX_T, BLOCK, NUM_WARP_Q, NUM_THREAD_Q, DIR)         \
    do {                                                                \
        pass1SelectLists2<INDEX_T, BLOCK, NUM_WARP_Q, NUM_THREAD_Q, DIR> \
                <<<grid, BLOCK, 0, stream>>>(                           \
                        prefixSumOffsets,                               \
                        distance,                                       \
                        nprobe,                                         \
                        k,                                              \
                        heapDistances,                                  \
                        heapIndices);                                   \
        return; /* success */                                           \
    } while (0)

#if GPU_MAX_SELECTION_K >= 2048

    // block size 128 for k <= 1024, 64 for k = 2048
#define RUN_PASS_DIR(INDEX_T, DIR)                                \
    do {                                                          \
        if (k == 1) {                                             \
            RUN_PASS(INDEX_T, 128, 1, 1, DIR);                    \
        } else if (k <= 32 && getWarpSizeCurrentDevice() == 32) { \
            RUN_PASS(INDEX_T, 256, 32, 2, DIR);                   \
        } else if (k <= 64) {                                     \
            RUN_PASS(INDEX_T, 512, 64, 2, DIR);                   \
        } else if (k <= 128 && keyconfig) {                       \
            RUN_PASS(INDEX_T, 512, 128, 4, DIR);                  \
        } else if (k <= 128 && !keyconfig) {                      \
            RUN_PASS(INDEX_T, 128, 128, 4, DIR);                  \
        } else if (k <= 256) {                                    \
            RUN_PASS(INDEX_T, 128, 256, 4, DIR);                  \
        } else if (k <= 512) {                                    \
            RUN_PASS(INDEX_T, 128, 512, 8, DIR);                  \
        } else if (k <= 1024) {                                   \
            RUN_PASS(INDEX_T, 128, 1024, 8, DIR);                 \
        } else if (k <= 2048) {                                   \
            RUN_PASS(INDEX_T, 64, 2048, 8, DIR);                  \
        }                                                         \
    } while (0)

#else

#define RUN_PASS_DIR(INDEX_T, DIR)                                \
    do {                                                          \
        if (k == 1) {                                             \
            RUN_PASS(INDEX_T, 128, 1, 1, DIR);                    \
        } else if (k <= 32 && getWarpSizeCurrentDevice() == 32) { \
            RUN_PASS(INDEX_T, 256, 32, 2, DIR);                   \
        } else if (k <= 64) {                                     \
            RUN_PASS(INDEX_T, 512, 64, 2, DIR);                   \
        } else if (k <= 128 && keyconfig) {                       \
            RUN_PASS(INDEX_T, 512, 128, 4, DIR);                  \
        } else if (k <= 128 && !keyconfig) {                      \
            RUN_PASS(INDEX_T, 128, 128, 4, DIR);                  \
        } else if (k <= 256) {                                    \
            RUN_PASS(INDEX_T, 128, 256, 4, DIR);                  \
        } else if (k <= 512) {                                    \
            RUN_PASS(INDEX_T, 128, 512, 8, DIR);                  \
        } else if (k <= 1024) {                                   \
            RUN_PASS(INDEX_T, 128, 1024, 8, DIR);                 \
        }                                                         \
    } while (0)

#endif // GPU_MAX_SELECTION_K

    if (use64BitSelection) {
        if (chooseLargest) {
            RUN_PASS_DIR(idx_t, true);
        } else {
            RUN_PASS_DIR(idx_t, false);
        }
    } else {
        if (chooseLargest) {
            RUN_PASS_DIR(int32_t, true);
        } else {
            RUN_PASS_DIR(int32_t, false);
        }
    }

#undef RUN_PASS_DIR
#undef RUN_PASS

    CUDA_TEST_ERROR();
}

} // namespace gpu
} // namespace faiss
