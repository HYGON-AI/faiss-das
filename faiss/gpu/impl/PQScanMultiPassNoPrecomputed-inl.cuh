/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Copyright (c) 2026 Hygon Information Technology Co., Ltd.
 * SPDX-License-Identifier: MIT
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/impl/IVFUtils.cuh>
#include <faiss/gpu/impl/PQCodeDistances.cuh>
#include <faiss/gpu/impl/PQCodeLoad.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/LoadStoreOperators.cuh>
#include <faiss/gpu/utils/NoTypeTensor.cuh>
#include <faiss/gpu/utils/WarpPackedBits.cuh>

namespace faiss {
namespace gpu {

// A basic implementation that works for the interleaved by vector layout for
// any number of sub-quantizers
template <typename EncodeT, int EncodeBits, typename CodeDistanceT>
__global__ void  __launch_bounds__(1024) pqScanInterleaved(
        Tensor<float, 2, true> queries,
        Tensor<float, 3, true> pqCentroids,
        Tensor<idx_t, 2, true> ivfListIds,
        Tensor<CodeDistanceT, 4, true> codeDistances,
        void** listCodes,
        idx_t* listLengths,
        Tensor<idx_t, 2, true> prefixSumOffsets,
        Tensor<float, 1, true> distance) {
    // Each block handles a single query
    auto queryId = blockIdx.y;
    auto probeId = blockIdx.x;

    idx_t listId = ivfListIds[queryId][probeId];
    // Safety guard in case NaNs in input cause no list ID to be generated
    if (listId == -1) {
        return;
    }

    int numWarps = blockDim.x / kWarpSize;
    // FIXME: some issue with getLaneId() and CUDA 10.1 and P4 GPUs?
    int laneId = threadIdx.x % kWarpSize;
    int warpId = threadIdx.x / kWarpSize;

    int numSubQuantizers = codeDistances.getSize(2);

    // This is where we start writing out data
    // We ensure that before the array (at offset -1), there is a 0 value
    auto outBase = *(prefixSumOffsets[queryId][probeId].data() - 1);
    float* distanceOut = distance[outBase].data();
    auto localCodeDistances = codeDistances[queryId][probeId];

    //=========================START
    int numCodes = codeDistances.getSize(3);
    int totalElements = numSubQuantizers * numCodes;
    extern __shared__ float s_localDistances[];
    for(int i = threadIdx.x; i<totalElements; i+=blockDim.x){
        int sq = i / numCodes;
        int code = i % numCodes;
        s_localDistances[i] = ConvertTo<float>::to(localCodeDistances[sq][code]);
    }
    __syncthreads();
    //=========================END
    // This is where the codes for our list start
    auto vecsBase = (EncodeT*)listCodes[listId];
    auto numVecs = listLengths[listId];

    // How many vector blocks of kWarpSize are in this list?
    idx_t numBlocks = utils::divUp(numVecs, (idx_t)kWarpSize);

    // Number of EncodeT words per each dimension of block of kWarpSize vecs
    constexpr int bytesPerVectorBlockDim = EncodeBits * kWarpSize / 8;
    constexpr int wordsPerVectorBlockDim =
            bytesPerVectorBlockDim / sizeof(EncodeT);
    int wordsPerVectorBlock = wordsPerVectorBlockDim * numSubQuantizers;

    for (idx_t block = warpId; block < numBlocks; block += numWarps) {
        float dist = 0;

        // This is the vector a given lane/thread handles
        idx_t vec = block * kWarpSize + laneId;
        bool valid = vec < numVecs;

        EncodeT* data = vecsBase + block * wordsPerVectorBlock;

        for (int sq = 0; sq < numSubQuantizers; ++sq) {
            EncodeT enc =
                    WarpPackedBits<EncodeT, EncodeBits>::read(laneId, data);
            EncodeT code =
                    WarpPackedBits<EncodeT, EncodeBits>::postRead(laneId, enc);

            dist += valid ? s_localDistances[sq * numCodes + code] : 0;
            data += wordsPerVectorBlockDim;
        }

        if (valid) {
            distanceOut[vec] = dist;
        }
    }
}

template <typename EncodeT, int EncodeBits, typename CodeDistanceT>
__global__ void __launch_bounds__(1024) pqScanInterleaved_multiblock(
        Tensor<float, 2, true> queries,
        Tensor<float, 3, true> pqCentroids,
        Tensor<idx_t, 2, true> ivfListIds,
        Tensor<CodeDistanceT, 4, true> codeDistances,
        void** listCodes,
        idx_t* listLengths,
        Tensor<idx_t, 2, true> prefixSumOffsets,
        Tensor<float, 1, true> distance) {
    // Each 4 block handles a single query because probeId incase always be 1
    auto queryId = blockIdx.y;
    auto probeId = blockIdx.x / 4;
    auto probeId_i = blockIdx.x % 4;


    idx_t listId = ivfListIds[queryId][probeId];
    // Safety guard in case NaNs in input cause no list ID to be generated
    if (listId == -1) {
        return;
    }

    int numWarps = blockDim.x / kWarpSize;
    // FIXME: some issue with getLaneId() and CUDA 10.1 and P4 GPUs?
    int laneId = threadIdx.x % kWarpSize;
    int warpId = threadIdx.x / kWarpSize;
    //32
    int numSubQuantizers = codeDistances.getSize(2);

    // This is where we start writing out data
    // We ensure that before the array (at offset -1), there is a 0 value

    auto outBase = *(prefixSumOffsets[queryId][probeId].data() - 1);
    float* distanceOut = distance[outBase].data();


    auto localCodeDistances = codeDistances[queryId][probeId];

    //=========================START
    int numCodes = codeDistances.getSize(3);
    int totalElements = numSubQuantizers * numCodes;
    extern __shared__ float s_localDistances[];
    for(int i = threadIdx.x; i<totalElements; i+=blockDim.x){
        int sq = i / numCodes;
        int code = i % numCodes;
        s_localDistances[i] = ConvertTo<float>::to(localCodeDistances[sq][code]);
    }
    __syncthreads();
    //=========================END

    // This is where the codes for our list start

    auto vecsBase = (EncodeT*)listCodes[listId];

    auto numVecs = listLengths[listId];

    // How many vector blocks of kWarpSize are in this list?
    idx_t numBlocks = utils::divUp(numVecs, (idx_t)kWarpSize);

    auto numBlocksDIVF = numBlocks / 4 ;

    auto numBlocksSF = numBlocks % 4 ;

    auto numBlocks_start = probeId_i * numBlocksDIVF;

    // Number of EncodeT words per each dimension of block of kWarpSize vecs
    constexpr int bytesPerVectorBlockDim = EncodeBits * kWarpSize / 8;

    constexpr int wordsPerVectorBlockDim =
            bytesPerVectorBlockDim / sizeof(EncodeT);
    int wordsPerVectorBlock = wordsPerVectorBlockDim * numSubQuantizers;

    for (idx_t block = warpId; block < numBlocksDIVF; block += numWarps) {
        float dist = 0;

        // This is the vector a given lane/thread handles
        idx_t vec = (numBlocks_start + block) * kWarpSize + laneId;
        bool valid = vec < numVecs;

        EncodeT* data = vecsBase + (numBlocks_start + block) * wordsPerVectorBlock;

        for (int sq = 0; sq < numSubQuantizers; ++sq) {
            EncodeT enc =
                    WarpPackedBits<EncodeT, EncodeBits>::read(laneId, data);
            EncodeT code =
                    WarpPackedBits<EncodeT, EncodeBits>::postRead(laneId, enc);

            dist += valid ? s_localDistances[sq * numCodes + code] : 0;
            data += wordsPerVectorBlockDim;
        }

        if (valid) {
            distanceOut[vec] = dist;
        }
    }
    if(probeId_i == 3){
        for (idx_t block = warpId; block < numBlocksSF; block += numWarps) {
        float dist = 0;

        // This is the vector a given lane/thread handles
        idx_t vec = (numBlocksDIVF * 4 + block) * kWarpSize + laneId;
        bool valid = vec < numVecs;

        EncodeT* data = vecsBase + ( numBlocksDIVF * 4 + block) * wordsPerVectorBlock;

        for (int sq = 0; sq < numSubQuantizers; ++sq) {
            EncodeT enc =
                    WarpPackedBits<EncodeT, EncodeBits>::read(laneId, data);
            EncodeT code =
                    WarpPackedBits<EncodeT, EncodeBits>::postRead(laneId, enc);

            dist += valid ? s_localDistances[sq * numCodes + code] : 0;
            data += wordsPerVectorBlockDim;
        }

        if (valid) {
            distanceOut[vec] = dist;
        }
    }
    }
}
template <typename LookupT, typename LookupVecT>
struct LoadCodeDistances {
    static inline __device__ void load(
            LookupT* __restrict__ smem,
            const LookupT* __restrict__ codes,
            int numCodes) {
        constexpr int kWordSize = sizeof(LookupVecT) / sizeof(LookupT);

        // We can only use the vector type if the data is guaranteed to be
        // aligned. The codes are innermost, so if it is evenly divisible,
        // then any slice will be aligned.
        if (numCodes % kWordSize == 0) {
            // Load the data by float4 for efficiency, and then handle any
            // remainder limitVec is the number of whole vec words we can load,
            // in terms of whole blocks performing the load
            constexpr int kUnroll = 2;
            int limitVec = numCodes / (kUnroll * kWordSize * blockDim.x);
            limitVec *= kUnroll * blockDim.x;

            LookupVecT* smemV = (LookupVecT*)smem;
            LookupVecT* codesV = (LookupVecT*)codes;

            for (int i = threadIdx.x; i < limitVec; i += kUnroll * blockDim.x) {
                LookupVecT vals[kUnroll];

#pragma unroll
                for (int j = 0; j < kUnroll; ++j) {
                    vals[j] = LoadStore<LookupVecT>::load(
                            &codesV[i + j * blockDim.x]);
                }

#pragma unroll
                for (int j = 0; j < kUnroll; ++j) {
                    LoadStore<LookupVecT>::store(
                            &smemV[i + j * blockDim.x], vals[j]);
                }
            }

            // This is where we start loading the remainder that does not evenly
            // fit into kUnroll x blockDim.x
            int remainder = limitVec * kWordSize;

            for (int i = remainder + threadIdx.x; i < numCodes;
                 i += blockDim.x) {
                smem[i] = codes[i];
            }
        } else {
            // Potential unaligned load
            constexpr int kUnroll = 4;

            int limit = utils::roundDown(numCodes, kUnroll * blockDim.x);

            int i = threadIdx.x;
            for (; i < limit; i += kUnroll * blockDim.x) {
                LookupT vals[kUnroll];

#pragma unroll
                for (int j = 0; j < kUnroll; ++j) {
                    vals[j] = codes[i + j * blockDim.x];
                }

#pragma unroll
                for (int j = 0; j < kUnroll; ++j) {
                    smem[i + j * blockDim.x] = vals[j];
                }
            }

            for (; i < numCodes; i += blockDim.x) {
                smem[i] = codes[i];
            }
        }
    }
};

template <
        int NumSubQuantizers,
        typename LookupT,
        int kThreadsPerBlock,
        int kItemsPerThread,
        int kCTAsPerList,
        typename LookupVecT>
__global__ void __launch_bounds__(kThreadsPerBlock) pqScanNoPrecomputedMultiPass(
        Tensor<float, 2, true> queries,
        Tensor<float, 3, true> pqCentroids,
        Tensor<idx_t, 2, true> ivfListIds,
        Tensor<LookupT, 4, true> codeDistances,
        void** listCodes,
        idx_t* listLengths,
        Tensor<idx_t, 2, true> prefixSumOffsets,
        Tensor<float, 1, true> distance) {
    static_assert(kItemsPerThread > 0, "kItemsPerThread must be positive");
    static_assert(kCTAsPerList > 0, "kCTAsPerList must be positive");

    // This kernel is used by the non-interleaved IVFPQ path.  On HCU/ROCm,
    // long lists are often limited by one CTA walking the whole inverted list.
    // kCTAsPerList allows the host path to split one (query, probe) list over
    // several CTAs while preserving the same contiguous distance output layout.
    auto queryId = blockIdx.y;
    int probeId = blockIdx.x / kCTAsPerList;
    int listBlockId = blockIdx.x - probeId * kCTAsPerList;

    if (probeId >= ivfListIds.getSize(1)) {
        return;
    }

    idx_t listId = ivfListIds[queryId][probeId];
    // Safety guard in case NaNs in input cause no list ID to be generated
    if (listId == -1) {
        return;
    }

    const int codesPerSubQuantizer = codeDistances.getSize(3);
    uint8_t* __restrict__ codeList = (uint8_t*)listCodes[listId];
    idx_t limit = listLengths[listId];

    if (limit <= 0) {
        return;
    }

    // This is where we start writing out data
    // We ensure that before the array (at offset -1), there is a 0 value
    auto outBase = *(prefixSumOffsets[queryId][probeId].data() - 1);
    float* __restrict__ distanceOut = distance[outBase].data();

    // Where the pq code -> residual distance is stored. Keeping the lookup
    // table in LDS is still faster on HCU than repeatedly reading global
    // memory, even when several CTAs split the same long list.
    extern __shared__ char smemCodeDistances[];
    LookupT* __restrict__ codeDist = (LookupT*)smemCodeDistances;

    LoadCodeDistances<LookupT, LookupVecT>::load(
            codeDist,
            codeDistances[queryId][probeId].data(),
            NumSubQuantizers * codesPerSubQuantizer);

    __syncthreads();

    constexpr int kNumCode32 =
            NumSubQuantizers <= 4 ? 1 : (NumSubQuantizers / 4);
    constexpr int kBytesPerCode32 =
            NumSubQuantizers < 4 ? NumSubQuantizers : 4;

    unsigned int code32[kNumCode32];

    const idx_t codesPerTile =
            (idx_t)kThreadsPerBlock * (idx_t)kItemsPerThread;
    const idx_t firstTile = (idx_t)listBlockId * codesPerTile;
    const idx_t tileStride = (idx_t)kCTAsPerList * codesPerTile;

    // Each CTA scans a disjoint set of contiguous tiles.  kItemsPerThread gives
    // each thread a small amount of serial work, reducing loop overhead and
    // improving LDS/code-load reuse without increasing register pressure as much
    // as the old double-buffered path.
    for (idx_t tileBase = firstTile; tileBase < limit; tileBase += tileStride) {
#pragma unroll
        for (int item = 0; item < kItemsPerThread; ++item) {
            idx_t codeIndex = tileBase +
                    (idx_t)item * (idx_t)kThreadsPerBlock + threadIdx.x;

            if (codeIndex >= limit) {
                continue;
            }

            LoadCode32<NumSubQuantizers>::load(code32, codeList, codeIndex);

            float dist = 0.0f;

#pragma unroll
            for (int word = 0; word < kNumCode32; ++word) {
                if (kBytesPerCode32 == 1) {
                    auto code = code32[0];
                    dist = ConvertTo<float>::to(codeDist[code]);
                } else {
#pragma unroll
                    for (int byte = 0; byte < kBytesPerCode32; ++byte) {
                        auto code = getByte(code32[word], byte * 8, 8);
                        auto offset = codesPerSubQuantizer *
                                (word * kBytesPerCode32 + byte);

                        dist += ConvertTo<float>::to(codeDist[offset + code]);
                    }
                }
            }

            // Write out intermediate distance result. We do not maintain
            // indices here, in order to reduce global memory traffic. Those
            // are recovered in the final selection step.
            distanceOut[codeIndex] = dist;
        }
    }
}

template <typename CentroidT>
void runMultiPassTile(
        GpuResources* res,
        Tensor<float, 2, true>& queries,
        Tensor<CentroidT, 2, true>& centroids,
        Tensor<float, 3, true>& pqCentroidsInnermostCode,
        NoTypeTensor<4, true>& codeDistances,
        Tensor<float, 2, true>& coarseDistances,
        Tensor<idx_t, 2, true>& coarseIndices,
        bool useFloat16Lookup,
        bool useMMCodeDistance,
        bool interleavedCodeLayout,
        int bitsPerSubQuantizer,
        int numSubQuantizers,
        int numSubQuantizerCodes,
        DeviceVector<void*>& listCodes,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        DeviceVector<idx_t>& listLengths,
        idx_t maxListLength,
        Tensor<char, 1, true>& thrustMem,
        Tensor<idx_t, 2, true>& prefixSumOffsets,
        Tensor<float, 1, true>& allDistances,
        Tensor<float, 3, true>& heapDistances,
        Tensor<idx_t, 3, true>& heapIndices,
        int k,
        bool use64BitSelection,
        faiss::MetricType metric,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        cudaStream_t stream) {
    // We only support two metrics at the moment
    FAISS_ASSERT(
            metric == MetricType::METRIC_INNER_PRODUCT ||
            metric == MetricType::METRIC_L2);

    bool l2Distance = metric == MetricType::METRIC_L2;

    // Calculate offset lengths, so we know where to write out
    // intermediate results
    runCalcListOffsets(
            res,
            coarseIndices,
            listLengths,
            prefixSumOffsets,
            thrustMem,
            stream);

    // Calculate residual code distances, since this is without
    // precomputed codes
    runPQCodeDistances(
            res,
            pqCentroidsInnermostCode,
            queries,
            centroids,
            coarseDistances,
            coarseIndices,
            codeDistances,
            useMMCodeDistance,
            l2Distance,
            useFloat16Lookup,
            stream);

    if (interleavedCodeLayout) {

    if(coarseIndices.getSize(1) == 1){

        auto kThreadsPerBlock = 1024;
        int numSubQuantizers = codeDistances.getSize(2);
        int numSubQuantizerCodes = codeDistances.getSize(3);
        size_t ldsBytes = numSubQuantizers * numSubQuantizerCodes * sizeof(float);
        auto grid = dim3(coarseIndices.getSize(1)  * 4 , coarseIndices.getSize(0));
        auto block = dim3(kThreadsPerBlock);

#define RUN_INTERLEAVED_multiblock(BITS_PER_CODE, CODE_DIST_T)            \
    do {                                                                  \
        pqScanInterleaved_multiblock<uint8_t, BITS_PER_CODE, CODE_DIST_T> \
                <<<grid, block, ldsBytes, stream>>>(                      \
                        queries,                               \
                        pqCentroidsInnermostCode,              \
                        coarseIndices,                         \
                        codeDistancesT,                        \
                        listCodes.data(),                      \
                        listLengths.data(),                    \
                        prefixSumOffsets,                      \
                        allDistances);                         \
    } while (0)

        if (useFloat16Lookup) {
            auto codeDistancesT = codeDistances.toTensor<half>();

            switch (bitsPerSubQuantizer) {
                case 4: {
                    RUN_INTERLEAVED_multiblock(4, half);
                } break;
                case 5: {
                    RUN_INTERLEAVED_multiblock(5, half);
                } break;
                case 6: {
                    RUN_INTERLEAVED_multiblock(6, half);
                } break;
                case 8: {
                    RUN_INTERLEAVED_multiblock(8, half);
                } break;
                default:
                    FAISS_ASSERT(false);
                    break;
            }
        } else {
            auto codeDistancesT = codeDistances.toTensor<float>();

            switch (bitsPerSubQuantizer) {
                case 4: {
                    RUN_INTERLEAVED_multiblock(4, float);
                } break;
                case 5: {
                    RUN_INTERLEAVED_multiblock(5, float);
                } break;
                case 6: {
                    RUN_INTERLEAVED_multiblock(6, float);
                } break;
                case 8: {
                    RUN_INTERLEAVED_multiblock(8, float);
                } break;
                default:
                    FAISS_ASSERT(false);
                    break;
            }
        }
    }
    else{
        // The vector interleaved layout implementation
        auto kThreadsPerBlock = 1024;
        int numSubQuantizers = codeDistances.getSize(2);
        int numSubQuantizerCodes = codeDistances.getSize(3);
        size_t ldsBytes = numSubQuantizers * numSubQuantizerCodes * sizeof(float);
        auto grid = dim3(coarseIndices.getSize(1), coarseIndices.getSize(0));
        auto block = dim3(kThreadsPerBlock);

#define RUN_INTERLEAVED(BITS_PER_CODE, CODE_DIST_T)            \
    do {                                                       \
        pqScanInterleaved<uint8_t, BITS_PER_CODE, CODE_DIST_T> \
                <<<grid, block, ldsBytes, stream>>>(                  \
                        queries,                               \
                        pqCentroidsInnermostCode,              \
                        coarseIndices,                         \
                        codeDistancesT,                        \
                        listCodes.data(),                      \
                        listLengths.data(),                    \
                        prefixSumOffsets,                      \
                        allDistances);                         \
    } while (0)

        if (useFloat16Lookup) {
            auto codeDistancesT = codeDistances.toTensor<half>();

            switch (bitsPerSubQuantizer) {
                case 4: {
                    RUN_INTERLEAVED(4, half);
                } break;
                case 5: {
                    RUN_INTERLEAVED(5, half);
                } break;
                case 6: {
                    RUN_INTERLEAVED(6, half);
                } break;
                case 8: {
                    RUN_INTERLEAVED(8, half);
                } break;
                default:
                    FAISS_ASSERT(false);
                    break;
            }
        } else {
            auto codeDistancesT = codeDistances.toTensor<float>();

            switch (bitsPerSubQuantizer) {
                case 4: {
                    RUN_INTERLEAVED(4, float);
                } break;
                case 5: {
                    RUN_INTERLEAVED(5, float);
                } break;
                case 6: {
                    RUN_INTERLEAVED(6, float);
                } break;
                case 8: {
                    RUN_INTERLEAVED(8, float);
                } break;
                default:
                    FAISS_ASSERT(false);
                    break;
            }
        }
    }

    } else {
        // Convert all codes to a distance, and write out (distance,
        // index) values for all intermediate results.
        //
        // HCU optimization: for a single probed list with a long inverted
        // list, split one list over several CTAs. The output is still written
        // to the original distanceOut[codeIndex] slot, so the later selection
        // passes do not need any layout changes. kItemsPerThread increases
        // useful work per wavefront and amortizes LDS lookup-table loading.
        constexpr int kThreadsPerBlock = 512;
        constexpr int kItemsPerThread = 2;
        int scanBlocksPerList =
                (coarseIndices.getSize(1) == 1 && maxListLength >= 4096) ? 4 : 1;

        auto block = dim3(kThreadsPerBlock);

        // pq centroid distances
        auto smem = useFloat16Lookup ? sizeof(half) : sizeof(float);

        smem *= numSubQuantizers * numSubQuantizerCodes;
        FAISS_ASSERT(smem <= getMaxSharedMemPerBlockCurrentDevice());

#define RUN_PQ_OPT(NUM_SUB_Q, LOOKUP_T, LOOKUP_VEC_T, CTAS_PER_LIST)    \
    do {                                                                \
        auto codeDistancesT = codeDistances.toTensor<LOOKUP_T>();       \
        auto grid = dim3(                                                \
                coarseIndices.getSize(1) * CTAS_PER_LIST,               \
                coarseIndices.getSize(0));                              \
                                                                        \
        pqScanNoPrecomputedMultiPass<                                   \
                NUM_SUB_Q,                                               \
                LOOKUP_T,                                                \
                kThreadsPerBlock,                                        \
                kItemsPerThread,                                         \
                CTAS_PER_LIST,                                           \
                LOOKUP_VEC_T><<<grid, block, smem, stream>>>(           \
                        queries,                                        \
                        pqCentroidsInnermostCode,                       \
                        coarseIndices,                                  \
                        codeDistancesT,                                 \
                        listCodes.data(),                               \
                        listLengths.data(),                             \
                        prefixSumOffsets,                               \
                        allDistances);                                  \
    } while (0)

#define RUN_PQ_CTAS(NUM_SUB_Q, CTAS_PER_LIST)     \
    do {                                          \
        if (useFloat16Lookup) {                   \
            RUN_PQ_OPT(NUM_SUB_Q, half, Half8, CTAS_PER_LIST);   \
        } else {                                  \
            RUN_PQ_OPT(NUM_SUB_Q, float, float4, CTAS_PER_LIST); \
        }                                         \
    } while (0)

#define RUN_PQ(NUM_SUB_Q)                         \
    do {                                          \
        if (scanBlocksPerList == 4) {             \
            RUN_PQ_CTAS(NUM_SUB_Q, 4);            \
        } else {                                  \
            RUN_PQ_CTAS(NUM_SUB_Q, 1);            \
        }                                         \
    } while (0)

        switch (numSubQuantizers) {
            case 1:
                RUN_PQ(1);
                break;
            case 2:
                RUN_PQ(2);
                break;
            case 3:
                RUN_PQ(3);
                break;
            case 4:
                RUN_PQ(4);
                break;
            case 8:
                RUN_PQ(8);
                break;
            case 12:
                RUN_PQ(12);
                break;
            case 16:
                RUN_PQ(16);
                break;
            case 20:
                RUN_PQ(20);
                break;
            case 24:
                RUN_PQ(24);
                break;
            case 28:
                RUN_PQ(28);
                break;
            case 32:
                RUN_PQ(32);
                break;
            case 40:
                RUN_PQ(40);
                break;
            case 48:
                RUN_PQ(48);
                break;
            case 56:
                RUN_PQ(56);
                break;
            case 64:
                RUN_PQ(64);
                break;
            case 96:
                RUN_PQ(96);
                break;
            default:
                FAISS_ASSERT(false);
                break;
        }

#undef RUN_PQ
#undef RUN_PQ_CTAS
#undef RUN_PQ_OPT
    }

    CUDA_TEST_ERROR();

    // k-select the output in chunks, to increase parallelism
    runPass1SelectLists(
            prefixSumOffsets,
            allDistances,
            coarseIndices.getSize(1),
            k,
            use64BitSelection,
            !l2Distance, // L2 distance chooses smallest
            heapDistances,
            heapIndices,
            stream,
            interleavedCodeLayout);

    // k-select final output
    auto flatHeapDistances = heapDistances.downcastInner<2>();
    auto flatHeapIndices = heapIndices.downcastInner<2>();

    runPass2SelectLists(
            flatHeapDistances,
            flatHeapIndices,
            listIndices,
            indicesOptions,
            prefixSumOffsets,
            coarseIndices,
            k,
            use64BitSelection,
            !l2Distance, // L2 distance chooses smallest
            outDistances,
            outIndices,
            stream);
}

template <typename CentroidT>
void runPQScanMultiPassNoPrecomputed(
        Tensor<float, 2, true>& queries,
        Tensor<CentroidT, 2, true>& centroids,
        Tensor<float, 3, true>& pqCentroidsInnermostCode,
        Tensor<float, 2, true>& coarseDistances,
        Tensor<idx_t, 2, true>& coarseIndices,
        bool useFloat16Lookup,
        bool useMMCodeDistance,
        bool interleavedCodeLayout,
        int bitsPerSubQuantizer,
        int numSubQuantizers,
        int numSubQuantizerCodes,
        DeviceVector<void*>& listCodes,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        DeviceVector<idx_t>& listLengths,
        idx_t maxListLength,
        int k,
        faiss::MetricType metric,
        // output
        Tensor<float, 2, true>& outDistances,
        // output
        Tensor<idx_t, 2, true>& outIndices,
        GpuResources* res) {
    auto stream = res->getDefaultStreamCurrentDevice();

    auto nprobe = coarseIndices.getSize(1);

    // If the maximum list length (in terms of number of vectors) times nprobe
    // (number of lists) is > 2^31 - 1, then we will use 64-bit indexing in the
    // selection kernels
    constexpr int k32Limit = idx_t(std::numeric_limits<int32_t>::max());

    bool use64BitSelection = (maxListLength * nprobe > k32Limit) ||
            (queries.getSize(0) > k32Limit);

    // Make a reservation for Thrust to do its dirty work (global memory
    // cross-block reduction space); hopefully this is large enough.
    constexpr idx_t kThrustMemSize = 16384;

    DeviceTensor<char, 1, true> thrustMem1(
            res, makeTempAlloc(AllocType::Other, stream), {kThrustMemSize});
    DeviceTensor<char, 1, true> thrustMem2(
            res, makeTempAlloc(AllocType::Other, stream), {kThrustMemSize});
    DeviceTensor<char, 1, true>* thrustMem[2] = {&thrustMem1, &thrustMem2};

    // How much temporary memory would we need to handle a single query?
    size_t sizePerQuery = getIVFPQPerQueryTempMemory(
            k,
            nprobe,
            maxListLength,
            false, /* no precomputed codes */
            numSubQuantizers,
            numSubQuantizerCodes);

    // How many queries do we wish to run at once?
    idx_t queryTileSize = getIVFQueryTileSize(
            queries.getSize(0),
            res->getTempMemoryAvailableCurrentDevice(),
            sizePerQuery);

    // Temporary memory buffers
    // Make sure there is space prior to the start which will be 0, and
    // will handle the boundary condition without branches
    DeviceTensor<idx_t, 1, true> prefixSumOffsetSpace1(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize * nprobe + 1});
    DeviceTensor<idx_t, 1, true> prefixSumOffsetSpace2(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize * nprobe + 1});

    DeviceTensor<idx_t, 2, true> prefixSumOffsets1(
            prefixSumOffsetSpace1[1].data(), {queryTileSize, nprobe});
    DeviceTensor<idx_t, 2, true> prefixSumOffsets2(
            prefixSumOffsetSpace2[1].data(), {queryTileSize, nprobe});
    DeviceTensor<idx_t, 2, true>* prefixSumOffsets[2] = {
            &prefixSumOffsets1, &prefixSumOffsets2};

    // Make sure the element before prefixSumOffsets is 0, since we
    // depend upon simple, boundary-less indexing to get proper results
    CUDA_VERIFY(cudaMemsetAsync(
            prefixSumOffsetSpace1.data(), 0, sizeof(idx_t), stream));
    CUDA_VERIFY(cudaMemsetAsync(
            prefixSumOffsetSpace2.data(), 0, sizeof(idx_t), stream));

    idx_t codeDistanceTypeSize =
            useFloat16Lookup ? sizeof(half) : sizeof(float);

    idx_t totalCodeDistancesSize = queryTileSize * nprobe * numSubQuantizers *
            numSubQuantizerCodes * codeDistanceTypeSize;

    DeviceTensor<char, 1, true> codeDistances1Mem(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {totalCodeDistancesSize});
    NoTypeTensor<4, true> codeDistances1(
            codeDistances1Mem.data(),
            codeDistanceTypeSize,
            {queryTileSize, nprobe, numSubQuantizers, numSubQuantizerCodes});

    DeviceTensor<char, 1, true> codeDistances2Mem(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {totalCodeDistancesSize});
    NoTypeTensor<4, true> codeDistances2(
            codeDistances2Mem.data(),
            codeDistanceTypeSize,
            {queryTileSize, nprobe, numSubQuantizers, numSubQuantizerCodes});

    NoTypeTensor<4, true>* codeDistances[2] = {
            &codeDistances1, &codeDistances2};

    DeviceTensor<float, 1, true> allDistances1(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize * nprobe * maxListLength});
    DeviceTensor<float, 1, true> allDistances2(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize * nprobe * maxListLength});
    DeviceTensor<float, 1, true>* allDistances[2] = {
            &allDistances1, &allDistances2};

    idx_t pass2Chunks = getIVFKSelectionPass2Chunks(nprobe);
    DeviceTensor<float, 3, true> heapDistances1(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize, pass2Chunks, k});
    DeviceTensor<float, 3, true> heapDistances2(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize, pass2Chunks, k});
    DeviceTensor<float, 3, true>* heapDistances[2] = {
            &heapDistances1, &heapDistances2};

    DeviceTensor<idx_t, 3, true> heapIndices1(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize, pass2Chunks, k});
    DeviceTensor<idx_t, 3, true> heapIndices2(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize, pass2Chunks, k});
    DeviceTensor<idx_t, 3, true>* heapIndices[2] = {
            &heapIndices1, &heapIndices2};

    auto streams = res->getAlternateStreamsCurrentDevice();
    streamWait(streams, {stream});

    int curStream = 0;

    for (idx_t query = 0; query < queries.getSize(0); query += queryTileSize) {
        idx_t numQueriesInTile =
                std::min(queryTileSize, queries.getSize(0) - query);

        auto prefixSumOffsetsView =
                prefixSumOffsets[curStream]->narrowOutermost(
                        0, numQueriesInTile);

        auto codeDistancesView =
                codeDistances[curStream]->narrowOutermost(0, numQueriesInTile);
        auto coarseDistancesView =
                coarseDistances.narrowOutermost(query, numQueriesInTile);
        auto coarseIndicesView =
                coarseIndices.narrowOutermost(query, numQueriesInTile);
        auto queryView = queries.narrowOutermost(query, numQueriesInTile);

        auto heapDistancesView =
                heapDistances[curStream]->narrowOutermost(0, numQueriesInTile);
        auto heapIndicesView =
                heapIndices[curStream]->narrowOutermost(0, numQueriesInTile);

        auto outDistanceView =
                outDistances.narrowOutermost(query, numQueriesInTile);
        auto outIndicesView =
                outIndices.narrowOutermost(query, numQueriesInTile);

        runMultiPassTile(
                res,
                queryView,
                centroids,
                pqCentroidsInnermostCode,
                codeDistancesView,
                coarseDistancesView,
                coarseIndicesView,
                useFloat16Lookup,
                useMMCodeDistance,
                interleavedCodeLayout,
                bitsPerSubQuantizer,
                numSubQuantizers,
                numSubQuantizerCodes,
                listCodes,
                listIndices,
                indicesOptions,
                listLengths,
                maxListLength,
                *thrustMem[curStream],
                prefixSumOffsetsView,
                *allDistances[curStream],
                heapDistancesView,
                heapIndicesView,
                k,
                use64BitSelection,
                metric,
                outDistanceView,
                outIndicesView,
                streams[curStream]);

        curStream = (curStream + 1) % 2;
    }

    streamWait({stream}, streams);
}

} // namespace gpu
} // namespace faiss
