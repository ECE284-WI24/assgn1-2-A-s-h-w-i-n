#include "seedTable.cuh"
#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/**
 * Prints information for each available GPU device on stdout
 */
void printGpuProperties () {
    int nDevices;

    // Store the number of available GPU device in nDevicess
    cudaError_t err = cudaGetDeviceCount(&nDevices);

    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaGetDeviceCount failed!\n");
        exit(1);
    }

    // For each GPU device found, print the information (memory, bandwidth etc.)
    // about the device
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Device memory: %lu\n", prop.totalGlobalMem);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
}

/**
 * Allocates arrays on the GPU device for (i) storing the compressed sequence
 * (ii) kmer offsets of the seed table (iii) kmer positions of the seed table
 * Size of the arrays depends on the input sequence length and kmer size
 */
void GpuSeedTable::DeviceArrays::allocateDeviceArrays (uint32_t* compressedSeq, uint32_t seqLen, uint32_t kmerSize) {
    cudaError_t err;

    d_seqLen = seqLen;
    uint32_t compressedSeqLen = (seqLen+15)/16;
    uint32_t maxKmers = (uint32_t) pow(4,kmerSize)+1;

    // Only (1) allocate and (2)transfer the 2-bit compressed sequence to GPU.
    // This reduces the memory transfer and storage overheads
    // 1. Allocate memory
    err = cudaMalloc(&d_compressedSeq, compressedSeqLen*sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    // 2. Transfer compressed sequence
    err = cudaMemcpy(d_compressedSeq, compressedSeq, compressedSeqLen*sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }

    // Allocate memory on GPU device for storing the kmer offset array
    err = cudaMalloc(&d_kmerOffset, maxKmers*sizeof(uint32_t));
    printf("maxKmers: %d\n", maxKmers);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    // Allocate memory on GPU device for storing the kmer position array
    // Each element is size_t (64-bit) because an intermediate step uses the
    // first 32-bits for kmer value and the last 32-bits for kmer positions
    err = cudaMalloc(&d_kmerPos, (seqLen-kmerSize+1)*sizeof(size_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }
    cudaDeviceSynchronize();
}

/**
 * Free allocated GPU device memory for different arrays
 */
void GpuSeedTable::DeviceArrays::deallocateDeviceArrays () {
    cudaFree(d_compressedSeq);
    cudaFree(d_kmerOffset);
    cudaFree(d_kmerPos);
}

/**
 * Finds kmers for the compressed sequence creates an array with elements
 * containing the 64-bit concatenated value consisting of the kmer value in the
 * first 32 bits and the kmer position in the last 32 bits. The values are
 * stored in the arrary kmerPos, with i-th element corresponding to the i-th
 * kmer in the sequence
 *
 * ASSIGNMENT 2 TASK: parallelize this function
 */
__global__ void kmerPosConcat(
    uint32_t* d_compressedSeq,
    uint32_t d_seqLen,
    uint32_t kmerSize,
    size_t* d_kmerPos) {

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    // Helps mask the non kmer bits from compressed sequence. E.g. for k=2,
    // mask=0x1111 and for k=3, mask=0x111111
    uint32_t mask = (1 << 2*kmerSize)-1;
    size_t kmer = 0;
    uint32_t N = d_seqLen;
    uint32_t k = kmerSize;
    uint32_t iterCount = (N-k+2)/(gs*bs);
    uint32_t currThread = bs*bx + tx;  
    uint32_t loopTerm;

    if (currThread == gs*bs-1) {
        loopTerm = N-k+1;
    }
    else {
        loopTerm = (currThread+1)*(iterCount);
    }
    for (uint32_t i = currThread*iterCount; i < loopTerm; i++) {
        if (i <= N-k) {
            uint32_t index = i/16;
            uint32_t shift1 = 2*(i%16);
            if (shift1 > 0) {
                uint32_t shift2 = 32-shift1;
                kmer = ((d_compressedSeq[index] >> shift1) | (d_compressedSeq[index+1] << shift2)) & mask;
            } else {
                kmer = d_compressedSeq[index] & mask;
            }

            // Concatenate kmer value (first 32-bits) with its position (last 32-bits)
            size_t kPosConcat = (kmer << 32) + i;
            d_kmerPos[i] = kPosConcat;
        }
    }
}

/**
 * Generates the kmerOffset array using the sorted kmerPos array consisting of
 * the kmer and positions. Requires iterating through the kmerPos array and
 * finding indexes where the kmer values change, depending on which the
 * kmerOffset values are determined.
 *
 * ASSIGNMENT 2 TASK: parallelize this function
 */
__global__ void kmerOffsetFill(
    uint32_t d_seqLen,
    uint32_t kmerSize,
    uint32_t numKmers,
    uint32_t* d_kmerOffset,
    size_t* d_kmerPos) {

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    uint32_t N = d_seqLen;
    uint32_t k = kmerSize;

    size_t mask = ((size_t) 1 << 32)-1;
    uint32_t kmer = 0;
    uint32_t nextKmer = 0;
    uint32_t iterCount = (N-k+2)/(gs*bs);
    uint32_t currThread = bs*bx + tx;
    uint32_t loopTerm;
    
    if (currThread == gs*bs-1) {
        loopTerm = N-k+1;
    }
    else {
        loopTerm = (currThread+1)*(iterCount);
    }
    for (uint32_t i = currThread*iterCount; i < loopTerm; i++) {
        if (i <= N-k) {
            kmer = (d_kmerPos[i] >> 32) & mask;
            nextKmer = (d_kmerPos[i+1] >> 32) & mask;
            if (kmer != nextKmer) {
                if (i == N-k)
                    d_kmerOffset[kmer] = i;
                else
                    d_kmerOffset[kmer] = i+1;
            }
        }
        else {
            kmer = (d_kmerPos[N-k] >> 32) & mask;
            if (d_kmerOffset[kmer] != 0) {
                d_kmerOffset[kmer] = N-k;
            }
        }
    }

    __syncthreads();

    if (currThread == bs*bx) {
        kmer = (d_kmerPos[currThread*iterCount] >> 32) & mask;
        nextKmer = (d_kmerPos[(currThread+bs)*iterCount-1] >> 32) & mask;
        for (auto j = kmer; j < nextKmer; j++) {    
            if (d_kmerOffset[j+1] == 0) {            
                d_kmerOffset[j+1] = d_kmerOffset[j];
            }
        }
    }
}

/**
 * Masks the first 32 bits of the elements in the kmerPos array
 *
 * ASSIGNMENT 2 TASK: parallelize this function
 */
__global__ void kmerPosMask(
    uint32_t d_seqLen,
    uint32_t kmerSize,
    size_t* d_kmerPos) {

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bs = blockDim.x;
    int gs = gridDim.x;

    uint32_t N = d_seqLen;
    uint32_t k = kmerSize;

    size_t mask = ((size_t) 1 << 32)-1;
    uint32_t iterCount = (N-k+2)/(gs*bs);
    uint32_t currThread = bs*bx + tx;
    uint32_t loopTerm;
    if (currThread == gs*bs-1) {
        loopTerm = N-k+1;
    }
    else {
        loopTerm = (currThread+1)*(iterCount);
    }
    for (uint32_t i = currThread*iterCount; i < loopTerm; i++) {
        if (i <= N-k) {
            d_kmerPos[i] = d_kmerPos[i] & mask;
        }
    }
}

/**
 * Constructs seed table, consisting of kmerOffset and kmerPos arrrays
 * on the GPU.
*/
void GpuSeedTable::seedTableOnGpu (
    uint32_t* compressedSeq,
    uint32_t seqLen,

    uint32_t kmerSize,

    uint32_t* kmerOffset,
    size_t* kmerPos) {

    // ASSIGNMENT 2 TASK: make sure to appropriately set the values below
    int blockSize = 512; // i.e. number of GPU threads per thread block
    int numBlocks = 1024; //(seqLen-kmerSize)/blockSize + 1; // i.e. number of thread blocks on the GPU
    kmerPosConcat<<<numBlocks, blockSize>>>(compressedSeq, seqLen, kmerSize, kmerPos);

    // Parallel sort the kmerPos array on the GPU device using the thrust
    // library (https://thrust.github.io/)
    thrust::device_ptr<size_t> kmerPosPtr(kmerPos);
    thrust::sort(kmerPosPtr, kmerPosPtr+seqLen-kmerSize+1);

    uint32_t numKmers = pow(4, kmerSize);;
    kmerOffsetFill<<<numBlocks, blockSize>>>(seqLen, kmerSize, numKmers, kmerOffset, kmerPos);
    kmerPosMask<<<numBlocks, blockSize>>>(seqLen, kmerSize, kmerPos);

    // Wait for all computation on GPU device to finish. Needed to ensure
    // correct runtime profiling results for this function.
    cudaDeviceSynchronize();
}

/**
 * Prints the fist N(=numValues) values of kmer offset and position tables to
 * help with the debugging of Assignment 2
 */
void GpuSeedTable::DeviceArrays::printValues(int numValues) {
    uint32_t* kmerOffset = new uint32_t[numValues];
    size_t* kmerPos = new size_t[numValues];

    cudaError_t err;

    err = cudaMemcpy(kmerOffset, d_kmerOffset, numValues*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }

    err = cudaMemcpy(kmerPos, d_kmerPos, numValues*sizeof(size_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }

    printf("i\tkmerOffset[i]\tkmerPos[i]\n");
    for (int i=0; i<numValues; i++) {
        printf("%i\t%u\t%zu\n", i, kmerOffset[i], kmerPos[i]);
    }
}