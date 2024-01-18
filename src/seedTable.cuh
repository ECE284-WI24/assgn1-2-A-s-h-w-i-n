#include <stdint.h>
#include "timer.hpp"

void printGpuProperties();

namespace GpuSeedTable {
struct DeviceArrays {
    uint32_t* d_compressedSeq;
    uint32_t d_seqLen;
    uint32_t* d_kmerOffset;
    size_t* d_kmerPos;
    // HINT: if needed, you add more device arrays for the GPU here (make sure to allocate and dellocate them in appropriate functions!)

    void allocateDeviceArrays (uint32_t* compressedSeq, uint32_t seqLen, uint32_t kmerSize);
    void printValues(int numValues);
    void deallocateDeviceArrays ();
};

static DeviceArrays deviceArrays;

void seedTableOnGpu (
    uint32_t* compressedSeq,
    uint32_t seqLen,

    uint32_t kmerSize,

    uint32_t* kmerOffset,
    size_t* kmerPos);
}

