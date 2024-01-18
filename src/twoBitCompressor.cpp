#include "twoBitCompressor.hpp"

/**
 * Compresses a DNA sequence (consisting of As, Cs, Gs and Ts) to a 2-bit
 * representation using the following encoding:
 *    A:2'b00
 *    C:2'b01
 *    G:2'b10
 *    T:2'b11
 * The final compressed string is stored in an array of 32-bit numbers, from
 * lowest to highest bits
 * For e.g., string GACT would be encoded as 0x11010010 (=210 as decimal number)
 *
 * ASSIGNMENT 1 TASK: parallelize this function
 *
 * HINT: use tbb::parallel_for using a new lambda function
 */
void twoBitCompress(char* seq, size_t seqLen, uint32_t* compressedSeq) {
    size_t compressedSeqLen = (seqLen+15)/16;

    for (size_t i=0; i < compressedSeqLen; i++) {
        compressedSeq[i] = 0;

        size_t start = 16*i;
        size_t end = std::min(seqLen, start+16);

        uint32_t twoBitVal = 0;
        uint32_t shift = 0;
        for (auto j=start; j<end; j++) {
            switch(seq[j]) {
            case 'A':
                twoBitVal = 0;
                break;
            case 'C':
                twoBitVal = 1;
                break;
            case 'G':
                twoBitVal = 2;
                break;
            case 'T':
                twoBitVal = 3;
                break;
            default:
                twoBitVal = 0;
                break;
            }

            compressedSeq[i] |= (twoBitVal << shift);
            shift += 2;
        }
    }
}

