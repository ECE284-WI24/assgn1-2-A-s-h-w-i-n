#include<string>
#include <iostream>
#include <tbb/parallel_for.h>

void twoBitCompress(char* seq, size_t seqLen, uint32_t* compressedSeq);
