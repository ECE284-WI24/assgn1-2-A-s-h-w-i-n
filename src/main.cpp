#include <iostream>
#include <vector>
#include <boost/program_options.hpp>
#include <tbb/task_scheduler_init.h>
#include "seedTable.cuh"
#include "twoBitCompressor.hpp"
#include "kseq.h"
#include "zlib.h"

// For parsing the command line values
namespace po = boost::program_options;

// For reading in the FASTA file
KSEQ_INIT2(, gzFile, gzread)

int main(int argc, char** argv) {
    // Timer below helps with the performance profiling (see timer.hpp for more
    // details)
    Timer timer;

    std::string refFilename;
    uint32_t kmerSize;
    uint32_t numThreads;

    // Parse the command line options
    po::options_description desc{"Options"};
    desc.add_options()
    ("reference,r", po::value<std::string>(&refFilename)->required(), "Input FASTA file name [REQUIRED].")
    ("kmerSize,k", po::value<uint32_t>(&kmerSize)->default_value(12), "kmer size (range: 2-15)")
    ("numThreads,T", po::value<uint32_t>(&numThreads)->default_value(4), "Number of Threads (range: 1-8)")
    ("help,h", "Print help messages");

    po::options_description allOptions;
    allOptions.add(desc);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(allOptions).run(), vm);
        po::notify(vm);
    } catch(std::exception &e) {
        std::cerr << desc << std::endl;
        exit(1);
    }

    if ((kmerSize < 2) || (kmerSize > 15)) {
        std::cerr << "ERROR! kmerSize should be between 2 and 15." << std::endl;
        exit(1);
    }
    if ((numThreads < 1) || (numThreads > 8)) {
        std::cerr << "ERROR! numThreads should be between 1 and 8." << std::endl;
        exit(1);
    }

    // Print GPU information
    timer.Start();
    fprintf(stdout, "Setting CPU threads to %u and printing GPU device properties.\n", numThreads);
    tbb::task_scheduler_init init(numThreads);
    printGpuProperties();
    fprintf(stdout, "Completed in %ld msec \n\n", timer.Stop());

    // Read input sequence as kseq_t object
    timer.Start();
    fprintf(stdout, "Reading input sequence and compressing to two-bit encoding.\n");
    gzFile fp = gzopen(refFilename.c_str(), "r");
    if (!fp) {
        fprintf(stdout, "ERROR: Cannot open file: %s\n", refFilename.c_str());
        exit(1);
    }
    kseq_t *record = kseq_init(fp);
    int n;
    if ((n = kseq_read(record)) < 0) {
        fprintf(stdout, "ERROR: No records found!\n");
        exit(1);
    }
    printf("Sequence name: %s\n", record->name.s);
    printf("Sequence size: %zu\n", record->seq.l);
    fprintf(stdout, "Completed in %ld msec \n\n", timer.Stop());

    // Compress the sequence using two-bit encoding
    // ASSIGNMENT 1 TASK: Parallelize the function twoBitCompress below (see
    // src/twoBitCompress.cpp for details)
    fprintf(stdout, "Compressing input sequence using two-bit encoding.\n");
    uint32_t twoBitCompressedSize = (record->seq.l+15)/16;
    uint32_t * twoBitCompressed = new uint32_t[twoBitCompressedSize];
    twoBitCompress(record->seq.s, record->seq.l, twoBitCompressed);
    uint32_t toPrint = std::min((uint32_t)10, twoBitCompressedSize);
    fprintf(stdout, "Completed in %ld msec \n\n", timer.Stop());

    fprintf(stdout, "First %u values of the compressed sequence:\ni\tHex value\n", toPrint);
    for (uint32_t i=0; i<toPrint; i++) {
        fprintf(stdout, "%u\t%x\n", i, twoBitCompressed[i]);
    }

    // Create arrays
    timer.Start();
    fprintf(stdout, "\nAllocating GPU device arrays.\n");
    GpuSeedTable::deviceArrays.allocateDeviceArrays(twoBitCompressed, record->seq.l, kmerSize);
    fprintf(stdout, "Completed in %ld msec \n\n", timer.Stop());

    // Construct the seed table on GPU
    // ASSIGNMENT 2 TASK: Parallelize the kernels used in seedTableOnGpu below
    // (see  src/seedTableOnGpu.cu for details)
    timer.Start();
    fprintf(stdout, "Constructing seed table on GPU.\n");
    GpuSeedTable::seedTableOnGpu(
        GpuSeedTable::deviceArrays.d_compressedSeq,
        GpuSeedTable::deviceArrays.d_seqLen,
        kmerSize,
        GpuSeedTable::deviceArrays.d_kmerOffset,
        GpuSeedTable::deviceArrays.d_kmerPos);
    fprintf(stdout, "Completed in %ld msec \n\n", timer.Stop());

    // Check correctness
    timer.Start();
    int numValues = 10;
    fprintf(stdout, "Printing first %i values of GPU arrays to check correctness.\n", numValues);
    GpuSeedTable::deviceArrays.printValues(numValues);
    fprintf(stdout, "Completed in %ld msec \n\n", timer.Stop());

    // Delete arrays
    timer.Start();
    fprintf(stdout, "Deallocating CPU and GPU arrays.\n");
    GpuSeedTable::deviceArrays.deallocateDeviceArrays();
    delete [] twoBitCompressed;
    fprintf(stdout, "Completed in %ld msec \n\n", timer.Stop());

    return 0;
}

