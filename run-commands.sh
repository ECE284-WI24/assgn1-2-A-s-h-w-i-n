#!/bin/sh

# Change directory (DO NOT CHANGE!)
repoDir=$(dirname "$(realpath "$0")")
echo $repoDir
cd $repoDir

# Recompile if necessary (DO NOT CHANGE!)
mkdir -p build
cd build
cmake  -DTBB_DIR=${HOME}/oneTBB-2019_U9  -DCMAKE_PREFIX_PATH=${HOME}/oneTBB-2019_U9/cmake ..
make -j4
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/tbb_cmake_build/tbb_cmake_build_subdir_release

## Basic run with 8 threads and kmerSize of 12
## HINT: needs changes to parameter values for Assignment 1
./seedTable --reference ../data/reference.fa --numThreads 8 --kmerSize 12

## Run command using nsys nvprof profiler
## HINT: Useful for profiling tasks in Assignment 2 
#nsys nvprof --print-gpu-trace ./seedTable -r ../data/reference.fa -T 8 -k 4 
