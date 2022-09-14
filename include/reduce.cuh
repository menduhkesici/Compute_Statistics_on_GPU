#pragma once

#include <cuda_runtime.h>
#include "./stats_data.cuh"

const int MAX_THREADS = 128; // max number of threads per block in reduce kernel
const int MAX_BLOCKS = 256;	 // max number of blocks per grid in reduce kernel

// takes data in T_in datatype
template <typename T_in>
void getStatsFromData(int nr_elems, T_in *inputData, intermediateStatsData *intermediateData,
					  finalStatsData *finalData, cudaStream_t stream);
