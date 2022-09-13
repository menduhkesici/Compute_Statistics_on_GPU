#pragma once

#include <cuda_runtime.h>
#include "./stats_data.cuh"

// takes data in T_in datatype
// makes calculations in T_mid datatype
// produces results in T_out datatype 
template <typename T_in, typename T_mid, typename T_out>
void getStatsFromData(int nr_elems, T_in* inputData, intermediateStatsData<T_mid>* intermediateData,
	finalStatsData<T_out>* finalData, cudaStream_t stream);
