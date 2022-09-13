#pragma once

#include <iostream>
#include <cuda_runtime.h>

// Checking error code in return values of CUDA calls
inline void checkCudaError(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess)
	{
		std::cerr << std::endl << "CUDA error (" << cudaGetErrorString(err) << ")!" << std::endl
			<< "File: '" << file << "', Line: " << line << std::endl;
		exit(EXIT_FAILURE);
	}
}
#define CHECK_CUDA_ERROR( err ) (checkCudaError((err), __FILE__, __LINE__))
