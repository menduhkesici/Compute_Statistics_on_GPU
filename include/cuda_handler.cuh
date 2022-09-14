#pragma once

#include <iostream>
#include <cuda_runtime.h>

inline void checkCudaError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        std::cout << std::endl
                  << cudaGetErrorString(err) << std::endl
                  << "File: '" << file << "', Line: " << line << std::endl;
        throw std::runtime_error("CUDA error occurred!");
    }
}
#define CHECK_CUDA_ERROR(err) (checkCudaError((err), __FILE__, __LINE__))

inline void resetCUDADevices(const char *file, int line)
{
    int deviceCount;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    for (int i = 0; i < deviceCount; i++)
    {
        CHECK_CUDA_ERROR(cudaSetDevice(i));
        CHECK_CUDA_ERROR(cudaDeviceReset());
    }
}
#define RESET_CUDA_DEVICES() (resetCUDADevices(__FILE__, __LINE__))
