#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "../include/cuda_handler.cuh"
#include "../include/stats_data.cuh"
#include "../include/reduce.cuh"

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <typename T>
struct SharedMemory
{
	__device__ inline operator T *()
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}

	__device__ inline operator const T *() const
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}
};

template <>
struct SharedMemory<double>
{
	__device__ inline operator double *()
	{
		extern __shared__ double __smem_d[];
		return (double *)__smem_d;
	}

	__device__ inline operator const double *() const
	{
		extern __shared__ double __smem_d[];
		return (double *)__smem_d;
	}
};

template <typename T_in, unsigned int blockSize>
__global__ void reduceDataToStatsKernel(T_in *g_idata, intermediateStatsData *g_odata, unsigned int n)
{
	intermediateStatsData *sdata = SharedMemory<intermediateStatsData>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockSize + threadIdx.x;
	unsigned int gridSize = blockSize * gridDim.x;

	intermediateStatsData myReduced;
	myReduced.initialize();

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		myReduced = intermediateStatsDataBinaryOp(myReduced, convertToIntermediateStatsData<T_in>(g_idata[i]));

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = myReduced;
	__syncthreads();

	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp(myReduced, sdata[tid + 256]);
	}

	__syncthreads();

	if ((blockSize >= 256) && (tid < 128))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp(myReduced, sdata[tid + 128]);
	}

	__syncthreads();

	if ((blockSize >= 128) && (tid < 64))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp(myReduced, sdata[tid + 64]);
	}

	__syncthreads();

	// fully unroll reduction within a single warp
	if ((blockSize >= 64) && (tid < 32))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp(myReduced, sdata[tid + 32]);
	}

	__syncthreads();

	if ((blockSize >= 32) && (tid < 16))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp(myReduced, sdata[tid + 16]);
	}

	__syncthreads();

	if ((blockSize >= 16) && (tid < 8))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp(myReduced, sdata[tid + 8]);
	}

	__syncthreads();

	if ((blockSize >= 8) && (tid < 4))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp(myReduced, sdata[tid + 4]);
	}

	__syncthreads();

	if ((blockSize >= 4) && (tid < 2))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp(myReduced, sdata[tid + 2]);
	}

	__syncthreads();

	if ((blockSize >= 2) && (tid < 1))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp(myReduced, sdata[tid + 1]);
	}

	__syncthreads();

	// write result for this block to global mem
	if (tid == 0)
	{
		g_odata[blockIdx.x] = myReduced;
	}
}

template <typename T_in>
inline void reduceDataToStats(int size, int threads, int blocks, T_in *d_idata, intermediateStatsData *d_odata, cudaStream_t stream)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? (2 * threads * sizeof(intermediateStatsData)) : (threads * sizeof(intermediateStatsData));

	switch (threads)
	{
	case 512:
		reduceDataToStatsKernel<T_in, 512><<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		break;

	case 256:
		reduceDataToStatsKernel<T_in, 256><<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		break;

	case 128:
		reduceDataToStatsKernel<T_in, 128><<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		break;

	case 64:
		reduceDataToStatsKernel<T_in, 64><<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		break;

	case 32:
		reduceDataToStatsKernel<T_in, 32><<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		break;

	case 16:
		reduceDataToStatsKernel<T_in, 16><<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		break;

	case 8:
		reduceDataToStatsKernel<T_in, 8><<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		break;

	case 4:
		reduceDataToStatsKernel<T_in, 4><<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		break;

	case 2:
		reduceDataToStatsKernel<T_in, 2><<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		break;

	case 1:
		reduceDataToStatsKernel<T_in, 1><<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		break;
	}
}

template <unsigned int blockSize>
__global__ void reduceStatsToStatsKernel(intermediateStatsData *g_idata, intermediateStatsData *g_odata, unsigned int n)
{
	intermediateStatsData *sdata = SharedMemory<intermediateStatsData>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockSize + threadIdx.x;
	unsigned int gridSize = blockSize * gridDim.x;

	intermediateStatsData myReduced;
	myReduced.initialize();

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		myReduced = intermediateStatsDataBinaryOp(myReduced, g_idata[i]);

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = myReduced;
	__syncthreads();

	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp(myReduced, sdata[tid + 256]);
	}

	__syncthreads();

	if ((blockSize >= 256) && (tid < 128))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp(myReduced, sdata[tid + 128]);
	}

	__syncthreads();

	if ((blockSize >= 128) && (tid < 64))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp(myReduced, sdata[tid + 64]);
	}

	__syncthreads();

	// fully unroll reduction within a single warp
	if ((blockSize >= 64) && (tid < 32))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp(myReduced, sdata[tid + 32]);
	}

	__syncthreads();

	if ((blockSize >= 32) && (tid < 16))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp(myReduced, sdata[tid + 16]);
	}

	__syncthreads();

	if ((blockSize >= 16) && (tid < 8))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp(myReduced, sdata[tid + 8]);
	}

	__syncthreads();

	if ((blockSize >= 8) && (tid < 4))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp(myReduced, sdata[tid + 4]);
	}

	__syncthreads();

	if ((blockSize >= 4) && (tid < 2))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp(myReduced, sdata[tid + 2]);
	}

	__syncthreads();

	if ((blockSize >= 2) && (tid < 1))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp(myReduced, sdata[tid + 1]);
	}

	__syncthreads();

	// write result for this block to global mem
	if (tid == 0)
	{
		g_odata[blockIdx.x] = myReduced;
	}
}

inline void reduceStatsToStats(int size, int threads, int blocks, intermediateStatsData *d_idata,
							   intermediateStatsData *d_odata, cudaStream_t stream)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? (2 * threads * sizeof(intermediateStatsData)) : (threads * sizeof(intermediateStatsData));

	switch (threads)
	{
	case 512:
		reduceStatsToStatsKernel<512><<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		break;

	case 256:
		reduceStatsToStatsKernel<256><<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		break;

	case 128:
		reduceStatsToStatsKernel<128><<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		break;

	case 64:
		reduceStatsToStatsKernel<64><<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		break;

	case 32:
		reduceStatsToStatsKernel<32><<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		break;

	case 16:
		reduceStatsToStatsKernel<16><<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		break;

	case 8:
		reduceStatsToStatsKernel<8><<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		break;

	case 4:
		reduceStatsToStatsKernel<4><<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		break;

	case 2:
		reduceStatsToStatsKernel<2><<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		break;

	case 1:
		reduceStatsToStatsKernel<1><<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size);
		break;
	}
}

__global__ void convertToFinalStatsDataKernel(intermediateStatsData *g_idata, finalStatsData *g_odata)
{
	g_odata[0] = convertToFinalStatsData(g_idata[0]);
}

inline unsigned int nextPow2(unsigned int x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

// Compute the number of threads and blocks to use for the reduction kernel.
// We set threads / block to the minimum of maxThreads and n/2.
// We also observe the maximum specified number of blocks, because each thread in
// that kernel can process a variable number of elements.
inline void getNumBlocksAndThreads(int n, int &blocks, int &threads)
{
	threads = (n < (MAX_THREADS * 2)) ? (nextPow2((n + 1) / 2)) : (MAX_THREADS);
	blocks = MIN(((n + threads * 2 - 1) / (threads * 2)), MAX_BLOCKS);

	return;
}

// takes data in T_in datatype
template <typename T_in>
void getStatsFromData(int nr_elems, T_in *inputData, intermediateStatsData *intermediateData,
					  finalStatsData *finalData, cudaStream_t stream)
{
	int numThreads, numBlocks;
	getNumBlocksAndThreads(nr_elems, numBlocks, numThreads);

	// Run reduction
	reduceDataToStats<T_in>(nr_elems, numThreads, numBlocks, inputData, intermediateData, stream);
	CHECK_CUDA_ERROR(cudaGetLastError());

	int currNumOfElems = numBlocks;
	while (currNumOfElems > 1)
	{
		int threads = 0, blocks = 0;
		getNumBlocksAndThreads(currNumOfElems, blocks, threads);

		reduceStatsToStats(currNumOfElems, threads, blocks, intermediateData, intermediateData, stream);
		CHECK_CUDA_ERROR(cudaGetLastError());

		currNumOfElems = (currNumOfElems + (threads * 2 - 1)) / (threads * 2);
	}

	convertToFinalStatsDataKernel<<<1, 1, 0, stream>>>(intermediateData, finalData);
	CHECK_CUDA_ERROR(cudaGetLastError());

	return;
}

// Instantiate the reduction function for different types

template void
getStatsFromData<int>(int nr_elems, int *inputData, intermediateStatsData *intermediateData,
					  finalStatsData *finalData, cudaStream_t stream);

template void
getStatsFromData<float>(int nr_elems, float *inputData, intermediateStatsData *intermediateData,
						finalStatsData *finalData, cudaStream_t stream);

template void
getStatsFromData<double>(int nr_elems, double *inputData, intermediateStatsData *intermediateData,
						 finalStatsData *finalData, cudaStream_t stream);
