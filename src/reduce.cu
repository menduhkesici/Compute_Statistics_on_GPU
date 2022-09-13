#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "../include/handler.cuh"
#include "../include/stats_data.cuh"
#include "../include/reduce.cuh"

extern const int MAX_THREADS;	// max number of threads per block in reduce kernel
extern const int MAX_BLOCKS;	// max number of blocks per grid in reduce kernel

inline bool isPow2(unsigned int x)
{
	return ((x & (x - 1)) == 0);
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

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <typename T>
struct SharedMemory
{
	__device__ inline operator T* ()
	{
		extern __shared__ int __smem[];
		return (T*)__smem;
	}

	__device__ inline operator const T* () const
	{
		extern __shared__ int __smem[];
		return (T*)__smem;
	}
};

template <typename T_in, typename T_mid, unsigned int blockSize, bool nIsPow2>
__global__ void reduceDataToStatsKernel(T_in* g_idata, intermediateStatsData<T_mid>* g_odata, unsigned int n)
{
	intermediateStatsData<T_mid>* sdata = SharedMemory<intermediateStatsData<T_mid>>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;

	intermediateStatsData<T_mid> myReduced;
	myReduced.initialize();

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		myReduced = intermediateStatsDataBinaryOp<T_mid>(myReduced, convertToIntermediateStatsData<T_in, T_mid>(g_idata[i]));

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || ((i + blockSize) < n))
		{
			myReduced = intermediateStatsDataBinaryOp<T_mid>(myReduced, convertToIntermediateStatsData<T_in, T_mid>(g_idata[i + blockSize]));
		}

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = myReduced;
	__syncthreads();

	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp<T_mid>(myReduced, sdata[tid + 256]);
	}

	__syncthreads();

	if ((blockSize >= 256) && (tid < 128))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp<T_mid>(myReduced, sdata[tid + 128]);
	}

	__syncthreads();

	if ((blockSize >= 128) && (tid < 64))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp<T_mid>(myReduced, sdata[tid + 64]);
	}

	__syncthreads();

	// fully unroll reduction within a single warp
	if ((blockSize >= 64) && (tid < 32))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp<T_mid>(myReduced, sdata[tid + 32]);
	}

	__syncthreads();

	if ((blockSize >= 32) && (tid < 16))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp<T_mid>(myReduced, sdata[tid + 16]);
	}

	__syncthreads();

	if ((blockSize >= 16) && (tid < 8))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp<T_mid>(myReduced, sdata[tid + 8]);
	}

	__syncthreads();

	if ((blockSize >= 8) && (tid < 4))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp<T_mid>(myReduced, sdata[tid + 4]);
	}

	__syncthreads();

	if ((blockSize >= 4) && (tid < 2))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp<T_mid>(myReduced, sdata[tid + 2]);
	}

	__syncthreads();

	if ((blockSize >= 2) && (tid < 1))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp<T_mid>(myReduced, sdata[tid + 1]);
	}

	__syncthreads();

	// write result for this block to global mem
	if (tid == 0)
	{
		g_odata[blockIdx.x] = myReduced;
	}
}

template <typename T_in, typename T_mid>
inline void reduceDataToStats(int size, int threads, int blocks, T_in* d_idata, intermediateStatsData<T_mid>* d_odata, cudaStream_t stream)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? (2 * threads * sizeof(intermediateStatsData<T_mid>)) : (threads * sizeof(intermediateStatsData<T_mid>));

	if (isPow2(size))
	{
		switch (threads)
		{
		case 512:
			reduceDataToStatsKernel<T_in, T_mid, 512, true> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case 256:
			reduceDataToStatsKernel<T_in, T_mid, 256, true> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case 128:
			reduceDataToStatsKernel<T_in, T_mid, 128, true> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case 64:
			reduceDataToStatsKernel<T_in, T_mid, 64, true> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case 32:
			reduceDataToStatsKernel<T_in, T_mid, 32, true> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case 16:
			reduceDataToStatsKernel<T_in, T_mid, 16, true> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case  8:
			reduceDataToStatsKernel<T_in, T_mid, 8, true> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case  4:
			reduceDataToStatsKernel<T_in, T_mid, 4, true> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case  2:
			reduceDataToStatsKernel<T_in, T_mid, 2, true> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case  1:
			reduceDataToStatsKernel<T_in, T_mid, 1, true> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;
		}
	}
	else
	{
		switch (threads)
		{
		case 512:
			reduceDataToStatsKernel<T_in, T_mid, 512, false> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case 256:
			reduceDataToStatsKernel<T_in, T_mid, 256, false> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case 128:
			reduceDataToStatsKernel<T_in, T_mid, 128, false> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case 64:
			reduceDataToStatsKernel<T_in, T_mid, 64, false> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case 32:
			reduceDataToStatsKernel<T_in, T_mid, 32, false> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case 16:
			reduceDataToStatsKernel<T_in, T_mid, 16, false> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case  8:
			reduceDataToStatsKernel<T_in, T_mid, 8, false> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case  4:
			reduceDataToStatsKernel<T_in, T_mid, 4, false> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case  2:
			reduceDataToStatsKernel<T_in, T_mid, 2, false> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case  1:
			reduceDataToStatsKernel<T_in, T_mid, 1, false> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;
		}
	}
}

template <typename T_mid, unsigned int blockSize, bool nIsPow2>
__global__ void reduceStatsToStatsKernel(intermediateStatsData<T_mid>* g_idata, intermediateStatsData<T_mid>* g_odata, unsigned int n)
{
	intermediateStatsData<T_mid>* sdata = SharedMemory<intermediateStatsData<T_mid>>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
	unsigned int gridSize = blockSize * 2 * gridDim.x;

	intermediateStatsData<T_mid> myReduced;
	myReduced.initialize();

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		myReduced = intermediateStatsDataBinaryOp<T_mid>(myReduced, g_idata[i]);

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (nIsPow2 || ((i + blockSize) < n))
		{
			myReduced = intermediateStatsDataBinaryOp<T_mid>(myReduced, g_idata[i + blockSize]);
		}

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = myReduced;
	__syncthreads();

	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp<T_mid>(myReduced, sdata[tid + 256]);
	}

	__syncthreads();

	if ((blockSize >= 256) && (tid < 128))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp<T_mid>(myReduced, sdata[tid + 128]);
	}

	__syncthreads();

	if ((blockSize >= 128) && (tid < 64))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp<T_mid>(myReduced, sdata[tid + 64]);
	}

	__syncthreads();

	// fully unroll reduction within a single warp
	if ((blockSize >= 64) && (tid < 32))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp<T_mid>(myReduced, sdata[tid + 32]);
	}

	__syncthreads();

	if ((blockSize >= 32) && (tid < 16))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp<T_mid>(myReduced, sdata[tid + 16]);
	}

	__syncthreads();

	if ((blockSize >= 16) && (tid < 8))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp<T_mid>(myReduced, sdata[tid + 8]);
	}

	__syncthreads();

	if ((blockSize >= 8) && (tid < 4))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp<T_mid>(myReduced, sdata[tid + 4]);
	}

	__syncthreads();

	if ((blockSize >= 4) && (tid < 2))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp<T_mid>(myReduced, sdata[tid + 2]);
	}

	__syncthreads();

	if ((blockSize >= 2) && (tid < 1))
	{
		sdata[tid] = myReduced = intermediateStatsDataBinaryOp<T_mid>(myReduced, sdata[tid + 1]);
	}

	__syncthreads();

	// write result for this block to global mem
	if (tid == 0)
	{
		g_odata[blockIdx.x] = myReduced;
	}
}

template <typename T_mid>
inline void reduceStatsToStats(int size, int threads, int blocks, intermediateStatsData<T_mid>* d_idata, 
	intermediateStatsData<T_mid>* d_odata, cudaStream_t stream)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? (2 * threads * sizeof(intermediateStatsData<T_mid>)) : (threads * sizeof(intermediateStatsData<T_mid>));

	if (isPow2(size))
	{
		switch (threads)
		{
		case 512:
			reduceStatsToStatsKernel<T_mid, 512, true> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case 256:
			reduceStatsToStatsKernel<T_mid, 256, true> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case 128:
			reduceStatsToStatsKernel<T_mid, 128, true> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case 64:
			reduceStatsToStatsKernel<T_mid, 64, true> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case 32:
			reduceStatsToStatsKernel<T_mid, 32, true> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case 16:
			reduceStatsToStatsKernel<T_mid, 16, true> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case  8:
			reduceStatsToStatsKernel<T_mid, 8, true> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case  4:
			reduceStatsToStatsKernel<T_mid, 4, true> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case  2:
			reduceStatsToStatsKernel<T_mid, 2, true> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case  1:
			reduceStatsToStatsKernel<T_mid, 1, true> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;
		}
	}
	else
	{
		switch (threads)
		{
		case 512:
			reduceStatsToStatsKernel<T_mid, 512, false> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case 256:
			reduceStatsToStatsKernel<T_mid, 256, false> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case 128:
			reduceStatsToStatsKernel<T_mid, 128, false> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case 64:
			reduceStatsToStatsKernel<T_mid, 64, false> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case 32:
			reduceStatsToStatsKernel<T_mid, 32, false> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case 16:
			reduceStatsToStatsKernel<T_mid, 16, false> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case  8:
			reduceStatsToStatsKernel<T_mid, 8, false> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case  4:
			reduceStatsToStatsKernel<T_mid, 4, false> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case  2:
			reduceStatsToStatsKernel<T_mid, 2, false> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;

		case  1:
			reduceStatsToStatsKernel<T_mid, 1, false> << < dimGrid, dimBlock, smemSize, stream >> > (d_idata, d_odata, size);
			break;
		}
	}
}

template <typename T_mid, typename T_out>
__global__ void convertToFinalStatsDataKernel(intermediateStatsData<T_mid>* g_idata, finalStatsData<T_out>* g_odata)
{
	g_odata[0] = convertToFinalStatsData<T_mid, T_out>(g_idata[0]);
}

// Compute the number of threads and blocks to use for the reduction kernel.
// We set threads / block to the minimum of maxThreads and n/2.
// We also observe the maximum specified number of blocks, because each thread in
// that kernel can process a variable number of elements.
inline void getNumBlocksAndThreads(int n, int& blocks, int& threads)
{
	threads = (n < (MAX_THREADS * 2)) ? (nextPow2((n + 1) / 2)) : (MAX_THREADS);
	blocks = MIN(((n + threads * 2 - 1) / (threads * 2)), MAX_BLOCKS);

	return;
}

// takes data in T_in datatype
// makes calculations in T_mid datatype
// produces results in T_out datatype 
template <typename T_in, typename T_mid, typename T_out>
void getStatsFromData(int nr_elems, T_in* inputData, intermediateStatsData<T_mid>* intermediateData, 
	finalStatsData<T_out>* finalData, cudaStream_t stream)
{
	int numThreads, numBlocks;
	getNumBlocksAndThreads(nr_elems, numBlocks, numThreads);

	// Run reduction
	reduceDataToStats<T_in, T_mid>(nr_elems, numThreads, numBlocks, inputData, intermediateData, stream);
	CHECK_CUDA_ERROR(cudaGetLastError());

	int currNumOfElems = numBlocks;
	while (currNumOfElems > 1)
	{
		int threads = 0, blocks = 0;
		getNumBlocksAndThreads(currNumOfElems, blocks, threads);

		reduceStatsToStats<T_mid>(currNumOfElems, threads, blocks, intermediateData, intermediateData, stream);
		CHECK_CUDA_ERROR(cudaGetLastError());

		currNumOfElems = (currNumOfElems + (threads * 2 - 1)) / (threads * 2);
	}

	convertToFinalStatsDataKernel<T_mid, T_out> << < 1, 1, 0, stream >> > (intermediateData, finalData);
	CHECK_CUDA_ERROR(cudaGetLastError());

	return;
}

// Instantiate the reduction function for different types

template void
getStatsFromData<int, float, double>(int nr_elems, int* inputData, intermediateStatsData<float>* intermediateData,
	finalStatsData<double>* finalData, cudaStream_t stream);

template void
getStatsFromData<float, float, double>(int nr_elems, float* inputData, intermediateStatsData<float>* intermediateData,
	finalStatsData<double>* finalData, cudaStream_t stream);

template void
getStatsFromData<double, double, double>(int nr_elems, double* inputData, intermediateStatsData<double>* intermediateData,
	finalStatsData<double>* finalData, cudaStream_t stream);
