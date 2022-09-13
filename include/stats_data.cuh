#pragma once

#include <float.h>
#include <cuda_runtime.h>

#ifndef MIN
#define MIN( x , y ) (((x) < (y)) ? (x) : (y))
#endif

#ifndef MAX
#define MAX( x , y ) (((x) > (y)) ? (x) : (y))
#endif

// structure used to accumulate the number of elements, min, max, 
// sum and squared sum values accumulated until that point
template <typename T_mid>
struct intermediateStatsData
{
	unsigned int num_elems;
	T_mid min;
	T_mid max;
	T_mid sum;
	T_mid sq_sum;

	// initialize to the identity element
	__host__ __device__ __forceinline__ void initialize()
	{
		num_elems = 0;
		min = FLT_MAX;
		max = -FLT_MAX;
		sum = sq_sum = 0;
	}
};

// structure to store final statistical results
template <typename T_out>
struct finalStatsData
{
	unsigned int num_elems;
	T_out min;
	T_out max;
	T_out sum;
	T_out mean;
	T_out variance;
	T_out std_dev;

	// initialize to the identity element
	__host__ __device__ __forceinline__ void initialize()
	{
		num_elems = 0;
		min = FLT_MAX;
		max = -FLT_MAX;
		sum = mean = variance = std_dev = 0;
	}
};

// convertToIntermediateStatsData is a function that takes in a value x and
// returns an intermediateStatsData used while accumulating sum and squared sum values
template <typename T_in, typename T_mid>
__host__ __device__ __forceinline__ intermediateStatsData<T_mid> convertToIntermediateStatsData(const T_in& x)
{
	intermediateStatsData<T_mid> result;
	result.num_elems = 1;
	result.min = x;
	result.max = x;
	result.sum = x;
	result.sq_sum = (T_mid)x * x;

	return result;
};

// intermediateStatsDataBinaryOp is a function that accepts
// two intermediateStatsData structs and returns their combination
template <typename T_mid>
__host__ __device__ __forceinline__ intermediateStatsData<T_mid>
	intermediateStatsDataBinaryOp(const intermediateStatsData<T_mid>& x, const intermediateStatsData <T_mid>& y)
{
	intermediateStatsData<T_mid> result;
	result.num_elems = x.num_elems + y.num_elems;
	result.min = MIN(x.min, y.min);
	result.max = MAX(x.max, y.max);
	result.sum = x.sum + y.sum;
	result.sq_sum = x.sq_sum + y.sq_sum;

	return result;
};

// converts final intermediateStatsData to finalStatsData
template <typename T_mid, typename T_out>
__host__ __device__ __forceinline__ finalStatsData<T_out> convertToFinalStatsData(const intermediateStatsData<T_mid>& x)
{
	finalStatsData<T_out> result;
	result.num_elems = x.num_elems;
	result.min = x.min;
	result.max = x.max;
	result.sum = x.sum;
	result.mean = x.sum / x.num_elems;
	result.variance = (x.sq_sum - x.sum * x.sum / x.num_elems) / (x.num_elems - 1);
	result.std_dev = sqrt(result.variance);

	return result;
};
