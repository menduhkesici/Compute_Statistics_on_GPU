#include "../include/kernel.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <thread>
#include <atomic>
#include <type_traits>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include "../include/handler.cuh"
#include "../include/stats_data.cuh"
#include "../include/reduce.cuh"

const int VECTOR_DIM = (int)(20e6); // number of elements in the vectors
const int NUM_OF_VECTORS = 100;		// number of vectors

extern const int MAX_THREADS = 128; // max number of threads per block in reduce kernel
extern const int MAX_BLOCKS = 256;	// max number of blocks per grid in reduce kernel

enum class reduceType
{
	INT_TO_FLOAT_TO_DOUBLE = 0,
	FLOAT_TO_FLOAT_TO_DOUBLE = 1,
	DOUBLE_TO_DOUBLE_TO_DOUBLE = 2
};

template <unsigned int NUM_OF_SLOTS>
struct threadDataStruct
{
	int deviceID;
	int turnCount;
	int nr_vectors;
	int nr_elems[NUM_OF_SLOTS];
	void *host_vec[NUM_OF_SLOTS];
	void *result[NUM_OF_SLOTS];
	reduceType reduce_type[NUM_OF_SLOTS];
};

template <unsigned int NUM_OF_SLOTS>
void threadCPU(void *pvoidData, std::atomic<int> &shared_val);

void kernel(void)
{
	// Start time for the program
	std::chrono::steady_clock::time_point begin_total = std::chrono::steady_clock::now();

	try
	{
		typedef double T_in;  // takes data in T_in datatype
		typedef double T_mid; // makes calculations in T_mid datatype
		typedef double T_out; // produces results in T_out datatype

		// Vector dimensions
		int nr_elems_A = VECTOR_DIM;

		// Print program information
		std::cout.precision(3);
		std::cout << "Vector Statistics with Transform-Reduce Operation" << std::endl
				  << "Calculates statistics of elements of " << NUM_OF_VECTORS << " vectors with " << (double)nr_elems_A << " elements." << std::endl;

		// Allocate and initialize the host vectors
		thrust::host_vector<T_in> h_A[NUM_OF_VECTORS];
		const T_in MAX_VALUE = 1e6;
		const T_in MIN_VALUE = 1.0;
		srand((unsigned int)time(NULL));
		for (int i = 0; i < NUM_OF_VECTORS; i++)
		{
			h_A[i] = thrust::host_vector<T_in>(nr_elems_A);
			for (int j = 0; j < nr_elems_A; j++)
			{
				h_A[i][j] = ((T_in)rand() / RAND_MAX) * (i + 1) * (MAX_VALUE - MIN_VALUE) + MIN_VALUE + i;
			}
		}

		// Start time for CPU part
		std::cout << std::endl
				  << "Calculation by using CPU:" << std::endl;
		std::chrono::steady_clock::time_point begin_cpu = std::chrono::steady_clock::now();

		T_out mean[NUM_OF_VECTORS], var[NUM_OF_VECTORS], min[NUM_OF_VECTORS], max[NUM_OF_VECTORS];

		for (int i = 0; i < NUM_OF_VECTORS; i++)
		{
			T_out sum = std::accumulate(h_A[i].begin(), h_A[i].end(), 0.0);
			mean[i] = sum / nr_elems_A;
			T_out accum = 0.0;
			std::for_each(h_A[i].begin(), h_A[i].end(), [&](const T_in d)
						  { accum += (d - mean[i]) * (d - mean[i]); });
			var[i] = accum / (nr_elems_A - 1);
			auto minmaxresult = std::minmax_element(h_A[i].begin(), h_A[i].end());
			min[i] = *minmaxresult.first;
			max[i] = *minmaxresult.second;
		}

		// End time for CPU part
		double time_cpu = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin_cpu).count();

		// Print results for some signals
		std::cout << "Vector No          : "
				  << std::setw(10) << std::right << "0"
				  << " |"
				  << std::setw(10) << std::right << "1"
				  << " |"
				  << std::setw(10) << std::right << "2"
				  << " |"
				  << "\t...\t|"
				  << std::setw(10) << std::right << NUM_OF_VECTORS - 3 << " |"
				  << std::setw(10) << std::right << NUM_OF_VECTORS - 2 << " |"
				  << std::setw(10) << std::right << NUM_OF_VECTORS - 1 << " |" << std::endl;
		std::cout << "Number of Elements : "
				  << std::setw(10) << std::right << nr_elems_A << " |"
				  << std::setw(10) << std::right << nr_elems_A << " |"
				  << std::setw(10) << std::right << nr_elems_A << " |"
				  << "\t...\t|"
				  << std::setw(10) << std::right << nr_elems_A << " |"
				  << std::setw(10) << std::right << nr_elems_A << " |"
				  << std::setw(10) << std::right << nr_elems_A << " |" << std::endl;
		std::cout << "Minimum Value      : "
				  << std::setw(10) << std::right << min[0] << " |"
				  << std::setw(10) << std::right << min[1] << " |"
				  << std::setw(10) << std::right << min[2] << " |"
				  << "\t...\t|"
				  << std::setw(10) << std::right << min[NUM_OF_VECTORS - 3] << " |"
				  << std::setw(10) << std::right << min[NUM_OF_VECTORS - 2] << " |"
				  << std::setw(10) << std::right << min[NUM_OF_VECTORS - 1] << " |" << std::endl;
		std::cout << "Maximum Value      : "
				  << std::setw(10) << std::right << max[0] << " |"
				  << std::setw(10) << std::right << max[1] << " |"
				  << std::setw(10) << std::right << max[2] << " |"
				  << "\t...\t|"
				  << std::setw(10) << std::right << max[NUM_OF_VECTORS - 3] << " |"
				  << std::setw(10) << std::right << max[NUM_OF_VECTORS - 2] << " |"
				  << std::setw(10) << std::right << max[NUM_OF_VECTORS - 1] << " |" << std::endl;
		std::cout << "Mean Value         : "
				  << std::setw(10) << std::right << mean[0] << " |"
				  << std::setw(10) << std::right << mean[1] << " |"
				  << std::setw(10) << std::right << mean[2] << " |"
				  << "\t...\t|"
				  << std::setw(10) << std::right << mean[NUM_OF_VECTORS - 3] << " |"
				  << std::setw(10) << std::right << mean[NUM_OF_VECTORS - 2] << " |"
				  << std::setw(10) << std::right << mean[NUM_OF_VECTORS - 1] << " |" << std::endl;
		std::cout << "Variance           : "
				  << std::setw(10) << std::right << var[0] << " |"
				  << std::setw(10) << std::right << var[1] << " |"
				  << std::setw(10) << std::right << var[2] << " |"
				  << "\t...\t|"
				  << std::setw(10) << std::right << var[NUM_OF_VECTORS - 3] << " |"
				  << std::setw(10) << std::right << var[NUM_OF_VECTORS - 2] << " |"
				  << std::setw(10) << std::right << var[NUM_OF_VECTORS - 1] << " |" << std::endl;
		std::cout << "Standard Deviation : "
				  << std::setw(10) << std::right << std::sqrt(var[0]) << " |"
				  << std::setw(10) << std::right << std::sqrt(var[1]) << " |"
				  << std::setw(10) << std::right << std::sqrt(var[2]) << " |"
				  << "\t...\t|"
				  << std::setw(10) << std::right << std::sqrt(var[NUM_OF_VECTORS - 3]) << " |"
				  << std::setw(10) << std::right << std::sqrt(var[NUM_OF_VECTORS - 2]) << " |"
				  << std::setw(10) << std::right << std::sqrt(var[NUM_OF_VECTORS - 1]) << " |" << std::endl;

		// Print elapsed time for CPU part
		std::cout << "Time elapsed with CPU = " << time_cpu << " sec." << std::endl;

		// Check GPU devices
		int deviceCount;
		CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
		if (deviceCount < 1)
		{
			std::cerr << std::endl
					  << "No CUDA Device found!" << std::endl;
			exit(EXIT_FAILURE);
		}
		// Uncomment below line do deactivate usage of multiple GPUs
		// deviceCount = 1;
		std::cout << std::endl
				  << "Calculation by using " << deviceCount << " GPU devices (";
		for (int i = 0; i < deviceCount; i++)
		{
			cudaDeviceProp prop;
			CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, i));
			std::cout << prop.name << ", ";
			if ((MAX_THREADS > prop.maxThreadsDim[0]) || (MAX_BLOCKS > prop.maxGridSize[0]))
			{
				std::cerr << std::endl
						  << "Required number of threads or blocks is too large for the device!" << std::endl;
				exit(EXIT_FAILURE);
			}
			if (prop.canMapHostMemory != 1)
			{
				std::cerr << std::endl
						  << "Device can not map host memory!" << std::endl;
				exit(EXIT_FAILURE);
			}
			int hostRegSupported;
			CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&hostRegSupported, cudaDevAttrHostRegisterSupported, i));
			if (hostRegSupported != 1)
			{
				std::cerr << std::endl
						  << "Device does not support host memory registration!" << std::endl;
				exit(EXIT_FAILURE);
			}
		}
		std::cout << "):" << std::endl;

		// Start time for GPU part
		cudaEvent_t start, stop;
		CHECK_CUDA_ERROR(cudaSetDevice(0));
		CHECK_CUDA_ERROR(cudaEventCreate(&start));
		CHECK_CUDA_ERROR(cudaEventCreate(&stop));
		CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
		std::chrono::steady_clock::time_point begin_gpu = std::chrono::steady_clock::now();
		CHECK_CUDA_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));

		// Share the data among GPUs
		const int NUM_OF_THREADS = 1;
		std::vector<threadDataStruct<NUM_OF_VECTORS>> threadData(NUM_OF_THREADS);
		std::atomic<int> shared_val{0};
		finalStatsData<T_out> result[NUM_OF_VECTORS];
		for (int i = 0; i < NUM_OF_VECTORS; i++)
		{
			result[i].initialize();
		}
		for (int i = 0; i < NUM_OF_THREADS; i++)
		{
			threadData[i].deviceID = i;
			threadData[i].turnCount = 10;
			threadData[i].nr_vectors = NUM_OF_VECTORS;
			for (int j = 0; j < NUM_OF_VECTORS; j++)
			{
				threadData[i].nr_elems[j] = VECTOR_DIM;
				threadData[i].host_vec[j] = (void *)thrust::raw_pointer_cast(h_A[j].data());
				threadData[i].result[j] = result + j;
				if ((std::is_same<T_in, int>::value) & (std::is_same<T_mid, float>::value) & (std::is_same<T_out, double>::value))
				{
					threadData[i].reduce_type[j] = reduceType::INT_TO_FLOAT_TO_DOUBLE;
				}
				else if ((std::is_same<T_in, float>::value) & (std::is_same<T_mid, float>::value) & (std::is_same<T_out, double>::value))
				{
					threadData[i].reduce_type[j] = reduceType::FLOAT_TO_FLOAT_TO_DOUBLE;
				}
				else if ((std::is_same<T_in, double>::value) & (std::is_same<T_mid, double>::value) & (std::is_same<T_out, double>::value))
				{
					threadData[i].reduce_type[j] = reduceType::DOUBLE_TO_DOUBLE_TO_DOUBLE;
				}
				else
				{
					std::cerr << std::endl
							  << "Unidentified reduce type on vector no: " << j << "!" << std::endl;
					exit(EXIT_FAILURE);
				}
			}
		}
		if (NUM_OF_THREADS > 1)
			threadData[1].turnCount = 1;

		// Pin host memory vectors
		for (int i = 0; i < NUM_OF_VECTORS; i++)
		{
			CHECK_CUDA_ERROR(cudaHostRegister((void *)thrust::raw_pointer_cast(h_A[i].data()), VECTOR_DIM * sizeof(T_in),
											  cudaHostRegisterMapped | cudaHostRegisterPortable));
		}
		CHECK_CUDA_ERROR(cudaHostRegister((void *)result, NUM_OF_VECTORS * sizeof(finalStatsData<T_out>),
										  cudaHostRegisterMapped | cudaHostRegisterPortable));

		// Execution on GPUs by using threads
		std::vector<std::thread> threadVector;
		for (int i = 1; i < NUM_OF_THREADS; i++)
		{
			threadVector.push_back(std::thread(threadCPU<NUM_OF_VECTORS>, &(threadData[i]), std::ref(shared_val)));
		}
		threadCPU<NUM_OF_VECTORS>(&(threadData[0]), std::ref(shared_val));
		for (std::thread &th : threadVector)
		{
			if (th.joinable())
				th.join();
		}

		// Unpin host memory vector
		for (int i = 0; i < NUM_OF_VECTORS; i++)
		{
			CHECK_CUDA_ERROR(cudaHostUnregister((void *)thrust::raw_pointer_cast(h_A[i].data())));
		}
		CHECK_CUDA_ERROR(cudaHostUnregister((void *)result));

		// End time for GPU part
		CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
		CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
		double time_gpu2 = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin_gpu).count();
		float time_gpu;
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_gpu, start, stop));
		CHECK_CUDA_ERROR(cudaEventDestroy(start));
		CHECK_CUDA_ERROR(cudaEventDestroy(stop));

		// Print results for some signals
		std::cout << "Vector No          : "
				  << std::setw(10) << std::right << "0"
				  << " |"
				  << std::setw(10) << std::right << "1"
				  << " |"
				  << std::setw(10) << std::right << "2"
				  << " |"
				  << "\t...\t|"
				  << std::setw(10) << std::right << NUM_OF_VECTORS - 3 << " |"
				  << std::setw(10) << std::right << NUM_OF_VECTORS - 2 << " |"
				  << std::setw(10) << std::right << NUM_OF_VECTORS - 1 << " |" << std::endl;
		std::cout << "Number of Elements : "
				  << std::setw(10) << std::right << result[0].num_elems << " |"
				  << std::setw(10) << std::right << result[1].num_elems << " |"
				  << std::setw(10) << std::right << result[2].num_elems << " |"
				  << "\t...\t|"
				  << std::setw(10) << std::right << result[NUM_OF_VECTORS - 3].num_elems << " |"
				  << std::setw(10) << std::right << result[NUM_OF_VECTORS - 2].num_elems << " |"
				  << std::setw(10) << std::right << result[NUM_OF_VECTORS - 1].num_elems << " |" << std::endl;
		std::cout << "Minimum Value      : "
				  << std::setw(10) << std::right << result[0].min << " |"
				  << std::setw(10) << std::right << result[1].min << " |"
				  << std::setw(10) << std::right << result[2].min << " |"
				  << "\t...\t|"
				  << std::setw(10) << std::right << result[NUM_OF_VECTORS - 3].min << " |"
				  << std::setw(10) << std::right << result[NUM_OF_VECTORS - 2].min << " |"
				  << std::setw(10) << std::right << result[NUM_OF_VECTORS - 1].min << " |" << std::endl;
		std::cout << "Maximum Value      : "
				  << std::setw(10) << std::right << result[0].max << " |"
				  << std::setw(10) << std::right << result[1].max << " |"
				  << std::setw(10) << std::right << result[2].max << " |"
				  << "\t...\t|"
				  << std::setw(10) << std::right << result[NUM_OF_VECTORS - 3].max << " |"
				  << std::setw(10) << std::right << result[NUM_OF_VECTORS - 2].max << " |"
				  << std::setw(10) << std::right << result[NUM_OF_VECTORS - 1].max << " |" << std::endl;
		std::cout << "Mean Value         : "
				  << std::setw(10) << std::right << result[0].mean << " |"
				  << std::setw(10) << std::right << result[1].mean << " |"
				  << std::setw(10) << std::right << result[2].mean << " |"
				  << "\t...\t|"
				  << std::setw(10) << std::right << result[NUM_OF_VECTORS - 3].mean << " |"
				  << std::setw(10) << std::right << result[NUM_OF_VECTORS - 2].mean << " |"
				  << std::setw(10) << std::right << result[NUM_OF_VECTORS - 1].mean << " |" << std::endl;
		std::cout << "Variance           : "
				  << std::setw(10) << std::right << result[0].variance << " |"
				  << std::setw(10) << std::right << result[1].variance << " |"
				  << std::setw(10) << std::right << result[2].variance << " |"
				  << "\t...\t|"
				  << std::setw(10) << std::right << result[NUM_OF_VECTORS - 3].variance << " |"
				  << std::setw(10) << std::right << result[NUM_OF_VECTORS - 2].variance << " |"
				  << std::setw(10) << std::right << result[NUM_OF_VECTORS - 1].variance << " |" << std::endl;
		std::cout << "Standard Deviation : "
				  << std::setw(10) << std::right << result[0].std_dev << " |"
				  << std::setw(10) << std::right << result[1].std_dev << " |"
				  << std::setw(10) << std::right << result[2].std_dev << " |"
				  << "\t...\t|"
				  << std::setw(10) << std::right << result[NUM_OF_VECTORS - 3].std_dev << " |"
				  << std::setw(10) << std::right << result[NUM_OF_VECTORS - 2].std_dev << " |"
				  << std::setw(10) << std::right << result[NUM_OF_VECTORS - 1].std_dev << " |" << std::endl;

		// Print elapsed time for GPU part
		time_gpu /= 1e3f; // conversion from msec to sec
		std::cout << "Time elapsed with GPU = " << time_gpu << " sec." << std::endl;

		// Print the performance comparison
		std::cout << std::endl
				  << "GPU performs the operation " << (time_cpu / time_gpu) << " times faster than CPU." << std::endl;
	}
	catch (const thrust::system_error &err)
	{
		std::cerr << std::endl
				  << "Thrust error (" << err.what() << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}
	catch (const std::bad_alloc &err)
	{
		std::cerr << std::endl
				  << "Memory allocation error (" << err.what() << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}
	catch (const std::runtime_error &err)
	{
		std::cerr << std::endl
				  << "Runtime error (" << err.what() << ")!" << std::endl;
		exit(EXIT_FAILURE);
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	int deviceCount;
	CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
	for (int i = 0; i < deviceCount; i++)
	{
		CHECK_CUDA_ERROR(cudaSetDevice(i));
		CHECK_CUDA_ERROR(cudaDeviceReset());
	}

	// Print elapsed time for the program
	double time_total = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin_total).count();
	std::cout << std::endl
			  << "The program finished in " << time_total << " sec." << std::endl;

	return;
}

template <unsigned int NUM_OF_SLOTS>
void threadCPU(void *pvoidData, std::atomic<int> &shared_val)
{
	threadDataStruct<NUM_OF_SLOTS> *data = (threadDataStruct<NUM_OF_SLOTS> *)pvoidData;
	if (data->deviceID != 0)
	{
		CHECK_CUDA_ERROR(cudaSetDevice(data->deviceID));
		CHECK_CUDA_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
	}

	// Create streams
	const int NUM_OF_STREAMS = 8;
	cudaStream_t stream[NUM_OF_STREAMS];
	for (int i = 0; i < NUM_OF_STREAMS; i++)
	{
		CHECK_CUDA_ERROR(cudaStreamCreate(&stream[i]));
	}

	// Allocate the device vectors
	void *d_A[NUM_OF_STREAMS];
	for (int i = 0; i < NUM_OF_STREAMS; i++)
	{
		CHECK_CUDA_ERROR(cudaMalloc(&d_A[i], VECTOR_DIM * sizeof(double)));
	}

	// Temporary storage variables for reduce operation
	void *d_temp_storage[NUM_OF_STREAMS];
	for (int i = 0; i < NUM_OF_STREAMS; i++)
	{
		CHECK_CUDA_ERROR(cudaMalloc(&d_temp_storage[i], MAX_BLOCKS * 2 * sizeof(intermediateStatsData<double>)));
	}

	int turnCount = 0;
	int val = shared_val.fetch_add(NUM_OF_STREAMS, std::memory_order_relaxed);
	while (val < data->nr_vectors)
	{
		// Copy host input vector to device
		for (int i = 0; i < NUM_OF_STREAMS; i++)
		{
			int curr_vec = val + i;
			if (curr_vec < data->nr_vectors)
			{
				size_t T_in_size;
				if (data->reduce_type[curr_vec] == reduceType::INT_TO_FLOAT_TO_DOUBLE)
				{
					T_in_size = sizeof(int);
				}
				else if (data->reduce_type[curr_vec] == reduceType::FLOAT_TO_FLOAT_TO_DOUBLE)
				{
					T_in_size = sizeof(float);
				}
				else if (data->reduce_type[curr_vec] == reduceType::DOUBLE_TO_DOUBLE_TO_DOUBLE)
				{
					T_in_size = sizeof(double);
				}
				else
				{
					std::cerr << std::endl
							  << "Unidentified reduce type on vector no: " << curr_vec << "!" << std::endl;
					exit(EXIT_FAILURE);
				}
				CHECK_CUDA_ERROR(cudaMemcpyAsync(d_A[i], data->host_vec[curr_vec], data->nr_elems[curr_vec] * T_in_size,
												 cudaMemcpyHostToDevice, stream[i]));
			}
			else
			{
				break;
			}
		}

		// Launch CUDA Kernel
		for (int i = 0; i < NUM_OF_STREAMS; i++)
		{
			int curr_vec = val + i;
			if (curr_vec < data->nr_vectors)
			{
				if (data->reduce_type[curr_vec] == reduceType::INT_TO_FLOAT_TO_DOUBLE)
				{
					getStatsFromData<int, float, double>(data->nr_elems[curr_vec], (int *)(d_A[i]),
														 (intermediateStatsData<float> *)d_temp_storage[i], (finalStatsData<double> *)d_temp_storage[i], stream[i]);
				}
				else if (data->reduce_type[curr_vec] == reduceType::FLOAT_TO_FLOAT_TO_DOUBLE)
				{
					getStatsFromData<float, float, double>(data->nr_elems[curr_vec], (float *)(d_A[i]),
														   (intermediateStatsData<float> *)d_temp_storage[i], (finalStatsData<double> *)d_temp_storage[i], stream[i]);
				}
				else if (data->reduce_type[curr_vec] == reduceType::DOUBLE_TO_DOUBLE_TO_DOUBLE)
				{
					getStatsFromData<double, double, double>(data->nr_elems[curr_vec], (double *)(d_A[i]),
															 (intermediateStatsData<double> *)d_temp_storage[i], (finalStatsData<double> *)d_temp_storage[i], stream[i]);
				}
				else
				{
					std::cerr << std::endl
							  << "Unidentified reduce type on vector no: " << curr_vec << "!" << std::endl;
					exit(EXIT_FAILURE);
				}
			}
			else
			{
				break;
			}
		}

		// Copy device output vector to host
		for (int i = 0; i < NUM_OF_STREAMS; i++)
		{
			int curr_vec = val + i;
			if (curr_vec < data->nr_vectors)
			{
				size_t T_out_size;
				if (data->reduce_type[curr_vec] == reduceType::INT_TO_FLOAT_TO_DOUBLE)
				{
					T_out_size = sizeof(finalStatsData<double>);
				}
				else if (data->reduce_type[curr_vec] == reduceType::FLOAT_TO_FLOAT_TO_DOUBLE)
				{
					T_out_size = sizeof(finalStatsData<double>);
				}
				else if (data->reduce_type[curr_vec] == reduceType::DOUBLE_TO_DOUBLE_TO_DOUBLE)
				{
					T_out_size = sizeof(finalStatsData<double>);
				}
				else
				{
					std::cerr << std::endl
							  << "Unidentified reduce type on vector no: " << curr_vec << "!" << std::endl;
					exit(EXIT_FAILURE);
				}
				CHECK_CUDA_ERROR(cudaMemcpyAsync(data->result[curr_vec], d_temp_storage[i], T_out_size,
												 cudaMemcpyDeviceToHost, stream[i]));
			}
			else
			{
				break;
			}
		}

		if ((++turnCount % data->turnCount) == 0)
		{
			CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		}

		val = shared_val.fetch_add(NUM_OF_STREAMS, std::memory_order_relaxed);
	}

	// Synchronize and destroy streams, deallocate device vectors and temporary storage
	for (int i = 0; i < NUM_OF_STREAMS; i++)
	{
		CHECK_CUDA_ERROR(cudaStreamSynchronize(stream[i]));
		CHECK_CUDA_ERROR(cudaStreamDestroy(stream[i]));
		CHECK_CUDA_ERROR(cudaFree(d_A[i]));
		CHECK_CUDA_ERROR(cudaFree(d_temp_storage[i]));
	}

	return;
}
