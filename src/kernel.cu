#include "../include/kernel.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <type_traits>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <thrust/device_vector.h>
#include "../include/cuda_handler.cuh"
#include "../include/stats_data.cuh"
#include "../include/reduce.cuh"

const int VECTOR_DIM = (int)(20e6); // number of elements in the vectors
const int NUM_OF_VECTORS = 100;		// number of vectors
typedef double T_in;				// takes data in T_in datatype

template <unsigned int NUM_OF_SLOTS>
struct threadDataStruct
{
	int deviceID;
	int turnCount;
	int nr_vectors;
	int nr_elems[NUM_OF_SLOTS];
	thrust::host_vector<T_in, thrust::cuda::experimental::pinned_allocator<T_in>> *host_vec[NUM_OF_SLOTS];
	finalStatsData *result[NUM_OF_SLOTS];
};

template <unsigned int NUM_OF_SLOTS>
void threadCPU(void *pvoidData, std::atomic<int> &shared_val);

void printResults(thrust::host_vector<finalStatsData, thrust::cuda::experimental::pinned_allocator<finalStatsData>> &result);

void kernel(void)
{
	// Start time for the program
	std::chrono::steady_clock::time_point begin_total = std::chrono::steady_clock::now();

	try
	{
		// Print program information
		std::cout.precision(3);
		std::cout << "Vector Statistics with Transform-Reduce Operation" << std::endl
				  << "Calculates statistics of elements of " << NUM_OF_VECTORS << " vectors with " << (double)VECTOR_DIM << " double elements." << std::endl;

		// Allocate and initialize the host vectors
		thrust::host_vector<T_in, thrust::cuda::experimental::pinned_allocator<T_in>> h_A[NUM_OF_VECTORS];
		const T_in MAX_VALUE = 1e3;
		const T_in MIN_VALUE = 1.0;
		srand((unsigned int)time(NULL));
		for (int i = 0; i < NUM_OF_VECTORS; i++)
		{
			h_A[i] = thrust::host_vector<T_in, thrust::cuda::experimental::pinned_allocator<T_in>>(VECTOR_DIM);
			for (int j = 0; j < VECTOR_DIM; j++)
				h_A[i][j] = ((double)rand() / RAND_MAX) * (i + 1) * (MAX_VALUE - MIN_VALUE) + MIN_VALUE + i;
		}
		thrust::host_vector<finalStatsData, thrust::cuda::experimental::pinned_allocator<finalStatsData>> result(NUM_OF_VECTORS);
		for (int i = 0; i < NUM_OF_VECTORS; i++)
			result[i].initialize();

		// Start time for CPU part
		std::cout << std::endl
				  << "Calculation by using CPU:" << std::endl;
		std::chrono::steady_clock::time_point begin_cpu = std::chrono::steady_clock::now();

		for (int i = 0; i < NUM_OF_VECTORS; i++)
		{
			intermediateStatsData statData;
			statData.initialize();
			for (int j = 0; j < VECTOR_DIM; j++)
				statData = intermediateStatsDataBinaryOp(statData, convertToIntermediateStatsData(h_A[i][j]));
			result[i] = convertToFinalStatsData(statData);
		}

		// End time for CPU part
		double time_cpu = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin_cpu).count();

		// Print results for some signals
		printResults(result);

		// Print elapsed time for CPU part
		std::cout << "Time elapsed with CPU = " << time_cpu << " sec." << std::endl;

		// Check GPU devices
		int deviceCount;
		CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
		if (deviceCount < 1)
			throw std::runtime_error("No CUDA Device found!");
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
				throw std::runtime_error("Required number of threads or blocks is too large for the device!");
			if (prop.canMapHostMemory != 1)
				throw std::runtime_error("Device can not map host memory!");
			int hostRegSupported;
			CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&hostRegSupported, cudaDevAttrHostRegisterSupported, i));
			if (hostRegSupported != 1)
				throw std::runtime_error("Device does not support host memory registration!");
		}
		std::cout << "):" << std::endl;

		// Reinitialize results vector
		for (int i = 0; i < NUM_OF_VECTORS; i++)
			result[i].initialize();

		// Share the data among GPUs
		const int NUM_OF_THREADS = deviceCount;
		std::vector<threadDataStruct<NUM_OF_VECTORS>> threadData(NUM_OF_THREADS);
		std::atomic<int> shared_val{0};
		for (int i = 0; i < NUM_OF_THREADS; i++)
		{
			threadData[i].deviceID = i;
			threadData[i].turnCount = (NUM_OF_VECTORS + 9) / 10;
			threadData[i].nr_vectors = NUM_OF_VECTORS;
			for (int j = 0; j < NUM_OF_VECTORS; j++)
			{
				threadData[i].nr_elems[j] = VECTOR_DIM;
				threadData[i].host_vec[j] = &h_A[j];
				threadData[i].result[j] = &result[j];
			}
		}

		// Start time for GPU part
		cudaEvent_t start, stop;
		CHECK_CUDA_ERROR(cudaSetDevice(0));
		CHECK_CUDA_ERROR(cudaEventCreate(&start));
		CHECK_CUDA_ERROR(cudaEventCreate(&stop));
		CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
		std::chrono::steady_clock::time_point begin_gpu = std::chrono::steady_clock::now();
		CHECK_CUDA_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));

		// Execution on GPUs by using threads
		std::vector<std::thread> threadVector;
		for (int i = 1; i < NUM_OF_THREADS; i++)
			threadVector.push_back(std::thread(threadCPU<NUM_OF_VECTORS>, &(threadData[i]), std::ref(shared_val)));
		threadCPU<NUM_OF_VECTORS>(&(threadData[0]), std::ref(shared_val));
		for (std::thread &th : threadVector)
			if (th.joinable())
				th.join();

		// End time for GPU part
		CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
		CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
		double time_gpu2 = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin_gpu).count();
		float time_gpu;
		CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_gpu, start, stop));
		CHECK_CUDA_ERROR(cudaEventDestroy(start));
		CHECK_CUDA_ERROR(cudaEventDestroy(stop));

		// Print results for some signals
		printResults(result);

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
	}
	catch (const std::bad_alloc &err)
	{
		std::cerr << std::endl
				  << "Memory allocation error (" << err.what() << ")!" << std::endl;
	}
	catch (const std::runtime_error &err)
	{
		std::cerr << std::endl
				  << "Runtime error (" << err.what() << ")!" << std::endl;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	RESET_CUDA_DEVICES();

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
	const int NUM_OF_STREAMS = 100;
	cudaStream_t stream[NUM_OF_STREAMS];
	for (int i = 0; i < NUM_OF_STREAMS; i++)
		CHECK_CUDA_ERROR(cudaStreamCreate(&stream[i]));

	// Allocate the device vectors
	thrust::device_vector<double> d_A[NUM_OF_STREAMS];
	for (int i = 0; i < NUM_OF_STREAMS; i++)
		d_A[i] = thrust::device_vector<double>(VECTOR_DIM);

	// Temporary storage variables for reduce operation
	thrust::device_vector<intermediateStatsData> d_temp_storage[NUM_OF_STREAMS];
	for (int i = 0; i < NUM_OF_STREAMS; i++)
		d_temp_storage[i] = thrust::device_vector<intermediateStatsData>(MAX_BLOCKS * 2);

	int turnCount = 0;
	int curr_vec = shared_val.fetch_add(1, std::memory_order_relaxed);
	int currStreamNo = 0;
	while (curr_vec < data->nr_vectors)
	{
		// Copy host input vector to device
		CHECK_CUDA_ERROR(cudaMemcpyAsync(thrust::raw_pointer_cast(d_A[currStreamNo].data()), thrust::raw_pointer_cast(data->host_vec[curr_vec]->data()),
										 data->nr_elems[curr_vec] * sizeof(T_in), cudaMemcpyHostToDevice, stream[currStreamNo]));

		// Launch CUDA Kernel
		getStatsFromData<T_in>(data->nr_elems[curr_vec], (T_in *)(thrust::raw_pointer_cast(d_A[currStreamNo].data())),
							   (intermediateStatsData *)thrust::raw_pointer_cast(d_temp_storage[currStreamNo].data()),
							   (finalStatsData *)thrust::raw_pointer_cast(d_temp_storage[currStreamNo].data()), stream[currStreamNo]);

		// Copy device output vector to host
		CHECK_CUDA_ERROR(cudaMemcpyAsync(data->result[curr_vec], thrust::raw_pointer_cast(d_temp_storage[currStreamNo].data()),
										 sizeof(finalStatsData), cudaMemcpyDeviceToHost, stream[currStreamNo]));

		turnCount++;
		if (turnCount >= data->turnCount)
		{
			CHECK_CUDA_ERROR(cudaDeviceSynchronize());
			turnCount = 0;
		}

		curr_vec = shared_val.fetch_add(1, std::memory_order_relaxed);
		currStreamNo = (currStreamNo + 1) % NUM_OF_STREAMS;
	}

	// Synchronize and destroy streams, deallocate device vectors and temporary storage
	for (int i = 0; i < NUM_OF_STREAMS; i++)
	{
		CHECK_CUDA_ERROR(cudaStreamSynchronize(stream[i]));
		CHECK_CUDA_ERROR(cudaStreamDestroy(stream[i]));
	}

	return;
}

void printResults(thrust::host_vector<finalStatsData, thrust::cuda::experimental::pinned_allocator<finalStatsData>> &result)
{
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
}
