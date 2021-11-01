#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <utility>
#include <iostream>

#include "cuda_helper_namespace.h"
#include "cuda_helper_intellisense.h"


// ========== Macros =============
#define LINEAR_THREAD_ID (blockDim.x * blockIdx.x + threadIdx.x)

#ifndef __CUDA_ARCH__
// host code
	#define CheckCuda(x) {\
		auto err = x;\
		if(err != cudaSuccess) {\
			printf("[CUDA Error][%s Line %i]\n\t[%s] %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err)); \
			throw err;\
		}\
	}
#else
// device code
	#define CheckCuda(x) {\
		auto err = x;\
		if(err != cudaSuccess) {\
			printf("[CUDA Error][%s Line %i]\n\t[%s] %s\n", __FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err)); \
			__trap();\
		}\
	}
#endif

#define CalBlockNum(total, per) ( (total - 1) / per + 1 )
// thread number & cuda block size -> grid size & block size
// grid size = thread number / cuda block size
// block size = cuda block size
#define TNBS2GSBS(threadNum, cudaBlockSize) CalBlockNum(threadNum, cudaBlockSize), cudaBlockSize
#define KERNEL_ARGS_SIMPLE(threadNum, cudaBlockSize) KERNEL_ARGS(TNBS2GSBS((unsigned int)(threadNum), (unsigned int)(cudaBlockSize)))





CUDA_HELPER_NAMESPACE_BEGIN
// ========== Kernels =============
namespace kernel {
	template<typename T>
	__global__ void Fill(T* arr, T val, size_t n) {
		auto ti = LINEAR_THREAD_ID;
		if (ti >= n)
			return;
		arr[ti] = val;
	}

	template<typename T_Data, typename T_Func>
	__global__ void ForEach(T_Data* dst, T_Data* src, T_Func func, size_t n) {
		auto ti = LINEAR_THREAD_ID;
		if (ti >= n)
			return;
		dst[ti] = func(src[ti]);
	}
}


// ========== Functions =============
template<typename T>
T* Malloc(size_t eleNum) {
	T* ptr = nullptr;
	CheckCuda(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * eleNum));
	return ptr;
}

template<typename T>
void Fill(T* d_arr, const T& val, size_t n, uint32_t cudaBlockSize=32) {
	kernel::Fill KERNEL_ARGS_SIMPLE(n, cudaBlockSize) (d_arr, val, n);
}

template<typename T>
void Copy(T* dst, const T* src, size_t n, cudaMemcpyKind kind = cudaMemcpyDefault) {
	CheckCuda(cudaMemcpy(dst, src, sizeof(T) * n, kind));
}

template<typename T>
__host__ __device__ void CopyAsync(T* dst, const T* src, size_t n, cudaStream_t stream = 0, cudaMemcpyKind kind = cudaMemcpyDefault) {
	CheckCuda(cudaMemcpyAsync(dst, src, sizeof(T) * n, kind, stream));
}

template<typename T>
T GetEntry(T* d_arr, size_t i) {
	T res;
	Copy(&res, d_arr + i, 1, cudaMemcpyDeviceToHost);
	return std::move(res);
};
template<typename T>
void SetEntry(T* d_arr, size_t i, const T& val) {
	Copy(d_arr + i, &val, 1, cudaMemcpyHostToDevice);
};





// ========== Experimental Functions =============
/*
template<typename... T_KernelArgs>
void LaunchKernel_usingAPI(void (*kernelFunc)(T_KernelArgs...),
	unsigned int blockNum, unsigned int threadNum, size_t sharedMemBtyes, cudaStream_t stream,
	T_KernelArgs... kernelArgs
) {
	// TODO: kernelArgs may ref instead of copy
	void* args[sizeof...(kernelArgs)] = { reinterpret_cast<void*>(&kernelArgs)... };
	CheckCuda(cudaLaunchKernel(reinterpret_cast<const void*>(kernelFunc),
		dim3(blockNum, 1, 1), dim3(threadNum, 1, 1),
		args,
		sharedMemBtyes, stream
	));
}

template<>
void LaunchKernel_usingAPI<>(void (*kernelFunc)(void),
	unsigned int blockNum, unsigned int threadNum, size_t sharedMemBtyes, cudaStream_t stream
) {
	CheckCuda(cudaLaunchKernel(reinterpret_cast<void*>(kernelFunc),
		dim3(blockNum, 1, 1), dim3(threadNum, 1, 1),
		nullptr,
		sharedMemBtyes, stream
	));
}
*/

CUDA_HELPER_NAMESPACE_END
