#pragma once

#include "cuda_helper_alias.h"

CUDA_HELPER_NAMESPACE_BEGIN

template<typename T_Data>
class CudaDevMem {
public:
	CudaDevMem(const CudaDevMem&) = delete;
	CudaDevMem& operator=(const CudaDevMem&) = delete;

public:
	CudaDevMem(size_t n):n(n) {
		CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(T_Data) * n)));
	}
	CudaDevMem(CudaDevMem&& b) {
		this->d_data = b.d_data;
		this->n = b.n;
		b.d_data = nullptr;
		b.n = 0;
	}
	~CudaDevMem() {
		CheckCuda(cudaFree(reinterpret_cast<void*>(d_data)));
		d_data = nullptr;
		n = 0;
	}

public:
	T_Data* GetData()const { return d_data; }
	size_t  GetN()   const { return n;      }

private:
	T_Data* d_data = nullptr;
	size_t n;
};

CUDA_HELPER_NAMESPACE_END
