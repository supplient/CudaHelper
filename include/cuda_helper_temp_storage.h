#pragma once

#include "cuda_helper_alias.h"

CUDA_HELPER_NAMESPACE_BEGIN

template<typename T>
class TempStorage {
public:
	TempStorage(float expandFactor = 1.5f): expandFactor(expandFactor) {}
	~TempStorage() { if (d_mem) CheckCuda(cudaFree(d_mem)); }
	
	// Return if there is enough space
	bool Check(size_t should) { return memSize >= should; }
	// Alloc at least newSize's memory
	void Alloc(size_t newSize) {
		if (newSize <= memSize)
			return;
		memSize = static_cast<size_t>(static_cast<float>(newSize) * expandFactor);
		if (memSize < newSize)
			memSize = newSize;

		if (d_mem)
			CheckCuda(cudaFree(d_mem));
		d_mem = cuh::Malloc<T>(memSize);
	}
	void CheckAndAlloc(size_t should) { if (!Check(should)) Alloc(should); }

	T* Get() { return d_mem; }
	operator T* () { return Get(); }

private:
	T* d_mem = nullptr;
	size_t memSize = 0;
	float expandFactor;
};

CUDA_HELPER_NAMESPACE_END
