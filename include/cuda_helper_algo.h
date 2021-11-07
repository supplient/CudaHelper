#pragma once

#include "cuda_helper_alias.h"


CUDA_HELPER_NAMESPACE_BEGIN
namespace algo {

namespace async {

namespace kernel_segOffset2segIndex {
template<typename T_Data>
__global__ void SetMark(T_Data* out, const T_Data* in, size_t n) {
	auto ti = LINEAR_THREAD_ID;
	if (ti >= n-1)
		return;
	out[in[ti]] = 1;
}
}

template<typename T_Data>
class segOffset2segIndex {
public:

	static size_t GetTempBytesize(size_t outputSize) {
		size_t scan_bytesize;
		{
			T_Data* p_useless = nullptr;

			CUB_WRAP(CheckCuda(cub::DeviceScan::InclusiveSum(nullptr, scan_bytesize, p_useless, p_useless, (int)outputSize)));
		}
		return scan_bytesize;
	}

	static void Work(
		T_Data* d_out, size_t outputSize,
		T_Data* d_in, size_t inputSize,
		void* d_temp, size_t tempBytesize,
		cudaStream_t stream = (cudaStream_t)0,
		uint32_t cudaBlockSize = 64
	) {
		CheckCuda(cudaMemsetAsync(d_out, 0, sizeof(T_Data) * outputSize, stream));

		kernel_segOffset2segIndex::SetMark KERNEL_ARGS_S3(inputSize, cudaBlockSize, stream) (
			d_out, d_in, inputSize
			);

		CUB_WRAP(CheckCuda(cub::DeviceScan::InclusiveSum(d_temp, tempBytesize, d_out, d_out, (int)outputSize, stream)));
	}
};

}


}
CUDA_HELPER_NAMESPACE_END
