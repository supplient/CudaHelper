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

/*
For example----
	=> segOffset: {3,6,7,10}
	<= segIndex:  {0,0,0,1,1,1,2,3,3,3}

Usage----
	size_t temp_bytesize = cuh::algo::async::segOffset2segIndex<uint32_t>::GetTempBytesize(m);
	void* d_tmp;
	cudaMalloc(&d_tmp, temp_bytesize);
	cuh::algo::async::segOffset2segIndex<uint32_t>::Work(d_out, outputSize, d_in, inputSize, d_tmp, temp_bytesize);
*/
template<typename T_Data>
class segOffset2segIndex {
public:

	/// <summary>
	/// 
	/// </summary>
	/// <param name="outputSize">: should equal with the segIndex's array size.</param>
	/// <returns>Temporary memory size in bytes</returns>
	static size_t GetTempBytesize(size_t outputSize) {
		size_t scan_bytesize;
		{
			T_Data* p_useless = nullptr;

			CUB_WRAP(CheckCuda(cub::DeviceScan::InclusiveSum(nullptr, scan_bytesize, p_useless, p_useless, (int)outputSize)));
		}
		return scan_bytesize;
	}

	/// <summary>
	/// 
	/// </summary>
	/// <param name="d_out">: segIndex. Start by 0. e.g. {0,0,0,1,1,1,2,3,3,3}</param>
	/// <param name="outputSize">: should equal with d_in[inputSize-1].</param>
	/// <param name="d_in">: segOffset. The i-th segment is [d_in[i-1], d_in[i]-1). e.g. {3,6,7,10}</param>
	/// <param name="inputSize">: the number of segment.</param>
	/// <param name="d_temp">: device accessible temporary memory.</param>
	/// <param name="tempBytesize">: should be gained by GetTempBytesize.</param>
	/// <param name="stream">: cuda stream to work on.</param>
	/// <param name="cudaBlockSize">: how many threads the cuda block should include.</param>
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
