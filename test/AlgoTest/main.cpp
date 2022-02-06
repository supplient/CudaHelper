#define CUDA_HELPER_NAMESPACE cuh
#include "cuda_helpers.h"

#include <random>
#include <iostream>
#include <functional>
#include <numeric>
using namespace std;


void Test_segOffset2segIndex() {
	constexpr uint32_t minSegSize = 0;
	constexpr uint32_t maxSegSize = 500;
	size_t n = 1000;
	constexpr uint32_t seed = 0x4fc2eec3;

	default_random_engine generator(seed);
	uniform_int_distribution<uint32_t> dist(minSegSize, maxSegSize);
	auto rint = bind(dist, generator);

	vector<uint32_t> A(n); { 
		for (size_t i = 0; i < n; i++) 
			A[i] = rint(); 
	}
	uint32_t m = accumulate(A.begin(), A.end(), 0);
	vector<uint32_t> B(m); {
		size_t bi = 0;
		for (uint32_t ai = 0; ai < A.size(); ai++) {
			uint32_t count = A[ai];
			while (count > 0) {
				B[bi] = ai;
				bi++;
				count--;
			}
		}
		assert(bi == m);
	}
	partial_sum(A.begin(), A.end(), A.begin());
	
	vector<uint32_t> resB(m);

	cuh::TempStorage<uint32_t> d_A;
	cuh::TempStorage<uint32_t> d_B;
	d_A.CheckAndAlloc(n);
	d_B.CheckAndAlloc(m);
	cuh::Copy<uint32_t>(d_A, A.data(), A.size());

	size_t temp_bytesize = cuh::algo::async::segOffset2segIndex<uint32_t>::GetTempBytesize(m);
	void* d_tmp;
	CheckCuda(cudaMalloc(&d_tmp, temp_bytesize));

	cudaStream_t stream;
	CheckCuda(cudaStreamCreate(&stream));

	cuh::algo::async::segOffset2segIndex<uint32_t>::Work(d_B, m, d_A, n, d_tmp, temp_bytesize, stream, 128);
	
	CheckCuda(cudaStreamDestroy(stream));
	cuh::Copy<uint32_t>(resB.data(), d_B, m);
	for (size_t i = 0; i < m; i++)
		assert(resB[i] == B[i]);
}



int main() {
	Test_segOffset2segIndex();

	return 0;
}
