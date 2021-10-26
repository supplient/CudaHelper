#include <iostream>

#define CUDA_HELPER_NAMESPACE cuh
#include "cuda_helpers.h"

using namespace std;


__global__ void kernel__Test1(int x) {
	printf("x: %i\n", x);
}

__global__ void kernel__Test2(int x, char y) {
	printf("x: %i, y: %c\n", x, y);
}

template<typename T>
void PrintArray(T* arr, size_t n) {
	for (size_t i = 0; i < n; i++) {
		cout << arr[i];
		if (i != n - 1)
			cout << ",";
	}
}

namespace kernel {
	__global__ void Test_KERNEL_ARGS_SIMPLE() {
	}
}
void Test_macro() {
	kernel::Test_KERNEL_ARGS_SIMPLE KERNEL_ARGS_SIMPLE(3, 2) ();
}

void Test_alias() {
	size_t N = 3;
	auto* d_x = cuh::Malloc<int>(N);
	cuh::Fill(d_x, 2, 3);
	int* x = new int[N];
	cuh::Copy(x, d_x, N);
	for (size_t i = 0; i < N; i++)
		assert(x[i] == 2);

	cuh::SetEntry(d_x, 1, 8);
	for (size_t i = 0; i < N; i++) {
		if(i != 1)
			assert(cuh::GetEntry(d_x, i) == 2);
		else
			assert(cuh::GetEntry(d_x, i) == 8);
	}
}

int main() {
	Test_macro();
	Test_alias();

	CheckCuda(cudaDeviceSynchronize());
	return 0;
}
