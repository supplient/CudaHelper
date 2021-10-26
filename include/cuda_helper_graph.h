#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <assert.h>

#include "lam2fp.h"
#include "cuda_helper_alias.h"

CUDA_HELPER_NAMESPACE_BEGIN

class cuda_graph {
public:
	static cudaGraph_t Create(unsigned int flags = 0) {
		cudaGraph_t graph;
		CheckCuda(cudaGraphCreate(&graph, flags));
		return graph;
	}

	/// <summary>
	/// Build dependencies between nodes
	/// </summary>
	/// <param name="graph"></param>
	/// <param name="args">from -> to (src -> dst)</param>
	static void Link(const cudaGraph_t& graph,
		std::initializer_list<cudaGraphNode_t> args
	) {
		// Args must be given in pair {from -> to}
		assert(args.size() % 2 == 0);
		size_t n = args.size() / 2;
		std::vector<cudaGraphNode_t> src(n);
		std::vector<cudaGraphNode_t> dst(n);

		size_t i = 0;
		for (auto node : args) {
			switch (i % 2) {
			case 0: src[i / 2] = node; break;
			case 1: dst[i / 2] = node; break;
			}
			i++;
		}

		CheckCuda(cudaGraphAddDependencies(graph,
			src.data(), dst.data(), n
		));
	}

	static cudaGraphExec_t Instantiate(const cudaGraph_t& graph) {
		cudaGraphExec_t graphExec;
		cudaGraphNode_t errorNode = nullptr;
		constexpr size_t LOG_BUFFER_SIZE = 16000;
		char logBuffer[LOG_BUFFER_SIZE];

		CheckCuda(cudaGraphInstantiate(&graphExec, graph, &errorNode, logBuffer, LOG_BUFFER_SIZE));
		if (errorNode != nullptr) {
			std::cerr << "Cuda Graph Instantiate Failed:\n";
			std::cerr << logBuffer << std::endl;
			// TODO: output errorNode's information
			throw "Cuda Graph Instantiate Failed.";
		}

		return graphExec;
	}

	class node {
	public:
		static cudaGraphNode_t child(cudaGraph_t graph,
			cudaGraph_t child
		) {
			cudaGraphNode_t node;
			CheckCuda(cudaGraphAddChildGraphNode(&node, graph,
				nullptr, 0,
				child
			));
			return node;
		}

		static cudaGraphNode_t event_record(cudaGraph_t graph,
			cudaEvent_t event
		) {
			cudaGraphNode_t node;
			CheckCuda(cudaGraphAddEventRecordNode(&node, graph,
				nullptr, 0,
				event
			));
			return node;
		}

		static cudaGraphNode_t event_wait(cudaGraph_t graph,
			cudaEvent_t event
		) {
			cudaGraphNode_t node;
			CheckCuda(cudaGraphAddEventWaitNode(&node, graph,
				nullptr, 0,
				event
			));
			return node;
		}

		template<typename T_Data>
		static cudaGraphNode_t memcpy(cudaGraph_t graph,
			T_Data* dst, const T_Data* src, size_t n,
			cudaMemcpyKind kind = cudaMemcpyDefault
		) {
			cudaGraphNode_t node;

			CheckCuda(cudaGraphAddMemcpyNode1D(&node, graph,
				nullptr, 0,
				reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src),
				n * sizeof(T_Data), kind
			));
			return node;
		}


		template<typename T_HostFunc>
		static cudaGraphNode_t host(cudaGraph_t graph,
			T_HostFunc hostFunc, void* userData
		) {
			cudaGraphNode_t node;
			cudaHostNodeParams params;
			params.fn = lam2fp(hostFunc);
			params.userData = userData;

			CheckCuda(cudaGraphAddHostNode(&node, graph,
				nullptr, 0,
				&params
			));
			return node;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <typeparam name="T_KernelFunc"></typeparam>
		/// <typeparam name="...T_KernelArgs"></typeparam>
		/// <param name="graph"></param>
		/// <param name="blockNum"></param>
		/// <param name="threadNum"></param>
		/// <param name="sharedMemBytes"></param>
		/// <param name="kernelFunc"></param>
		/// <param name="...kernelArgs">: args must be reference parameter, and the parameters must be alive when the graph is launched.</param>
		/// <returns></returns>
		template<typename T_KernelFunc, typename... T_KernelArgs>
		static cudaGraphNode_t kernel(cudaGraph_t graph,
			size_t blockNum, size_t threadNum, unsigned int sharedMemBytes,
			T_KernelFunc kernelFunc, T_KernelArgs&... kernelArgs
		) {
			cudaGraphNode_t node;
			cudaKernelNodeParams params;
			params.gridDim = dim3(static_cast<unsigned int>(blockNum), 1, 1);
			params.blockDim = dim3(static_cast<unsigned int>(threadNum), 1, 1);
			params.sharedMemBytes = sharedMemBytes;
			params.extra = nullptr;

			params.func = reinterpret_cast<void*>(kernelFunc);
			void* args[sizeof...(kernelArgs)] = { reinterpret_cast<void*>(&kernelArgs)... };
			params.kernelParams = args;

			CheckCuda(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params));
			return node;
		}

	};
};

CUDA_HELPER_NAMESPACE_END
