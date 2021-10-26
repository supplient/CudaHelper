#pragma once

#include <type_traits>
#include <utility>

#include "cuda_helper_namespace.h"

CUDA_HELPER_NAMESPACE_BEGIN

// Source: https://stackoverflow.com/a/48368508/17132546

// Entry template
//	extract the lambda's operaor() function signature
template <class F, class T=F>
struct lambda_traits: lambda_traits<decltype(&std::remove_reference<F>::type::operator()), F>
{};

// For mutable lambda, See https://en.cppreference.com/w/cpp/language/lambda
//	mutable lambda's operator() is not const,
//	not mutable lambda's operator() is const
template <typename rF, typename F, typename R, typename... Args>
struct lambda_traits<R(rF::*)(Args...), F>: lambda_traits<R(rF::*)(Args...) const, F>
{};

// Workhorse
//	every lambda has an unique signature
//	=> lambda_traits will be specialized for every lambda, even if their function signature are the same.
template <typename rF, typename F, typename R, typename... Args>
struct lambda_traits<R(rF::*)(Args...) const, F> {
	static auto cify(F&& f) {
		static rF fn = std::forward<F>(f);
		return [](Args... args) {
			return fn(std::forward<Args>(args)...);
		};
	}
};

// Wrapper, for convenience
template <class F>
inline auto lam2fp(F&& f) {
	return lambda_traits<F>::cify(std::forward<F>(f));
}

CUDA_HELPER_NAMESPACE_END
