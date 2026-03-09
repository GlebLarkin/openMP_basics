#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <ranges>
#include "../matrixes/Dense.hpp"
#include "../matrixes/CSR.hpp"


template <Arithmetic T>
T norm(const std::vector<T>& vec)
{
    T sum = std::inner_product(vec.begin(), vec.end(), vec.begin(), T(0));
    return std::sqrt(sum);
}

template <Arithmetic T>
std::vector<T> operator+(const std::vector<T>& lhs, const std::vector<T>& rhs)
{
    if (lhs.size() != rhs.size()) [[unlikely]]
    {
        throw std::runtime_error("Can't add vectors with different sizes");
    }

    std::vector<T> ans(lhs.size());
    std::ranges::transform(lhs, rhs, ans.begin(), std::plus<T>{});
    return ans;
}

template <Arithmetic T>
std::vector<T> operator-(const std::vector<T>& lhs, const std::vector<T>& rhs)
{
    if (lhs.size() != rhs.size()) [[unlikely]]
        throw std::runtime_error("Can't subtract vectors with different sizes");

    std::vector<T> ans(lhs.size());
    std::ranges::transform(lhs, rhs, ans.begin(), std::minus<T>{});
    return ans;
}

template <Arithmetic T>
T operator*(const std::vector<T>& lhs, const std::vector<T>& rhs)
{
    return std::inner_product(lhs.begin(), lhs.end(), rhs.begin(), T(0));
}

template <Arithmetic T>
std::vector<T> operator*(const std::vector<T>& vec, T scalar)
{
    std::vector<T> ans(vec.size());
    std::ranges::transform(vec, ans.begin(), [&](T val) { return val * scalar; });
    return ans;
}

template <Arithmetic T>
std::vector<T> operator*(T scalar, const std::vector<T>& vec)
{
    return vec * scalar;
}