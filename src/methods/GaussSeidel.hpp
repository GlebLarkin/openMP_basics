//
// Created by gleblarkin on 06.03.2026.
//

#pragma once

#include "../matrixes/CSR.hpp"
#include "../common_stuff/custom_concepts.hpp"
#include "../common_stuff/operators.hpp"
#include "../common_stuff/check_residual.hpp"


template <Arithmetic T, Matrix<T> M>
std::vector<T> Gauss_Seidel_method(const M& A,
                                   const std::vector<T>& b,
                                   const T max_residual = std::numeric_limits<T>::epsilon() * 1e3,
                                   const size_t max_iterations = 100)
{
    const auto [rows, cols] = A.size();
    for (size_t i = 0; i < rows; ++i)
    {
        if (A[i, i] == 0) [[unlikely]]
        {
            throw std::invalid_argument("Zero element is in the matrix. Gauss Seidel method can not be used.");
        }
    }

    std::vector<T> ans(b.size());

    size_t iteration = 0;
    while (!check_residual(A, ans, b, max_residual) && iteration < max_iterations)
    {
        for (size_t k = 0; k < rows; ++k)
        {
            T sum1 = 0, sum2 = 0;

            for (size_t j = k + 1; j < cols; ++j) { sum1 += A[k, j] * ans[j]; }
            for (size_t j = 0; j < k; ++j) { sum2 += A[k, j] * ans[j]; }

            ans[k] = (b[k] - sum1 - sum2) / A[k, k];
        }
        ++iteration;
    }
    return ans;
}