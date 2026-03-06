#pragma once

#include <vector>
#include <limits>
#include <iostream>
#include "../matrixes/CSR.hpp"
#include "../common_stuff/custom_concepts.hpp"
#include "../common_stuff/operators.hpp"
#include "../common_stuff/check_residual.hpp"


template <Arithmetic T, Matrix<T> M>
std::vector<T> Jacoby_Classic(const M& A,
                      const std::vector<T>& b,
                      const T max_residual    = std::numeric_limits<T>::epsilon() * 1e3,
                      const size_t max_iterations = 100)
{
    const auto [rows, cols] = A.size();

    for (size_t i = 0; i < rows; ++i)
    {
        const T diag = std::abs(A[i, i]);
        T off_diag_sum = T(0);

        for (size_t j = 0; j < cols; ++j)
            if (i != j) [[likely]] off_diag_sum += std::abs(A[i, j]);

        if (diag <= off_diag_sum) [[unlikely]]
        {
            std::cout << "Matrix does not have diagonal dominance, Jacobi method may be unstable.\n";
            break;
        }
    }

    std::vector<T> diag(rows);
    for (size_t i = 0; i < rows; ++i)
        diag[i] = A[i, i];

    std::vector<T> ans(rows, T(0));
    std::vector<T> ans_new(rows);

    for (size_t iteration = 0; iteration < max_iterations; ++iteration)
    {
        for (size_t i = 0; i < rows; ++i)
        {
            T sigma = T(0);
            for (size_t j = 0; j < cols; ++j)
            {
                if (i != j) [[likely]] sigma += A[i, j] * ans[j];
            }

            ans_new[i] = (b[i] - sigma) / diag[i];
        }

        ans = ans_new;

        if (check_residual(A, ans, b, max_residual)) [[unlikely]] break;
    }

    return ans;
}