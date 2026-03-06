//
// Created by gleblarkin on 06.03.2026.
//

#pragma once

#include <limits>
#include "custom_concepts.hpp"
#include "operators.hpp"
#include "../matrixes/CSR.hpp"

template <Arithmetic T, Matrix<T> M>
bool check_residual(const M &             A,
                    const std::vector<T>& x,
                    const std::vector<T>& b,
                    T                     r_max)
{
    return norm(A * x - b) < r_max;
}

template <Arithmetic T>
bool check_residual(const CSR_Matrix<T>&  A,
                    const std::vector<T>& x,
                    const std::vector<T>& b,
                    T                     r_max)
{
    return norm(A.parallel_multiply_dynamic(x) - b) < r_max;
}
