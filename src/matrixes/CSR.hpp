//
// Created by gleblarkin on 06.03.2026.
//

#pragma once

#include <map>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include "../common_stuff/custom_concepts.hpp"

template <Arithmetic T>
class CSR_Matrix
{
public:
	CSR_Matrix(const std::map<std::tuple<size_t, size_t>, T> & init_matrix,
					 size_t y_size,
					 size_t x_size);

	T operator[] (size_t i, size_t j) const;

	std::vector<T> operator*                (const std::vector<T> & vec) const;
	std::vector<T> unparallel_multiply      (const std::vector<T> & vec) const;
	std::vector<T> parallel_multiply_static (const std::vector<T> & vec) const;
	std::vector<T> parallel_multiply_dynamic(const std::vector<T> & vec) const;

	[[nodiscard]] std::tuple<size_t, size_t> size() const;
    [[nodiscard]] const std::vector<T>&      get_values() const { return values; }
    [[nodiscard]] const std::vector<size_t>& get_cols()   const { return cols;   }
    [[nodiscard]] const std::vector<size_t>& get_rows()   const { return rows;   }

private:
	std::vector<T> values;
	std::vector<size_t> cols;
	std::vector<size_t> rows;
	std::tuple<size_t, size_t> sizes;
};

//-----------------------------------------------------------

template <Arithmetic T>
CSR_Matrix<T>::CSR_Matrix(const std::map<std::tuple<size_t, size_t>, T> & init_matrix,
                          size_t y_size,
                          size_t x_size)
    : sizes(std::make_tuple(y_size, x_size))
{
    size_t init_matrix_size = init_matrix.size();
    values.reserve(init_matrix_size);
    cols.reserve(init_matrix_size);
    rows.resize(y_size + 1, 0);

    for (const auto & element : init_matrix)
    {
        values.push_back(element.second);
        cols.push_back(std::get<1>(element.first));
        ++rows[std::get<0>(element.first) + 1];
    }

    for (size_t i = 1; i <= y_size; ++i)
    {
        rows[i] += rows[i - 1];
    }
}


template <Arithmetic T>
T CSR_Matrix<T>::operator[] (const size_t i, const size_t j) const
{
    if (i >= std::get<0>(sizes) || j >= std::get<1>(sizes)) [[unlikely]]
    {
        throw std::out_of_range("Trying to get index out of matrix size");
    }

    else [[likely]]
    {
        const size_t value_row_begin = rows[i];
        const size_t value_row_end = rows[i + 1];

        for (size_t k = value_row_begin; k < value_row_end; ++k)
        {
            if (cols[k] == j) return values[k];
        }

        return T(0);
    }
}

template<Arithmetic T>
std::vector<T> CSR_Matrix<T>::operator*(const std::vector<T> &vec) const
{
    const size_t vec_size = vec.size();
    const auto [y_size, x_size] = sizes;

    if (vec_size != x_size) [[unlikely]]
    {
        throw std::invalid_argument("Size of vector and size of matrix do not match");
    }

    else [[likely]]
    {
        std::vector<T> ans(y_size, 0);

        for (size_t i = 0; i < y_size; ++i)
        {
            const size_t row_start = rows[i];
            const size_t row_end = rows[i + 1];

            for (size_t j = row_start; j < row_end; ++j)
            {
                ans[i] += values[j] * vec[cols[j]];
            }
        }

        return ans;
    }
}

template <Arithmetic T>
std::vector<T> CSR_Matrix<T>::unparallel_multiply(const std::vector<T> &vec) const
{
    return operator*(vec);
}

template<Arithmetic T>
std::vector<T> CSR_Matrix<T>::parallel_multiply_static(const std::vector<T> &vec) const
{
    const size_t vec_size = vec.size();
    const auto [y_size, x_size] = sizes;

    if (vec_size != x_size) [[unlikely]]
    {
        throw std::invalid_argument("Size of vector and size of matrix do not match");
    }

    else [[likely]]
    {
        std::vector<T> ans(y_size, 0);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < y_size; ++i)
        {
            const size_t row_start = rows[i];
            const size_t row_end = rows[i + 1];

            for (size_t j = row_start; j < row_end; ++j)
            {
                ans[i] += values[j] * vec[cols[j]];
            }
        }

        return ans;
    }
}

template<Arithmetic T>
std::vector<T> CSR_Matrix<T>::parallel_multiply_dynamic(const std::vector<T> &vec) const
{
    const size_t vec_size = vec.size();
    const auto [y_size, x_size] = sizes;

    if (vec_size != x_size) [[unlikely]]
    {
        throw std::invalid_argument("Size of vector and size of matrix do not match");
    }

    else [[likely]]
    {
        std::vector<T> ans(y_size, 0);

        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < y_size; ++i)
        {
            const size_t row_start = rows[i];
            const size_t row_end = rows[i + 1];

            for (size_t j = row_start; j < row_end; ++j)
            {
                ans[i] += values[j] * vec[cols[j]];
            }
        }

        return ans;
    }
}


template <Arithmetic T>
std::tuple<size_t, size_t> CSR_Matrix<T>::size() const
{
    return sizes;
}