#pragma once

#include <vector>
#include <stdexcept>
#include <iostream>
#include "../common_stuff/custom_concepts.hpp"

template <Arithmetic T>
class Dense_Matrix
{
public:
    Dense_Matrix() = default;
    Dense_Matrix(const std::vector<std::vector<T>>& init_matrix, size_t rows, size_t cols);
    Dense_Matrix(const Dense_Matrix<T>& other) = default;
    Dense_Matrix(Dense_Matrix<T>&& other) noexcept = default;
    Dense_Matrix<T>& operator=(const Dense_Matrix<T>& other) = default;
    Dense_Matrix<T>& operator=(Dense_Matrix<T>&& other) noexcept = default;

    T&             operator[]        (size_t i, size_t j);
    T              operator[]        (size_t i, size_t j)         const;
    std::vector<T> operator*         (const std::vector<T>& vec)  const;
    std::vector<T> parallel_multiply (const std::vector<T> & vec) const;
    Dense_Matrix<T> operator*        (T scalar)                   const;

    Dense_Matrix<T>  transposed() const;
    Dense_Matrix<T>& transpose_inplace();

    [[nodiscard]] std::tuple<size_t, size_t> size() const;

    void print() const;

private:
    size_t nx_ = 0;
    size_t ny_ = 0;
    std::vector<T> data_;
};

//----------------------------------------------------------------------------------------------------------------------

template <Arithmetic T>
Dense_Matrix<T>::Dense_Matrix(const std::vector<std::vector<T>>& init_matrix, size_t rows, size_t cols)
    : nx_(cols), ny_(rows), data_(rows * cols)
{
    if (init_matrix.size() != rows) [[unlikely]]
        throw std::invalid_argument("Number of rows in init matrix does not match");

    for (size_t i = 0; i < rows; ++i)
    {
        if (init_matrix[i].size() != cols) [[unlikely]]
            throw std::invalid_argument("All rows must have the same number of columns");

        for (size_t j = 0; j < cols; ++j)
            data_[i * cols + j] = init_matrix[i][j];
    }
}

template <Arithmetic T>
T& Dense_Matrix<T>::operator[](const size_t i, const size_t j)
{
    if (i >= ny_ || j >= nx_) [[unlikely]]
    {
        throw std::out_of_range("Trying to access an element outside the dense matrix");
    }
    return data_[i * nx_ + j];
}

template <Arithmetic T>
T Dense_Matrix<T>::operator[](const size_t i, const size_t j) const
{
    if (i >= ny_ || j >= nx_) [[unlikely]]
    {
        throw std::out_of_range("Trying to access an element outside the dense matrix");
    }
    return data_[i * nx_ + j];
}

template <Arithmetic T>
std::vector<T> Dense_Matrix<T>::operator*(const std::vector<T>& vec) const
{
    if (vec.size() != nx_) [[unlikely]]
        throw std::invalid_argument("Size of vector and size of matrix do not match");

    std::vector<T> ans(ny_, T(0));
    for (size_t i = 0; i < ny_; ++i)
        for (size_t j = 0; j < nx_; ++j)
            ans[i] += (*this)[i, j] * vec[j];

    return ans;
}

template <Arithmetic T>
std::vector<T> Dense_Matrix<T>::parallel_multiply(const std::vector<T>& vec) const
{
    if (vec.size() != nx_) [[unlikely]]
    {
        throw std::invalid_argument("Size of vector and size of matrix do not match");
    }

    std::vector<T> ans(ny_, T(0));
    #pragma omp parallel
    for (size_t i = 0; i < ny_; ++i)
        for (size_t j = 0; j < nx_; ++j)
            ans[i] += (*this)[i, j] * vec[j];

    return ans;
}

template <Arithmetic T>
Dense_Matrix<T> Dense_Matrix<T>::operator*(T scalar) const
{
    Dense_Matrix<T> result(*this);
    for (auto& val : result.data_)
        val *= scalar;
    return result;
}

template <Arithmetic T>
Dense_Matrix<T> Dense_Matrix<T>::transposed() const
{
    std::vector<std::vector<T>> t(nx_, std::vector<T>(ny_));
    for (size_t i = 0; i < ny_; ++i)
        for (size_t j = 0; j < nx_; ++j)
            t[j][i] = (*this)[i, j];
    return Dense_Matrix<T>(t, nx_, ny_);
}

template <Arithmetic T>
Dense_Matrix<T>& Dense_Matrix<T>::transpose_inplace()
{
    std::vector<T> t(nx_ * ny_);
    for (size_t i = 0; i < ny_; ++i)
        for (size_t j = 0; j < nx_; ++j)
            t[j * ny_ + i] = data_[i * nx_ + j];
    std::swap(nx_, ny_);
    data_ = std::move(t);
    return *this;
}

template <Arithmetic T>
std::tuple<size_t, size_t> Dense_Matrix<T>::size() const
{
    return {ny_, nx_};
}

template <Arithmetic T>
void Dense_Matrix<T>::print() const
{
    for (size_t i = 0; i < ny_; ++i)
    {
        for (size_t j = 0; j < nx_; ++j)
            std::cout << (*this)[i, j] << " ";
        std::cout << "\n";
    }
}