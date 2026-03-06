//
// Created by gleblarkin on 06.03.2026.
//

#pragma once
#include <type_traits>
#include <vector>
#include <concepts>

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template<typename M, typename T>
concept Matrix = requires(M mat, size_t i, size_t j, const std::vector<T>& vec) {
    { mat[i, j]  }            -> std::convertible_to<T>;
    { mat.size() }            -> std::convertible_to<std::tuple<size_t, size_t>>;
    { mat * vec  }            -> std::same_as<std::vector<T>>;
};
