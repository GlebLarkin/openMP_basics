//
// Created by gleblarkin on 06.03.2026.
//
// benchmark.cpp

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <tuple>
#include <random>
#include <chrono>
#include <filesystem>

#include "../src/matrixes/CSR.hpp"
#include "../src/matrixes/Dense.hpp"
#include "../src/common_stuff/operators.hpp"
#include "../src/methods/Jacoby.hpp"
#include "../src/methods/JacobyClassic.hpp"
#include "../src/methods/GaussSeidel.hpp"

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;

double elapsed_ms(Clock::time_point t0, Clock::time_point t1)
{
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ── generators ────────────────────────────────────────────────────────────────

CSR_Matrix<double> generate_sparse_dd(size_t N, size_t nnz_per_row, unsigned seed = 42)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> col_dist(0, N - 1);
    std::uniform_real_distribution<double> val_dist(0.1, 1.0);

    std::map<std::tuple<size_t, size_t>, double> entries;

    for (size_t i = 0; i < N; ++i)
    {
        double off_sum = 0.0;
        size_t filled  = 0;

        while (filled < nnz_per_row)
        {
            size_t j = col_dist(rng);
            if (j == i || entries.count({i, j})) continue;
            double v = val_dist(rng);
            entries[{i, j}] = v;
            off_sum += v;
            ++filled;
        }
        entries[{i, i}] = off_sum + val_dist(rng);
    }

    return CSR_Matrix<double>(entries, N, N);
}

// build a Dense_Matrix from the same sparsity pattern for fair comparison
Dense_Matrix<double> csr_to_dense(const CSR_Matrix<double>& A)
{
    const auto [rows, cols] = A.size();
    std::vector<std::vector<double>> data(rows, std::vector<double>(cols, 0.0));

    const auto& rv = A.get_rows();
    const auto& cv = A.get_cols();
    const auto& vv = A.get_values();

    for (size_t i = 0; i < rows; ++i)
        for (size_t k = rv[i]; k < rv[i + 1]; ++k)
            data[i][cv[k]] = vv[k];

    return Dense_Matrix<double>(data, rows, cols);
}

std::vector<double> generate_rhs(size_t N, unsigned seed = 7)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> d(0.5, 2.0);
    std::vector<double> b(N);
    for (auto& v : b) v = d(rng);
    return b;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main()
{
    fs::create_directories("../data");

    constexpr double tol         = 1e-6;
    constexpr size_t max_iter    = 1000;
    constexpr size_t nnz         = 10;
    constexpr int    matvec_reps = 5000;

    const std::vector<size_t> sizes = { 200, 500, 1000, 2000, 3000, 4000 };

    std::ofstream csv("../data/benchmark.csv");
    csv << "N,"
        << "ms_jacobi_seq,ms_jacobi_par,ms_gs_par,"
        << "ms_csr_seq,ms_csr_static,ms_csr_dynamic,"
        << "ms_dense_seq,ms_dense_par\n";

    for (size_t N : sizes)
    {
        std::cout << "N = " << N << " ..." << std::flush;

        auto A_csr   = generate_sparse_dd(N, nnz);
        auto A_dense = csr_to_dense(A_csr);
        auto b       = generate_rhs(N);

        // ── solvers ───────────────────────────────────────────────────────────
        auto bench_solver = [&](auto fn) -> double {
            auto t0 = Clock::now();
            fn(A_csr, b, tol, max_iter);
            return elapsed_ms(t0, Clock::now());
        };

        double ms_jac_seq = bench_solver([](auto& A, auto& b, auto tol, auto mi){ return Jacoby_Classic(A, b, tol, mi); });
        double ms_jac_par = bench_solver([](auto& A, auto& b, auto tol, auto mi){ return Jacoby(A, b, tol, mi); });
        double ms_gs_par  = bench_solver([](auto& A, auto& b, auto tol, auto mi){ return Gauss_Seidel_method(A, b, tol, mi); });

        // ── CSR matvec ────────────────────────────────────────────────────────
        auto bench_matvec = [&](auto fn) -> double {
            auto t0 = Clock::now();
            for (int r = 0; r < matvec_reps; ++r) { volatile auto res = fn(b); }
            return elapsed_ms(t0, Clock::now()) / matvec_reps;
        };

        double ms_csr_seq = bench_matvec([&](const auto& v){ return A_csr.unparallel_multiply(v); });
        double ms_csr_st  = bench_matvec([&](const auto& v){ return A_csr.parallel_multiply_static(v); });
        double ms_csr_dyn = bench_matvec([&](const auto& v){ return A_csr.parallel_multiply_dynamic(v); });

        // ── Dense matvec ──────────────────────────────────────────────────────
        double ms_dense_seq = bench_matvec([&](const auto& v){ return A_dense * v; });
        double ms_dense_par = bench_matvec([&](const auto& v){ return A_dense.parallel_multiply(v); });

        csv << N << ','
            << ms_jac_seq << ',' << ms_jac_par << ',' << ms_gs_par  << ','
            << ms_csr_seq << ',' << ms_csr_st  << ',' << ms_csr_dyn << ','
            << ms_dense_seq << ',' << ms_dense_par << '\n';

        std::cout << " jac=" << ms_jac_seq << "/" << ms_jac_par
                  << " gs="  << ms_gs_par
                  << " csr=" << ms_csr_seq << "/" << ms_csr_st << "/" << ms_csr_dyn
                  << " dense=" << ms_dense_seq << "/" << ms_dense_par << " ms\n";
    }

    std::cout << "\nsaved: ../data/benchmark.csv\n";
    return 0;
}